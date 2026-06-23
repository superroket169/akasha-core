use akasha_core::nn::akasha_model::AkashaModel;
use std::collections::HashMap;
use std::io::{self, Write};
use std::sync::Arc;
use wilupgu::context::WgpuContext;
use wilupgu::tensor::Tensor;

const PAD_WORD: &str = "<pad>";

fn build_vocab() -> (HashMap<String, u32>, HashMap<u32, String>, Vec<u32>) {
    let sentence_words = ["Akasha", "is", "a", "custom", "language", "model."];

    let mut vocab = HashMap::new();
    let mut inverse_vocab = HashMap::new();
    for (i, w) in sentence_words.iter().enumerate() {
        vocab.insert(w.to_string(), i as u32);
        inverse_vocab.insert(i as u32, w.to_string());
    }
    let pad_id = sentence_words.len() as u32;
    vocab.insert(PAD_WORD.to_string(), pad_id);
    inverse_vocab.insert(pad_id, PAD_WORD.to_string());

    let sentence_tokens: Vec<u32> = (0..sentence_words.len() as u32).collect();
    (vocab, inverse_vocab, sentence_tokens)
}

fn sgd_step(weight: &Tensor, grad: &Tensor, lr: f32) {
    let w: Vec<f32> = weight.to_cpu();
    let g: Vec<f32> = grad.to_cpu();
    let updated: Vec<f32> = w
        .iter()
        .zip(g.iter())
        .map(|(wi, gi)| wi - lr * gi)
        .collect();
    weight.copy_from_cpu(&updated);
}

fn sgd_update_all(model: &AkashaModel, lr: f32) {
    sgd_step(&model.embedding.table, &model.embedding.grad_table, lr);
    for block in &model.layers {
        sgd_step(&block.norm_1.weight, &block.norm_1.grad_weight, lr);
        sgd_step(&block.q_proj.weight, &block.q_proj.grad_weight, lr);
        sgd_step(&block.k_proj.weight, &block.k_proj.grad_weight, lr);
        sgd_step(&block.v_proj.weight, &block.v_proj.grad_weight, lr);
        sgd_step(&block.out_proj.weight, &block.out_proj.grad_weight, lr);
        sgd_step(&block.norm_2.weight, &block.norm_2.grad_weight, lr);
        sgd_step(&block.ffn_up.weight, &block.ffn_up.grad_weight, lr);
        sgd_step(&block.ffn_down.weight, &block.ffn_down.grad_weight, lr);
    }
    sgd_step(&model.final_norm.weight, &model.final_norm.grad_weight, lr);
    sgd_step(&model.lm_head.weight, &model.lm_head.grad_weight, lr);
}

fn cross_entropy_loss_and_grad(
    logits: &[f32],
    targets: &[u32],
    seq_len: usize,
    vocab_size: usize,
) -> (f32, Vec<f32>) {
    let mut grad = vec![0.0f32; seq_len * vocab_size];
    let mut total_loss = 0.0f32;
    let num_targets = targets.len() as f32;

    for (row, &target) in targets.iter().enumerate() {
        let row_logits = &logits[row * vocab_size..(row + 1) * vocab_size];
        let max_logit = row_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Vec<f32> = row_logits.iter().map(|&x| (x - max_logit).exp()).collect();
        let sum_exp: f32 = exp_vals.iter().sum();
        let probs: Vec<f32> = exp_vals.iter().map(|&e| e / sum_exp).collect();

        total_loss += -(probs[target as usize].max(1e-9)).ln();

        for (v, &p) in probs.iter().enumerate() {
            let one_hot = if v as u32 == target { 1.0 } else { 0.0 };
            grad[row * vocab_size + v] = (p - one_hot) / num_targets;
        }
    }

    (total_loss / num_targets, grad)
}

fn argmax_excluding(row_logits: &[f32], exclude_id: u32) -> u32 {
    row_logits
        .iter()
        .enumerate()
        .filter(|&(i, _)| i as u32 != exclude_id)
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

fn main() {
    println!("[OVERFIT DEMO] Starting...");

    let ctx = Arc::new(pollster::block_on(WgpuContext::new()));

    let (vocab, inverse_vocab, sentence_tokens) = build_vocab();
    let pad_id = vocab[PAD_WORD];

    let vocab_size = vocab.len() as u32;
    let seq_len = sentence_tokens.len() as u32;
    let dim = 256;
    let num_layers = 4;
    let epochs = 1000;
    let lr = 0.1f32;

    println!(
        "Vocab: {} words (incl. <pad>), seq_len: {}, dim: {}, layers: {}",
        vocab_size, seq_len, dim, num_layers
    );

    let targets: Vec<u32> = sentence_tokens[1..].to_vec();

    let t_input_tokens = Arc::new(Tensor::init_from_cpu(ctx.clone(), &sentence_tokens));

    let model = AkashaModel::new(
        ctx.clone(),
        vocab_size,
        dim,
        seq_len,
        num_layers,
        &t_input_tokens,
    );

    println!("Training (overfitting on the single sentence)...");
    for epoch in 0..epochs {
        model.forward();

        let logits: Vec<f32> = model.lm_head.out_buffer.to_cpu();
        let (loss, grad) =
            cross_entropy_loss_and_grad(&logits, &targets, seq_len as usize, vocab_size as usize);
        model.grad_logits.copy_from_cpu(&grad);

        model.zero_grad();
        model.backward();

        sgd_update_all(&model, lr);

        if epoch % 25 == 0 || epoch == epochs - 1 {
            println!("  epoch {:4} | loss {:.4}", epoch, loss);
        }
    }
    println!("Training complete.");

    println!();
    println!("[INTERACTIVE MODE] Type a prompt using the training vocabulary:");
    println!(
        "  {}",
        ["Akasha", "is", "a", "custom", "language", "model."].join(", ")
    );
    println!("Type \"exit\" to quit.");

    let stdin = io::stdin();
    loop {
        print!("> ");
        io::stdout().flush().unwrap();

        let mut line = String::new();
        if stdin.read_line(&mut line).unwrap() == 0 {
            break; // EOF
        }
        let line = line.trim();
        if line.eq_ignore_ascii_case("exit") {
            break;
        }
        if line.is_empty() {
            continue;
        }

        let mut prompt_tokens: Vec<u32> = Vec::new();
        let mut had_unknown = false;
        for word in line.split_whitespace() {
            match vocab.get(word) {
                Some(&id) if id != pad_id => prompt_tokens.push(id),
                _ => {
                    println!("  (unknown word, skipped: \"{}\")", word);
                    had_unknown = true;
                }
            }
        }
        if prompt_tokens.is_empty() {
            println!("  (no recognized words - try one from the training vocabulary)");
            continue;
        }
        if prompt_tokens.len() as u32 >= seq_len {
            println!("  (prompt already fills the model's fixed context window)");
            continue;
        }
        if had_unknown {
            println!("  (continuing with the recognized words only)");
        }

        print!("  completion: {}", line);
        io::stdout().flush().unwrap();

        loop {
            let mut window = prompt_tokens.clone();
            window.resize(seq_len as usize, pad_id);
            t_input_tokens.copy_from_cpu(&window);

            model.forward();
            let logits: Vec<f32> = model.lm_head.out_buffer.to_cpu();

            let last_real_pos = prompt_tokens.len() - 1;
            let row = &logits
                [last_real_pos * vocab_size as usize..(last_real_pos + 1) * vocab_size as usize];
            let next_id = argmax_excluding(row, pad_id);

            if next_id == pad_id {
                break;
            }

            print!(
                " {}",
                inverse_vocab
                    .get(&next_id)
                    .map(|s| s.as_str())
                    .unwrap_or("?")
            );
            io::stdout().flush().unwrap();

            prompt_tokens.push(next_id);
            if prompt_tokens.len() as u32 >= seq_len {
                break;
            }
        }
        println!();
    }

    println!("Bye.");
}

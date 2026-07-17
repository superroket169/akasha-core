use std::sync::Arc;

use akasha_core::config::*;
use akasha_core::data::Dataset;
use akasha_core::nn::checkpoint;
use akasha_core::nn::{InferenceSession, ModelWeights, Trainer};
use akasha_core::tokenizer::AkashaTokenizer;
use wilupgu::{Backend, Tensor, WgpuBackend};

fn find_latest_checkpoint(dir: &str) -> Option<(String, usize)> {
    std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let name = e.file_name().into_string().ok()?;
            let step = name
                .strip_prefix("model_step_")?
                .strip_suffix(".bin")?
                .parse::<usize>()
                .ok()?;
            Some((e.path().to_str()?.to_string(), step))
        })
        .max_by_key(|(_, step)| *step)
}

struct EvalSet {
    inputs: Vec<u32>,
    targets: Vec<u32>,
    windows: usize,
}

fn load_eval_set(tokenizer: &AkashaTokenizer, seq_len: usize) -> Option<EvalSet> {
    let text = std::fs::read_to_string("data/eval.txt").ok()?;
    let set = eval_windows(&tokenizer.encode(&text), seq_len, BATCH_SIZE);
    if set.is_none() {
        eprintln!(
            "WARNING: data/eval.txt is too small (need > {} tokens), eval disabled",
            BATCH_SIZE * seq_len
        );
    }
    set
}

fn eval_windows(tokens: &[u32], seq_len: usize, batch_size: usize) -> Option<EvalSet> {
    let max_windows = tokens.len().saturating_sub(1) / seq_len;
    let windows = max_windows.min(EVAL_WINDOWS) / batch_size * batch_size;
    if windows == 0 {
        return None;
    }
    let mut inputs = Vec::with_capacity(windows * seq_len);
    let mut targets = Vec::with_capacity(windows * seq_len);
    for w in 0..windows {
        let start = w * seq_len;
        inputs.extend_from_slice(&tokens[start..start + seq_len]);
        targets.extend_from_slice(&tokens[start + 1..start + seq_len + 1]);
    }
    Some(EvalSet {
        inputs,
        targets,
        windows,
    })
}

fn eval_loss<B: Backend>(model: &Trainer<B>, set: &EvalSet) -> f32 {
    let rows = model.cross_entropy.seq_len as usize;
    let passes = set.inputs.len() / rows;
    let mut total = 0.0;
    for p in 0..passes {
        let span = p * rows..(p + 1) * rows;
        model.input_tokens.copy_from_cpu(&set.inputs[span.clone()]);
        model
            .cross_entropy
            .target_tokens
            .copy_from_cpu(&set.targets[span]);
        model.forward_fused();
        total += model.cross_entropy.loss();
    }
    total / passes as f32
}

fn run_eval<B: Backend>(model: &Trainer<B>, set: &EvalSet, step: usize) {
    let loss = eval_loss(model, set);
    let ppl = loss.exp();
    println!(
        "--- eval @ step {}: loss {:.4} | ppl {:.2} ({} windows) ---",
        step, loss, ppl, set.windows
    );
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("checkpoints/eval_log.txt")
    {
        use std::io::Write;
        let _ = writeln!(f, "{}\t{:.4}\t{:.2}", step, loss, ppl);
    }
}

fn run_chat<B: Backend>(ctx: Arc<B>, weights_path: &str) {
    let tokenizer = AkashaTokenizer::from_pretrained();
    let cfg = ModelConfig::akasha_hall_1();

    let weights = Arc::new(ModelWeights::zeros(ctx.clone(), &cfg));
    checkpoint::load(&weights, weights_path)
        .unwrap_or_else(|e| panic!("Failed to load {weights_path}: {e}"));
    println!("Weights: {weights_path}");

    let mut session = InferenceSession::new(ctx, weights, SEQ_LEN);

    println!("Model loaded. Type a prompt (Ctrl+C to exit):\n");
    loop {
        print!("> ");
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
        let mut input = String::new();
        if std::io::stdin().read_line(&mut input).unwrap() == 0 {
            break;
        }
        let input = input.trim();
        if input.is_empty() {
            continue;
        }
        let tokens = tokenizer.encode(input);

        session.take_cache();
        // temperature 0.8, top-k 40, top-p 0.95 - llama.cpp-style defaults - for now
        match session.generate(&tokenizer, &tokens, 200, 0.8, 40, 0.95) {
            Ok(output) => println!("{}\n", output),
            Err(e) => eprintln!("generation failed: {e}\n"),
        }
    }
}

fn run_training<B: Backend>(ctx: Arc<B>) {
    let tokenizer = AkashaTokenizer::from_pretrained();
    println!("Vocab size: {}", tokenizer.vocab_size());

    let mut dataset = Dataset::from_file("data/train.txt", &tokenizer, SEQ_LEN as usize);
    println!("Dataset: {} tokens", dataset.token_count());

    let cfg = ModelConfig::akasha_hall_1().with_batch_size(BATCH_SIZE as u32);
    let weights = Arc::new(ModelWeights::random(ctx.clone(), &cfg));
    let input_tokens = Arc::new(Tensor::init_from_cpu(
        ctx.clone(),
        &vec![0u32; (cfg.batch_size * cfg.seq_len) as usize],
    ));
    let model = Trainer::new(ctx, weights, &input_tokens);
    println!(
        "Model ready - ~162M parameters (12-head attention, batch {})",
        cfg.batch_size
    );

    std::fs::create_dir_all("checkpoints").unwrap();

    let start_step = match find_latest_checkpoint("checkpoints") {
        Some((path, name_step)) => {
            let file_step = model
                .load_checkpoint(&path)
                .expect("Failed to load checkpoint") as usize;
            // The step recorded in the file wins; the filename is only a
            // fallback for migrated files (they carry train_step 0).
            let step = if file_step > 0 { file_step } else { name_step };
            println!("Resumed from: {} (step {})", path, step);
            step + 1
        }
        // Continued-pretraining entry point: a migrated final checkpoint
        // starts a fresh schedule (step 0, cold optimizer) on trained weights.
        None if std::path::Path::new("checkpoints/model_final.v3.bin").exists() => {
            model
                .load_checkpoint("checkpoints/model_final.v3.bin")
                .expect("Failed to load checkpoints/model_final.v3.bin");
            println!("Starting from migrated final weights (fresh schedule)");
            0
        }
        None => {
            println!("Starting fresh training run");
            0
        }
    };

    let mut rng = rand::thread_rng();
    let mut best_loss = f32::MAX;

    let eval_set = load_eval_set(&tokenizer, SEQ_LEN as usize);
    match &eval_set {
        // Baseline BEFORE any continued-pretraining step: the whole point is
        // seeing the curve move from this number.
        Some(set) => run_eval(&model, set, start_step),
        None => println!("(no data/eval.txt - held-out eval disabled)"),
    }

    println!("Training started.");
    println!("{:>8} | {:>8} | {:>10}", "step", "loss", "lr");
    println!("{}", "-".repeat(35));

    for step in start_step..MAX_STEPS {
        let (inputs, targets) = dataset.random_batch(BATCH_SIZE, &mut rng);

        // BATCH_SIZE handles the fused execute inside the model.
        // The host-loop argument must strictly remain 1.
        let loss = model.train_step(&inputs, &targets, 1, step, ACCUMULATION_STEPS);

        if let Some(l) = loss {
            if l < best_loss {
                best_loss = l;
            }

            if step % LOG_EVERY == 0 {
                let (_, lr) = model.optimizer.current_schedule();
                println!("step {:6} | loss {:.4} | lr {:.2e}", step, l, lr);
            }

            if l.is_nan() || l.is_infinite() {
                eprintln!("ERROR: Loss is NaN at step {}. Stopping.", step);
                eprintln!("Try reducing LR_MAX to 1e-4 and restart.");
                std::process::exit(1);
            }
        }

        if step % SAVE_EVERY == 0 && step > 0 {
            let path = format!("checkpoints/model_step_{}.bin", step);
            model.save_checkpoint(&path, step as u64).unwrap();
            println!("--- Checkpoint saved: {} ---", path);
        }

        if step % EVAL_EVERY == 0 && step > start_step {
            if let Some(set) = &eval_set {
                run_eval(&model, set, step);
            }
        }
    }

    // NOT model_final.bin: that name is the untouchable v1 memento.
    model
        .save_checkpoint("checkpoints/model_final.v3.bin", MAX_STEPS as u64)
        .unwrap();

    let config_json = format!(
        r#"{{
  "dim": {},
  "num_layers": {},
  "seq_len": {},
  "ffn_hidden": {},
  "vocab_size": {},
  "trained_steps": {},
  "best_loss": {:.4}
}}"#,
        DIM, NUM_LAYERS, SEQ_LEN, FFN_HIDDEN, VOCAB_SIZE, MAX_STEPS, best_loss
    );
    std::fs::write("checkpoints/config.json", config_json).unwrap();

    println!("Training complete!");
    println!("Best loss: {:.4}", best_loss);
    println!("Model saved: checkpoints/model_final.v3.bin");
    println!("Run with: cargo run --release --bin akasha-core -- --chat");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let is_chat = args.iter().any(|a| a == "--chat");
    let weights_path = args
        .iter()
        .position(|a| a == "--weights")
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
        .unwrap_or("checkpoints/model_final.v3.bin")
        .to_string();
    #[allow(unused_variables)]
    let force_cpu = args.iter().any(|a| a == "--cpu");

    #[cfg(not(feature = "cpu"))]
    if force_cpu {
        eprintln!(
            "[wilupgu] WARNING: --cpu ignored -- this binary was built without the `cpu` feature."
        );
        eprintln!("          rebuild with: cargo run --release --features cpu -- --chat --cpu");
    }

    #[cfg(feature = "cpu")]
    if force_cpu {
        use wilupgu::CpuBackend;
        println!("[wilupgu] CPU backend selected");
        let ctx = Arc::new(CpuBackend::new());
        if is_chat {
            run_chat(ctx, &weights_path);
        } else {
            run_training(ctx);
        }
        return;
    }

    #[cfg(feature = "cuda")]
    {
        use wilupgu::CudaBackend;
        if let Ok(ctx) = CudaBackend::new(0) {
            println!("[wilupgu] CUDA backend selected");
            if !is_chat && TRAIN_BF16_MATMUL {
                ctx.set_bf16_matmul(true);
                println!("[wilupgu] bf16 tensor-core matmul compute enabled");
            }
            let ctx = Arc::new(ctx);
            if is_chat {
                run_chat(ctx, &weights_path);
            } else {
                run_training(ctx);
            }
            return;
        }
    }
    println!("[wilupgu] Vulkan backend selected");
    let ctx = Arc::new(pollster::block_on(WgpuBackend::new()));
    if is_chat {
        run_chat(ctx, &weights_path);
    } else {
        run_training(ctx);
    }
}

#[cfg(test)]
mod eval_harness {
    use super::eval_windows;

    #[test]
    fn eval_windows_are_fixed_shifted_and_batch_aligned() {
        let tokens: Vec<u32> = (0..41).collect();
        let set = eval_windows(&tokens, 4, 2).unwrap();

        // 40 usable tokens / 4 = 10 windows, floored to a multiple of 2.
        assert_eq!(set.windows, 10);
        assert_eq!(set.inputs.len(), 40);
        // Non-overlapping consecutive windows, targets shifted by one.
        assert_eq!(&set.inputs[..8], &[0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(&set.targets[..8], &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(set.inputs[36..40], [36, 37, 38, 39]);
        assert_eq!(set.targets[39], 40);

        // Batch alignment floors odd window counts.
        let set = eval_windows(&(0..29).collect::<Vec<u32>>(), 4, 2).unwrap();
        assert_eq!(set.windows, 6);

        // Too small for a single batch -> None, not a panic.
        assert!(eval_windows(&(0..8).collect::<Vec<u32>>(), 4, 2).is_none());
        assert!(eval_windows(&[], 4, 2).is_none());
    }
}

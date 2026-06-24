use akasha_core::nn::akasha_model::AkashaModel;
use akasha_core::tokenizer::Tokenizer;
use std::collections::VecDeque;
use std::io::{self, Write};
use std::sync::Arc;
use wilupgu::context::WgpuContext;
use wilupgu::tensor::Tensor;

// CONSTRAINS
const NUM_WORDS: u32 = 6;
const PAD_ID: u32 = NUM_WORDS;
const VOCAB_SIZE: u32 = NUM_WORDS + 1;

const DIM: u32 = 128;
const NUM_LAYERS: usize = 2;
const BATCH_SIZE: usize = 4;
const ROLLING_WINDOW: usize = 50;
const DEFAULT_WEIGHTS_FILE: &str = "akasha.bin";
const DEFAULT_LR: f32 = 0.002;
const DATA_VOCAB_SIZE: u32 = 64;
const MAX_GEN_TOKENS: usize = 20;
const ACCUMULATION_STEPS: usize = 10;

fn parse_flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
}

fn make_batch(
    corpus: &[u32],
    seq_len: usize,
    batch_size: usize,
    pad_id: u32,
    cursor: usize,
) -> (Vec<u32>, Vec<u32>) {
    let mut inputs = Vec::with_capacity(seq_len * batch_size);
    let mut targets = Vec::with_capacity(seq_len * batch_size);
    for b in 0..batch_size {
        let start = (cursor + b * seq_len) % corpus.len();
        for i in 0..seq_len {
            let idx = (start + i) % corpus.len();
            inputs.push(corpus[idx]);
            let next_idx = idx + 1;
            targets.push(if next_idx < corpus.len() {
                corpus[next_idx]
            } else {
                pad_id
            });
        }
    }
    (inputs, targets)
}

struct Dashboard {
    step: usize,
    epoch: usize,
    last_loss: f32,
    recent_losses: VecDeque<f32>,
    cursor: usize,
}

impl Dashboard {
    fn new() -> Self {
        Self {
            step: 0,
            epoch: 0,
            last_loss: 0.0,
            recent_losses: VecDeque::with_capacity(ROLLING_WINDOW),
            cursor: 0,
        }
    }

    fn push_loss(&mut self, loss: f32) {
        self.epoch += 1;
        self.last_loss = loss;
        if self.recent_losses.len() == ROLLING_WINDOW {
            self.recent_losses.pop_front();
        }
        self.recent_losses.push_back(loss);
    }

    fn rolling_avg(&self) -> f32 {
        if self.recent_losses.is_empty() {
            return 0.0;
        }
        self.recent_losses.iter().sum::<f32>() / self.recent_losses.len() as f32
    }

    fn print(&self) {
        println!("--- Training Dashboard ---");
        println!("  Epoch:               {}", self.epoch);
        println!("  Last loss:           {:.4}", self.last_loss);
        println!(
            "  Rolling avg ({:>2} ep): {:.4}",
            self.recent_losses.len(),
            self.rolling_avg()
        );
        println!("--------------------------");
    }
}

fn run_epochs(
    model: &AkashaModel,
    dashboard: &mut Dashboard,
    corpus_tokens: &[u32],
    pad_id: u32,
    lr: f32,
    n: usize,
) {
    let seq_len = NUM_WORDS as usize;
    let step_size = seq_len * BATCH_SIZE;

    for _ in 0..n {
        let (batch_inputs, batch_targets) =
            make_batch(corpus_tokens, seq_len, BATCH_SIZE, pad_id, dashboard.cursor);
        dashboard.cursor = (dashboard.cursor + step_size) % corpus_tokens.len();

        let loss = model.train_step(
            &batch_inputs,
            &batch_targets,
            BATCH_SIZE,
            lr,
            dashboard.step,
            ACCUMULATION_STEPS,
        );
        dashboard.step += 1;
        if let Some(loss) = loss {
            dashboard.push_loss(loss);
            println!("  epoch {:6} | loss {:.4}", dashboard.epoch, loss);
        }
    }
}

fn generate(model: &AkashaModel, tokenizer: &Tokenizer, prompt: &str, seq_len: usize, pad_id: u32) {
    let prompt_tokens = tokenizer.encode(prompt);

    let mut window: Vec<u32> = if prompt_tokens.len() >= seq_len {
        prompt_tokens[prompt_tokens.len() - seq_len..].to_vec()
    } else {
        let mut padded = vec![pad_id; seq_len - prompt_tokens.len()];
        padded.extend_from_slice(&prompt_tokens);
        padded
    };

    print!("{}", prompt);
    io::stdout().flush().unwrap();

    for _ in 0..MAX_GEN_TOKENS {
        model.input_tokens.copy_from_cpu(&window);
        model.forward_fused();

        let logits: Vec<f32> = model.lm_head.out_buffer.to_cpu();
        let vocab_size = logits.len() / seq_len;
        let last_row = &logits[(seq_len - 1) * vocab_size..seq_len * vocab_size];

        let next_id = last_row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx as u32)
            .unwrap_or(pad_id);

        if next_id == pad_id {
            break;
        }

        print!("{}", tokenizer.decode(&[next_id]));
        io::stdout().flush().unwrap();

        window.remove(0);
        window.push(next_id);
    }

    println!();
}

fn main() {
    println!("[TRAIN DASHBOARD] Starting...");

    let args: Vec<String> = std::env::args().collect();
    let load_path = parse_flag(&args, "--load");
    let data_path = parse_flag(&args, "--data");
    let lr: f32 = parse_flag(&args, "--lr")
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_LR);

    let ctx = Arc::new(pollster::block_on(WgpuContext::new()));

    let (corpus_tokens, vocab_size, pad_id, tokenizer): (Vec<u32>, u32, u32, Option<Tokenizer>) =
        match data_path {
            Some(path) => {
                let text = std::fs::read_to_string(path)
                    .unwrap_or_else(|e| panic!("Failed to read --data file '{}': {}", path, e));
                let mut tokenizer = Tokenizer::new();
                tokenizer.train(&text, DATA_VOCAB_SIZE);
                let encoded = tokenizer.encode(&text);
                assert!(
                    encoded.len() > NUM_WORDS as usize,
                    "Corpus '{}' encodes to only {} tokens, need more than seq_len={}",
                    path,
                    encoded.len(),
                    NUM_WORDS
                );
                let trained_vocab = tokenizer.vocab.len() as u32;
                println!(
                    "Loaded corpus '{}': {} chars -> {} tokens, trained vocab {}",
                    path,
                    text.len(),
                    encoded.len(),
                    trained_vocab
                );
                (encoded, trained_vocab + 1, trained_vocab, Some(tokenizer))
            }
            None => ((0..NUM_WORDS).collect(), VOCAB_SIZE, PAD_ID, None),
        };

    let t_input_tokens = Arc::new(Tensor::init_from_cpu(
        ctx.clone(),
        &corpus_tokens[..NUM_WORDS as usize],
    ));
    let model = AkashaModel::new(
        ctx.clone(),
        vocab_size,
        DIM,
        NUM_WORDS,
        NUM_LAYERS,
        &t_input_tokens,
    );

    if let Some(path) = load_path {
        match model.load_weights(path) {
            Ok(()) => println!("Resumed from '{}'", path),
            Err(e) => eprintln!("Could not load '{}': {} (starting from scratch)", path, e),
        }
    }

    println!(
        "Vocab: {} (incl. <pad>), seq_len: {}, dim: {}, layers: {}, batch_size: {}, lr: {}",
        vocab_size, NUM_WORDS, DIM, NUM_LAYERS, BATCH_SIZE, lr
    );
    println!(
        "Commands: run <epochs> | save [path] | load [path] | generate \"<text>\" | print | clear | exit"
    );

    let mut dashboard = Dashboard::new();
    let stdin = io::stdin();

    loop {
        print!("akasha> ");
        io::stdout().flush().unwrap();

        let mut line = String::new();
        if stdin.read_line(&mut line).unwrap() == 0 {
            break;
        }

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let mut parts = trimmed.split_whitespace();
        let cmd = parts.next().unwrap_or("").to_lowercase();
        let arg = parts.next();

        match cmd.as_str() {
            "run" => {
                let n: usize = arg.and_then(|a| a.parse().ok()).unwrap_or(1);
                println!("Running {} epoch(s) (blocking)...", n);
                run_epochs(&model, &mut dashboard, &corpus_tokens, pad_id, lr, n);
                println!(
                    "Done. epoch {} | last loss {:.4}",
                    dashboard.epoch, dashboard.last_loss
                );
            }
            "save" => {
                let path = arg.unwrap_or(DEFAULT_WEIGHTS_FILE);
                match model.save_weights(path) {
                    Ok(()) => println!("Saved to '{}'.", path),
                    Err(e) => eprintln!("Save failed: {}", e),
                }
            }
            "load" => {
                let path = arg.unwrap_or(DEFAULT_WEIGHTS_FILE);
                match model.load_weights(path) {
                    Ok(()) => println!("Loaded '{}'.", path),
                    Err(e) => eprintln!("Load failed: {}", e),
                }
            }
            "generate" => match (tokenizer.as_ref(), trimmed.find('"'), trimmed.rfind('"')) {
                (Some(tk), Some(start), Some(end)) if end > start => {
                    let prompt = &trimmed[start + 1..end];
                    generate(&model, tk, prompt, NUM_WORDS as usize, pad_id);
                }
                (None, _, _) => {
                    println!("No tokenizer loaded — start with --data <path> to enable generation.")
                }
                _ => println!("Usage: generate \"<text>\""),
            },
            "print" => dashboard.print(),
            "clear" => {
                print!("\x1B[2J\x1B[1;1H");
                io::stdout().flush().unwrap();
            }
            "exit" | "quit" => break,
            _ => println!("Unknown command: '{}'", cmd),
        }
    }

    println!("Saving final weights to '{}'...", DEFAULT_WEIGHTS_FILE);
    match model.save_weights(DEFAULT_WEIGHTS_FILE) {
        Ok(()) => println!("Saved."),
        Err(e) => eprintln!("Warning: failed to save weights: {}", e),
    }
    println!("Bye.");
}

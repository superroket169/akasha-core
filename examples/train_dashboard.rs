use akasha_core::nn::akasha_model::AkashaModel;
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
const DEFAULT_LR: f32 = 0.005;

fn parse_flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
}

struct Dashboard {
    step: usize,
    epoch: usize,
    last_loss: f32,
    recent_losses: VecDeque<f32>,
}

impl Dashboard {
    fn new() -> Self {
        Self {
            step: 0,
            epoch: 0,
            last_loss: 0.0,
            recent_losses: VecDeque::with_capacity(ROLLING_WINDOW),
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
    batch_inputs: &[u32],
    batch_targets: &[u32],
    lr: f32,
    n: usize,
) {
    for _ in 0..n {
        let loss = model.train_step(batch_inputs, batch_targets, BATCH_SIZE, lr, dashboard.step);
        dashboard.step += 1;
        if let Some(loss) = loss {
            dashboard.push_loss(loss);
            println!("  epoch {:6} | loss {:.4}", dashboard.epoch, loss);
        }
    }
}

fn main() {
    println!("[TRAIN DASHBOARD] Starting...");

    let args: Vec<String> = std::env::args().collect();
    let load_path = parse_flag(&args, "--load");
    let lr: f32 = parse_flag(&args, "--lr")
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_LR);

    let ctx = Arc::new(pollster::block_on(WgpuContext::new()));

    let sentence_tokens: Vec<u32> = (0..NUM_WORDS).collect();
    let mut targets = sentence_tokens[1..].to_vec();
    targets.push(PAD_ID);

    let batch_inputs: Vec<u32> = sentence_tokens
        .iter()
        .cloned()
        .cycle()
        .take(sentence_tokens.len() * BATCH_SIZE)
        .collect();
    let batch_targets: Vec<u32> = targets
        .iter()
        .cloned()
        .cycle()
        .take(targets.len() * BATCH_SIZE)
        .collect();

    let t_input_tokens = Arc::new(Tensor::init_from_cpu(ctx.clone(), &sentence_tokens));
    let model = AkashaModel::new(
        ctx.clone(),
        VOCAB_SIZE,
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
        VOCAB_SIZE, NUM_WORDS, DIM, NUM_LAYERS, BATCH_SIZE, lr
    );
    println!("Commands: run <epochs> | save [path] | load [path] | print | clear | exit");

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
                run_epochs(&model, &mut dashboard, &batch_inputs, &batch_targets, lr, n);
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

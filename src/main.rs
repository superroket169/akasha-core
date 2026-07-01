use std::sync::Arc;

use akasha_core::config::*;
use akasha_core::data::Dataset;
use akasha_core::nn::akasha_model::AkashaModel;
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

fn build_model<B: Backend>(ctx: Arc<B>) -> AkashaModel<B> {
    let input_tokens = Arc::new(Tensor::init_from_cpu(
        ctx.clone(),
        &vec![0u32; SEQ_LEN as usize],
    ));
    AkashaModel::new(
        ctx,
        VOCAB_SIZE,
        DIM,
        SEQ_LEN,
        NUM_LAYERS,
        NUM_HEADS,
        &input_tokens,
    )
}

fn run_chat<B: Backend>(ctx: Arc<B>) {
    let tokenizer = AkashaTokenizer::from_pretrained();
    let model = build_model(ctx);
    model
        .load_weights("checkpoints/model_final.bin")
        .expect("Failed to load checkpoints/model_final.bin -- train the model first");

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
        let output = model.generate(&tokenizer, &tokens, 200, 0.8);
        println!("{}\n", output);
    }
}

fn run_training<B: Backend>(ctx: Arc<B>) {
    let tokenizer = AkashaTokenizer::from_pretrained();
    println!("Vocab size: {}", tokenizer.vocab_size());

    let dataset = Dataset::from_file("data/train.txt", &tokenizer, SEQ_LEN as usize);
    println!("Dataset: {} tokens", dataset.token_count());

    let model = build_model(ctx);
    println!("Model ready - ~117M parameters (12-head attention)");

    std::fs::create_dir_all("checkpoints").unwrap();

    let start_step = match find_latest_checkpoint("checkpoints") {
        Some((path, step)) => {
            model
                .load_weights(&path)
                .expect("Failed to load checkpoint");
            println!("Resumed from: {} (step {})", path, step);
            step + 1
        }
        None => {
            println!("Starting fresh training run");
            0
        }
    };

    let mut rng = rand::thread_rng();
    let mut best_loss = f32::MAX;

    println!("Training started.");
    println!("{:>8} | {:>8} | {:>10}", "step", "loss", "lr");
    println!("{}", "-".repeat(35));

    for step in start_step..MAX_STEPS {
        let lr = cosine_lr(step, WARMUP_STEPS, MAX_STEPS, LR_MAX, LR_MIN);
        let (inputs, targets) = dataset.random_batch(BATCH_SIZE, &mut rng);

        let loss = model.train_step(&inputs, &targets, BATCH_SIZE, lr, step, ACCUMULATION_STEPS);

        if let Some(l) = loss {
            if l < best_loss {
                best_loss = l;
            }

            if step % LOG_EVERY == 0 {
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
            model.save_weights(&path).unwrap();
            println!("--- Checkpoint saved: {} ---", path);
        }
    }

    model.save_weights("checkpoints/model_final.bin").unwrap();

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
    println!("Model saved: checkpoints/model_final.bin");
    println!("Run with: cargo run --release --bin akasha-core -- --chat");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let is_chat = args.iter().any(|a| a == "--chat");

    #[cfg(feature = "cuda")]
    {
        use wilupgu::CudaBackend;
        if let Ok(ctx) = CudaBackend::new(0) {
            println!("[wilupgu] CUDA backend selected");
            let ctx = Arc::new(ctx);
            if is_chat {
                run_chat(ctx);
            } else {
                run_training(ctx);
            }
            return;
        }
    }
    println!("[wilupgu] Vulkan backend selected");
    let ctx = Arc::new(pollster::block_on(WgpuBackend::new()));
    if is_chat {
        run_chat(ctx);
    } else {
        run_training(ctx);
    }
}

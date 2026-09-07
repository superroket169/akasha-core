use std::sync::Arc;

use sequexa_core::config::{ModelConfig, TrainConfig};
use sequexa_core::diagnostic::{DiagnosticCheck, DiagnosticSuite};
use sequexa_core::nn::{Model, ModelWeights};
use rand::Rng;
use wilupgu::{Backend, Tensor, WgpuBackend};

static DIAG_RNG: std::sync::OnceLock<std::sync::Mutex<rand::rngs::StdRng>> =
    std::sync::OnceLock::new();
fn diag_rng() -> std::sync::MutexGuard<'static, rand::rngs::StdRng> {
    DIAG_RNG
        .get_or_init(|| std::sync::Mutex::new(rand::SeedableRng::seed_from_u64(42)))
        .lock()
        .unwrap()
}

fn rand_u32_vec(n: usize, max_exclusive: u32) -> Vec<u32> {
    let mut rng = diag_rng();
    (0..n).map(|_| rng.gen_range(0..max_exclusive)).collect()
}

fn l2_norm<B: Backend>(t: &Tensor<B>) -> f32 {
    let data: Vec<f32> = t.to_cpu();
    data.iter().map(|x| x * x).sum::<f32>().sqrt()
}

// weight decay grouping (was CHECK 6, advisory-only against the old Trainer)
// is no longer a runtime check here -- it's a static, code-level guarantee
// now: EmbeddingOp::param() and RmsNormOp::param() hardcode decay=false,
// LinearOp::param() forwards whatever `decay` its NodeSpec passed (always
// true in blocks::transformer's specs). AdamW actually honors this per-param
// flag (see optim/adamw.rs), unlike the old Trainer's single uniform group
// -- the exact issue the old advisory flagged is fixed by construction.

struct ParamCountCheck<B: Backend> {
    ctx: Arc<B>,
    vocab_size: u32,
}

impl<B: Backend> DiagnosticCheck for ParamCountCheck<B> {
    fn name(&self) -> &'static str {
        "CHECK 1 (param count)"
    }

    fn run(&self) -> bool {
        let arch = ModelConfig::hall_1();
        let cfg = ModelConfig::new(
            self.vocab_size,
            arch.dim,
            arch.num_heads,
            arch.num_layers,
            arch.seq_len,
        );
        let weights = ModelWeights::random(self.ctx.clone(), &cfg);

        let total: u64 = weights.params().iter().map(|w| w.size / 4).sum();
        let pass = total > 10_000_000;
        self.log(&format!(
            "total trainable parameters = {} ({:.1}M){}",
            total,
            total as f64 / 1e6,
            if pass { "" } else { "  <-- RED FLAG: far below 117M" }
        ));
        pass
    }
}

struct GradFlowCheck<B: Backend> {
    ctx: Arc<B>,
    vocab_size: u32,
}

impl<B: Backend> DiagnosticCheck for GradFlowCheck<B> {
    fn name(&self) -> &'static str {
        "CHECK 2 (gradient flow)"
    }

    fn run(&self) -> bool {
        let ctx = self.ctx.clone();
        let vocab_size = self.vocab_size;
        let arch = ModelConfig::hall_1();
        let seq_len = 16u32;

        let cfg = ModelConfig::new(
            vocab_size,
            arch.dim,
            arch.num_heads,
            arch.num_layers,
            seq_len,
        );
        let weights = ModelWeights::random(ctx.clone(), &cfg);
        let mut model =
            Model::for_training(ctx.clone(), weights, cfg, TrainConfig::hall1_pretrain());

        let tokens = rand_u32_vec(seq_len as usize, vocab_size);
        let targets = rand_u32_vec(seq_len as usize, vocab_size);
        model.zero_grad();
        let loss = model.train_step(&tokens, &targets);

        self.log(&format!(
            "1-step grad flow test (seq_len={seq_len}, dim={}, layers={}, heads={})",
            arch.dim, arch.num_layers, arch.num_heads
        ));
        self.log(&format!("forward loss = {loss:.4}"));

        let params = model.params();
        let norms: Vec<f32> = params.iter().map(|(_, g, _)| l2_norm(g)).collect();
        let any_zero = norms.iter().any(|&n| n == 0.0);
        let any_explosion = norms.iter().any(|&n| n > 100.0);

        // params() is laid out front-to-back as embedding, block0..blockN,
        // final_norm, lm_head -- front-half vs back-half sums are a coarse
        // stand-in for "early layers vs late layers" without needing to know
        // the exact per-block param count here.
        let half = norms.len() / 2;
        let front_sum: f32 = norms[..half].iter().sum();
        let back_sum: f32 = norms[half..].iter().sum();
        let vanishing = if front_sum > 1e-9 {
            let ratio = front_sum / back_sum.max(1e-12);
            if ratio > 1e3 {
                self.log(&format!(
                    "RED FLAG: front-half grad sum ({front_sum:.4}) / back-half grad sum ({back_sum:.4}) = {ratio:.1} -- looks like vanishing gradient"
                ));
                true
            } else {
                false
            }
        } else {
            false
        };

        if any_zero {
            self.log(
                "RED FLAG: at least one grad norm is exactly 0.0 -- gradient not flowing there",
            );
        }
        if any_explosion {
            self.log("RED FLAG: at least one grad norm > 100 -- exploding gradient");
        }

        !any_zero && !any_explosion && !vanishing
    }
}

struct AccumulationCheck<B: Backend> {
    ctx: Arc<B>,
    vocab_size: u32,
}

impl<B: Backend> DiagnosticCheck for AccumulationCheck<B> {
    fn name(&self) -> &'static str {
        "CHECK 5 (grad accumulation)"
    }

    fn run(&self) -> bool {
        let ctx = self.ctx.clone();
        let vocab_size = self.vocab_size;
        let arch = ModelConfig::hall_1();
        let seq_len = 16u32;

        let cfg = ModelConfig::new(
            vocab_size,
            arch.dim,
            arch.num_heads,
            arch.num_layers,
            seq_len,
        );
        let weights = ModelWeights::random(ctx.clone(), &cfg);
        let mut model =
            Model::for_training(ctx.clone(), weights, cfg, TrainConfig::hall1_pretrain());

        let inputs = rand_u32_vec(seq_len as usize, vocab_size);
        let targets = rand_u32_vec(seq_len as usize, vocab_size);

        let (lm_weight, lm_grad, _) = model.params().last().unwrap().clone();
        let w_before: Vec<f32> = lm_weight.to_cpu();

        model.zero_grad();
        model.train_step(&inputs, &targets); // step 0 of a 2-step accumulation cycle (mid-cycle)
        let w_after_step0: Vec<f32> = lm_weight.to_cpu();
        let grad_after_step0 = l2_norm(&lm_grad);
        let delta0: f32 = w_before
            .iter()
            .zip(w_after_step0.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();

        model.train_step(&inputs, &targets); // step 1: accumulate more
        model.optimizer_step(); // cycle boundary
        let w_after_step1: Vec<f32> = lm_weight.to_cpu();
        let delta1: f32 = w_after_step0
            .iter()
            .zip(w_after_step1.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();

        self.log(&format!(
            "after step 0 (mid-cycle): weight delta norm = {delta0:.8}, grad_weight norm = {grad_after_step0:.6}"
        ));
        self.log(&format!(
            "after step 1 (cycle boundary): weight delta norm = {delta1:.8}"
        ));

        let pass = delta0 < 1e-7 && grad_after_step0 > 0.0 && delta1 > 1e-7;
        if delta0 >= 1e-7 {
            self.log(
                "RED FLAG: weights changed mid-cycle (before optimizer_step() should have run)",
            );
        }
        if delta1 < 1e-7 {
            self.log(
                "RED FLAG: weights did NOT change at accumulation boundary -- optimizer broken",
            );
        }
        pass
    }
}

struct MemorizationCheck<B: Backend> {
    ctx: Arc<B>,
}

impl<B: Backend> DiagnosticCheck for MemorizationCheck<B> {
    fn name(&self) -> &'static str {
        "CHECK 8 (memorization)"
    }

    fn run(&self) -> bool {
        memorization_run(self, self.ctx.clone(), 3e-3)
    }
}

fn memorization_run<B: Backend>(check: &MemorizationCheck<B>, ctx: Arc<B>, lr: f32) -> bool {
    let dim = 128u32; // head_dim=64: Model's AttentionOp always uses flash attention, hardcoded to it
    let num_heads = 2u32;
    let num_layers = 1usize;
    let vocab_size = 100u32;
    let seq_len = 16u32;
    let batch_size = 4u32;

    // real batching (cfg.batch_size) stands in for the old host-loop batch
    // -- Model has no separate host-loop-batching path.
    let cfg = ModelConfig::new(vocab_size, dim, num_heads, num_layers, seq_len)
        .with_batch_size(batch_size);
    let weights = ModelWeights::random(ctx.clone(), &cfg);

    let mut train_cfg = TrainConfig::hall1_pretrain();
    train_cfg.name = "diagnose_check8";
    train_cfg.lr_max = lr;
    train_cfg.lr_min = lr;
    train_cfg.warmup_steps = 0;
    train_cfg.max_steps = 600;
    train_cfg.optimizer.lr_max = lr;
    train_cfg.optimizer.lr_min = lr;
    train_cfg.optimizer.warmup_steps = 0;
    train_cfg.optimizer.max_steps = 600;
    train_cfg.run.accumulation_steps = 1;

    let mut model = Model::for_training(ctx.clone(), weights, cfg, train_cfg);

    let rows = (batch_size * seq_len) as usize;
    let inputs = rand_u32_vec(rows, vocab_size);
    let targets = rand_u32_vec(rows, vocab_size);

    check.log(&format!(
        "single-layer memorization test (dim={dim}, heads={num_heads}, layers={num_layers}, vocab={vocab_size}, seq_len={seq_len}, batch={batch_size}, lr={lr})"
    ));

    let mut final_loss = f32::MAX;
    for step in 0..600usize {
        model.zero_grad();
        let loss = model.train_step(&inputs, &targets);
        model.optimizer_step();

        final_loss = loss;
        if step % 40 == 0 || step == 599 {
            check.log(&format!("step {step:3} | loss {loss:.4}"));
        }
        if loss.is_nan() {
            check.log(&format!("RED FLAG: loss is NaN at step {step}"));
            return false;
        }
    }

    let pass = final_loss < 0.1;
    check.log(&format!(
        "final loss = {final_loss:.4} -> {}",
        if pass { "PASS" } else { "FAIL" }
    ));
    if !pass {
        check.log("RED FLAG: tiny single-layer model could not memorize a fixed batch in 600 steps.");
        check.log(
            "This points to a bug in the training loop itself (optimizer, backward, or loss),",
        );
        check.log("not just a hyperparameter/scale issue with the full 117M model.");
    }
    pass
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

// CHECK 9/10 replace the old InferenceSession-vs-Trainer comparison (deleted
// along with InferenceSession/Cache): they exercise nn::Model's public
// prefill/decode primitives directly, comparing the cache/decode-step code
// path against the full-prefill code path.
//
// First version of CHECK 9 compared sampled (argmax) tokens instead of raw
// logits and produced a false FAIL: on random untrained weights logits have
// no dominant peak, so the tiny fp differences between flash-attention
// (prefill) and attn_qk_cached/attn_av_cached (decode) can flip which vocab
// entry wins argmax without either kernel being wrong. Comparing logits
// numerically (like every other kernel-parity check in this codebase) is the
// correct signal.
struct PrefillDecodeParityCheck<B: Backend> {
    ctx: Arc<B>,
}

impl<B: Backend> DiagnosticCheck for PrefillDecodeParityCheck<B> {
    fn name(&self) -> &'static str {
        "CHECK 9 (prefill/decode parity)"
    }

    fn run(&self) -> bool {
        let ctx = self.ctx.clone();
        let vocab_size = 61u32;
        let cfg = ModelConfig::new(vocab_size, 128, 2, 2, 24); // head_dim=64
        let max_context_len = 24u32;
        let weights = ModelWeights::random(ctx.clone(), &cfg);
        let base_prompt = rand_u32_vec(6, vocab_size);
        let extra_token = rand_u32_vec(1, vocab_size)[0];

        // A: prefill(base_prompt) writes cache[0..6), then one decode step
        // for `extra_token` at pos=6 -- its logits predict what follows.
        let mut model_a = Model::for_chat(ctx.clone(), weights.clone(), cfg, max_context_len);
        model_a
            .prefill_logits(&base_prompt)
            .expect("prefill A failed");
        let decode_logits = model_a
            .decode_step_logits(extra_token, base_prompt.len() as u32)
            .expect("decode A failed");

        // B: prefill(base_prompt + [extra_token]) in one shot -- same
        // context as A, computed entirely by the full-sequence prefill path.
        let mut extended_prompt = base_prompt.clone();
        extended_prompt.push(extra_token);
        let mut model_b = Model::for_chat(ctx.clone(), weights, cfg, max_context_len);
        let prefill_logits = model_b
            .prefill_logits(&extended_prompt)
            .expect("prefill B failed");

        let diff = max_abs_diff(&decode_logits, &prefill_logits);
        let pass = diff < 1e-2;
        self.log(&format!(
            "prefill-vs-decode parity -- max logit diff = {diff:.6} -> {}",
            if pass { "PASS" } else { "FAIL" }
        ));

        if !pass {
            self.log("RED FLAG: decode-step (KV-cache) path disagrees numerically with the full-prefill path for the same context -- cache write, RoPE offset, or cached-attention kernel is likely wrong.");
        }
        pass
    }
}

struct DecodeCacheSpeedCheck<B: Backend> {
    ctx: Arc<B>,
}

impl<B: Backend> DiagnosticCheck for DecodeCacheSpeedCheck<B> {
    fn name(&self) -> &'static str {
        "CHECK 10 (decode cache speed)"
    }

    fn run(&self) -> bool {
        let ctx = self.ctx.clone();
        let vocab_size = 61u32;
        let cfg = ModelConfig::new(vocab_size, 128, 2, 2, 64); // head_dim=64
        let max_context_len = 64u32;
        let weights = ModelWeights::random(ctx.clone(), &cfg);
        let prompt = rand_u32_vec(8, vocab_size);
        let extra_tokens = 10usize;

        // naive: re-prefill from scratch for every new token (no cache reuse
        // across calls -- each generate() call is a fresh Model/fresh cache).
        let naive_start = std::time::Instant::now();
        let mut naive_seq = prompt.clone();
        for _ in 0..extra_tokens {
            let mut m = Model::for_chat(ctx.clone(), weights.clone(), cfg, max_context_len);
            let next = m
                .generate(&naive_seq, 1, 0.0, 0, 1.0, 1.0)
                .expect("naive generate failed");
            naive_seq.push(next[0]);
        }
        let naive_elapsed = naive_start.elapsed();

        // cached: one Model, one generate() call -- extra_tokens - 1 of the
        // new tokens come from decode steps against the KV cache.
        let cached_start = std::time::Instant::now();
        let mut m = Model::for_chat(ctx.clone(), weights, cfg, max_context_len);
        m.generate(&prompt, extra_tokens, 0.0, 0, 1.0, 1.0)
            .expect("cached generate failed");
        let cached_elapsed = cached_start.elapsed();

        let speedup = naive_elapsed.as_secs_f64() / cached_elapsed.as_secs_f64().max(1e-9);
        self.log(&format!(
            "{extra_tokens} tokens -- naive re-prefill = {naive_elapsed:.2?}, cached decode = {cached_elapsed:.2?}, speedup = {speedup:.2}x"
        ));
        let pass = cached_elapsed < naive_elapsed;
        if !pass {
            self.log("RED FLAG: cached decode was not faster than re-prefilling from scratch each step -- KV-cache isn't providing a speed benefit.");
        }
        pass
    }
}

fn run_diagnostics<B: Backend>(ctx: Arc<B>) {
    if std::env::var("DIAGNOSE_ONLY_CHECK8").is_ok() {
        DiagnosticSuite::new()
            .add(Box::new(MemorizationCheck { ctx }))
            .run();
        return;
    }

    println!("\n================= SEQUEXA TRAINING DIAGNOSTICS =================\n");

    let arch = ModelConfig::hall_1();

    let vocab_size: u32 = std::env::var("DIAGNOSE_VOCAB_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(arch.vocab_size);
    if vocab_size != arch.vocab_size {
        println!(
            "NOTE: DIAGNOSE_VOCAB_SIZE override active -- using vocab_size={vocab_size} instead of {}\n",
            arch.vocab_size
        );
    }

    DiagnosticSuite::new()
        .add(Box::new(ParamCountCheck { ctx: ctx.clone(), vocab_size }))
        .add(Box::new(GradFlowCheck { ctx: ctx.clone(), vocab_size }))
        .add(Box::new(AccumulationCheck { ctx: ctx.clone(), vocab_size }))
        .add(Box::new(MemorizationCheck { ctx: ctx.clone() }))
        .add(Box::new(PrefillDecodeParityCheck { ctx: ctx.clone() }))
        .add(Box::new(DecodeCacheSpeedCheck { ctx }))
        .run();
}

fn main() {
    #[cfg(feature = "cuda")]
    {
        use wilupgu::CudaBackend;
        if let Ok(ctx) = CudaBackend::new(0) {
            println!("[diagnose] CUDA backend selected");
            run_diagnostics(Arc::new(ctx));
            return;
        }
        println!("[diagnose] CUDA backend unavailable, falling back to Vulkan");
    }
    println!("[wilupgu] Vulkan backend selected");
    run_diagnostics(Arc::new(pollster::block_on(WgpuBackend::new())));
}

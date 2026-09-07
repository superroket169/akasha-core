use std::sync::Arc;

use akasha_core::config::{
    GradClipConfig, GradClipKind, ModelConfig, OptimizerConfig, OptimizerKind, RunConfig,
    TrainConfig,
};
use akasha_core::diagnostic::{DiagnosticCheck, run_all};
use akasha_core::nn::{Model, ModelWeights, Trainer};
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

struct ParamCountCheck<B: Backend> {
    model: Arc<Trainer<B>>,
}

impl<B: Backend> DiagnosticCheck for ParamCountCheck<B> {
    fn name(&self) -> &'static str {
        "CHECK 1 (param count)"
    }

    fn run(&self) -> bool {
        let total: u64 = self
            .model
            .trainable_params()
            .iter()
            .map(|(w, _)| w.size / 4)
            .sum();
        let pass = total > 10_000_000;
        println!(
            "CHECK 1: total trainable parameters = {} ({:.1}M){}",
            total,
            total as f64 / 1e6,
            if pass {
                ""
            } else {
                "  <-- RED FLAG: far below 117M"
            }
        );
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
        let arch = ModelConfig::akasha_hall_1();
        let seq_len = 16u32;

        let input_tokens = Arc::new(Tensor::init_from_cpu(
            ctx.clone(),
            &rand_u32_vec(seq_len as usize, vocab_size),
        ));
        let cfg = ModelConfig::new(
            vocab_size,
            arch.dim,
            arch.num_heads,
            arch.num_layers,
            seq_len,
        );
        let weights = Arc::new(ModelWeights::random(ctx.clone(), &cfg));
        let model = Trainer::new(
            ctx.clone(),
            weights,
            &input_tokens,
            TrainConfig::hall1_pretrain(),
        );
        let num_layers = model.layers.len();

        model.zero_grad();
        model.zero_transient_grads();
        model
            .cross_entropy
            .target_tokens
            .copy_from_cpu(&rand_u32_vec(seq_len as usize, vocab_size));
        model.cross_entropy.set_grad_scale(1.0 / seq_len as f32);

        model.forward_fused();
        let loss = model.cross_entropy.loss();
        model.backward_fused();

        println!(
            "CHECK 2: 1-step grad flow test (seq_len={seq_len}, dim={}, layers={num_layers}, heads={})",
            arch.dim, arch.num_heads
        );
        println!("  forward loss = {loss:.4}");

        let mut any_zero = false;
        let mut any_explosion = false;
        let mut layer_total_norms = Vec::with_capacity(num_layers);

        for (i, layer) in model.layers.iter().enumerate() {
            let entries = [
                ("QKV_proj", l2_norm(&layer.qkv_proj.grad_weight)),
                ("O_proj", l2_norm(&layer.out_proj.grad_weight)),
                ("FFN_up", l2_norm(&layer.ffn_up.grad_weight)),
                ("FFN_down", l2_norm(&layer.ffn_down.grad_weight)),
                ("RMSNorm1", l2_norm(&layer.norm_1.grad_weight)),
                ("RMSNorm2", l2_norm(&layer.norm_2.grad_weight)),
            ];
            let mut layer_sum = 0.0f32;
            for (name, norm) in entries {
                println!("  Layer {i}: {name} grad norm = {norm:.4}");
                if norm == 0.0 {
                    any_zero = true;
                }
                if norm > 100.0 {
                    any_explosion = true;
                }
                layer_sum += norm;
            }
            layer_total_norms.push(layer_sum);
        }

        let emb_norm = l2_norm(&model.embedding.grad_table);
        let lmhead_norm = l2_norm(&model.lm_head.grad_weight);

        println!("  Embedding grad norm = {emb_norm:.4}");
        println!("  LM_head grad norm = {lmhead_norm:.4}");

        if emb_norm == 0.0 || lmhead_norm == 0.0 {
            any_zero = true;
        }

        let vanishing = if layer_total_norms[0] > 1e-9 {
            let ratio = layer_total_norms[0] / layer_total_norms[num_layers - 1].max(1e-12);
            if ratio > 1e3 {
                println!(
                    "  RED FLAG: layer-0 grad sum ({:.4}) / layer-{} grad sum ({:.4}) = {:.1} -- looks like vanishing gradient",
                    layer_total_norms[0],
                    num_layers - 1,
                    layer_total_norms[num_layers - 1],
                    ratio
                );
                true
            } else {
                false
            }
        } else {
            false
        };

        if any_zero {
            println!(
                "  RED FLAG: at least one grad norm is exactly 0.0 -- gradient not flowing there"
            );
        }
        if any_explosion {
            println!("  RED FLAG: at least one grad norm > 100 -- exploding gradient");
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
        let arch = ModelConfig::akasha_hall_1();
        let seq_len = 16u32;

        let input_tokens = Arc::new(Tensor::init_from_cpu(
            ctx.clone(),
            &rand_u32_vec(seq_len as usize, vocab_size),
        ));
        let cfg = ModelConfig::new(
            vocab_size,
            arch.dim,
            arch.num_heads,
            arch.num_layers,
            seq_len,
        );
        let weights = Arc::new(ModelWeights::random(ctx.clone(), &cfg));
        let model = Trainer::new(
            ctx.clone(),
            weights,
            &input_tokens,
            TrainConfig::hall1_pretrain(),
        );

        let inputs = rand_u32_vec(seq_len as usize, vocab_size);
        let targets = rand_u32_vec(seq_len as usize, vocab_size);

        let w_before: Vec<f32> = model.lm_head.weight.to_cpu();

        model.train_step(&inputs, &targets, 1, 0, 2); // step 0 of a 2-step accumulation cycle
        let w_after_step0: Vec<f32> = model.lm_head.weight.to_cpu();
        let grad_after_step0 = l2_norm(&model.lm_head.grad_weight);
        let delta0: f32 = w_before
            .iter()
            .zip(w_after_step0.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();

        model.train_step(&inputs, &targets, 1, 1, 2); // step 1 = accumulation boundary
        let w_after_step1: Vec<f32> = model.lm_head.weight.to_cpu();
        let delta1: f32 = w_after_step0
            .iter()
            .zip(w_after_step1.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();

        println!("CHECK 5: gradient accumulation");
        println!(
            "  after step 0 (mid-cycle): weight delta norm = {delta0:.8}, grad_weight norm = {grad_after_step0:.6}"
        );
        println!("  after step 1 (cycle boundary): weight delta norm = {delta1:.8}");

        let pass = delta0 < 1e-7 && grad_after_step0 > 0.0 && delta1 > 1e-7;
        if delta0 >= 1e-7 {
            println!(
                "  RED FLAG: weights changed mid-cycle (before optimizer.step() should have run)"
            );
        }
        if delta1 < 1e-7 {
            println!(
                "  RED FLAG: weights did NOT change at accumulation boundary -- optimizer broken"
            );
        }
        println!("CHECK 5: {}", if pass { "PASS" } else { "FAIL" });
        pass
    }
}

struct WeightDecayGroupsCheck<B: Backend> {
    model: Arc<Trainer<B>>,
    weight_decay: f32,
}

impl<B: Backend> DiagnosticCheck for WeightDecayGroupsCheck<B> {
    fn name(&self) -> &'static str {
        "CHECK 6 (weight decay groups, advisory)"
    }

    fn run(&self) -> bool {
        let params = self.model.trainable_params();
        let emb_in_group = params
            .iter()
            .any(|(w, _)| Arc::ptr_eq(w, &self.model.embedding.table));
        let norm_in_group = params
            .iter()
            .any(|(w, _)| Arc::ptr_eq(w, &self.model.final_norm.weight));

        println!("CHECK 6: AdamW weight-decay grouping");
        println!(
            "  AdamW::new() is called with a single uniform parameter list and a single shared"
        );
        println!("  StepConfig{{weight_decay,...}} applied identically to every tensor in it --");
        println!("  this codebase has no separate no_decay group at all.");
        println!("  Embedding table in the (only) weight_decay group: {emb_in_group}");
        println!("  RMSNorm (final_norm) weight in the (only) weight_decay group: {norm_in_group}");
        println!(
            "  ADVISORY (not a correctness bug): embeddings and RMSNorm scale weights ARE being"
        );
        println!(
            "  weight-decayed at adam_weight_decay={}, which is non-standard --",
            self.weight_decay
        );
        println!(
            "  most GPT-2-style training setups exempt 1D params (norms, embeddings, biases) from decay."
        );
        emb_in_group && norm_in_group
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
        for &(lr, clip) in &[(3e-3f32, false)] {
            println!("--- trying lr={lr} grad_clip={clip}, extended to 600 steps ---");
            if memorization_run(self.ctx.clone(), lr, clip) {
                return true;
            }
        }
        false
    }
}

fn memorization_run<B: Backend>(ctx: Arc<B>, lr: f32, use_clip: bool) -> bool {
    let dim = 64u32;
    let num_heads = 4u32;
    let num_layers = 1usize;
    let vocab_size = 100u32;
    let seq_len = 16u32;
    let batch_size = 4usize;

    let input_tokens = Arc::new(Tensor::init_from_cpu(
        ctx.clone(),
        &vec![0u32; seq_len as usize],
    ));

    let cfg = ModelConfig::new(vocab_size, dim, num_heads, num_layers, seq_len);
    let weights = Arc::new(ModelWeights::random(ctx.clone(), &cfg));
    let train_cfg = TrainConfig {
        name: "diagnose_check8",
        batch_size,
        accumulation_steps: 1,
        lr_max: lr,
        lr_min: lr,
        warmup_steps: 0,
        max_steps: 600,
        save_every: usize::MAX,
        log_every: 40,
        eval_every: usize::MAX,
        eval_windows: 0,
        adam_weight_decay: 0.01,
        grad_clip_norm: 1.0,
        train_bf16_matmul: false,
        optimizer: OptimizerConfig {
            kind: OptimizerKind::AdamW,
            beta1: 0.9,
            beta2: 0.95,
            weight_decay: 0.01,
            lr_max: lr,
            lr_min: lr,
            warmup_steps: 0,
            max_steps: 600,
        },
        grad_clip: GradClipConfig {
            kind: GradClipKind::GlobalNorm,
            max_norm: 1.0,
        },
        run: RunConfig {
            batch_size,
            accumulation_steps: 1,
            save_every: usize::MAX,
            log_every: 40,
            eval_every: usize::MAX,
            eval_windows: 0,
            train_bf16_matmul: false,
        },
    };
    let model = Trainer::new(ctx.clone(), weights, &input_tokens, train_cfg);

    let inputs = rand_u32_vec(batch_size * seq_len as usize, vocab_size);
    let targets = rand_u32_vec(batch_size * seq_len as usize, vocab_size);

    model
        .cross_entropy
        .set_grad_scale(1.0 / (seq_len as f32 * batch_size as f32));

    println!(
        "CHECK 8: single-layer memorization test (dim={dim}, heads={num_heads}, layers={num_layers}, vocab={vocab_size}, seq_len={seq_len}, batch={batch_size}, lr={lr})"
    );

    let mut final_loss = f32::MAX;
    for step in 0..600usize {
        model.zero_grad();
        let mut total_loss = 0.0f32;
        for i in 0..batch_size {
            let window = i * seq_len as usize..(i + 1) * seq_len as usize;

            model.input_tokens.copy_from_cpu(&inputs[window.clone()]);
            model
                .cross_entropy
                .target_tokens
                .copy_from_cpu(&targets[window]);

            model.zero_transient_grads();
            model.forward_fused();

            if step == 0 && i == 0 {
                let norm_f32 = |t: &Arc<Tensor<B>>| -> f64 {
                    let data: Vec<f32> = t.to_cpu();
                    data.iter()
                        .map(|&x| (x as f64) * (x as f64))
                        .sum::<f64>()
                        .sqrt()
                };
                println!(
                    "  [fwd-fp] layer0.add_2(resid into final_norm)={:.8} final_norm.out={:.8} softmax_probs={:.8}",
                    norm_f32(&model.layers[0].add_2.out_buffer),
                    norm_f32(&model.final_norm.out_buffer),
                    norm_f32(&model.lm_head.out_buffer),
                );
            }

            total_loss += model.cross_entropy.loss();
            model.backward_fused();
        }
        if use_clip {
            model.clip_grad_norm();
        }
        model.optimizer.step();

        let avg_loss = total_loss / batch_size as f32;
        final_loss = avg_loss;
        if step % 40 == 0 || step == 599 {
            println!("  step {step:3} | loss {avg_loss:.4}");
        }

        if step == 0 {
            let norm = |t: &Arc<Tensor<B>>| -> f64 {
                let data: Vec<f32> = t.to_cpu();
                data.iter()
                    .map(|&x| (x as f64) * (x as f64))
                    .sum::<f64>()
                    .sqrt()
            };
            println!(
                "  [fp-detail] lm_head.grad_input(=dY into final_norm.backward)={:.8}",
                norm(&model.lm_head.grad_input)
            );
            for (i, layer) in model.layers.iter().enumerate() {
                println!(
                    "  [fp-detail] layer {i}: QKV={:.6} O={:.6} FFNup={:.6} FFNdown={:.6} Norm1={:.6} Norm2={:.6}",
                    norm(&layer.qkv_proj.grad_weight),
                    norm(&layer.out_proj.grad_weight),
                    norm(&layer.ffn_up.grad_weight),
                    norm(&layer.ffn_down.grad_weight),
                    norm(&layer.norm_1.grad_weight),
                    norm(&layer.norm_2.grad_weight)
                );
            }

            println!(
                "  [fp-detail] embedding={:.6} final_norm={:.6} lm_head={:.6}",
                norm(&model.embedding.grad_table),
                norm(&model.final_norm.grad_weight),
                norm(&model.lm_head.grad_weight)
            );
        }
        if step < 10 {
            let grad_norm: f64 = model
                .trainable_params()
                .iter()
                .map(|(_, g)| {
                    let data: Vec<f32> = g.to_cpu();
                    data.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>()
                })
                .sum::<f64>()
                .sqrt();

            let m_norm: f64 = model
                .optimizer
                .moments
                .iter()
                .map(|(m, _)| {
                    let data: Vec<f32> = m.to_cpu();
                    data.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>()
                })
                .sum::<f64>()
                .sqrt();

            let v_norm: f64 = model
                .optimizer
                .moments
                .iter()
                .map(|(_, v)| {
                    let data: Vec<f32> = v.to_cpu();
                    data.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>()
                })
                .sum::<f64>()
                .sqrt();
            println!(
                "  [fingerprint] step {step} | loss {avg_loss:.6} | grad_norm {grad_norm:.6} | m_norm {m_norm:.6} | v_norm {v_norm:.6}"
            );
        }
        if avg_loss.is_nan() {
            println!("  RED FLAG: loss is NaN at step {step}");
            return false;
        }
    }

    let pass = final_loss < 0.1;

    println!(
        "CHECK 8: final loss = {final_loss:.4} -> {}",
        if pass { "PASS" } else { "FAIL" }
    );
    if !pass {
        println!(
            "  RED FLAG: tiny single-layer model could not memorize a fixed batch in 200 steps."
        );
        println!(
            "  This points to a bug in the training loop itself (optimizer, backward, or loss),"
        );
        println!("  not just a hyperparameter/scale issue with the full 117M model.");
    }
    pass
}

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

        // A: prefill(base_prompt), then one decode step, then another --
        // gen_a[1] is produced by the cache/decode-step path.
        let mut model_a = Model::for_chat(ctx.clone(), weights.clone(), cfg, max_context_len);
        let gen_a = model_a
            .generate(&base_prompt, 2, 0.0, 0, 1.0, 1.0)
            .expect("generate A failed");

        // B: prefill(base_prompt + gen_a[0]) in one shot -- same context as A
        // right before its second token, but computed entirely by the
        // full-sequence prefill path instead of cache + decode-step.
        let mut extended_prompt = base_prompt.clone();
        extended_prompt.push(gen_a[0]);

        let mut model_b = Model::for_chat(ctx.clone(), weights, cfg, max_context_len);
        let gen_b = model_b
            .generate(&extended_prompt, 1, 0.0, 0, 1.0, 1.0)
            .expect("generate B failed");

        let pass = gen_a.len() >= 2 && !gen_b.is_empty() && gen_a[1] == gen_b[0];

        println!(
            "CHECK 9: prefill-vs-decode parity -- A(prefill+decode)={gen_a:?} B(prefill-only, one token further)={gen_b:?} -> {}",
            if pass { "PASS" } else { "FAIL" }
        );

        if !pass {
            println!(
                "  RED FLAG: decode-step (KV-cache) path disagrees with the full-prefill path for the same context -- cache write, RoPE offset, or cached-attention kernel is likely wrong."
            );
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
        println!(
            "CHECK 10: {extra_tokens} tokens -- naive re-prefill = {naive_elapsed:.2?}, cached decode = {cached_elapsed:.2?}, speedup = {speedup:.2}x"
        );
        let pass = cached_elapsed < naive_elapsed;
        if !pass {
            println!(
                "  RED FLAG: cached decode was not faster than re-prefilling from scratch each step -- KV-cache isn't providing a speed benefit."
            );
        }
        pass
    }
}

fn run_diagnostics<B: Backend>(ctx: Arc<B>) {
    if std::env::var("DIAGNOSE_ONLY_CHECK8").is_ok() {
        let check = MemorizationCheck { ctx };
        let pass = check.run();
        println!(
            "CHECK 8 (memorization): {}",
            if pass { "PASS" } else { "FAIL" }
        );
        return;
    }

    println!("\n================= AKASHA TRAINING DIAGNOSTICS =================\n");

    let arch = ModelConfig::akasha_hall_1();
    let train_cfg = TrainConfig::hall1_pretrain();

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

    let input_tokens = Arc::new(Tensor::init_from_cpu(
        ctx.clone(),
        &vec![0u32; arch.seq_len as usize],
    ));
    let cfg = ModelConfig::new(
        vocab_size,
        arch.dim,
        arch.num_heads,
        arch.num_layers,
        arch.seq_len,
    );
    let weights = Arc::new(ModelWeights::random(ctx.clone(), &cfg));
    let full_model = Arc::new(Trainer::new(ctx.clone(), weights, &input_tokens, train_cfg));

    let checks: Vec<Box<dyn DiagnosticCheck>> = vec![
        Box::new(ParamCountCheck {
            model: full_model.clone(),
        }),
        Box::new(GradFlowCheck {
            ctx: ctx.clone(),
            vocab_size,
        }),
        Box::new(AccumulationCheck {
            ctx: ctx.clone(),
            vocab_size,
        }),
        Box::new(WeightDecayGroupsCheck {
            model: full_model,
            weight_decay: train_cfg.adam_weight_decay,
        }),
        Box::new(MemorizationCheck { ctx: ctx.clone() }),
        Box::new(PrefillDecodeParityCheck { ctx: ctx.clone() }),
        Box::new(DecodeCacheSpeedCheck { ctx }),
    ];
    run_all(&checks);
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

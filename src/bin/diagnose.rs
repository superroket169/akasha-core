use std::sync::Arc;
use std::time::Instant;

use akasha_core::config::{ADAM_WEIGHT_DECAY, GRAD_CLIP_NORM, ModelConfig};
use akasha_core::nn::akasha_model::AkashaModel;
use akasha_core::nn::cross_entropy::CrossEntropy;
use akasha_core::nn::rmsnorm::RMSNorm;
use akasha_core::nn::traits::Layer;
use akasha_core::nn::{Cache, InferenceSession};
use rand::Rng;
use wilupgu::{Backend, Binding, ComputeGraph, Tensor, TensorMode, WgpuBackend};

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .fold((0u32, f32::MIN), |(bi, bv), (i, &v)| {
            if v > bv { (i as u32, v) } else { (bi, bv) }
        })
        .0
}

fn l2_norm<B: Backend>(t: &Tensor<B>) -> f32 {
    let data: Vec<f32> = t.to_cpu();
    data.iter().map(|x| x * x).sum::<f32>().sqrt()
}

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

fn rand_f32_vec(n: usize, scale: f32) -> Vec<f32> {
    let mut rng = diag_rng();
    (0..n).map(|_| rng.gen_range(-scale..scale)).collect()
}

fn check1_param_count<B: Backend>(model: &AkashaModel<B>) -> (bool, u64) {
    let total: u64 = model
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
    (pass, total)
}

fn check2_grad_flow<B: Backend>(ctx: Arc<B>, vocab_size: u32) -> bool {
    use akasha_core::config::{DIM, NUM_HEADS, NUM_LAYERS};
    let seq_len = 16u32;

    let input_tokens = Arc::new(Tensor::init_from_cpu(
        ctx.clone(),
        &rand_u32_vec(seq_len as usize, vocab_size),
    ));
    let cfg = ModelConfig::new(vocab_size, DIM, NUM_HEADS, NUM_LAYERS, seq_len);
    let model = AkashaModel::new(ctx.clone(), &cfg, &input_tokens);

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
        "CHECK 2: 1-step grad flow test (seq_len={seq_len}, dim={DIM}, layers={NUM_LAYERS}, heads={NUM_HEADS})"
    );
    println!("  forward loss = {loss:.4}");

    let mut any_zero = false;
    let mut any_explosion = false;
    let mut layer_total_norms = Vec::with_capacity(NUM_LAYERS);

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
        let ratio = layer_total_norms[0] / layer_total_norms[NUM_LAYERS - 1].max(1e-12);
        if ratio > 1e3 {
            println!(
                "  RED FLAG: layer-0 grad sum ({:.4}) / layer-{} grad sum ({:.4}) = {:.1} -- looks like vanishing gradient",
                layer_total_norms[0],
                NUM_LAYERS - 1,
                layer_total_norms[NUM_LAYERS - 1],
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
        println!("  RED FLAG: at least one grad norm is exactly 0.0 -- gradient not flowing there");
    }
    if any_explosion {
        println!("  RED FLAG: at least one grad norm > 100 -- exploding gradient");
    }

    !any_zero && !any_explosion && !vanishing
}

fn check3_head_gather_scatter<B: Backend>(ctx: Arc<B>) -> bool {
    let seq_len = 4u32;
    let dim = 768u32;
    let head_dim = 64u32;
    let num_heads = dim / head_dim;

    let input_data = rand_f32_vec((seq_len * dim) as usize, 1.0);
    let input = Arc::new(Tensor::init_from_cpu(ctx.clone(), &input_data));

    let reconstructed = Arc::new(Tensor::init_from_cpu(
        ctx.clone(),
        &vec![0.0f32; (seq_len * dim) as usize],
    ));
    let mut g = ComputeGraph::new(ctx.clone());

    let mut head_bufs = Vec::with_capacity(num_heads as usize);
    for h in 0..num_heads {
        let head_offset = h * head_dim;
        let meta = Arc::new(Tensor::init_from_cpu(
            ctx.clone(),
            &[seq_len, dim, head_dim, head_offset],
        ));
        let head_buf = Arc::new(Tensor::init_from_cpu(
            ctx.clone(),
            &vec![0.0f32; (seq_len * head_dim) as usize],
        ));

        g.add_node(
            "HeadGather",
            &[
                Binding::new(0, &input.buffer, TensorMode::Input),
                Binding::new(1, &head_buf.buffer, TensorMode::Output),
                Binding::new(2, &meta.buffer, TensorMode::Meta),
            ],
            [(head_dim + 15) / 16, (seq_len + 15) / 16, 1],
        );
        g.add_node(
            "HeadScatter",
            &[
                Binding::new(0, &head_buf.buffer, TensorMode::Input),
                Binding::new(1, &reconstructed.buffer, TensorMode::Output),
                Binding::new(2, &meta.buffer, TensorMode::Meta),
            ],
            [(head_dim + 15) / 16, (seq_len + 15) / 16, 1],
        );
        head_bufs.push((head_buf, meta, head_offset));
    }
    g.execute();

    let got: Vec<f32> = reconstructed.to_cpu();
    let max_roundtrip_diff = got
        .iter()
        .zip(input_data.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let roundtrip_pass = max_roundtrip_diff < 1e-6;
    println!(
        "CHECK 3: HeadGather/HeadScatter round-trip identity, max diff = {max_roundtrip_diff:.8} -> {}",
        if roundtrip_pass { "PASS" } else { "FAIL" }
    );

    let (_, meta0, head_offset0) = &head_bufs[0];
    let ones = Arc::new(Tensor::init_from_cpu(
        ctx.clone(),
        &vec![1.0f32; (seq_len * head_dim) as usize],
    ));
    let analytic_grad = Arc::new(Tensor::init_from_cpu(
        ctx.clone(),
        &vec![0.0f32; (seq_len * dim) as usize],
    ));
    let mut g2 = ComputeGraph::new(ctx.clone());
    g2.add_node(
        "HeadScatter",
        &[
            Binding::new(0, &ones.buffer, TensorMode::Input),
            Binding::new(1, &analytic_grad.buffer, TensorMode::Output),
            Binding::new(2, &meta0.buffer, TensorMode::Meta),
        ],
        [(head_dim + 15) / 16, (seq_len + 15) / 16, 1],
    );
    g2.execute();
    let analytic: Vec<f32> = analytic_grad.to_cpu();

    let eps = 1e-2f32;
    let mut max_grad_diff = 0.0f32;
    let test_indices: Vec<usize> = (0..(seq_len * dim) as usize).step_by(37).collect();
    for &idx in &test_indices {
        let row = idx as u32 / dim;
        let col = idx as u32 % dim;
        let in_head = col >= *head_offset0 && col < *head_offset0 + head_dim;

        let mut xp = input_data.clone();
        xp[idx] += eps;
        let fp: f32 = if in_head {
            (0..head_dim)
                .map(|d| xp[(row * dim + head_offset0 + d) as usize])
                .sum()
        } else {
            (0..head_dim)
                .map(|d| input_data[(row * dim + head_offset0 + d) as usize])
                .sum()
        };
        let mut xm = input_data.clone();
        xm[idx] -= eps;
        let fm: f32 = if in_head {
            (0..head_dim)
                .map(|d| xm[(row * dim + head_offset0 + d) as usize])
                .sum()
        } else {
            (0..head_dim)
                .map(|d| input_data[(row * dim + head_offset0 + d) as usize])
                .sum()
        };
        let numeric = (fp - fm) / (2.0 * eps);
        let diff = (numeric - analytic[idx]).abs();
        max_grad_diff = max_grad_diff.max(diff);
    }
    let grad_pass = max_grad_diff < 1e-3;
    println!(
        "CHECK 3: HeadGather/HeadScatter backward, max numeric-vs-analytic diff = {max_grad_diff:.6} -> {}",
        if grad_pass { "PASS" } else { "FAIL" }
    );

    roundtrip_pass && grad_pass
}

fn cpu_rmsnorm_row_sum_f64(row: &[f32], w: &[f32], dim: usize) -> f64 {
    let eps = 1e-5f64;
    let ms: f64 = row.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / dim as f64;
    let rsqrt = 1.0 / (ms + eps).sqrt();
    (0..dim)
        .map(|d| (row[d] as f64) * rsqrt * (w[d] as f64))
        .sum()
}

fn cpu_rmsnorm_sum_f64(x: &[f32], w: &[f32], seq_len: usize, dim: usize) -> f64 {
    (0..seq_len)
        .map(|i| cpu_rmsnorm_row_sum_f64(&x[i * dim..(i + 1) * dim], w, dim))
        .sum()
}

fn check4_rmsnorm_backward<B: Backend>(ctx: Arc<B>) -> bool {
    let seq_len = 8u32;
    let dim = 768u32;
    let n = (seq_len * dim) as usize;

    let x_data = rand_f32_vec(n, 1.0);
    let w_data = rand_f32_vec(dim as usize, 1.0);

    let x_buf = Arc::new(Tensor::init_from_cpu(ctx.clone(), &x_data));
    let grad_out = Arc::new(Tensor::init_from_cpu(ctx.clone(), &vec![1.0f32; n]));
    let grad_in = Arc::new(Tensor::init_from_cpu(ctx.clone(), &vec![0.0f32; n]));

    let norm = RMSNorm::new(
        ctx.clone(),
        dim,
        seq_len,
        &w_data,
        &x_buf,
        &grad_out,
        &grad_in,
    );
    norm.forward();
    norm.backward();

    let got_grad_x: Vec<f32> = norm.grad_input.to_cpu();
    let got_grad_w: Vec<f32> = norm.grad_weight.to_cpu();

    let eps = 1e-3f64;
    let mut max_diff = 0.0f64;

    // x only affects its own row -- evaluate that row alone to avoid cancellation
    // from the other rows' unrelated magnitude.
    let x_indices: Vec<usize> = (0..n).step_by(53).collect();
    for &idx in &x_indices {
        let row_idx = idx / dim as usize;
        let row_start = row_idx * dim as usize;
        let row = &x_data[row_start..row_start + dim as usize];
        let local_idx = idx - row_start;

        let mut rp = row.to_vec();
        rp[local_idx] += eps as f32;
        let fp = cpu_rmsnorm_row_sum_f64(&rp, &w_data, dim as usize);
        let mut rm = row.to_vec();
        rm[local_idx] -= eps as f32;
        let fm = cpu_rmsnorm_row_sum_f64(&rm, &w_data, dim as usize);
        let numeric = (fp - fm) / (2.0 * eps);
        max_diff = max_diff.max((numeric - got_grad_x[idx] as f64).abs());
    }

    let w_indices: Vec<usize> = (0..dim as usize).step_by(11).collect();
    for &idx in &w_indices {
        let mut wp = w_data.clone();
        wp[idx] += eps as f32;
        let fp = cpu_rmsnorm_sum_f64(&x_data, &wp, seq_len as usize, dim as usize);
        let mut wm = w_data.clone();
        wm[idx] -= eps as f32;
        let fm = cpu_rmsnorm_sum_f64(&x_data, &wm, seq_len as usize, dim as usize);
        let numeric = (fp - fm) / (2.0 * eps);
        max_diff = max_diff.max((numeric - got_grad_w[idx] as f64).abs());
    }

    let pass = max_diff < 1e-3;
    println!(
        "CHECK 4: RMSNorm backward, max numeric-vs-analytic diff = {max_diff:.6} -> {}",
        if pass { "PASS" } else { "FAIL" }
    );
    pass
}

fn check5_accumulation<B: Backend>(ctx: Arc<B>, vocab_size: u32) -> bool {
    use akasha_core::config::{DIM, NUM_HEADS, NUM_LAYERS};
    let seq_len = 16u32;

    let input_tokens = Arc::new(Tensor::init_from_cpu(
        ctx.clone(),
        &rand_u32_vec(seq_len as usize, vocab_size),
    ));
    let cfg = ModelConfig::new(vocab_size, DIM, NUM_HEADS, NUM_LAYERS, seq_len);
    let model = AkashaModel::new(ctx.clone(), &cfg, &input_tokens);

    let inputs = rand_u32_vec(seq_len as usize, vocab_size);
    let targets = rand_u32_vec(seq_len as usize, vocab_size);

    let w_before: Vec<f32> = model.lm_head.weight.to_cpu();

    model.train_step(&inputs, &targets, 1, 1e-3, 0, 2); // step 0 of a 2-step accumulation cycle
    let w_after_step0: Vec<f32> = model.lm_head.weight.to_cpu();
    let grad_after_step0 = l2_norm(&model.lm_head.grad_weight);
    let delta0: f32 = w_before
        .iter()
        .zip(w_after_step0.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt();

    model.train_step(&inputs, &targets, 1, 1e-3, 1, 2); // step 1 = accumulation boundary
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
        println!("  RED FLAG: weights changed mid-cycle (before optimizer.step() should have run)");
    }
    if delta1 < 1e-7 {
        println!("  RED FLAG: weights did NOT change at accumulation boundary -- optimizer broken");
    }
    println!("CHECK 5: {}", if pass { "PASS" } else { "FAIL" });
    pass
}

fn check6_weight_decay_groups<B: Backend>(model: &AkashaModel<B>) -> bool {
    let params = model.trainable_params();
    let emb_in_group = params
        .iter()
        .any(|(w, _)| Arc::ptr_eq(w, &model.embedding.table));
    let norm_in_group = params
        .iter()
        .any(|(w, _)| Arc::ptr_eq(w, &model.final_norm.weight));

    println!("CHECK 6: AdamW weight-decay grouping");
    println!("  AdamW::new() is called with a single uniform parameter list and a single shared");
    println!("  StepConfig{{weight_decay,...}} applied identically to every tensor in it --");
    println!("  this codebase has no separate no_decay group at all.");
    println!("  Embedding table in the (only) weight_decay group: {emb_in_group}");
    println!("  RMSNorm (final_norm) weight in the (only) weight_decay group: {norm_in_group}");
    println!("  ADVISORY (not a correctness bug): embeddings and RMSNorm scale weights ARE being");
    println!("  weight-decayed at ADAM_WEIGHT_DECAY={ADAM_WEIGHT_DECAY}, which is non-standard --");
    println!(
        "  most GPT-2-style training setups exempt 1D params (norms, embeddings, biases) from decay."
    );
    emb_in_group && norm_in_group
}

fn check7_cross_entropy<B: Backend>(ctx: Arc<B>) -> bool {
    let vocab_size = 50257u32;
    let seq_len = 4u32;

    let logits = Arc::new(Tensor::init_from_cpu(
        ctx.clone(),
        &vec![0.0f32; (vocab_size * seq_len) as usize],
    ));
    let grad_logits = Arc::new(Tensor::init_from_cpu(
        ctx.clone(),
        &vec![0.0f32; (vocab_size * seq_len) as usize],
    ));

    let ce = CrossEntropy::new(ctx.clone(), vocab_size, seq_len, &logits, &grad_logits);
    ce.target_tokens.copy_from_cpu(&vec![0u32, 1, 2, 3]);
    ce.forward();

    let got = ce.loss();
    let expected = (vocab_size as f32).ln();
    let diff = (got - expected).abs();
    let pass = diff < 0.01;
    println!(
        "CHECK 7: cross-entropy on all-zero logits, expected ln({vocab_size}) = {expected:.4}, got = {got:.4}, diff = {diff:.4} -> {}",
        if pass { "PASS" } else { "FAIL" }
    );
    pass
}

fn check8_memorization<B: Backend>(ctx: Arc<B>) -> bool {
    for &(lr, clip) in &[(3e-3f32, false)] {
        println!("--- trying lr={lr} grad_clip={clip}, extended to 600 steps ---");
        if check8_run(ctx.clone(), lr, clip) {
            return true;
        }
    }
    false
}

fn check8_run<B: Backend>(ctx: Arc<B>, lr: f32, use_clip: bool) -> bool {
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
    let model = AkashaModel::new(ctx.clone(), &cfg, &input_tokens);

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
                    "  [fwd-fp] layer0.add_2(resid into final_norm)={:.8} final_norm.out={:.8} lm_head.out={:.8}",
                    norm_f32(&model.layers[0].add_2.in_out_buffer),
                    norm_f32(&model.final_norm.out_buffer),
                    norm_f32(&model.lm_head.out_buffer),
                );
            }
            total_loss += model.cross_entropy.loss();
            model.backward_fused();
        }
        if use_clip {
            model.clip_grad_norm(GRAD_CLIP_NORM);
        }
        model.optimizer.step(lr, 0.9, 0.95, 0.0);

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

fn check9_kv_cache_equivalence<B: Backend>(ctx: Arc<B>) -> bool {
    let dim = 64u32;
    let num_heads = 4u32;
    let num_layers = 2usize;
    let vocab_size = 64u32;
    let seq_len = 16u32;
    let prompt_len = 4usize;
    let total_len = 14usize;

    let full_sequence = rand_u32_vec(total_len, vocab_size);
    let prompt = &full_sequence[..prompt_len];

    let input_tokens = Arc::new(Tensor::init_from_cpu(
        ctx.clone(),
        &vec![0u32; seq_len as usize],
    ));
    let cfg = ModelConfig::new(vocab_size, dim, num_heads, num_layers, seq_len);
    let model = AkashaModel::new(ctx.clone(), &cfg, &input_tokens);

    let mut logits_a: Vec<Vec<f32>> = Vec::new();
    let seq_len_us = seq_len as usize;
    let vocab_us = vocab_size as usize;
    for cur_len in prompt_len..full_sequence.len() {
        let (start, pred_pos) = if cur_len >= seq_len_us {
            (cur_len - seq_len_us, seq_len_us - 1)
        } else {
            (0, cur_len - 1)
        };
        let mut window = vec![0u32; seq_len_us];
        let slice = &full_sequence[start..cur_len];
        window[..slice.len()].copy_from_slice(slice);
        model.input_tokens.copy_from_cpu(&window);
        model.forward_fused();
        let logits = model.lm_head.out_buffer.to_cpu();
        logits_a.push(logits[pred_pos * vocab_us..(pred_pos + 1) * vocab_us].to_vec());
    }

    // ---- KV-cache prefill + decode ----
    let model_arc = Arc::new(model);
    let mut session = InferenceSession::new(ctx.clone(), model_arc, seq_len);
    session.replace_cache(Cache::new(ctx.clone(), num_layers, dim, seq_len));

    let mut logits_b: Vec<Vec<f32>> = vec![session.prefill(prompt)];
    for &t in &full_sequence[prompt_len..full_sequence.len() - 1] {
        logits_b.push(session.decode_step(t));
    }

    assert_eq!(logits_a.len(), logits_b.len(), "step count mismatch");
    let max_diff = logits_a
        .iter()
        .zip(logits_b.iter())
        .flat_map(|(a, b)| a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()))
        .fold(0.0f32, f32::max);

    let pass = max_diff < 1e-3;
    println!(
        "CHECK 9: KV-cache decode vs naive sliding-window forward, {} teacher-forced steps, max logit diff = {max_diff:.6} -> {}",
        logits_a.len(),
        if pass { "PASS" } else { "FAIL" }
    );
    pass
}

fn check10_kv_cache_speed<B: Backend>(ctx: Arc<B>) {
    let dim = 64u32;
    let num_heads = 4u32;
    let num_layers = 2usize;
    let vocab_size = 64u32;
    let seq_len = 64u32;
    let prompt_len = 4usize;
    let n_new_tokens = 50usize;

    let prompt = rand_u32_vec(prompt_len, vocab_size);

    // ---- naive sliding-window path ----
    let input_tokens_a = Arc::new(Tensor::init_from_cpu(
        ctx.clone(),
        &vec![0u32; seq_len as usize],
    ));
    let cfg = ModelConfig::new(vocab_size, dim, num_heads, num_layers, seq_len);
    let model_a = AkashaModel::new(ctx.clone(), &cfg, &input_tokens_a);
    let mut tokens = prompt.clone();
    let seq_len_us = seq_len as usize;
    let vocab_us = vocab_size as usize;
    let start = Instant::now();
    for _ in 0..n_new_tokens {
        let cur_len = tokens.len();
        let (start_idx, pred_pos) = if cur_len >= seq_len_us {
            (cur_len - seq_len_us, seq_len_us - 1)
        } else {
            (0, cur_len - 1)
        };
        let mut window = vec![0u32; seq_len_us];
        let slice = &tokens[start_idx..];
        window[..slice.len()].copy_from_slice(slice);
        model_a.input_tokens.copy_from_cpu(&window);
        model_a.forward_fused();
        let logits = model_a.lm_head.out_buffer.to_cpu();
        let row = &logits[pred_pos * vocab_us..(pred_pos + 1) * vocab_us];
        tokens.push(argmax(row));
    }
    let naive_elapsed = start.elapsed();

    // ---- KV-cache path ----
    let input_tokens_b = Arc::new(Tensor::init_from_cpu(
        ctx.clone(),
        &vec![0u32; seq_len as usize],
    ));
    let model_b = Arc::new(AkashaModel::new(ctx.clone(), &cfg, &input_tokens_b));
    let mut session = InferenceSession::new(ctx.clone(), model_b, seq_len);
    session.replace_cache(Cache::new(ctx.clone(), num_layers, dim, seq_len));

    let start = Instant::now();
    let mut logits = session.prefill(&prompt);
    for _ in 0..n_new_tokens - 1 {
        logits = session.decode_step(argmax(&logits));
    }
    let _ = logits;
    let cached_elapsed = start.elapsed();

    println!(
        "CHECK 10: KV-cache decode speed (tiny config: dim={dim}, layers={num_layers}, heads={num_heads}, vocab={vocab_size}, seq_len={seq_len}, {n_new_tokens} generated tokens)"
    );
    println!(
        "  naive sliding-window: {naive_elapsed:?} ({:.1} tok/s)",
        n_new_tokens as f64 / naive_elapsed.as_secs_f64()
    );
    println!(
        "  KV-cache decode:      {cached_elapsed:?} ({:.1} tok/s)",
        n_new_tokens as f64 / cached_elapsed.as_secs_f64()
    );
    println!(
        "  NOTE: at this tiny scale GPU dispatch overhead dominates -- the production-scale (dim=768, seq_len=512) speedup is expected to be far larger."
    );
}

fn run_diagnostics<B: Backend>(ctx: Arc<B>) {
    if std::env::var("DIAGNOSE_ONLY_CHECK8").is_ok() {
        let c8_pass = check8_memorization(ctx.clone());
        println!(
            "CHECK 8 (memorization): {}",
            if c8_pass { "PASS" } else { "FAIL" }
        );
        return;
    }

    if std::env::var("DIAGNOSE_ONLY_CHECK9").is_ok() {
        let c9_pass = check9_kv_cache_equivalence(ctx.clone());
        println!();
        check10_kv_cache_speed(ctx.clone());
        println!(
            "\nCHECK 9 (KV-cache equivalence): {}",
            if c9_pass { "PASS" } else { "FAIL" }
        );
        return;
    }

    println!("\n================= AKASHA TRAINING DIAGNOSTICS =================\n");

    use akasha_core::config::{DIM, NUM_HEADS, NUM_LAYERS, SEQ_LEN, VOCAB_SIZE};

    let vocab_size: u32 = std::env::var("DIAGNOSE_VOCAB_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(VOCAB_SIZE);
    if vocab_size != VOCAB_SIZE {
        println!(
            "NOTE: DIAGNOSE_VOCAB_SIZE override active -- using vocab_size={vocab_size} instead of {VOCAB_SIZE}\n"
        );
    }

    let input_tokens = Arc::new(Tensor::init_from_cpu(
        ctx.clone(),
        &vec![0u32; SEQ_LEN as usize],
    ));
    let cfg = ModelConfig::new(vocab_size, DIM, NUM_HEADS, NUM_LAYERS, SEQ_LEN);
    let full_model = AkashaModel::new(ctx.clone(), &cfg, &input_tokens);

    let (c1_pass, c1_count) = check1_param_count(&full_model);
    println!();
    let c2_pass = check2_grad_flow(ctx.clone(), vocab_size);
    println!();
    let c3_pass = check3_head_gather_scatter(ctx.clone());
    println!();
    let c4_pass = check4_rmsnorm_backward(ctx.clone());
    println!();
    let c5_pass = check5_accumulation(ctx.clone(), vocab_size);
    println!();
    let c6_pass = check6_weight_decay_groups(&full_model);
    println!();
    let c7_pass = check7_cross_entropy(ctx.clone());
    println!();
    let c8_pass = check8_memorization(ctx.clone());
    println!();
    let c9_pass = check9_kv_cache_equivalence(ctx.clone());
    println!();
    check10_kv_cache_speed(ctx.clone());

    println!("\n================= SUMMARY =================");
    println!(
        "CHECK 1 (param count):          {} -- {} parameters",
        if c1_pass { "PASS" } else { "FAIL" },
        c1_count
    );
    println!(
        "CHECK 2 (gradient flow):        {}",
        if c2_pass { "PASS" } else { "FAIL" }
    );
    println!(
        "CHECK 3 (HeadGather/Scatter):   {}",
        if c3_pass { "PASS" } else { "FAIL" }
    );
    println!(
        "CHECK 4 (RMSNorm backward):     {}",
        if c4_pass { "PASS" } else { "FAIL" }
    );
    println!(
        "CHECK 5 (grad accumulation):    {}",
        if c5_pass { "PASS" } else { "FAIL" }
    );
    println!(
        "CHECK 6 (weight decay groups):  {} (advisory)",
        if c6_pass { "PASS" } else { "FAIL" }
    );
    println!(
        "CHECK 7 (loss function):        {}",
        if c7_pass { "PASS" } else { "FAIL" }
    );
    println!(
        "CHECK 8 (memorization):          {}",
        if c8_pass { "PASS" } else { "FAIL" }
    );
    println!(
        "CHECK 9 (KV-cache equivalence):  {}",
        if c9_pass { "PASS" } else { "FAIL" }
    );
    println!("CHECK 10 (KV-cache speed):       informational, see above");
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

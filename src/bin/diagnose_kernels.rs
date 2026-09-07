use std::sync::Arc;

use akasha_core::diagnostic::{DiagnosticCheck, run_all};
use akasha_core::nn::{Layer, RMSNorm};
use akasha_core::shaders;
use rand::Rng;
use wilupgu::{Backend, Binding, ComputeGraph, Tensor, TensorMode, WgpuBackend};

static DIAG_RNG: std::sync::OnceLock<std::sync::Mutex<rand::rngs::StdRng>> =
    std::sync::OnceLock::new();
fn diag_rng() -> std::sync::MutexGuard<'static, rand::rngs::StdRng> {
    DIAG_RNG
        .get_or_init(|| std::sync::Mutex::new(rand::SeedableRng::seed_from_u64(42)))
        .lock()
        .unwrap()
}

fn rand_f32_vec(n: usize, scale: f32) -> Vec<f32> {
    let mut rng = diag_rng();
    (0..n).map(|_| rng.gen_range(-scale..scale)).collect()
}

struct HeadGatherScatterCheck<B: Backend> {
    ctx: Arc<B>,
}

impl<B: Backend> DiagnosticCheck for HeadGatherScatterCheck<B> {
    fn name(&self) -> &'static str {
        "CHECK 3 (HeadGather/Scatter)"
    }

    fn run(&self) -> bool {
        let ctx = self.ctx.clone();
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
                &shaders::HEAD_GATHER,
                &[
                    Binding::new(0, &input.buffer, TensorMode::Input),
                    Binding::new(1, &head_buf.buffer, TensorMode::Output),
                    Binding::new(2, &meta.buffer, TensorMode::Meta),
                ],
                [(head_dim + 15) / 16, (seq_len + 15) / 16, 1],
            );
            g.add_node(
                &shaders::HEAD_SCATTER,
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
            &shaders::HEAD_SCATTER,
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

struct RmsNormBackwardCheck<B: Backend> {
    ctx: Arc<B>,
}

impl<B: Backend> DiagnosticCheck for RmsNormBackwardCheck<B> {
    fn name(&self) -> &'static str {
        "CHECK 4 (RMSNorm backward)"
    }

    fn run(&self) -> bool {
        let ctx = self.ctx.clone();
        let seq_len = 8u32;
        let dim = 768u32;
        let n = (seq_len * dim) as usize;

        let x_data = rand_f32_vec(n, 1.0);
        let w_data = rand_f32_vec(dim as usize, 1.0);

        let x_buf = Arc::new(Tensor::init_from_cpu(ctx.clone(), &x_data));
        let w_buf = Arc::new(Tensor::init_from_cpu(ctx.clone(), &w_data));
        let grad_out = Arc::new(Tensor::init_from_cpu(ctx.clone(), &vec![1.0f32; n]));
        let grad_in = Arc::new(Tensor::init_from_cpu(ctx.clone(), &vec![0.0f32; n]));

        let norm = RMSNorm::new(
            ctx.clone(),
            dim,
            seq_len,
            1e-5,
            &w_buf,
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
}

struct CrossEntropyCheck<B: Backend> {
    ctx: Arc<B>,
}

impl<B: Backend> DiagnosticCheck for CrossEntropyCheck<B> {
    fn name(&self) -> &'static str {
        "CHECK 7 (loss function)"
    }

    fn run(&self) -> bool {
        use akasha_core::nn::CrossEntropy;

        let ctx = self.ctx.clone();
        let vocab_size = 50257u32;
        let seq_len = 4u32;

        let logits = Arc::new(Tensor::init_from_cpu(
            ctx.clone(),
            &vec![0.0f32; (vocab_size * seq_len) as usize],
        ));

        let ce = CrossEntropy::new(ctx.clone(), vocab_size, seq_len, &logits);
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
}

fn run_diagnostics<B: Backend>(ctx: Arc<B>) {
    println!("\n================= AKASHA KERNEL DIAGNOSTICS =================\n");
    let checks: Vec<Box<dyn DiagnosticCheck>> = vec![
        Box::new(HeadGatherScatterCheck { ctx: ctx.clone() }),
        Box::new(RmsNormBackwardCheck { ctx: ctx.clone() }),
        Box::new(CrossEntropyCheck { ctx }),
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

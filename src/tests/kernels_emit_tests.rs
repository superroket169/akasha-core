use super::*;

#[cfg(test)]
mod flash_attention_validation {
    use super::*;
    use wilupgu::{ComputeGraph, WgpuBackend};

    fn rand_vec(n: usize, seed: u64) -> Vec<Real> {
        let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let bits = ((state >> 40) as u32) & 0x00FF_FFFF;
                (bits as f32 / 0x00FF_FFFF as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn max_abs_diff(a: &[Real], b: &[Real]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    #[allow(clippy::needless_range_loop)]
    fn cpu_attention(
        q: &[Real],
        k: &[Real],
        v: &[Real],
        grad_out: &[Real],
        seq_len: usize,
        dim: usize,
        head_dim: usize,
    ) -> (Vec<Real>, Vec<Real>, Vec<Real>, Vec<Real>) {
        let scale = 1.0 / (head_dim as f32).sqrt();
        let n = seq_len * dim;
        let (mut out, mut dq, mut dk, mut dv) =
            (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);

        for h0 in (0..dim).step_by(head_dim) {
            let at = |i: usize, c: usize| i * dim + h0 + c;

            let mut p = vec![0.0f32; seq_len * seq_len];
            for i in 0..seq_len {
                let row = &mut p[i * seq_len..(i + 1) * seq_len];
                for j in 0..=i {
                    row[j] = scale
                        * (0..head_dim)
                            .map(|c| q[at(i, c)] * k[at(j, c)])
                            .sum::<f32>();
                }
                let max = row[..=i].iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let mut sum = 0.0;
                for j in 0..=i {
                    row[j] = (row[j] - max).exp();
                    sum += row[j];
                }
                for j in 0..=i {
                    row[j] /= sum;
                }
            }

            // out = P V ; dV = P^T dO
            for i in 0..seq_len {
                for j in 0..=i {
                    let pij = p[i * seq_len + j];
                    for c in 0..head_dim {
                        out[at(i, c)] += pij * v[at(j, c)];
                        dv[at(j, c)] += pij * grad_out[at(i, c)];
                    }
                }
            }

            // dS = P o (dP - rowsum(P o dP)), dP = dO V^T; then dQ/dK
            for i in 0..seq_len {
                let dp: Vec<f32> = (0..=i)
                    .map(|j| {
                        (0..head_dim)
                            .map(|c| grad_out[at(i, c)] * v[at(j, c)])
                            .sum()
                    })
                    .collect();
                let dot: f32 = (0..=i).map(|j| p[i * seq_len + j] * dp[j]).sum();
                for j in 0..=i {
                    let ds = scale * p[i * seq_len + j] * (dp[j] - dot);
                    for c in 0..head_dim {
                        dq[at(i, c)] += ds * k[at(j, c)];
                        dk[at(j, c)] += ds * q[at(i, c)];
                    }
                }
            }
        }
        (out, dq, dk, dv)
    }

    #[test]
    fn flash_attention_matches_cpu_reference() {
        // head_dim is pinned at 64 everywhere -- the wgsl kernel is
        // hardcoded to it (see assert_flash_head_dim); seq_len/num_heads
        // still vary for coverage.
        check(8, 2, 64);
        check(37, 3, 64);
        check(65, 12, 64);
    }

    #[test]
    #[should_panic(expected = "head_dim must be 64")]
    fn flash_attention_rejects_wrong_head_dim() {
        check(8, 2, 16);
    }

    fn check(seq_len: u32, num_heads: u32, head_dim: u32) {
        let ctx = Arc::new(pollster::block_on(WgpuBackend::new()));

        let dim: u32 = num_heads * head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let n = (seq_len * dim) as usize;

        let q_cpu = rand_vec(n, 1);
        let k_cpu = rand_vec(n, 2);
        let v_cpu = rand_vec(n, 3);
        let grad_out_cpu = rand_vec(n, 4);

        let q_buf = Arc::new(Tensor::init_from_cpu(ctx.clone(), &q_cpu));
        let k_buf = Arc::new(Tensor::init_from_cpu(ctx.clone(), &k_cpu));
        let v_buf = Arc::new(Tensor::init_from_cpu(ctx.clone(), &v_cpu));
        let grad_output = Arc::new(Tensor::init_from_cpu(ctx.clone(), &grad_out_cpu));

        let zeros = || Arc::new(Tensor::init_from_cpu(ctx.clone(), &vec![0.0 as Real; n]));

        let (ref_out, ref_dq, ref_dk, ref_dv) = cpu_attention(
            &q_cpu,
            &k_cpu,
            &v_cpu,
            &grad_out_cpu,
            seq_len as usize,
            dim as usize,
            head_dim as usize,
        );

        // ---- new: flash_attention + flash_attention_bwd ----
        let new_out = zeros();
        let (new_grad_q, new_grad_k, new_grad_v) = (zeros(), zeros(), zeros());
        let shape = FlashAttnMeta {
            seq_len,
            dim,
            head_dim,
            scale,
            row_offset: 0,
        };

        let mut new_fwd = ComputeGraph::new(ctx.clone());
        let mut gb = GraphBuilder::train(&mut new_fwd);
        let saved = flash_attention(&mut gb, &q_buf, &k_buf, &v_buf, &new_out, shape);
        new_fwd.execute();
        ctx.synchronize();

        let mut new_bwd = ComputeGraph::new(ctx.clone());
        let mut gb = GraphBuilder::train(&mut new_bwd);
        flash_attention_bwd(
            &mut gb,
            &q_buf,
            &k_buf,
            &v_buf,
            &saved,
            &grad_output,
            &new_grad_q,
            &new_grad_k,
            &new_grad_v,
            shape,
        );
        new_bwd.execute();
        ctx.synchronize();

        let tol = 1e-3;

        let ctx_msg = format!("seq_len={seq_len} num_heads={num_heads} head_dim={head_dim}");

        let out_diff = max_abs_diff(&ref_out, &new_out.to_cpu::<Real>());
        assert!(
            out_diff < tol,
            "forward output mismatch ({ctx_msg}): max_abs_diff={out_diff}"
        );

        let dq_diff = max_abs_diff(&ref_dq, &new_grad_q.to_cpu::<Real>());
        assert!(
            dq_diff < tol,
            "dQ mismatch ({ctx_msg}): max_abs_diff={dq_diff}"
        );

        let dk_diff = max_abs_diff(&ref_dk, &new_grad_k.to_cpu::<Real>());
        assert!(
            dk_diff < tol,
            "dK mismatch ({ctx_msg}): max_abs_diff={dk_diff}"
        );

        let dv_diff = max_abs_diff(&ref_dv, &new_grad_v.to_cpu::<Real>());
        assert!(
            dv_diff < tol,
            "dV mismatch ({ctx_msg}): max_abs_diff={dv_diff}"
        );
    }
}

#[cfg(test)]
mod kernel_fusion_validation {
    use super::*;
    use wilupgu::{ComputeGraph, WgpuBackend};

    fn rand_vec(n: usize, seed: u64) -> Vec<Real> {
        let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let bits = ((state >> 40) as u32) & 0x00FF_FFFF;
                (bits as f32 / 0x00FF_FFFF as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn max_abs_diff(a: &[Real], b: &[Real]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn rope_qk_matches_two_rope_calls() {
        check_rope(8, 8, 4);
        check_rope(37, 12, 4);
        check_rope(65, 48, 16);
    }

    fn check_rope(seq_len: u32, dim: u32, head_dim: u32) {
        let ctx = Arc::new(pollster::block_on(WgpuBackend::new()));
        let n = (seq_len * dim) as usize;
        let shape = RopeMeta {
            seq_len,
            dim,
            head_dim,
            row_offset: 0,
        };

        let q_data = rand_vec(n, 10);
        let k_data = rand_vec(n, 20);
        let dq_data = rand_vec(n, 30);
        let dk_data = rand_vec(n, 40);

        // ---- forward ----
        let old_q = Arc::new(Tensor::init_from_cpu(ctx.clone(), &q_data));
        let old_k = Arc::new(Tensor::init_from_cpu(ctx.clone(), &k_data));
        let mut old_graph = ComputeGraph::new(ctx.clone());
        let mut gb = GraphBuilder::train(&mut old_graph);
        rope(&mut gb, &old_q, shape);
        rope(&mut gb, &old_k, shape);
        old_graph.execute();
        ctx.synchronize();

        let new_q = Arc::new(Tensor::init_from_cpu(ctx.clone(), &q_data));
        let new_k = Arc::new(Tensor::init_from_cpu(ctx.clone(), &k_data));
        let mut new_graph = ComputeGraph::new(ctx.clone());
        let mut gb = GraphBuilder::train(&mut new_graph);
        rope_qk(&mut gb, &new_q, &new_k, shape);
        new_graph.execute();
        ctx.synchronize();

        let ctx_msg = format!("seq_len={seq_len} dim={dim} head_dim={head_dim}");
        let q_diff = max_abs_diff(&old_q.to_cpu::<Real>(), &new_q.to_cpu::<Real>());
        assert!(q_diff < 1e-4, "rope_qk Q mismatch ({ctx_msg}): {q_diff}");
        let k_diff = max_abs_diff(&old_k.to_cpu::<Real>(), &new_k.to_cpu::<Real>());
        assert!(k_diff < 1e-4, "rope_qk K mismatch ({ctx_msg}): {k_diff}");

        // ---- backward ----
        let old_dq = Arc::new(Tensor::init_from_cpu(ctx.clone(), &dq_data));
        let old_dk = Arc::new(Tensor::init_from_cpu(ctx.clone(), &dk_data));
        let mut old_bwd = ComputeGraph::new(ctx.clone());
        let mut gb = GraphBuilder::train(&mut old_bwd);
        rope_bwd(&mut gb, &old_dq, shape);
        rope_bwd(&mut gb, &old_dk, shape);
        old_bwd.execute();
        ctx.synchronize();

        let new_dq = Arc::new(Tensor::init_from_cpu(ctx.clone(), &dq_data));
        let new_dk = Arc::new(Tensor::init_from_cpu(ctx.clone(), &dk_data));
        let mut new_bwd = ComputeGraph::new(ctx.clone());
        let mut gb = GraphBuilder::train(&mut new_bwd);
        rope_bwd_qk(&mut gb, &new_dq, &new_dk, shape);
        new_bwd.execute();
        ctx.synchronize();

        let dq_diff = max_abs_diff(&old_dq.to_cpu::<Real>(), &new_dq.to_cpu::<Real>());
        assert!(
            dq_diff < 1e-4,
            "rope_bwd_qk dQ mismatch ({ctx_msg}): {dq_diff}"
        );
        let dk_diff = max_abs_diff(&old_dk.to_cpu::<Real>(), &new_dk.to_cpu::<Real>());
        assert!(
            dk_diff < 1e-4,
            "rope_bwd_qk dK mismatch ({ctx_msg}): {dk_diff}"
        );
    }

    #[test]
    fn qkv_split_matches_three_head_gathers() {
        check_qkv(8, 4);
        check_qkv(37, 12);
        check_qkv(65, 48);
    }

    fn check_qkv(seq_len: u32, dim: u32) {
        let ctx = Arc::new(pollster::block_on(WgpuBackend::new()));
        let n = (seq_len * dim) as usize;
        let src_data = rand_vec(n * 3, 50);
        let src = Arc::new(Tensor::init_from_cpu(ctx.clone(), &src_data));
        let zeros = || Arc::new(Tensor::init_from_cpu(ctx.clone(), &vec![0.0 as Real; n]));

        // ---- forward ----
        let (old_q, old_k, old_v) = (zeros(), zeros(), zeros());
        let mut old_graph = ComputeGraph::new(ctx.clone());
        let mut gb = GraphBuilder::train(&mut old_graph);
        for (buf, off) in [(&old_q, 0), (&old_k, dim), (&old_v, 2 * dim)] {
            head_gather(
                &mut gb,
                &src,
                buf,
                HeadMoveMeta::qkv_slice(seq_len, dim, off),
            );
        }
        old_graph.execute();
        ctx.synchronize();

        let (new_q, new_k, new_v) = (zeros(), zeros(), zeros());
        let mut new_graph = ComputeGraph::new(ctx.clone());
        let mut gb = GraphBuilder::train(&mut new_graph);
        qkv_split(
            &mut gb,
            &src,
            &new_q,
            &new_k,
            &new_v,
            HeadMoveMeta::qkv_slice(seq_len, dim, 0),
        );
        new_graph.execute();
        ctx.synchronize();

        let ctx_msg = format!("seq_len={seq_len} dim={dim}");
        assert!(
            max_abs_diff(&old_q.to_cpu::<Real>(), &new_q.to_cpu::<Real>()) < 1e-6,
            "qkv_split Q mismatch ({ctx_msg})"
        );
        assert!(
            max_abs_diff(&old_k.to_cpu::<Real>(), &new_k.to_cpu::<Real>()) < 1e-6,
            "qkv_split K mismatch ({ctx_msg})"
        );
        assert!(
            max_abs_diff(&old_v.to_cpu::<Real>(), &new_v.to_cpu::<Real>()) < 1e-6,
            "qkv_split V mismatch ({ctx_msg})"
        );

        // ---- backward ----
        let grad_q = Arc::new(Tensor::init_from_cpu(ctx.clone(), &rand_vec(n, 60)));
        let grad_k = Arc::new(Tensor::init_from_cpu(ctx.clone(), &rand_vec(n, 70)));
        let grad_v = Arc::new(Tensor::init_from_cpu(ctx.clone(), &rand_vec(n, 80)));

        let old_dst = Arc::new(Tensor::init_from_cpu(
            ctx.clone(),
            &vec![0.0 as Real; n * 3],
        ));
        let mut old_bwd = ComputeGraph::new(ctx.clone());
        let mut gb = GraphBuilder::train(&mut old_bwd);
        for (buf, off) in [(&grad_q, 0), (&grad_k, dim), (&grad_v, 2 * dim)] {
            head_scatter(
                &mut gb,
                buf,
                &old_dst,
                HeadMoveMeta::qkv_slice(seq_len, dim, off),
            );
        }
        old_bwd.execute();
        ctx.synchronize();

        let new_dst = Arc::new(Tensor::init_from_cpu(
            ctx.clone(),
            &vec![0.0 as Real; n * 3],
        ));
        let mut new_bwd = ComputeGraph::new(ctx.clone());
        let mut gb = GraphBuilder::train(&mut new_bwd);
        qkv_scatter(
            &mut gb,
            &grad_q,
            &grad_k,
            &grad_v,
            &new_dst,
            HeadMoveMeta::qkv_slice(seq_len, dim, 0),
        );
        new_bwd.execute();
        ctx.synchronize();

        assert!(
            max_abs_diff(&old_dst.to_cpu::<Real>(), &new_dst.to_cpu::<Real>()) < 1e-6,
            "qkv_scatter mismatch ({ctx_msg})"
        );
    }
}

// Decode-path kernels (strided-cache attention, GEMV) vs plain-Rust references.
#[cfg(test)]
mod decode_kernel_validation {
    use super::*;
    use crate::nn::kernels::meta::AttnCachedMeta;
    use wilupgu::{ComputeGraph, WgpuBackend};

    fn rand_vec(n: usize, seed: u64) -> Vec<Real> {
        let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let bits = ((state >> 40) as u32) & 0x00FF_FFFF;
                (bits as f32 / 0x00FF_FFFF as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn max_abs_diff(a: &[Real], b: &[Real]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn cached_attention_matches_cpu_reference() {
        check_attn(1, 2, 4);
        check_attn(5, 3, 16);
        check_attn(33, 12, 64);
    }

    fn check_attn(attn_len: u32, num_heads: u32, head_dim: u32) {
        let ctx = Arc::new(pollster::block_on(WgpuBackend::new()));
        let dim = num_heads * head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();
        // Grid sized for a larger context than attn_len, like a real decode
        // step mid-generation: the meta must bound the live work.
        let max_ctx = 64u32;

        let q_cpu = rand_vec(dim as usize, 1);
        let k_cpu = rand_vec((max_ctx * dim) as usize, 2);
        let v_cpu = rand_vec((max_ctx * dim) as usize, 3);

        let q = Arc::new(Tensor::init_from_cpu(ctx.clone(), &q_cpu));
        let k_cache = Arc::new(Tensor::init_from_cpu(ctx.clone(), &k_cpu));
        let v_cache = Arc::new(Tensor::init_from_cpu(ctx.clone(), &v_cpu));
        let scores = Arc::new(Tensor::init_from_cpu(
            ctx.clone(),
            &vec![0.0 as Real; (num_heads * max_ctx) as usize],
        ));
        let out = Arc::new(Tensor::init_from_cpu(
            ctx.clone(),
            &vec![0.0 as Real; dim as usize],
        ));

        let softmax_shape = SoftmaxRectMeta {
            num_rows: num_heads,
            width: attn_len,
            scale,
        };
        let attn_meta = AttnCachedMeta {
            attn_len,
            dim,
            head_dim,
        }
        .upload(&ctx);
        let softmax_meta = softmax_shape.upload(&ctx);

        let mut graph = ComputeGraph::new(ctx.clone());
        let mut gb = GraphBuilder::decode(&mut graph);
        attn_qk_cached_with(
            &mut gb, &q, &k_cache, &scores, num_heads, max_ctx, &attn_meta,
        );
        softmax_rect_with(&mut gb, &scores, softmax_shape, &softmax_meta);
        attn_av_cached_with(&mut gb, &scores, &v_cache, &out, dim, &attn_meta);
        graph.execute();
        ctx.synchronize();

        // CPU reference: per head, softmax(scale * q.K^T) @ V off the cache
        let (al, d, hd) = (attn_len as usize, dim as usize, head_dim as usize);
        let mut ref_out = vec![0.0f32; d];
        for h in 0..num_heads as usize {
            let q_off = h * hd;
            let mut p: Vec<f32> = (0..al)
                .map(|j| {
                    scale
                        * (0..hd)
                            .map(|c| q_cpu[q_off + c] * k_cpu[j * d + q_off + c])
                            .sum::<f32>()
                })
                .collect();
            let max = p.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let sum: f32 = p
                .iter_mut()
                .map(|x| {
                    *x = (*x - max).exp();
                    *x
                })
                .sum();
            for j in 0..al {
                p[j] /= sum;
                for c in 0..hd {
                    ref_out[q_off + c] += p[j] * v_cpu[j * d + q_off + c];
                }
            }
        }

        let diff = max_abs_diff(&ref_out, &out.to_cpu::<Real>());
        assert!(
            diff < 1e-4,
            "cached attention mismatch (attn_len={attn_len} num_heads={num_heads} head_dim={head_dim}): {diff}"
        );
    }

    // m=1 matmuls route to GEMV/GEMV_ADD inside matmul_with/matmul_add_with;
    // n,k chosen off the 256/16 grid boundaries to exercise bounds checks.
    #[test]
    fn gemv_routing_matches_cpu_reference() {
        let ctx = Arc::new(pollster::block_on(WgpuBackend::new()));
        let (n, k) = (301u32, 19u32);

        let a_cpu = rand_vec(k as usize, 10);
        let b_cpu = rand_vec((k * n) as usize, 20);
        let c0_cpu = rand_vec(n as usize, 30);

        let a = Arc::new(Tensor::init_from_cpu(ctx.clone(), &a_cpu));
        let b = Arc::new(Tensor::init_from_cpu(ctx.clone(), &b_cpu));
        let c = Arc::new(Tensor::init_from_cpu(
            ctx.clone(),
            &vec![0.0 as Real; n as usize],
        ));
        let c_add = Arc::new(Tensor::init_from_cpu(ctx.clone(), &c0_cpu));

        let shape = MatMulMeta { m: 1, n, k };
        let mut graph = ComputeGraph::new(ctx.clone());
        let mut gb = GraphBuilder::train(&mut graph);
        matmul(&mut gb, &a, &b, &c, shape);
        matmul_add(&mut gb, &a, &b, &c_add, shape);
        graph.execute();
        ctx.synchronize();

        let dot = |col: usize| -> f32 {
            (0..k as usize)
                .map(|i| a_cpu[i] * b_cpu[i * n as usize + col])
                .sum()
        };
        let ref_c: Vec<f32> = (0..n as usize).map(dot).collect();
        let ref_c_add: Vec<f32> = (0..n as usize).map(|j| c0_cpu[j] + dot(j)).collect();

        assert!(
            max_abs_diff(&ref_c, &c.to_cpu::<Real>()) < 1e-4,
            "gemv mismatch"
        );
        assert!(
            max_abs_diff(&ref_c_add, &c_add.to_cpu::<Real>()) < 1e-4,
            "gemv_add mismatch"
        );
    }
}

#[cfg(test)]
mod elementwise_grid_validation {
    use super::*;
    use wilupgu::{ComputeGraph, WgpuBackend};

    #[test]
    fn elementwise_ops_past_1d_grid_limit() {
        const LEN: usize = 17_000_003; // > 65535 * 256, not a multiple of 256
        const BOUNDARY: usize = 65535 * 256;
        let ctx = Arc::new(pollster::block_on(WgpuBackend::new()));

        let data: Vec<Real> = (0..LEN)
            .map(|i| ((i % 1000) as f32 - 500.0) / 250.0)
            .collect();
        let res: Vec<Real> = (0..LEN).map(|i| ((i % 777) as f32) / 777.0).collect();

        let buf = Arc::new(Tensor::init_from_cpu(ctx.clone(), &data));
        let res_buf = Arc::new(Tensor::init_from_cpu(ctx.clone(), &res));
        let mut graph = ComputeGraph::new(ctx.clone());
        let mut gb = GraphBuilder::train(&mut graph);
        silu(&mut gb, &buf, LEN as u32);
        residual_add(&mut gb, &buf, &res_buf, LEN as u32);
        graph.execute();
        ctx.synchronize();

        let out = buf.to_cpu::<Real>();
        for &i in &[0usize, BOUNDARY - 1, BOUNDARY, 16_900_000, LEN - 1] {
            let v = data[i];
            let expected = v / (1.0 + (-v).exp()) + res[i];
            assert!(
                (out[i] - expected).abs() < 1e-5,
                "idx {i}: got {} expected {expected}",
                out[i]
            );
        }
    }
}

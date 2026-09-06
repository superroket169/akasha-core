use super::*;

#[cfg(test)]
mod checkpoint_roundtrip {
    use super::*;
    use wilupgu::WgpuBackend;

    #[test]
    fn v3_save_load_roundtrip() {
        let ctx = Arc::new(pollster::block_on(WgpuBackend::new()));
        let cfg = ModelConfig::new(37, 128, 2, 2, 11); // head_dim=64: flash attention's wgsl is hardcoded to it
        let tokens: Vec<u32> = (0..cfg.seq_len).map(|i| i % cfg.vocab_size).collect();
        let targets: Vec<u32> = (0..cfg.seq_len).map(|i| (i + 1) % cfg.vocab_size).collect();

        // A few real steps so moments and the schedule counter are nonzero.
        let input_a = Arc::new(Tensor::init_from_cpu(ctx.clone(), &tokens));
        let weights_a = Arc::new(ModelWeights::random(ctx.clone(), &cfg));
        let a = Trainer::new(
            ctx.clone(),
            weights_a,
            &input_a,
            TrainConfig::hall1_pretrain(),
        );

        for step in 0..3 {
            a.train_step(&tokens, &targets, 1, step, 1);
        }

        ctx.synchronize();

        let path = std::env::temp_dir().join("akasha_v3_roundtrip_test.bin");
        let path = path.to_str().unwrap();
        a.save_checkpoint(path, 42).unwrap();

        let input_b = Arc::new(Tensor::init_from_cpu(ctx.clone(), &tokens));
        let weights_b = Arc::new(ModelWeights::zeros(ctx.clone(), &cfg));

        let b = Trainer::new(
            ctx.clone(),
            weights_b,
            &input_b,
            TrainConfig::hall1_pretrain(),
        );

        let train_step = b.load_checkpoint(path).unwrap();

        ctx.synchronize();
        std::fs::remove_file(path).ok();

        assert_eq!(train_step, 42);
        assert_eq!(
            a.optimizer.current_schedule().0,
            b.optimizer.current_schedule().0,
            "schedule step didn't survive the roundtrip"
        );
        for (i, (wa, wb)) in a
            .weights
            .params()
            .iter()
            .zip(b.weights.params())
            .enumerate()
        {
            assert_eq!(
                wa.to_cpu::<Real>(),
                wb.to_cpu::<Real>(),
                "weight tensor {i} differs after roundtrip"
            );
        }
        for (i, ((ma, va), (mb, vb))) in a
            .optimizer
            .moments
            .iter()
            .zip(b.optimizer.moments.iter())
            .enumerate()
        {
            assert_eq!(
                ma.to_cpu::<Real>(),
                mb.to_cpu::<Real>(),
                "m moment {i} differs after roundtrip"
            );
            assert_eq!(
                va.to_cpu::<Real>(),
                vb.to_cpu::<Real>(),
                "v moment {i} differs after roundtrip"
            );
        }
    }
}

#[cfg(test)]
mod flat_weights_roundtrip {
    use super::*;
    use wilupgu::WgpuBackend;

    #[test]
    fn set_flat_weights_overwrites_weights_not_optimizer() {
        let ctx = Arc::new(pollster::block_on(WgpuBackend::new()));
        let cfg = ModelConfig::new(37, 128, 2, 2, 11); // head_dim=64: flash attention's wgsl is hardcoded to it
        let tokens: Vec<u32> = (0..cfg.seq_len).map(|i| i % cfg.vocab_size).collect();
        let targets: Vec<u32> = (0..cfg.seq_len).map(|i| (i + 1) % cfg.vocab_size).collect();
        let input_tokens = Arc::new(Tensor::init_from_cpu(ctx.clone(), &tokens));
        let weights = Arc::new(ModelWeights::random(ctx.clone(), &cfg));
        let trainer = Trainer::new(
            ctx.clone(),
            weights,
            &input_tokens,
            TrainConfig::hall1_pretrain(),
        );

        // A few real steps so AdamW moments are nonzero -- otherwise "moments
        // unchanged" would be trivially true.
        for step in 0..3 {
            trainer.train_step(&tokens, &targets, 1, step, 1);
        }

        ctx.synchronize();
        let moments_before: Vec<Real> = trainer.optimizer.moments[0].0.to_cpu();

        let flat = trainer.to_flat_weights();
        let mutated: Vec<Real> = flat.iter().map(|w| w + 1.0).collect();
        trainer.set_flat_weights(&mutated);
        ctx.synchronize();

        let roundtrip = trainer.to_flat_weights();
        assert_eq!(roundtrip, mutated, "set_flat_weights didn't apply exactly");

        let moments_after: Vec<Real> = trainer.optimizer.moments[0].0.to_cpu();
        assert_eq!(
            moments_before, moments_after,
            "set_flat_weights must not touch optimizer state"
        );
    }

    #[test]
    #[should_panic(expected = "doesn't match")]
    fn set_flat_weights_rejects_wrong_length() {
        let ctx = Arc::new(pollster::block_on(WgpuBackend::new()));
        let cfg = ModelConfig::new(37, 128, 2, 2, 11); // head_dim=64: flash attention's wgsl is hardcoded to it
        let tokens: Vec<u32> = (0..cfg.seq_len).map(|i| i % cfg.vocab_size).collect();
        let input_tokens = Arc::new(Tensor::init_from_cpu(ctx.clone(), &tokens));
        let weights = Arc::new(ModelWeights::random(ctx.clone(), &cfg));
        let trainer = Trainer::new(
            ctx.clone(),
            weights,
            &input_tokens,
            TrainConfig::hall1_pretrain(),
        );

        trainer.set_flat_weights(&[0.0; 3]);
    }
}

#[cfg(test)]
mod fused_ops_integration {
    use super::*;
    use wilupgu::WgpuBackend;

    #[test]
    fn trainer_forward_backward_stays_finite_with_fused_ops() {
        let ctx = Arc::new(pollster::block_on(WgpuBackend::new()));

        let cfg = ModelConfig::new(37, 128, 2, 2, 11); // head_dim=64: flash attention's wgsl is hardcoded to it

        let tokens: Vec<u32> = (0..cfg.seq_len).map(|i| i % cfg.vocab_size).collect();
        let targets: Vec<u32> = (0..cfg.seq_len).map(|i| (i + 1) % cfg.vocab_size).collect();
        let input_tokens = Arc::new(Tensor::init_from_cpu(ctx.clone(), &tokens));

        let weights = Arc::new(ModelWeights::random(ctx.clone(), &cfg));
        let trainer = Trainer::new(
            ctx.clone(),
            weights,
            &input_tokens,
            TrainConfig::hall1_pretrain(),
        );
        trainer.cross_entropy.target_tokens.copy_from_cpu(&targets);
        trainer.zero_grad();
        trainer.zero_transient_grads();

        trainer.fused_forward_graph.execute_captured();
        let loss = trainer.cross_entropy.loss();
        trainer.backward_fused();
        ctx.synchronize();

        assert!(loss.is_finite(), "loss is not finite: {loss}");
        assert!(
            loss > 0.0,
            "loss should be positive cross-entropy, got {loss}"
        );

        let embed_grad = trainer.embedding.grad_table.to_cpu::<Real>();
        assert!(
            embed_grad.iter().all(|g| g.is_finite()),
            "embedding grad contains non-finite values"
        );
        assert!(
            embed_grad.iter().any(|&g| g != 0.0),
            "embedding grad is all zero -- gradient did not flow back through the fused attention/rope/qkv ops"
        );
    }
}

/// Numerical gradcheck through the REAL fused forward/backward chain test
#[cfg(test)]
mod full_chain_gradcheck {
    use super::*;
    use wilupgu::WgpuBackend;

    fn loss_at(trainer: &Trainer<WgpuBackend>) -> Real {
        trainer.fused_forward_graph.execute_captured();
        trainer.cross_entropy.loss()
    }

    fn check_param(
        trainer: &Trainer<WgpuBackend>,
        name: &str,
        weight: &Arc<Tensor<WgpuBackend>>,
        analytic: &[Real],
        indices: &[usize],
    ) {
        let eps = 1e-2 as Real;
        for &i in indices {
            let mut w: Vec<Real> = weight.to_cpu();
            let orig = w[i];

            w[i] = orig + eps;
            weight.copy_from_cpu(&w);
            let loss_plus = loss_at(trainer);

            w[i] = orig - eps;
            weight.copy_from_cpu(&w);
            let loss_minus = loss_at(trainer);

            w[i] = orig;
            weight.copy_from_cpu(&w);

            let numeric = (loss_plus - loss_minus) / (2.0 * eps);
            let denom = numeric.abs().max(analytic[i].abs()).max(1e-3);
            let rel = (numeric - analytic[i]).abs() / denom;
            assert!(
                rel < 0.08,
                "{name}[{i}]: analytic={} numeric={numeric} rel_err={rel}",
                analytic[i]
            );
        }
    }

    #[test]
    fn backward_matches_numerical_gradients_through_full_chain() {
        let ctx = Arc::new(pollster::block_on(WgpuBackend::new()));
        let cfg = ModelConfig::new(37, 128, 2, 2, 11); // head_dim=64: flash attention's wgsl is hardcoded to it
        let seq_len = cfg.seq_len as usize;

        let tokens: Vec<u32> = (0..cfg.seq_len)
            .map(|i| (i * 7 + 3) % cfg.vocab_size)
            .collect();
        let targets: Vec<u32> = (0..cfg.seq_len)
            .map(|i| (i * 5 + 1) % cfg.vocab_size)
            .collect();
        let input_tokens = Arc::new(Tensor::init_from_cpu(ctx.clone(), &tokens));

        let weights = Arc::new(ModelWeights::random(ctx.clone(), &cfg));
        let trainer = Trainer::new(
            ctx.clone(),
            weights,
            &input_tokens,
            TrainConfig::hall1_pretrain(),
        );
        trainer.cross_entropy.target_tokens.copy_from_cpu(&targets);
        trainer.cross_entropy.set_grad_scale(1.0 / seq_len as Real);

        trainer.zero_grad();
        trainer.zero_transient_grads();
        trainer.fused_forward_graph.execute_captured();
        trainer.backward_fused();
        ctx.synchronize();

        let l0 = &trainer.layers[0];
        let l1 = &trainer.layers[1];
        let idx_norm = [0usize, 5, 11, 15];
        check_param(
            &trainer,
            "layer0.norm_1.weight",
            &l0.norm_1.weight,
            &l0.norm_1.grad_weight.to_cpu::<Real>(),
            &idx_norm,
        );
        check_param(
            &trainer,
            "layer1.norm_2.weight",
            &l1.norm_2.weight,
            &l1.norm_2.grad_weight.to_cpu::<Real>(),
            &idx_norm,
        );

        let idx_mat = [0usize, 100, 500, 1023];
        check_param(
            &trainer,
            "layer0.ffn_up.weight",
            &l0.ffn_up.weight,
            &l0.ffn_up.grad_weight.to_cpu::<Real>(),
            &idx_mat,
        );

        let idx_qkv = [0usize, 200, 767];
        check_param(
            &trainer,
            "layer1.qkv_proj.weight",
            &l1.qkv_proj.weight,
            &l1.qkv_proj.grad_weight.to_cpu::<Real>(),
            &idx_qkv,
        );
    }
}

/// Proves the row_offset-based real-batching design
#[cfg(test)]
mod batching_validation {
    use super::*;
    use crate::nn::weights::BlockWeights;
    use wilupgu::WgpuBackend;

    fn clone_weights<B: Backend>(w: &ModelWeights<B>, cfg: ModelConfig) -> ModelWeights<B> {
        ModelWeights {
            cfg,
            embedding: w.embedding.clone(),
            blocks: w
                .blocks
                .iter()
                .map(|b| BlockWeights {
                    norm_1: b.norm_1.clone(),
                    qkv_proj: b.qkv_proj.clone(),
                    out_proj: b.out_proj.clone(),
                    norm_2: b.norm_2.clone(),
                    ffn_up: b.ffn_up.clone(),
                    ffn_down: b.ffn_down.clone(),
                })
                .collect(),
            final_norm: w.final_norm.clone(),
            lm_head: w.lm_head.clone(),
        }
    }

    fn rand_tokens(n: usize, vocab: u32, seed: u64) -> Vec<u32> {
        let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 33) as u32) % vocab
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
    fn real_batching_matches_sequential_accumulation() {
        batching_parity(Arc::new(pollster::block_on(WgpuBackend::new())));
    }

    /// Same parity, on the REAL CUDA backend — the wgpu variant above proves
    /// the design, this one proves the CUDA kernel twins (row_offset in
    /// rope_qk/flash launches). Only compiles/runs on the nvidia machine.
    #[cfg(feature = "cuda")]
    #[test]
    fn real_batching_matches_sequential_accumulation_cuda() {
        let ctx = wilupgu::CudaBackend::new(0).expect("CUDA backend unavailable");
        batching_parity(Arc::new(ctx));
    }

    fn batching_parity<B: Backend>(ctx: Arc<B>) {
        let batch: u32 = 3;
        let base_cfg = ModelConfig::new(37, 128, 2, 2, 11); // head_dim=64: flash attention's wgsl is hardcoded to it
        let seq_len = base_cfg.seq_len as usize;
        let vocab = base_cfg.vocab_size;
        let dim = base_cfg.dim as usize;
        let scale = 1.0 / (batch as usize * seq_len) as Real;

        let cfg_batched = base_cfg.with_batch_size(batch);
        let weights_batched = Arc::new(ModelWeights::random(ctx.clone(), &cfg_batched));
        let weights_ref = Arc::new(clone_weights(&weights_batched, base_cfg));

        let all_inputs: Vec<u32> = (0..batch as usize)
            .flat_map(|b| rand_tokens(seq_len, vocab, 100 + b as u64))
            .collect();
        let all_targets: Vec<u32> = (0..batch as usize)
            .flat_map(|b| rand_tokens(seq_len, vocab, 200 + b as u64))
            .collect();

        // Reference: Sequential trainer (batch_size=1) called in a loop to accumulate gradients.
        // Used to verify mathematical parity with the batched forward/backward pass.
        let ref_input_tokens = Arc::new(Tensor::init_from_cpu(ctx.clone(), &vec![0u32; seq_len]));
        let ref_trainer = Trainer::new(
            ctx.clone(),
            weights_ref,
            &ref_input_tokens,
            TrainConfig::hall1_pretrain(),
        );
        ref_trainer.cross_entropy.set_grad_scale(scale);
        ref_trainer.zero_grad();
        let mut ref_total_loss = 0.0 as Real;
        for b in 0..batch as usize {
            let window = b * seq_len..(b + 1) * seq_len;
            ref_trainer
                .input_tokens
                .copy_from_cpu(&all_inputs[window.clone()]);
            ref_trainer
                .cross_entropy
                .target_tokens
                .copy_from_cpu(&all_targets[window]);
            ref_trainer.zero_transient_grads();
            ref_trainer.fused_forward_graph.execute_captured();
            ref_total_loss += ref_trainer.cross_entropy.loss();
            ref_trainer.backward_fused();
        }
        ctx.synchronize();
        let ref_avg_loss = ref_total_loss / batch as Real;

        // last iteration's (b = batch-1) forward output is what's left in
        // ref_trainer's buffers -- compare it against the batched trainer's
        // matching row_offset slice below.
        let last_block = ref_trainer.layers.last().unwrap();
        let ref_last_out = last_block.add_2.out_buffer.to_cpu::<Real>();

        // ---- new: one batch_size=3 Trainer, ONE execute over all 33 rows ----
        let rows = batch as usize * seq_len;
        let batched_input_tokens = Arc::new(Tensor::init_from_cpu(ctx.clone(), &vec![0u32; rows]));
        let batched_trainer = Trainer::new(
            ctx.clone(),
            weights_batched,
            &batched_input_tokens,
            TrainConfig::hall1_pretrain(),
        );
        batched_trainer.cross_entropy.set_grad_scale(scale);
        batched_trainer.zero_grad();
        batched_trainer.input_tokens.copy_from_cpu(&all_inputs);
        batched_trainer
            .cross_entropy
            .target_tokens
            .copy_from_cpu(&all_targets);
        batched_trainer.zero_transient_grads();
        batched_trainer.fused_forward_graph.execute_captured();
        let batched_loss = batched_trainer.cross_entropy.loss();
        batched_trainer.backward_fused();
        ctx.synchronize();

        assert!(
            (ref_avg_loss - batched_loss).abs() < 1e-3,
            "loss mismatch: sequential avg={ref_avg_loss} batched={batched_loss}"
        );

        let batched_last_block = batched_trainer.layers.last().unwrap();
        let batched_out_full = batched_last_block.add_2.out_buffer.to_cpu::<Real>();
        let last_window = (batch as usize - 1) * seq_len * dim..batch as usize * seq_len * dim;
        let batched_last_out = &batched_out_full[last_window];
        let out_diff = max_abs_diff(&ref_last_out, batched_last_out);
        assert!(
            out_diff < 1e-3,
            "forward residual-stream mismatch at last batch item: max_abs_diff={out_diff}"
        );

        let ref_params = ref_trainer.trainable_params();
        let batched_params = batched_trainer.trainable_params();
        assert_eq!(ref_params.len(), batched_params.len());
        for (i, ((_, ref_grad), (_, batched_grad))) in
            ref_params.iter().zip(batched_params.iter()).enumerate()
        {
            let a = ref_grad.to_cpu::<Real>();
            let b = batched_grad.to_cpu::<Real>();
            let diff = max_abs_diff(&a, &b);
            assert!(
                diff < 1e-3,
                "grad mismatch at trainable_params()[{i}]: max_abs_diff={diff}"
            );
        }
    }
}

/// GPU grad clip vs. the old host-side reference formula
#[cfg(test)]
mod grad_clip_validation {
    use super::*;
    use wilupgu::WgpuBackend;

    fn check_clip(amplitude: Real) {
        let ctx = Arc::new(pollster::block_on(WgpuBackend::new()));
        let cfg = ModelConfig::new(37, 128, 2, 2, 11); // head_dim=64: flash attention's wgsl is hardcoded to it

        let tokens: Vec<u32> = (0..cfg.seq_len).map(|i| i % cfg.vocab_size).collect();
        let input_tokens = Arc::new(Tensor::init_from_cpu(ctx.clone(), &tokens));
        let weights = Arc::new(ModelWeights::random(ctx.clone(), &cfg));
        let train_cfg = TrainConfig::hall1_pretrain();
        let trainer = Trainer::new(ctx.clone(), weights, &input_tokens, train_cfg);

        let params = trainer.trainable_params();
        let mut host_grads: Vec<Vec<Real>> = Vec::new();
        for (t, (_, grad)) in params.iter().enumerate() {
            let len = (grad.size / std::mem::size_of::<Real>() as u64) as usize;
            let data: Vec<Real> = (0..len)
                .map(|i| ((t * 31 + i) as Real * 0.7).sin() * amplitude)
                .collect();
            grad.copy_from_cpu(&data);
            host_grads.push(data);
        }

        trainer.clip_grad_norm();
        ctx.synchronize();

        let total_sq: f64 = host_grads
            .iter()
            .flatten()
            .map(|&g| (g as f64) * (g as f64))
            .sum();
        let norm = total_sq.sqrt() as f32;
        let scale = if norm > train_cfg.grad_clip_norm {
            train_cfg.grad_clip_norm / (norm + 1e-6)
        } else {
            1.0
        };

        for (host, (_, grad)) in host_grads.iter().zip(params.iter()) {
            let gpu: Vec<Real> = grad.to_cpu();
            for (i, (&h, &g)) in host.iter().zip(gpu.iter()).enumerate() {
                let expected = h * scale;
                assert!(
                    (expected - g).abs() < 1e-6 + expected.abs() * 1e-4,
                    "amplitude {amplitude}: grad[{i}] expected {expected}, gpu {g} (norm {norm}, scale {scale})"
                );
            }
        }
    }

    #[test]
    fn clips_when_norm_exceeds_max() {
        check_clip(0.1);
    }

    #[test]
    fn leaves_grads_alone_under_max() {
        check_clip(1e-4);
    }
}

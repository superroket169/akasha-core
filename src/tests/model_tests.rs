use super::*;

#[cfg(test)]
mod full_chain_gradcheck {
    use super::*;
    use wilupgu::WgpuBackend;

    fn loss_at(model: &mut Model<WgpuBackend>, tokens: &[u32], targets: &[u32]) -> Real {
        model.train_step(tokens, targets)
    }

    fn check_param(
        model: &mut Model<WgpuBackend>,
        tokens: &[u32],
        targets: &[u32],
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
            let loss_plus = loss_at(model, tokens, targets);

            w[i] = orig - eps;
            weight.copy_from_cpu(&w);
            let loss_minus = loss_at(model, tokens, targets);

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
    fn model_backward_matches_numerical_gradients() {
        let ctx = Arc::new(pollster::block_on(WgpuBackend::new()));
        let cfg = ModelConfig::new(37, 128, 2, 2, 11); // head_dim=64: flash attention's wgsl is hardcoded to it
        let seq_len = cfg.seq_len;

        let tokens: Vec<u32> = (0..seq_len).map(|i| (i * 7 + 3) % cfg.vocab_size).collect();
        let targets: Vec<u32> = (0..seq_len).map(|i| (i * 5 + 1) % cfg.vocab_size).collect();

        let weights = ModelWeights::random(ctx.clone(), &cfg);
        let mut model =
            Model::for_training(ctx.clone(), weights, cfg, TrainConfig::hall1_pretrain());

        // clean per-token-mean scale for the gradcheck, independent of the
        // real training config's accumulation_steps.
        model
            .train
            .as_ref()
            .unwrap()
            .loss
            .set_grad_scale(1.0 / seq_len as Real);

        model.zero_grad();
        model.train_step(&tokens, &targets);
        ctx.synchronize();

        let (n1_weight, n1_analytic, qkv_weight, qkv_analytic, lm_weight, lm_analytic) = {
            let t = model.train.as_ref().unwrap();
            let params0 = t.blocks[0].tape.params();
            let (n1_w, n1_g, _) = params0[0];
            let (qkv_w, qkv_g, _) = params0[1];
            let tail_params = t.tail.params();
            let (lm_w, lm_g, _) = tail_params[1];
            (
                n1_w.clone(),
                n1_g.to_cpu(),
                qkv_w.clone(),
                qkv_g.to_cpu(),
                lm_w.clone(),
                lm_g.to_cpu(),
            )
        };

        check_param(
            &mut model,
            &tokens,
            &targets,
            "block0.norm_1",
            &n1_weight,
            &n1_analytic,
            &[0, 10, 50],
        );
        check_param(
            &mut model,
            &tokens,
            &targets,
            "block0.qkv_proj",
            &qkv_weight,
            &qkv_analytic,
            &[0, 100, 500],
        );
        check_param(
            &mut model,
            &tokens,
            &targets,
            "tail.lm_head",
            &lm_weight,
            &lm_analytic,
            &[0, 50, 200],
        );
    }
}

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

        // A few real accumulated steps + one optimizer_step so moments and
        // the schedule counter are nonzero.
        let weights_a = ModelWeights::random(ctx.clone(), &cfg);
        let mut a = Model::for_training(ctx.clone(), weights_a, cfg, TrainConfig::hall1_pretrain());
        a.zero_grad();
        for _ in 0..3 {
            a.train_step(&tokens, &targets);
        }
        a.optimizer_step();
        ctx.synchronize();

        let path = std::env::temp_dir().join("akasha_model_v3_roundtrip_test.bin");
        let path = path.to_str().unwrap();
        a.save_checkpoint(path, 42).unwrap();

        let weights_b = ModelWeights::zeros(ctx.clone(), &cfg);
        let b = Model::for_training(ctx.clone(), weights_b, cfg, TrainConfig::hall1_pretrain());
        let train_step = b.load_checkpoint(path).unwrap();

        ctx.synchronize();
        std::fs::remove_file(path).ok();

        assert_eq!(train_step, 42);

        let a_schedule = a.train.as_ref().unwrap().optimizer.current_schedule().0;
        let b_schedule = b.train.as_ref().unwrap().optimizer.current_schedule().0;
        assert_eq!(
            a_schedule, b_schedule,
            "schedule step didn't survive the roundtrip"
        );

        for (i, (wa, wb)) in a
            .weights()
            .params()
            .iter()
            .zip(b.weights().params())
            .enumerate()
        {
            assert_eq!(
                wa.to_cpu::<Real>(),
                wb.to_cpu::<Real>(),
                "weight tensor {i} differs after roundtrip"
            );
        }

        let a_moments = a.train.as_ref().unwrap().optimizer.moments();
        let b_moments = b.train.as_ref().unwrap().optimizer.moments();
        for (i, ((ma, va), (mb, vb))) in a_moments.iter().zip(b_moments.iter()).enumerate() {
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
        let weights = ModelWeights::random(ctx.clone(), &cfg);
        let mut model =
            Model::for_training(ctx.clone(), weights, cfg, TrainConfig::hall1_pretrain());

        // A few real steps so AdamW moments are nonzero -- otherwise "moments
        // unchanged" would be trivially true.
        model.zero_grad();
        for _ in 0..3 {
            model.train_step(&tokens, &targets);
        }
        model.optimizer_step();
        ctx.synchronize();

        let moments_before: Vec<Real> = model.train.as_ref().unwrap().optimizer.moments()[0]
            .0
            .to_cpu();

        let flat = model.to_flat_weights();
        let mutated: Vec<Real> = flat.iter().map(|w| w + 1.0).collect();
        model.set_flat_weights(&mutated);
        ctx.synchronize();

        let roundtrip = model.to_flat_weights();
        assert_eq!(roundtrip, mutated, "set_flat_weights didn't apply exactly");

        let moments_after: Vec<Real> = model.train.as_ref().unwrap().optimizer.moments()[0]
            .0
            .to_cpu();
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
        let weights = ModelWeights::random(ctx.clone(), &cfg);
        let model = Model::for_training(ctx.clone(), weights, cfg, TrainConfig::hall1_pretrain());
        model.set_flat_weights(&[0.0; 3]);
    }
}

/// Proves the Tape/Model system accumulates a batch_size=N forward+backward
/// pass identically to N sequential batch_size=1 passes (the row_offset-based
/// real-batching design).
#[cfg(test)]
mod batching_validation {
    use super::*;
    use crate::test_common::max_abs_diff;
    use wilupgu::WgpuBackend;

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

    #[test]
    fn real_batching_matches_sequential_accumulation() {
        batching_parity(Arc::new(pollster::block_on(WgpuBackend::new())));
    }

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
        let weights_batched = ModelWeights::random(ctx.clone(), &cfg_batched);
        let weights_ref = weights_batched.clone();

        let all_inputs: Vec<u32> = (0..batch as usize)
            .flat_map(|b| rand_tokens(seq_len, vocab, 100 + b as u64))
            .collect();
        let all_targets: Vec<u32> = (0..batch as usize)
            .flat_map(|b| rand_tokens(seq_len, vocab, 200 + b as u64))
            .collect();

        // Reference: batch_size=1 Model called in a loop to accumulate gradients.
        let mut ref_model = Model::for_training(
            ctx.clone(),
            weights_ref,
            base_cfg,
            TrainConfig::hall1_pretrain(),
        );
        ref_model.train.as_ref().unwrap().loss.set_grad_scale(scale);
        ref_model.zero_grad();

        let mut ref_total_loss = 0.0 as Real;

        for b in 0..batch as usize {
            let window = b * seq_len..(b + 1) * seq_len;
            ref_total_loss +=
                ref_model.train_step(&all_inputs[window.clone()], &all_targets[window]);
        }
        ctx.synchronize();
        let ref_avg_loss = ref_total_loss / batch as Real;

        // last iteration's (b = batch-1) forward output is what's left in
        // ref_model's last block -- compare it against the batched model's
        // matching row_offset slice below.
        let ref_last_out: Vec<Real> = {
            let t = ref_model.train.as_ref().unwrap();
            let last_block = t.blocks.last().unwrap();
            last_block.tape.output(last_block.output).to_cpu()
        };

        // ---- batched: one batch_size=3 Model, ONE train_step over all 33 rows ----
        let mut batched_model = Model::for_training(
            ctx.clone(),
            weights_batched,
            cfg_batched,
            TrainConfig::hall1_pretrain(),
        );
        batched_model
            .train
            .as_ref()
            .unwrap()
            .loss
            .set_grad_scale(scale);
        batched_model.zero_grad();
        let batched_loss = batched_model.train_step(&all_inputs, &all_targets);
        ctx.synchronize();

        assert!(
            (ref_avg_loss - batched_loss).abs() < 1e-3,
            "loss mismatch: sequential avg={ref_avg_loss} batched={batched_loss}"
        );

        let batched_out_full: Vec<Real> = {
            let t = batched_model.train.as_ref().unwrap();
            let last_block = t.blocks.last().unwrap();
            last_block.tape.output(last_block.output).to_cpu()
        };
        let last_window = (batch as usize - 1) * seq_len * dim..batch as usize * seq_len * dim;
        let batched_last_out = &batched_out_full[last_window];
        let out_diff = max_abs_diff(&ref_last_out, batched_last_out);
        assert!(
            out_diff < 1e-3,
            "forward residual-stream mismatch at last batch item: max_abs_diff={out_diff}"
        );

        let ref_t = ref_model.train.as_ref().unwrap();
        let batched_t = batched_model.train.as_ref().unwrap();
        let ref_params: Vec<_> = ref_t
            .head
            .params()
            .into_iter()
            .chain(ref_t.blocks.iter().flat_map(|b| b.tape.params()))
            .chain(ref_t.tail.params())
            .collect();
        let batched_params: Vec<_> = batched_t
            .head
            .params()
            .into_iter()
            .chain(batched_t.blocks.iter().flat_map(|b| b.tape.params()))
            .chain(batched_t.tail.params())
            .collect();
        assert_eq!(ref_params.len(), batched_params.len());
        for (i, ((_, ref_grad, _), (_, batched_grad, _))) in
            ref_params.iter().zip(batched_params.iter()).enumerate()
        {
            let a: Vec<Real> = ref_grad.to_cpu();
            let b: Vec<Real> = batched_grad.to_cpu();
            let diff = max_abs_diff(&a, &b);
            assert!(
                diff < 1e-3,
                "grad mismatch at params()[{i}]: max_abs_diff={diff}"
            );
        }
    }
}

/// GPU grad clip (Model's AnyGradClip) vs. the host-side reference formula.
#[cfg(test)]
mod grad_clip_validation {
    use super::*;
    use wilupgu::WgpuBackend;

    fn check_clip(amplitude: Real) {
        let ctx = Arc::new(pollster::block_on(WgpuBackend::new()));
        let cfg = ModelConfig::new(37, 128, 2, 2, 11); // head_dim=64: flash attention's wgsl is hardcoded to it
        let weights = ModelWeights::random(ctx.clone(), &cfg);
        let train_cfg = TrainConfig::hall1_pretrain();
        let model = Model::for_training(ctx.clone(), weights, cfg, train_cfg);

        let t = model.train.as_ref().unwrap();
        let params: Vec<_> = t
            .head
            .params()
            .into_iter()
            .chain(t.blocks.iter().flat_map(|b| b.tape.params()))
            .chain(t.tail.params())
            .collect();

        let mut host_grads: Vec<Vec<Real>> = Vec::new();
        for (i, (_, grad, _)) in params.iter().enumerate() {
            let len = (grad.size / std::mem::size_of::<Real>() as u64) as usize;
            let data: Vec<Real> = (0..len)
                .map(|j| ((i * 31 + j) as Real * 0.7).sin() * amplitude)
                .collect();
            grad.copy_from_cpu(&data);
            host_grads.push(data);
        }

        t.grad_clip.clip();
        ctx.synchronize();

        let total_sq: f64 = host_grads
            .iter()
            .flatten()
            .map(|&g| (g as f64) * (g as f64))
            .sum();
        let norm = total_sq.sqrt() as f32;
        let scale = if norm > train_cfg.grad_clip.max_norm {
            train_cfg.grad_clip.max_norm / (norm + 1e-6)
        } else {
            1.0
        };

        for (host, (_, grad, _)) in host_grads.iter().zip(params.iter()) {
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

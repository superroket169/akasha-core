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

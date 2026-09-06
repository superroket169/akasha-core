    use super::*;
    use crate::config::cosine_lr;
    use wilupgu::WgpuBackend;

    #[test]
    fn schedule_matches_host_formula_and_updates_weights() {
        let ctx = Arc::new(pollster::block_on(WgpuBackend::new()));

        let weight_data = vec![1.0 as Real, 2.0, 3.0, 4.0];
        let grad_data = vec![0.1 as Real, -0.2, 0.3, -0.4];
        let weight = Arc::new(Tensor::init_from_cpu(ctx.clone(), &weight_data));
        let grad = Arc::new(Tensor::init_from_cpu(ctx.clone(), &grad_data));

        let (lr_max, lr_min, warmup_steps, max_steps) = (6e-4 as Real, 6e-5 as Real, 5u32, 50u32);
        let opt = AdamW::new(
            ctx.clone(),
            &[(weight.clone(), grad.clone(), true)],
            AdamWSchedule {
                lr_max,
                lr_min,
                warmup_steps,
                max_steps,
            },
            0.9,
            0.95,
            0.01,
        );

        for expected_step in 1..=15u32 {
            opt.step();
            ctx.synchronize();

            let (step, lr) = opt.current_schedule();
            assert_eq!(step, expected_step, "on-device step counter drifted");

            let expected_lr = cosine_lr(
                expected_step as usize,
                warmup_steps as usize,
                max_steps as usize,
                lr_max,
                lr_min,
            );
            let diff = (lr - expected_lr).abs();
            assert!(
                diff < 1e-6,
                "step {expected_step}: on-device lr {lr} vs. host cosine_lr {expected_lr} (diff {diff})"
            );
        }

        // weight[0]  -> should decrease
        // weight[1]  -> should increase
        let final_weights: Vec<Real> = weight.to_cpu();
        assert!(
            final_weights[0] < weight_data[0],
            "weight[0] should have decreased (positive grad): {} -> {}",
            weight_data[0],
            final_weights[0]
        );
        assert!(
            final_weights[1] > weight_data[1],
            "weight[1] should have increased (negative grad): {} -> {}",
            weight_data[1],
            final_weights[1]
        );
        assert!(
            final_weights.iter().all(|w| w.is_finite()),
            "weights contain non-finite values after {} steps",
            final_weights.len()
        );
    }

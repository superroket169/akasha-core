    use super::*;

    /// B4 boundary cases: t=0, warmup edge, max_steps edge and beyond,
    /// degenerate max_steps == warmup_steps.
    #[test]
    fn cosine_lr_boundaries() {
        let (warmup, max, lr_max, lr_min) = (10, 100, 1.0, 0.1);

        assert_eq!(cosine_lr(0, warmup, max, lr_max, lr_min), 0.0);
        assert!((cosine_lr(warmup, warmup, max, lr_max, lr_min) - lr_max).abs() < 1e-6);
        assert!((cosine_lr(max, warmup, max, lr_max, lr_min) - lr_min).abs() < 1e-6);

        // Past max_steps the LR must stay pinned at lr_min, not climb back.
        for step in [max + 1, max + 50, max * 10] {
            let lr = cosine_lr(step, warmup, max, lr_max, lr_min);
            assert!(
                (lr - lr_min).abs() < 1e-6,
                "lr climbed back after max_steps: step={step} lr={lr}"
            );
        }

        // Degenerate config: max_steps == warmup_steps must not produce NaN.
        let lr = cosine_lr(warmup, warmup, warmup, lr_max, lr_min);
        assert!(lr.is_finite());
        assert!((lr - lr_min).abs() < 1e-6);
    }

    #[test]
    fn resolve_profile_known_names_roundtrip() {
        for name in ["hall1_pretrain", "dolly_finetune", "pidgeon_pretrain"] {
            let (_, train) =
                resolve_profile(name).unwrap_or_else(|| panic!("missing profile: {name}"));
            assert_eq!(train.name, name);
        }
        assert!(resolve_profile("nonexistent").is_none());
    }

    #[test]
    fn named_model_profiles_pass_head_dim_guard() {
        assert_eq!(ModelConfig::hall_1().head_dim(), 64);
        assert_eq!(ModelConfig::pidgeon().head_dim(), 64);
    }

    #[test]
    fn nested_config_matches_flat_fields_in_every_profile() {
        for train in [
            TrainConfig::hall1_pretrain(),
            TrainConfig::dolly_finetune(),
            TrainConfig::pidgeon_pretrain(),
        ] {
            assert_eq!(train.optimizer.weight_decay, train.adam_weight_decay);
            assert_eq!(train.optimizer.lr_max, train.lr_max);
            assert_eq!(train.optimizer.lr_min, train.lr_min);
            assert_eq!(train.optimizer.warmup_steps, train.warmup_steps);
            assert_eq!(train.optimizer.max_steps, train.max_steps);
            assert_eq!(train.grad_clip.max_norm, train.grad_clip_norm);
            assert_eq!(train.run.batch_size, train.batch_size);
            assert_eq!(train.run.accumulation_steps, train.accumulation_steps);
            assert_eq!(train.run.save_every, train.save_every);
            assert_eq!(train.run.log_every, train.log_every);
            assert_eq!(train.run.eval_every, train.eval_every);
            assert_eq!(train.run.eval_windows, train.eval_windows);
            assert_eq!(train.run.train_bf16_matmul, train.train_bf16_matmul);
        }
    }

    #[test]
    fn model_config_layers_are_all_transformer() {
        let cfg = ModelConfig::hall_1();
        assert_eq!(cfg.layers().len(), cfg.num_layers);
        assert!(cfg.layers().iter().all(|k| *k == BlockKind::Transformer));
    }

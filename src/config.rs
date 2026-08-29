/// What kind of block occupies a layer slot. One variant today
/// (transformer-only stacks); a second (e.g. Mamba) is a new arm here, not
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockKind {
    Transformer,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModelConfig {
    pub vocab_size: u32,
    pub dim: u32,
    pub num_heads: u32,
    pub num_layers: usize,
    pub seq_len: u32,
    pub ffn_hidden: u32,
    pub norm_eps: f32,
    pub batch_size: u32,
    pub eos_token: u32,
}

impl ModelConfig {
    pub fn new(vocab_size: u32, dim: u32, num_heads: u32, num_layers: usize, seq_len: u32) -> Self {
        assert!(num_layers >= 1, "At least one layer is required!");
        assert_eq!(dim % num_heads, 0, "dim must be divisible by num_heads");
        Self {
            vocab_size,
            dim,
            num_heads,
            num_layers,
            seq_len,
            ffn_hidden: dim * 4,
            norm_eps: 1e-5,
            batch_size: 1,
            // GPT-2 BPE convention: <|endoftext|> is the last vocab id.
            eos_token: vocab_size - 1,
        }
    }

    pub fn with_batch_size(mut self, batch_size: u32) -> Self {
        assert!(batch_size >= 1, "batch_size must be >= 1");
        self.batch_size = batch_size;
        self
    }

    pub fn head_dim(&self) -> u32 {
        self.dim / self.num_heads
    }

    pub fn layers(&self) -> Vec<BlockKind> {
        vec![BlockKind::Transformer; self.num_layers]
    }

    /// about last flash attention patch. head_dim must be 64
    fn assert_flash_attention_head_dim(self) -> Self {
        assert_eq!(
            self.head_dim(),
            64,
            "head_dim={} but wilupgu's flash-attention shaders assume 64 -- see \
             wilupgu/REFACTOR.md before shipping this profile",
            self.head_dim()
        );
        self
    }

    pub fn akasha_hall_1() -> Self {
        Self::new(50257, 768, 12, 12, 512).assert_flash_attention_head_dim()
    }

    pub fn pidgeon() -> Self {
        Self::new(50257, 320, 5, 8, 512).assert_flash_attention_head_dim()
    }
}

/// One variant today (AdamW) -- a second optimizer is a new arm here and
/// in `AnyOptimizer`, not a new config shape. See ARCHITECTURE.md.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OptimizerKind {
    AdamW,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OptimizerConfig {
    pub kind: OptimizerKind,
    pub beta1: f32,
    pub beta2: f32,
    pub weight_decay: f32,
    pub lr_max: f32,
    pub lr_min: f32,
    pub warmup_steps: usize,
    pub max_steps: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GradClipKind {
    GlobalNorm,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GradClipConfig {
    pub kind: GradClipKind,
    pub max_norm: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RunConfig {
    pub batch_size: usize,
    pub accumulation_steps: usize,
    pub save_every: usize,
    pub log_every: usize,
    pub eval_every: usize,
    pub eval_windows: usize,
    pub train_bf16_matmul: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrainConfig {
    pub name: &'static str,
    pub batch_size: usize,
    pub accumulation_steps: usize,
    pub lr_max: f32,
    pub lr_min: f32,
    pub warmup_steps: usize,
    pub max_steps: usize,
    pub save_every: usize,
    pub log_every: usize,
    pub eval_every: usize,
    pub eval_windows: usize,
    pub adam_weight_decay: f32,
    pub grad_clip_norm: f32,
    pub train_bf16_matmul: bool,
    pub optimizer: OptimizerConfig,
    pub grad_clip: GradClipConfig,
    pub run: RunConfig,
}

impl TrainConfig {
    pub fn hall1_pretrain() -> Self {
        Self {
            name: "hall1_pretrain",
            batch_size: 2,
            accumulation_steps: 32, // effective batch = 64
            lr_max: 6e-5,
            lr_min: 6e-6,
            warmup_steps: 1000,
            max_steps: 3_000_000,
            save_every: 1000,
            log_every: 50,
            eval_every: 1000,
            eval_windows: 32,
            adam_weight_decay: 0.01,
            grad_clip_norm: 1.0,
            train_bf16_matmul: true,
            optimizer: OptimizerConfig {
                kind: OptimizerKind::AdamW,
                beta1: 0.9,
                beta2: 0.95,
                weight_decay: 0.01,
                lr_max: 6e-5,
                lr_min: 6e-6,
                warmup_steps: 1000,
                max_steps: 3_000_000,
            },
            grad_clip: GradClipConfig {
                kind: GradClipKind::GlobalNorm,
                max_norm: 1.0,
            },
            run: RunConfig {
                batch_size: 2,
                accumulation_steps: 32,
                save_every: 1000,
                log_every: 50,
                eval_every: 1000,
                eval_windows: 32,
                train_bf16_matmul: true,
            },
        }
    }

    pub fn dolly_finetune() -> Self {
        Self {
            name: "dolly_finetune",
            batch_size: 2,
            accumulation_steps: 32,
            lr_max: 3e-5,
            lr_min: 3e-6,
            warmup_steps: 40,
            max_steps: 800,
            save_every: 50,
            log_every: 10,
            eval_every: 25,
            eval_windows: 32,
            adam_weight_decay: 0.01,
            grad_clip_norm: 1.0,
            train_bf16_matmul: true,
            optimizer: OptimizerConfig {
                kind: OptimizerKind::AdamW,
                beta1: 0.9,
                beta2: 0.95,
                weight_decay: 0.01,
                lr_max: 3e-5,
                lr_min: 3e-6,
                warmup_steps: 40,
                max_steps: 800,
            },
            grad_clip: GradClipConfig {
                kind: GradClipKind::GlobalNorm,
                max_norm: 1.0,
            },
            run: RunConfig {
                batch_size: 2,
                accumulation_steps: 32,
                save_every: 50,
                log_every: 10,
                eval_every: 25,
                eval_windows: 32,
                train_bf16_matmul: true,
            },
        }
    }

    pub fn pidgeon_pretrain() -> Self {
        Self {
            name: "pidgeon_pretrain",
            batch_size: 10,
            accumulation_steps: 6, // effective batch = 60
            lr_max: 6e-5,
            lr_min: 6e-6,
            warmup_steps: 1000,
            max_steps: 200_000,
            save_every: 1000,
            log_every: 50,
            eval_every: 1000,
            eval_windows: 32,
            adam_weight_decay: 0.01,
            grad_clip_norm: 1.0,
            train_bf16_matmul: true,
            optimizer: OptimizerConfig {
                kind: OptimizerKind::AdamW,
                beta1: 0.9,
                beta2: 0.95,
                weight_decay: 0.01,
                lr_max: 6e-5,
                lr_min: 6e-6,
                warmup_steps: 1000,
                max_steps: 200_000,
            },
            grad_clip: GradClipConfig {
                kind: GradClipKind::GlobalNorm,
                max_norm: 1.0,
            },
            run: RunConfig {
                batch_size: 10,
                accumulation_steps: 6,
                save_every: 1000,
                log_every: 50,
                eval_every: 1000,
                eval_windows: 32,
                train_bf16_matmul: true,
            },
        }
    }
}

pub fn resolve_profile(name: &str) -> Option<(ModelConfig, TrainConfig)> {
    match name {
        "hall1_pretrain" => Some((ModelConfig::akasha_hall_1(), TrainConfig::hall1_pretrain())),
        "dolly_finetune" => Some((ModelConfig::akasha_hall_1(), TrainConfig::dolly_finetune())),
        "pidgeon_pretrain" => Some((ModelConfig::pidgeon(), TrainConfig::pidgeon_pretrain())),
        _ => None,
    }
}

pub fn cosine_lr(
    step: usize,
    warmup_steps: usize,
    max_steps: usize,
    lr_max: f32,
    lr_min: f32,
) -> f32 {
    if step < warmup_steps {
        return lr_max * step as f32 / warmup_steps as f32;
    }

    // Clamp: past max_steps the cosine must not wrap back up (resume/fine-tune scenario)
    // max_steps == warmup_steps would divide 0/0
    let progress = if max_steps > warmup_steps {
        ((step - warmup_steps) as f32 / (max_steps - warmup_steps) as f32).min(1.0)
    } else {
        1.0
    };
    lr_min + 0.5 * (lr_max - lr_min) * (1.0 + (std::f32::consts::PI * progress).cos())
}

#[cfg(test)]
mod tests {
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
        assert_eq!(ModelConfig::akasha_hall_1().head_dim(), 64);
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
        let cfg = ModelConfig::akasha_hall_1();
        assert_eq!(cfg.layers().len(), cfg.num_layers);
        assert!(cfg.layers().iter().all(|k| *k == BlockKind::Transformer));
    }
}

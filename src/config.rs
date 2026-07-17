// ~162M parameters (lm_head untied from embedding) - GPT-2 Small scale
pub const DIM: u32 = 768;
pub const NUM_HEADS: u32 = 12;
pub const HEAD_DIM: u32 = DIM / NUM_HEADS; // 64
pub const NUM_LAYERS: usize = 12;
pub const SEQ_LEN: u32 = 512;
pub const FFN_HIDDEN: u32 = 3072; // 4 x DIM
pub const VOCAB_SIZE: u32 = 50257; // GPT-2 tokenizer

// Real fused batch execution. VRAM bounded: logits buffer alone is BATCH_SIZE x ~103MB.
// Calibrate BATCH_SIZE based on GPU limits (e.g., RTX 4050 6GB handles ~4).
pub const BATCH_SIZE: usize = 4;
pub const ACCUMULATION_STEPS: usize = 16; // Effective batch = 64

pub const LR_MAX: f32 = 6e-5;
pub const LR_MIN: f32 = 6e-6;
pub const WARMUP_STEPS: usize = 1000;
pub const MAX_STEPS: usize = 200_000;
pub const SAVE_EVERY: usize = 1000;
pub const LOG_EVERY: usize = 50;

pub const ADAM_WEIGHT_DECAY: f32 = 0.01;
pub const GRAD_CLIP_NORM: f32 = 1.0;

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
        }
    }

    pub fn with_batch_size(mut self, batch_size: u32) -> Self {
        assert!(batch_size >= 1, "batch_size must be >= 1");
        self.batch_size = batch_size;
        self
    }

    /// The shipped akasha-hall 1.0 architecture (matches the constants above).
    pub fn akasha_hall_1() -> Self {
        Self::new(VOCAB_SIZE, DIM, NUM_HEADS, NUM_LAYERS, SEQ_LEN)
    }

    pub fn head_dim(&self) -> u32 {
        self.dim / self.num_heads
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
    use super::cosine_lr;

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
}

// ~162M parameters (lm_head untied from embedding) - GPT-2 Small scale
pub const DIM: u32 = 768;
pub const NUM_HEADS: u32 = 12;
pub const HEAD_DIM: u32 = DIM / NUM_HEADS; // 64
pub const NUM_LAYERS: usize = 12;
pub const SEQ_LEN: u32 = 512;
pub const FFN_HIDDEN: u32 = 3072; // 4 x DIM
pub const VOCAB_SIZE: u32 = 50257; // GPT-2 tokenizer

pub const BATCH_SIZE: usize = 2;
pub const ACCUMULATION_STEPS: usize = 32; // effective batch = 64

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
        }
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
    let progress = (step - warmup_steps) as f32 / (max_steps - warmup_steps) as f32;
    lr_min + 0.5 * (lr_max - lr_min) * (1.0 + (std::f32::consts::PI * progress).cos())
}

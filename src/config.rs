// ~117M parameters - GPT-2 Small scale
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

pub fn cosine_lr(step: usize, warmup_steps: usize, max_steps: usize, lr_max: f32, lr_min: f32) -> f32 {
    if step < warmup_steps {
        return lr_max * step as f32 / warmup_steps as f32;
    }
    let progress = (step - warmup_steps) as f32 / (max_steps - warmup_steps) as f32;
    lr_min + 0.5 * (lr_max - lr_min) * (1.0 + (std::f32::consts::PI * progress).cos())
}

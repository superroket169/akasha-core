//! Raw CUDA C kernel sources, consumed only as `&str` by `CudaShape::Generic` in
//! `shaders/mod.rs` -- dispatch (buffer locking, launch) lives entirely in wilupgu's
//! generic dispatcher now, so this file needs no wilupgu/cudarc imports at all.

pub(crate) const EMBEDDING: &str = include_str!("cuda/fwd/embedding.cu");
pub(crate) const EMBEDDING_BWD: &str = include_str!("cuda/bwd/embedding_bwd.cu");
pub(crate) const SILU: &str = include_str!("cuda/fwd/silu.cu");
pub(crate) const SILU_OUT: &str = include_str!("cuda/fwd/silu_out.cu");
pub(crate) const ADD: &str = include_str!("cuda/fwd/add.cu");
pub(crate) const SILU_BWD: &str = include_str!("cuda/bwd/silu_bwd.cu");
pub(crate) const ROPE: &str = include_str!("cuda/fwd/rope.cu");
pub(crate) const ROPE_BWD: &str = include_str!("cuda/bwd/rope_bwd.cu");
pub(crate) const ROPE_QK: &str = include_str!("cuda/fwd/rope_qk.cu");
pub(crate) const ROPE_BWD_QK: &str = include_str!("cuda/bwd/rope_bwd_qk.cu");
pub(crate) const ROPE_OFFSET: &str = include_str!("cuda/fwd/rope_offset.cu");
pub(crate) const HEAD_GATHER: &str = include_str!("cuda/head_gather.cu");
pub(crate) const HEAD_SCATTER: &str = include_str!("cuda/head_scatter.cu");
pub(crate) const QKV_SPLIT: &str = include_str!("cuda/fwd/qkv_split.cu");
pub(crate) const QKV_SCATTER: &str = include_str!("cuda/bwd/qkv_scatter.cu");
pub(crate) const ATTN_QK_CACHED: &str = include_str!("cuda/fwd/attn_qk_cached.cu");
pub(crate) const ATTN_AV_CACHED: &str = include_str!("cuda/fwd/attn_av_cached.cu");
pub(crate) const SOFTMAX_RECT: &str = include_str!("cuda/fwd/softmax_rect.cu");
pub(crate) const RMSNORM: &str = include_str!("cuda/fwd/rmsnorm.cu");
pub(crate) const RMSNORM_BWD: &str = include_str!("cuda/bwd/rmsnorm_bwd.cu");
pub(crate) const RMSNORM_WEIGHT_BWD: &str = include_str!("cuda/bwd/rmsnorm_weight_bwd.cu");
pub(crate) const CROSS_ENTROPY: &str = include_str!("cuda/fwd/cross_entropy.cu");
pub(crate) const CROSS_ENTROPY_BWD: &str = include_str!("cuda/bwd/cross_entropy_bwd.cu");
pub(crate) const CACHE_WRITE: &str = include_str!("cuda/cache_write.cu");
pub(crate) const FLASH_ATTENTION: &str = include_str!("cuda/fwd/flash_attention.cu");
pub(crate) const FLASH_ATTENTION_BWD_DQ: &str = include_str!("cuda/bwd/flash_attention_bwd_dq.cu");
pub(crate) const FLASH_ATTENTION_BWD_DKDV: &str = include_str!("cuda/bwd/flash_attention_bwd_dkdv.cu");
pub(crate) const GRAD_SUMSQ: &str = include_str!("cuda/bwd/grad_sumsq.cu");
pub(crate) const GRAD_NORM_SCALE: &str = include_str!("cuda/bwd/grad_norm_scale.cu");
pub(crate) const GRAD_SCALE: &str = include_str!("cuda/bwd/grad_scale.cu");

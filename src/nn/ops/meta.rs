//! Single source of truth for every kernel's Meta-buffer layout.
//!
//! Each struct mirrors, field by field and in order, the meta struct the
//! corresponding WGSL/CUDA/CPU kernel reads — so uploading one is
//! bit-identical to the old positional `&[u32]` arrays. Bare positional meta
//! arrays are banned: the `qkv_proj_meta` N/K-swap bug happened precisely
//! because nothing stopped `[1, dim, dim*3]` from compiling where
//! `[1, dim*3, dim]` was meant.

use std::sync::Arc;
use wilupgu::{Backend, Tensor};

/// Upload/update helpers shared by every meta struct.
pub trait KernelMeta: bytemuck::Pod {
    /// Allocate a device meta tensor holding `self`.
    fn upload<B: Backend>(&self, ctx: &Arc<B>) -> Arc<Tensor<B>> {
        Arc::new(Tensor::init_from_cpu(
            ctx.clone(),
            std::slice::from_ref(self),
        ))
    }

    /// Overwrite an existing (persistent) meta tensor in place.
    fn write_to<B: Backend>(&self, tensor: &Tensor<B>) {
        tensor.copy_from_cpu(std::slice::from_ref(self));
    }
}

/// `MatMul` / `MatMulTrp` / `MatMulAdd` / `MatMulWeightBwd`:
/// C is `[m, n]`, the contraction length is `k`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MatMulMeta {
    pub m: u32,
    pub n: u32,
    pub k: u32,
}
impl KernelMeta for MatMulMeta {}

/// `RMSNorm` / `RMSNormBwd` / `RMSNormWeightBwd`: `seq_len` rows of `size`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct NormMeta {
    pub seq_len: u32,
    pub size: u32,
    pub eps: f32,
}
impl KernelMeta for NormMeta {}

/// `HeadGather` / `HeadScatter`: copy a `head_dim`-wide column slice at
/// `head_offset` between a `full_dim`-wide buffer and a compact
/// `[seq_len, head_dim]` buffer. Also reused for fused-QKV splitting
/// (`head_dim = dim`, `full_dim = 3*dim`).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct HeadMoveMeta {
    pub seq_len: u32,
    pub full_dim: u32,
    pub head_dim: u32,
    pub head_offset: u32,
}
impl KernelMeta for HeadMoveMeta {}

/// `CausalSoftmax` / `SoftmaxBwd`: square `[seq_len, seq_len]` scores,
/// pre-softmax scale `1/sqrt(head_dim)`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AttnScaleMeta {
    pub seq_len: u32,
    pub scale: f32,
}
impl KernelMeta for AttnScaleMeta {}

/// `SoftmaxRect`: rectangular `[num_rows, width]` scores (decode attention,
/// where `width = attn_len` grows each step).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SoftmaxRectMeta {
    pub num_rows: u32,
    pub width: u32,
    pub scale: f32,
}
impl KernelMeta for SoftmaxRectMeta {}

/// `RoPE` / `RoPEBwd`: rotate `seq_len` rows of `dim`, positions `0..seq_len`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RopeMeta {
    pub seq_len: u32,
    pub dim: u32,
    pub head_dim: u32,
}
impl KernelMeta for RopeMeta {}

/// `RoPEOffset`: like `RoPE` but rotating at an absolute position `pos`
/// (decode: `seq_len = 1`, `pos` = cache length).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RopeOffsetMeta {
    pub seq_len: u32,
    pub dim: u32,
    pub head_dim: u32,
    pub pos: u32,
}
impl KernelMeta for RopeOffsetMeta {}

/// `CacheWrite`: append `row_count` rows of `width` into a persistent cache
/// starting at row `dst_row_offset`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CacheWriteMeta {
    pub row_count: u32,
    pub width: u32,
    pub dst_row_offset: u32,
}
impl KernelMeta for CacheWriteMeta {}

/// `Embedding` / `EmbeddingBwd`: table is `[vocab_size, dim]`, `seq_len`
/// token rows.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct EmbeddingMeta {
    pub vocab_size: u32,
    pub dim: u32,
    pub seq_len: u32,
}
impl KernelMeta for EmbeddingMeta {}

/// `CrossEntropy` / `CrossEntropyBwd`: `num_rows` rows of `vocab_size` logits.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CrossEntropyMeta {
    pub vocab_size: u32,
    pub num_rows: u32,
}
impl KernelMeta for CrossEntropyMeta {}

/// `ZeroTensor`: zero `len` elements.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ZeroMeta {
    pub len: u32,
}
impl KernelMeta for ZeroMeta {}

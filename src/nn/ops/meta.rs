use std::sync::Arc;
use wilupgu::{Backend, Tensor};

pub trait KernelMeta: bytemuck::Pod {
    fn upload<B: Backend>(&self, ctx: &Arc<B>) -> Arc<Tensor<B>> {
        Arc::new(Tensor::init_from_cpu(
            ctx.clone(),
            std::slice::from_ref(self),
        ))
    }

    /// Overwrite a persistent meta tensor in place.
    fn write_to<B: Backend>(&self, tensor: &Tensor<B>) {
        tensor.copy_from_cpu(std::slice::from_ref(self));
    }
}

/// MatMul family: C is `[m, n]`, contraction length `k`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MatMulMeta {
    pub m: u32,
    pub n: u32,
    pub k: u32,
}
impl KernelMeta for MatMulMeta {}

/// RMSNorm kernels: `seq_len` rows of `size`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct NormMeta {
    pub seq_len: u32,
    pub size: u32,
    pub eps: f32,
}
impl KernelMeta for NormMeta {}

/// HeadGather/HeadScatter: `head_dim`-wide slice at `head_offset` in a
/// `full_dim`-wide buffer. Reused for fused-QKV slices (`head_dim = dim`).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct HeadMoveMeta {
    pub seq_len: u32,
    pub full_dim: u32,
    pub head_dim: u32,
    pub head_offset: u32,
}
impl KernelMeta for HeadMoveMeta {}

impl HeadMoveMeta {
    /// One dim-wide Q/K/V role slice of a fused `[seq_len, 3*dim]` QKV buffer.
    pub(crate) fn qkv_slice(seq_len: u32, dim: u32, role_offset: u32) -> Self {
        Self {
            seq_len,
            full_dim: 3 * dim,
            head_dim: dim,
            head_offset: role_offset,
        }
    }
}

/// CausalSoftmax/SoftmaxBwd: square `[seq_len, seq_len]` scores.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AttnScaleMeta {
    pub seq_len: u32,
    pub scale: f32,
}
impl KernelMeta for AttnScaleMeta {}

/// SoftmaxRect: rectangular `[num_rows, width]` scores (decode).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SoftmaxRectMeta {
    pub num_rows: u32,
    pub width: u32,
    pub scale: f32,
}
impl KernelMeta for SoftmaxRectMeta {}

/// RoPE/RoPEBwd: rotate `seq_len` rows at positions `0..seq_len`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RopeMeta {
    pub seq_len: u32,
    pub dim: u32,
    pub head_dim: u32,
    pub row_offset: u32,
}
impl KernelMeta for RopeMeta {}

/// RoPEOffset: one row at absolute position `pos` (decode).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RopeOffsetMeta {
    pub seq_len: u32,
    pub dim: u32,
    pub head_dim: u32,
    pub pos: u32,
}
impl KernelMeta for RopeOffsetMeta {}

/// CacheWrite: `row_count` rows of `width` appended at `dst_row_offset`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CacheWriteMeta {
    pub row_count: u32,
    pub width: u32,
    pub dst_row_offset: u32,
}
impl KernelMeta for CacheWriteMeta {}

/// Embedding/EmbeddingBwd: `[vocab_size, dim]` table, `seq_len` token rows.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct EmbeddingMeta {
    pub vocab_size: u32,
    pub dim: u32,
    pub seq_len: u32,
}
impl KernelMeta for EmbeddingMeta {}

/// CrossEntropy kernels: `num_rows` rows of `vocab_size` logits.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CrossEntropyMeta {
    pub vocab_size: u32,
    pub num_rows: u32,
}
impl KernelMeta for CrossEntropyMeta {}

/// ZeroTensor / GradScale: `len` elements, nothing else.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ZeroMeta {
    pub len: u32,
}
impl KernelMeta for ZeroMeta {}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GradSumSqMeta {
    pub len: u32,
    pub out_offset: u32,
}
impl KernelMeta for GradSumSqMeta {}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GradNormMeta {
    pub num_partials: u32,
    pub max_norm: f32,
}
impl KernelMeta for GradNormMeta {}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FlashAttnMeta {
    pub seq_len: u32,
    pub dim: u32,
    pub head_dim: u32,
    pub scale: f32,
    pub row_offset: u32,
}
impl KernelMeta for FlashAttnMeta {}

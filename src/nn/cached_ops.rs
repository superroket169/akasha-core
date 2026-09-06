use super::core_ops::{
    AddOp, AttentionOp, EmbeddingOp, LinearOp, QkvSplitOp, RmsNormOp, RopeQkOp, SiluOp,
};
use super::ops;
use super::ops::meta::{
    AttnCachedMeta, CacheWriteMeta, HeadMoveMeta, KernelMeta, RopeOffsetMeta, SoftmaxRectMeta,
};
use super::ops::{CachedPhase, Decode, GraphBuilder};
use super::tape::{Forward, Leaf, zeros};
use std::sync::Arc;
use wilupgu::{Backend, Tensor};

pub(crate) struct CacheWriteOp<B: Backend> {
    pub(super) cache: Arc<Tensor<B>>,
    pub(super) meta: Arc<Tensor<B>>,
    pub(super) shape: CacheWriteMeta,
}

impl<B: Backend> CacheWriteOp<B> {
    pub(crate) fn new(cache: Arc<Tensor<B>>, row_count: u32, width: u32) -> Self {
        let shape = CacheWriteMeta {
            row_count,
            width,
            dst_row_offset: 0,
        };
        let meta = shape.upload(&cache.ctx);
        Self { cache, meta, shape }
    }
}

impl<B: Backend, P: CachedPhase> Forward<B, P> for CacheWriteOp<B> {
    fn forward(
        &mut self,
        gb: &mut GraphBuilder<'_, B, P>,
        xs: &[Arc<Tensor<B>>],
    ) -> Vec<Arc<Tensor<B>>> {
        ops::cache_write_with(gb, &xs[0], &self.cache, self.shape, &self.meta);
        vec![xs[0].clone()]
    }
}

pub(crate) struct RopeOffsetOp<B: Backend> {
    pub(super) meta: Arc<Tensor<B>>,
    pub(super) shape: RopeOffsetMeta,
}

impl<B: Backend> RopeOffsetOp<B> {
    pub(crate) fn new(ctx: &Arc<B>, dim: u32, head_dim: u32) -> Self {
        let shape = RopeOffsetMeta {
            seq_len: 1,
            dim,
            head_dim,
            pos: 0,
        };
        Self {
            meta: shape.upload(ctx),
            shape,
        }
    }
}

impl<B: Backend> Forward<B, Decode> for RopeOffsetOp<B> {
    fn forward(
        &mut self,
        gb: &mut GraphBuilder<'_, B, Decode>,
        xs: &[Arc<Tensor<B>>],
    ) -> Vec<Arc<Tensor<B>>> {
        ops::rope_offset_with(gb, &xs[0], self.shape, &self.meta);
        vec![xs[0].clone()]
    }
}

pub(crate) struct HeadGatherOp<B: Backend> {
    dst: Arc<Tensor<B>>,
    meta: Arc<Tensor<B>>,
    shape: HeadMoveMeta,
}

impl<B: Backend> HeadGatherOp<B> {
    pub(crate) fn new(ctx: &Arc<B>, dim: u32, role_offset: u32) -> Self {
        let shape = HeadMoveMeta::qkv_slice(1, dim, role_offset);
        Self {
            dst: zeros(ctx, dim),
            meta: shape.upload(ctx),
            shape,
        }
    }
}

impl<B: Backend> Forward<B, Decode> for HeadGatherOp<B> {
    fn forward(
        &mut self,
        gb: &mut GraphBuilder<'_, B, Decode>,
        xs: &[Arc<Tensor<B>>],
    ) -> Vec<Arc<Tensor<B>>> {
        ops::head_gather_with(gb, &xs[0], &self.dst, self.shape, &self.meta);
        vec![self.dst.clone()]
    }
}

pub(crate) struct CachedAttentionOp<B: Backend> {
    cache_k: Arc<Tensor<B>>,
    cache_v: Arc<Tensor<B>>,
    scores: Arc<Tensor<B>>,
    out: Arc<Tensor<B>>,
    pub(super) attn_meta: Arc<Tensor<B>>,
    pub(super) softmax_meta: Arc<Tensor<B>>,
    num_heads: u32,
    dim: u32,
    max_attn_len: u32,
    pub(super) attn_shape: AttnCachedMeta,
    pub(super) softmax_shape: SoftmaxRectMeta,
}

impl<B: Backend> CachedAttentionOp<B> {
    pub(crate) fn new(
        cache_k: Arc<Tensor<B>>,
        cache_v: Arc<Tensor<B>>,
        num_heads: u32,
        dim: u32,
        head_dim: u32,
        max_context_len: u32,
    ) -> Self {
        let ctx = cache_k.ctx.clone();
        let attn_shape = AttnCachedMeta {
            attn_len: 1,
            dim,
            head_dim,
        };
        let scale = 1.0 / (head_dim as f32).sqrt();
        let softmax_shape = SoftmaxRectMeta {
            num_rows: num_heads,
            width: 1,
            scale,
        };
        Self {
            cache_k,
            cache_v,
            scores: zeros(&ctx, num_heads * max_context_len),
            out: zeros(&ctx, dim),
            attn_meta: attn_shape.upload(&ctx),
            softmax_meta: softmax_shape.upload(&ctx),
            num_heads,
            dim,
            max_attn_len: max_context_len,
            attn_shape,
            softmax_shape,
        }
    }
}

impl<B: Backend> Forward<B, Decode> for CachedAttentionOp<B> {
    fn forward(
        &mut self,
        gb: &mut GraphBuilder<'_, B, Decode>,
        xs: &[Arc<Tensor<B>>],
    ) -> Vec<Arc<Tensor<B>>> {
        let q = &xs[0];
        ops::attn_qk_cached_with(
            gb,
            q,
            &self.cache_k,
            &self.scores,
            self.num_heads,
            self.max_attn_len,
            &self.attn_meta,
        );
        ops::softmax_rect_with(gb, &self.scores, self.softmax_shape, &self.softmax_meta);
        ops::attn_av_cached_with(
            gb,
            &self.scores,
            &self.cache_v,
            &self.out,
            self.dim,
            &self.attn_meta,
        );
        vec![self.out.clone()]
    }
}

pub(crate) enum PrefillOp<B: Backend> {
    Embedding(EmbeddingOp<B>),
    Linear(LinearOp<B>),
    RmsNorm(RmsNormOp<B>),
    Silu(SiluOp<B>),
    Add(AddOp<B>),
    RopeQk(RopeQkOp),
    QkvSplit(QkvSplitOp<B>),
    Attention(AttentionOp<B>),
    CacheWrite(CacheWriteOp<B>),
    Leaf(Leaf<B>),
}

pub(crate) enum DecodeOp<B: Backend> {
    Embedding(EmbeddingOp<B>),
    Linear(LinearOp<B>),
    RmsNorm(RmsNormOp<B>),
    Silu(SiluOp<B>),
    Add(AddOp<B>),
    RopeOffset(RopeOffsetOp<B>),
    HeadGather(HeadGatherOp<B>),
    CacheWrite(CacheWriteOp<B>),
    CachedAttention(CachedAttentionOp<B>),
    Leaf(Leaf<B>),
}

// From<X> impls -> from.rs. Forward dispatch -> forward.rs. Advance -> advance.rs.
// (all three generated/collected by their respective files.)

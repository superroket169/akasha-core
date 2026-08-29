//! Train's concrete ops + `TransformerOp` (Tape/Op tasarımı: ARCHITECTURE.md → Big Refactor).

use super::ops;
use super::ops::FlashAttnBuffers;
use super::ops::meta::{
    EmbeddingMeta, FlashAttnMeta, HeadMoveMeta, KernelMeta, MatMulMeta, NormMeta, RopeMeta,
};
use super::ops::{FullSeqPhase, FwdPhase, GraphBuilder, Train};
use super::tape::{Backward, Forward, Identity, zeros, zeros_like};
use std::sync::Arc;
use wilupgu::{Backend, Tensor};

pub(crate) struct EmbeddingOp<B: Backend> {
    tokens: Arc<Tensor<B>>,
    table: Arc<Tensor<B>>,
    grad_table: Arc<Tensor<B>>,
    out: Arc<Tensor<B>>,
    shape: EmbeddingMeta,
}

impl<B: Backend> EmbeddingOp<B> {
    pub(crate) fn new(table: &Arc<Tensor<B>>, seq_len: u32, vocab_size: u32, dim: u32) -> Self {
        let ctx = &table.ctx;
        Self {
            tokens: zeros(ctx, seq_len),
            table: table.clone(),
            grad_table: zeros_like(table),
            out: zeros(ctx, seq_len * dim),
            shape: EmbeddingMeta {
                vocab_size,
                dim,
                seq_len,
            },
        }
    }

    pub(crate) fn tokens_handle(&self) -> Arc<Tensor<B>> {
        self.tokens.clone()
    }
}

impl<B: Backend, P: FwdPhase> Forward<B, P> for EmbeddingOp<B> {
    fn forward(
        &mut self,
        gb: &mut GraphBuilder<'_, B, P>,
        _xs: &[Arc<Tensor<B>>],
    ) -> Vec<Arc<Tensor<B>>> {
        ops::embedding(gb, &self.tokens, &self.table, &self.out, self.shape);
        vec![self.out.clone()]
    }
}

impl<B: Backend> Backward<B> for EmbeddingOp<B> {
    fn backward(
        &mut self,
        gb: &mut GraphBuilder<'_, B, Train>,
        grad_outputs: &[Arc<Tensor<B>>],
    ) -> Vec<Arc<Tensor<B>>> {
        ops::embedding_bwd(
            gb,
            &self.tokens,
            &grad_outputs[0],
            &self.grad_table,
            self.shape,
        );
        vec![]
    }

    fn param(&self) -> Option<(&Arc<Tensor<B>>, &Arc<Tensor<B>>, bool)> {
        // the embedding table is decay-exempt, same as norm gains.
        Some((&self.table, &self.grad_table, false))
    }
}

/// `y = x @ weight`; everything else self-allocated from `weight`.
pub(crate) struct LinearOp<B: Backend> {
    weight: Arc<Tensor<B>>,
    grad_weight: Arc<Tensor<B>>,
    out: Arc<Tensor<B>>,
    grad_in: Arc<Tensor<B>>,
    meta: Arc<Tensor<B>>,
    shape: MatMulMeta,
    saved_input: Option<Arc<Tensor<B>>>,
    decay: bool,
}

impl<B: Backend> LinearOp<B> {
    pub(crate) fn new(weight: &Arc<Tensor<B>>, shape: MatMulMeta, decay: bool) -> Self {
        let ctx = &weight.ctx;
        Self {
            weight: weight.clone(),
            grad_weight: zeros_like(weight),
            out: zeros(ctx, shape.m * shape.n),
            grad_in: zeros(ctx, shape.m * shape.k),
            meta: shape.upload(ctx),
            shape,
            saved_input: None,
            decay,
        }
    }
}

impl<B: Backend, P: FwdPhase> Forward<B, P> for LinearOp<B> {
    fn forward(
        &mut self,
        gb: &mut GraphBuilder<'_, B, P>,
        xs: &[Arc<Tensor<B>>],
    ) -> Vec<Arc<Tensor<B>>> {
        let x = &xs[0];
        ops::matmul_with(gb, x, &self.weight, &self.out, self.shape, &self.meta);
        self.saved_input = Some(x.clone());
        vec![self.out.clone()]
    }
}

impl<B: Backend> Backward<B> for LinearOp<B> {
    fn backward(
        &mut self,
        gb: &mut GraphBuilder<'_, B, Train>,
        grad_outputs: &[Arc<Tensor<B>>],
    ) -> Vec<Arc<Tensor<B>>> {
        let grad_output = &grad_outputs[0];
        let x = self
            .saved_input
            .take()
            .expect("LinearOp::backward called before forward");
        ops::matmul_weight_bwd(gb, &x, grad_output, &self.grad_weight, self.shape);

        // n/k swapped vs the forward shape -- matches layers.rs::Linear's backward.
        let trp_shape = MatMulMeta {
            m: self.shape.m,
            n: self.shape.k,
            k: self.shape.n,
        };
        ops::matmul_trp(gb, grad_output, &self.weight, &self.grad_in, trp_shape);
        vec![self.grad_in.clone()]
    }

    fn param(&self) -> Option<(&Arc<Tensor<B>>, &Arc<Tensor<B>>, bool)> {
        Some((&self.weight, &self.grad_weight, self.decay))
    }
}

/// `weight` comes straight from `BlockWeights` (e.g. `&bw.norm_1`).
pub(crate) struct RmsNormOp<B: Backend> {
    weight: Arc<Tensor<B>>,
    grad_weight: Arc<Tensor<B>>,
    out: Arc<Tensor<B>>,
    grad_in: Arc<Tensor<B>>,
    rsqrt_cache: Arc<Tensor<B>>,
    meta: Arc<Tensor<B>>,
    shape: NormMeta,
    saved_input: Option<Arc<Tensor<B>>>,
}

impl<B: Backend> RmsNormOp<B> {
    pub(crate) fn new(weight: &Arc<Tensor<B>>, shape: NormMeta) -> Self {
        let ctx = &weight.ctx;
        Self {
            weight: weight.clone(),
            grad_weight: zeros_like(weight),
            out: zeros(ctx, shape.seq_len * shape.size),
            grad_in: zeros(ctx, shape.seq_len * shape.size),
            rsqrt_cache: zeros(ctx, shape.seq_len),
            meta: shape.upload(ctx),
            shape,
            saved_input: None,
        }
    }
}

impl<B: Backend, P: FwdPhase> Forward<B, P> for RmsNormOp<B> {
    fn forward(
        &mut self,
        gb: &mut GraphBuilder<'_, B, P>,
        xs: &[Arc<Tensor<B>>],
    ) -> Vec<Arc<Tensor<B>>> {
        let x = &xs[0];
        ops::rmsnorm_with(gb, x, &self.weight, &self.out, self.shape, &self.meta);
        self.saved_input = Some(x.clone());
        vec![self.out.clone()]
    }
}

impl<B: Backend> Backward<B> for RmsNormOp<B> {
    fn backward(
        &mut self,
        gb: &mut GraphBuilder<'_, B, Train>,
        grad_outputs: &[Arc<Tensor<B>>],
    ) -> Vec<Arc<Tensor<B>>> {
        let x = self
            .saved_input
            .take()
            .expect("RmsNormOp::backward called before forward");
        ops::rmsnorm_bwd(
            gb,
            &grad_outputs[0],
            &x,
            &self.weight,
            &self.grad_in,
            &self.rsqrt_cache,
            &self.grad_weight,
            self.shape,
        );
        vec![self.grad_in.clone()]
    }

    fn param(&self) -> Option<(&Arc<Tensor<B>>, &Arc<Tensor<B>>, bool)> {
        // norm gains are decay-exempt (E1, ARCHITECTURE.md Invariantlar).
        Some((&self.weight, &self.grad_weight, false))
    }
}

/// Weightless; silu_out/silu_bwd (not the in-place `silu`) since backward needs the pre-activation.
pub(crate) struct SiluOp<B: Backend> {
    out: Arc<Tensor<B>>,
    grad_in: Arc<Tensor<B>>,
    len: u32,
    saved_input: Option<Arc<Tensor<B>>>,
}

impl<B: Backend> SiluOp<B> {
    pub(crate) fn new(ctx: &Arc<B>, len: u32) -> Self {
        Self {
            out: zeros(ctx, len),
            grad_in: zeros(ctx, len),
            len,
            saved_input: None,
        }
    }
}

impl<B: Backend, P: FwdPhase> Forward<B, P> for SiluOp<B> {
    fn forward(
        &mut self,
        gb: &mut GraphBuilder<'_, B, P>,
        xs: &[Arc<Tensor<B>>],
    ) -> Vec<Arc<Tensor<B>>> {
        let x = &xs[0];
        ops::silu_out(gb, x, &self.out, self.len);
        self.saved_input = Some(x.clone());
        vec![self.out.clone()]
    }
}

impl<B: Backend> Backward<B> for SiluOp<B> {
    fn backward(
        &mut self,
        gb: &mut GraphBuilder<'_, B, Train>,
        grad_outputs: &[Arc<Tensor<B>>],
    ) -> Vec<Arc<Tensor<B>>> {
        let x = self
            .saved_input
            .take()
            .expect("SiluOp::backward called before forward");
        ops::silu_bwd(gb, &x, &grad_outputs[0], &self.grad_in, self.len);
        vec![self.grad_in.clone()]
    }
}

/// `y = a + b` (residual); backward hands the same gradient to both inputs unchanged.
pub(crate) struct AddOp<B: Backend> {
    out: Arc<Tensor<B>>,
    len: u32,
}

impl<B: Backend> AddOp<B> {
    pub(crate) fn new(ctx: &Arc<B>, len: u32) -> Self {
        Self {
            out: zeros(ctx, len),
            len,
        }
    }
}

impl<B: Backend, P: FwdPhase> Forward<B, P> for AddOp<B> {
    fn forward(
        &mut self,
        gb: &mut GraphBuilder<'_, B, P>,
        xs: &[Arc<Tensor<B>>],
    ) -> Vec<Arc<Tensor<B>>> {
        ops::add_out(gb, &xs[0], &xs[1], &self.out, self.len);
        vec![self.out.clone()]
    }
}

impl<B: Backend> Backward<B> for AddOp<B> {
    fn backward(
        &mut self,
        _gb: &mut GraphBuilder<'_, B, Train>,
        grad_outputs: &[Arc<Tensor<B>>],
    ) -> Vec<Arc<Tensor<B>>> {
        vec![grad_outputs[0].clone(), grad_outputs[0].clone()]
    }
}

/// Fused in-place RoPE on Q+K, `P: FullSeqPhase` (Train+Prefill); Decode uses `RopeOffsetOp`.
pub(crate) struct RopeQkOp {
    shape: RopeMeta,
    batch_size: u32,
}

impl RopeQkOp {
    pub(crate) fn new(seq_len: u32, dim: u32, head_dim: u32, batch_size: u32) -> Self {
        Self {
            shape: RopeMeta {
                seq_len,
                dim,
                head_dim,
                row_offset: 0,
            },
            batch_size,
        }
    }

    fn shape_for(&self, b: u32) -> RopeMeta {
        RopeMeta {
            row_offset: b * self.shape.seq_len,
            ..self.shape
        }
    }
}

impl<B: Backend, P: FullSeqPhase> Forward<B, P> for RopeQkOp {
    fn forward(
        &mut self,
        gb: &mut GraphBuilder<'_, B, P>,
        xs: &[Arc<Tensor<B>>],
    ) -> Vec<Arc<Tensor<B>>> {
        for b in 0..self.batch_size {
            ops::rope_qk(gb, &xs[0], &xs[1], self.shape_for(b));
        }
        vec![xs[0].clone(), xs[1].clone()]
    }
}

impl<B: Backend> Backward<B> for RopeQkOp {
    fn backward(
        &mut self,
        gb: &mut GraphBuilder<'_, B, Train>,
        grad_outputs: &[Arc<Tensor<B>>],
    ) -> Vec<Arc<Tensor<B>>> {
        for b in 0..self.batch_size {
            ops::rope_bwd_qk(gb, &grad_outputs[0], &grad_outputs[1], self.shape_for(b));
        }
        vec![grad_outputs[0].clone(), grad_outputs[1].clone()]
    }
}

/// Fused qkv split/scatter, `P: FullSeqPhase`; Decode uses unfused `HeadGatherOp` instead.
pub(crate) struct QkvSplitOp<B: Backend> {
    q: Arc<Tensor<B>>,
    k: Arc<Tensor<B>>,
    v: Arc<Tensor<B>>,
    grad_qkv: Arc<Tensor<B>>,
    shape: HeadMoveMeta,
}

impl<B: Backend> QkvSplitOp<B> {
    pub(crate) fn new(ctx: &Arc<B>, rows: u32, dim: u32) -> Self {
        Self {
            q: zeros(ctx, rows * dim),
            k: zeros(ctx, rows * dim),
            v: zeros(ctx, rows * dim),
            grad_qkv: zeros(ctx, rows * dim * 3),
            shape: HeadMoveMeta::qkv_slice(rows, dim, 0),
        }
    }
}

impl<B: Backend, P: FullSeqPhase> Forward<B, P> for QkvSplitOp<B> {
    fn forward(
        &mut self,
        gb: &mut GraphBuilder<'_, B, P>,
        xs: &[Arc<Tensor<B>>],
    ) -> Vec<Arc<Tensor<B>>> {
        ops::qkv_split(gb, &xs[0], &self.q, &self.k, &self.v, self.shape);
        vec![self.q.clone(), self.k.clone(), self.v.clone()]
    }
}

impl<B: Backend> Backward<B> for QkvSplitOp<B> {
    fn backward(
        &mut self,
        gb: &mut GraphBuilder<'_, B, Train>,
        grad_outputs: &[Arc<Tensor<B>>],
    ) -> Vec<Arc<Tensor<B>>> {
        ops::qkv_scatter(
            gb,
            &grad_outputs[0],
            &grad_outputs[1],
            &grad_outputs[2],
            &self.grad_qkv,
            self.shape,
        );
        vec![self.grad_qkv.clone()]
    }
}

/// Flash attention, `P: FullSeqPhase`, no weight. Decode uses `CachedAttentionOp` (a different kernel, not phase-generic).
pub(crate) struct AttentionOp<B: Backend> {
    out: Arc<Tensor<B>>,
    grad_q: Arc<Tensor<B>>,
    grad_k: Arc<Tensor<B>>,
    grad_v: Arc<Tensor<B>>,
    seq_len: u32,
    dim: u32,
    head_dim: u32,
    batch_size: u32,
    saved: Option<(
        Arc<Tensor<B>>,
        Arc<Tensor<B>>,
        Arc<Tensor<B>>,
        Vec<FlashAttnBuffers<B>>,
    )>,
}

impl<B: Backend> AttentionOp<B> {
    pub(crate) fn new(
        ctx: &Arc<B>,
        seq_len: u32,
        dim: u32,
        head_dim: u32,
        batch_size: u32,
    ) -> Self {
        let rows_size = seq_len * batch_size * dim;
        Self {
            out: zeros(ctx, rows_size),
            grad_q: zeros(ctx, rows_size),
            grad_k: zeros(ctx, rows_size),
            grad_v: zeros(ctx, rows_size),
            seq_len,
            dim,
            head_dim,
            batch_size,
            saved: None,
        }
    }

    fn shape_for(&self, b: u32) -> FlashAttnMeta {
        FlashAttnMeta {
            seq_len: self.seq_len,
            dim: self.dim,
            head_dim: self.head_dim,
            scale: 1.0 / (self.head_dim as f32).sqrt(),
            row_offset: b * self.seq_len,
        }
    }
}

impl<B: Backend, P: FullSeqPhase> Forward<B, P> for AttentionOp<B> {
    fn forward(
        &mut self,
        gb: &mut GraphBuilder<'_, B, P>,
        xs: &[Arc<Tensor<B>>],
    ) -> Vec<Arc<Tensor<B>>> {
        let (q, k, v) = (xs[0].clone(), xs[1].clone(), xs[2].clone());
        let bufs = (0..self.batch_size)
            .map(|b| ops::flash_attention(gb, &q, &k, &v, &self.out, self.shape_for(b)))
            .collect();
        self.saved = Some((q, k, v, bufs));
        vec![self.out.clone()]
    }
}

impl<B: Backend> Backward<B> for AttentionOp<B> {
    fn backward(
        &mut self,
        gb: &mut GraphBuilder<'_, B, Train>,
        grad_outputs: &[Arc<Tensor<B>>],
    ) -> Vec<Arc<Tensor<B>>> {
        let (q, k, v, bufs) = self
            .saved
            .take()
            .expect("AttentionOp::backward called before forward");
        for (b, saved_bufs) in bufs.into_iter().enumerate() {
            ops::flash_attention_bwd(
                gb,
                &q,
                &k,
                &v,
                &saved_bufs,
                &grad_outputs[0],
                &self.grad_q,
                &self.grad_k,
                &self.grad_v,
                self.shape_for(b as u32),
            );
        }
        vec![
            self.grad_q.clone(),
            self.grad_k.clone(),
            self.grad_v.clone(),
        ]
    }
}

/// What `Tape<B, TransformerOp<B>>` (training) holds -- closed enum, static dispatch.
pub(crate) enum TransformerOp<B: Backend> {
    Embedding(EmbeddingOp<B>),
    Linear(LinearOp<B>),
    RmsNorm(RmsNormOp<B>),
    Silu(SiluOp<B>),
    Add(AddOp<B>),
    RopeQk(RopeQkOp),
    QkvSplit(QkvSplitOp<B>),
    Attention(AttentionOp<B>),
    Identity(Identity<B>),
}

impl<B: Backend> From<Identity<B>> for TransformerOp<B> {
    fn from(id: Identity<B>) -> Self {
        TransformerOp::Identity(id)
    }
}

impl<B: Backend> Forward<B, Train> for TransformerOp<B> {
    fn forward(
        &mut self,
        gb: &mut GraphBuilder<'_, B, Train>,
        xs: &[Arc<Tensor<B>>],
    ) -> Vec<Arc<Tensor<B>>> {
        match self {
            TransformerOp::Embedding(op) => op.forward(gb, xs),
            TransformerOp::Linear(op) => op.forward(gb, xs),
            TransformerOp::RmsNorm(op) => op.forward(gb, xs),
            TransformerOp::Silu(op) => op.forward(gb, xs),
            TransformerOp::Add(op) => op.forward(gb, xs),
            TransformerOp::RopeQk(op) => op.forward(gb, xs),
            TransformerOp::QkvSplit(op) => op.forward(gb, xs),
            TransformerOp::Attention(op) => op.forward(gb, xs),
            TransformerOp::Identity(op) => op.forward(gb, xs),
        }
    }
}

impl<B: Backend> Backward<B> for TransformerOp<B> {
    fn backward(
        &mut self,
        gb: &mut GraphBuilder<'_, B, Train>,
        grad_outputs: &[Arc<Tensor<B>>],
    ) -> Vec<Arc<Tensor<B>>> {
        match self {
            TransformerOp::Embedding(op) => op.backward(gb, grad_outputs),
            TransformerOp::Linear(op) => op.backward(gb, grad_outputs),
            TransformerOp::RmsNorm(op) => op.backward(gb, grad_outputs),
            TransformerOp::Silu(op) => op.backward(gb, grad_outputs),
            TransformerOp::Add(op) => op.backward(gb, grad_outputs),
            TransformerOp::RopeQk(op) => op.backward(gb, grad_outputs),
            TransformerOp::QkvSplit(op) => op.backward(gb, grad_outputs),
            TransformerOp::Attention(op) => op.backward(gb, grad_outputs),
            TransformerOp::Identity(op) => op.backward(gb, grad_outputs),
        }
    }

    fn param(&self) -> Option<(&Arc<Tensor<B>>, &Arc<Tensor<B>>, bool)> {
        match self {
            TransformerOp::Embedding(op) => op.param(),
            TransformerOp::Linear(op) => op.param(),
            TransformerOp::RmsNorm(op) => op.param(),
            TransformerOp::Silu(op) => op.param(),
            TransformerOp::Add(op) => op.param(),
            TransformerOp::RopeQk(op) => op.param(),
            TransformerOp::QkvSplit(op) => op.param(),
            TransformerOp::Attention(op) => op.param(),
            TransformerOp::Identity(op) => op.param(),
        }
    }
}

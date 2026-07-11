use super::meta::{
    AttnScaleMeta, CacheWriteMeta, CrossEntropyMeta, EmbeddingMeta, FlashAttnMeta, GradNormMeta,
    GradSumSqMeta, HeadMoveMeta, KernelMeta, MatMulMeta, NormMeta, RopeMeta, RopeOffsetMeta,
    SoftmaxRectMeta, ZeroMeta,
};
use super::{CachedPhase, Decode, FullSeqPhase, FwdPhase, GraphBuilder, Phase, Train};
use crate::Real;
use crate::shaders;
use std::sync::Arc;
use wilupgu::builtin;
use wilupgu::{Backend, Binding, Shader, Tensor, TensorMode};

// ---- matmul ----

fn grid_nm(shape: MatMulMeta) -> [u32; 3] {
    [(shape.n + 15) / 16, (shape.m + 15) / 16, 1]
}

/// `C[m,n] = A[m,k] @ B[k,n]`
pub(crate) fn matmul_with<B: Backend, P: FwdPhase>(
    gb: &mut GraphBuilder<'_, B, P>,
    a: &Arc<Tensor<B>>,
    b: &Arc<Tensor<B>>,
    c: &Arc<Tensor<B>>,
    shape: MatMulMeta,
    meta: &Arc<Tensor<B>>,
) {
    gb.graph.add_node(
        &builtin::MATMUL,
        &[
            Binding::new(0, &a.buffer, TensorMode::Input),
            Binding::new(1, &b.buffer, TensorMode::Input),
            Binding::new(2, &c.buffer, TensorMode::Output),
            Binding::new(3, &meta.buffer, TensorMode::Meta),
        ],
        grid_nm(shape),
    );
}

pub(crate) fn matmul<B: Backend, P: FwdPhase>(
    gb: &mut GraphBuilder<'_, B, P>,
    a: &Arc<Tensor<B>>,
    b: &Arc<Tensor<B>>,
    c: &Arc<Tensor<B>>,
    shape: MatMulMeta,
) {
    let meta = shape.upload(&a.ctx);
    matmul_with(gb, a, b, c, shape, &meta);
}

/// `C[m,n] = A[m,k] @ B[n,k]^T`
pub(crate) fn matmul_trp_with<B: Backend, P: FwdPhase>(
    gb: &mut GraphBuilder<'_, B, P>,
    a: &Arc<Tensor<B>>,
    b: &Arc<Tensor<B>>,
    c: &Arc<Tensor<B>>,
    shape: MatMulMeta,
    meta: &Arc<Tensor<B>>,
) {
    gb.graph.add_node(
        &builtin::MATMUL_TRP,
        &[
            Binding::new(0, &a.buffer, TensorMode::Input),
            Binding::new(1, &b.buffer, TensorMode::Input),
            Binding::new(2, &c.buffer, TensorMode::Output),
            Binding::new(3, &meta.buffer, TensorMode::Meta),
        ],
        grid_nm(shape),
    );
}

pub(crate) fn matmul_trp<B: Backend, P: FwdPhase>(
    gb: &mut GraphBuilder<'_, B, P>,
    a: &Arc<Tensor<B>>,
    b: &Arc<Tensor<B>>,
    c: &Arc<Tensor<B>>,
    shape: MatMulMeta,
) {
    let meta = shape.upload(&a.ctx);
    matmul_trp_with(gb, a, b, c, shape, &meta);
}

/// `C[m,n] += A[m,k] @ B[k,n]` (fused residual, `c` is InOut)
pub(crate) fn matmul_add_with<B: Backend, P: FwdPhase>(
    gb: &mut GraphBuilder<'_, B, P>,
    a: &Arc<Tensor<B>>,
    b: &Arc<Tensor<B>>,
    c: &Arc<Tensor<B>>,
    shape: MatMulMeta,
    meta: &Arc<Tensor<B>>,
) {
    gb.graph.add_node(
        &builtin::MATMUL_ADD,
        &[
            Binding::new(0, &a.buffer, TensorMode::Input),
            Binding::new(1, &b.buffer, TensorMode::Input),
            Binding::new(2, &c.buffer, TensorMode::InOut),
            Binding::new(3, &meta.buffer, TensorMode::Meta),
        ],
        grid_nm(shape),
    );
}

pub(crate) fn matmul_add<B: Backend, P: FwdPhase>(
    gb: &mut GraphBuilder<'_, B, P>,
    a: &Arc<Tensor<B>>,
    b: &Arc<Tensor<B>>,
    c: &Arc<Tensor<B>>,
    shape: MatMulMeta,
) {
    let meta = shape.upload(&a.ctx);
    matmul_add_with(gb, a, b, c, shape, &meta);
}

/// `dW[k,n] += A[m,k]^T @ dY[m,n]` -- accumulates, zero `grad_weight` first.
pub(crate) fn matmul_weight_bwd<B: Backend>(
    gb: &mut GraphBuilder<'_, B, Train>,
    input: &Arc<Tensor<B>>,
    grad_output: &Arc<Tensor<B>>,
    grad_weight: &Arc<Tensor<B>>,
    shape: MatMulMeta,
) {
    let meta = shape.upload(&input.ctx);
    gb.graph.add_node(
        &builtin::MATMUL_WEIGHT_BWD,
        &[
            Binding::new(0, &input.buffer, TensorMode::Input),
            Binding::new(1, &grad_output.buffer, TensorMode::Input),
            Binding::new(2, &grad_weight.buffer, TensorMode::Output),
            Binding::new(3, &meta.buffer, TensorMode::Meta),
        ],
        [(shape.n + 15) / 16, (shape.k + 15) / 16, 1],
    );
}

// ---- norm ----

pub(crate) fn rmsnorm_with<B: Backend, P: FwdPhase>(
    gb: &mut GraphBuilder<'_, B, P>,
    input: &Arc<Tensor<B>>,
    weight: &Arc<Tensor<B>>,
    output: &Arc<Tensor<B>>,
    shape: NormMeta,
    meta: &Arc<Tensor<B>>,
) {
    gb.graph.add_node(
        &shaders::RMSNORM,
        &[
            Binding::new(0, &input.buffer, TensorMode::Input),
            Binding::new(1, &weight.buffer, TensorMode::Input),
            Binding::new(2, &output.buffer, TensorMode::Output),
            Binding::new(3, &meta.buffer, TensorMode::Meta),
        ],
        [shape.seq_len, 1, 1],
    );
}

pub(crate) fn rmsnorm<B: Backend, P: FwdPhase>(
    gb: &mut GraphBuilder<'_, B, P>,
    input: &Arc<Tensor<B>>,
    weight: &Arc<Tensor<B>>,
    output: &Arc<Tensor<B>>,
    shape: NormMeta,
) {
    let meta = shape.upload(&input.ctx);
    rmsnorm_with(gb, input, weight, output, shape, &meta);
}

/// Both backward nodes (input grad + weight grad, linked by `rsqrt_cache`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn rmsnorm_bwd<B: Backend>(
    gb: &mut GraphBuilder<'_, B, Train>,
    grad_output: &Arc<Tensor<B>>,
    input: &Arc<Tensor<B>>,
    weight: &Arc<Tensor<B>>,
    grad_input: &Arc<Tensor<B>>,
    rsqrt_cache: &Arc<Tensor<B>>,
    grad_weight: &Arc<Tensor<B>>,
    shape: NormMeta,
) {
    let meta = shape.upload(&input.ctx);

    gb.graph.add_node(
        &shaders::RMSNORM_BWD,
        &[
            Binding::new(0, &grad_output.buffer, TensorMode::Input),
            Binding::new(1, &input.buffer, TensorMode::Input),
            Binding::new(2, &weight.buffer, TensorMode::Input),
            Binding::new(3, &grad_input.buffer, TensorMode::Output),
            Binding::new(4, &rsqrt_cache.buffer, TensorMode::Output),
            Binding::new(5, &meta.buffer, TensorMode::Meta),
        ],
        [shape.seq_len, 1, 1],
    );

    gb.graph.add_node(
        &shaders::RMSNORM_WEIGHT_BWD,
        &[
            Binding::new(0, &grad_output.buffer, TensorMode::Input),
            Binding::new(1, &input.buffer, TensorMode::Input),
            Binding::new(2, &rsqrt_cache.buffer, TensorMode::Input),
            Binding::new(3, &grad_weight.buffer, TensorMode::Output),
            Binding::new(4, &meta.buffer, TensorMode::Meta),
        ],
        [(shape.size + 255) / 256, 1, 1],
    );
}

// ---- embedding ----

fn grid_embedding(shape: EmbeddingMeta) -> [u32; 3] {
    [(shape.dim + 255) / 256, shape.seq_len, 1]
}

pub(crate) fn embedding_with<B: Backend, P: FwdPhase>(
    gb: &mut GraphBuilder<'_, B, P>,
    tokens: &Arc<Tensor<B>>,
    table: &Arc<Tensor<B>>,
    output: &Arc<Tensor<B>>,
    shape: EmbeddingMeta,
    meta: &Arc<Tensor<B>>,
) {
    gb.graph.add_node(
        &shaders::EMBEDDING,
        &[
            Binding::new(0, &tokens.buffer, TensorMode::Input),
            Binding::new(1, &table.buffer, TensorMode::Input),
            Binding::new(2, &output.buffer, TensorMode::Output),
            Binding::new(3, &meta.buffer, TensorMode::Meta),
        ],
        grid_embedding(shape),
    );
}

pub(crate) fn embedding<B: Backend, P: FwdPhase>(
    gb: &mut GraphBuilder<'_, B, P>,
    tokens: &Arc<Tensor<B>>,
    table: &Arc<Tensor<B>>,
    output: &Arc<Tensor<B>>,
    shape: EmbeddingMeta,
) {
    let meta = shape.upload(&tokens.ctx);
    embedding_with(gb, tokens, table, output, shape, &meta);
}

pub(crate) fn embedding_bwd<B: Backend>(
    gb: &mut GraphBuilder<'_, B, Train>,
    tokens: &Arc<Tensor<B>>,
    grad_output: &Arc<Tensor<B>>,
    grad_table: &Arc<Tensor<B>>,
    shape: EmbeddingMeta,
) {
    let meta = shape.upload(&tokens.ctx);
    gb.graph.add_node(
        &shaders::EMBEDDING_BWD,
        &[
            Binding::new(0, &tokens.buffer, TensorMode::Input),
            Binding::new(1, &grad_output.buffer, TensorMode::Input),
            Binding::new(2, &grad_table.buffer, TensorMode::Output),
            Binding::new(3, &meta.buffer, TensorMode::Meta),
        ],
        grid_embedding(shape),
    );
}

// ---- rope ----

fn inout_meta_node<B: Backend, P: Phase>(
    gb: &mut GraphBuilder<'_, B, P>,
    shader: &'static Shader,
    buf: &Arc<Tensor<B>>,
    meta: &Arc<Tensor<B>>,
    grid: [u32; 3],
) {
    gb.graph.add_node(
        shader,
        &[
            Binding::new(0, &buf.buffer, TensorMode::InOut),
            Binding::new(1, &meta.buffer, TensorMode::Meta),
        ],
        grid,
    );
}

fn grid_full(shape: RopeMeta) -> [u32; 3] {
    [
        (shape.head_dim / 2 + 15) / 16,
        (shape.seq_len + 15) / 16,
        shape.dim / shape.head_dim,
    ]
}

pub(crate) fn rope<B: Backend, P: FullSeqPhase>(
    gb: &mut GraphBuilder<'_, B, P>,
    buf: &Arc<Tensor<B>>,
    shape: RopeMeta,
) {
    let meta = shape.upload(&buf.ctx);
    inout_meta_node(gb, &shaders::ROPE, buf, &meta, grid_full(shape));
}

pub(crate) fn rope_bwd<B: Backend>(
    gb: &mut GraphBuilder<'_, B, Train>,
    grad: &Arc<Tensor<B>>,
    shape: RopeMeta,
) {
    let meta = shape.upload(&grad.ctx);
    inout_meta_node(gb, &shaders::ROPE_BWD, grad, &meta, grid_full(shape));
}

pub(crate) fn rope_offset_with<B: Backend>(
    gb: &mut GraphBuilder<'_, B, Decode>,
    buf: &Arc<Tensor<B>>,
    shape: RopeOffsetMeta,
    meta: &Arc<Tensor<B>>,
) {
    inout_meta_node(
        gb,
        &shaders::ROPE_OFFSET,
        buf,
        meta,
        [
            (shape.head_dim / 2 + 15) / 16,
            1,
            shape.dim / shape.head_dim,
        ],
    );
}

pub(crate) fn rope_qk<B: Backend, P: FullSeqPhase>(
    gb: &mut GraphBuilder<'_, B, P>,
    q_buf: &Arc<Tensor<B>>,
    k_buf: &Arc<Tensor<B>>,
    shape: RopeMeta,
) {
    let meta = shape.upload(&q_buf.ctx);
    gb.graph.add_node(
        &shaders::ROPE_QK,
        &[
            Binding::new(0, &q_buf.buffer, TensorMode::InOut),
            Binding::new(1, &k_buf.buffer, TensorMode::InOut),
            Binding::new(2, &meta.buffer, TensorMode::Meta),
        ],
        grid_full(shape),
    );
}

pub(crate) fn rope_bwd_qk<B: Backend>(
    gb: &mut GraphBuilder<'_, B, Train>,
    grad_q: &Arc<Tensor<B>>,
    grad_k: &Arc<Tensor<B>>,
    shape: RopeMeta,
) {
    let meta = shape.upload(&grad_q.ctx);
    gb.graph.add_node(
        &shaders::ROPE_BWD_QK,
        &[
            Binding::new(0, &grad_q.buffer, TensorMode::InOut),
            Binding::new(1, &grad_k.buffer, TensorMode::InOut),
            Binding::new(2, &meta.buffer, TensorMode::Meta),
        ],
        grid_full(shape),
    );
}

// ---- head_move ----

fn grid_head(shape: HeadMoveMeta) -> [u32; 3] {
    [(shape.head_dim + 15) / 16, (shape.seq_len + 15) / 16, 1]
}

fn move_node<B: Backend, P: Phase>(
    gb: &mut GraphBuilder<'_, B, P>,
    shader: &'static Shader,
    src: &Arc<Tensor<B>>,
    dst: &Arc<Tensor<B>>,
    shape: HeadMoveMeta,
    meta: &Arc<Tensor<B>>,
) {
    gb.graph.add_node(
        shader,
        &[
            Binding::new(0, &src.buffer, TensorMode::Input),
            Binding::new(1, &dst.buffer, TensorMode::Output),
            Binding::new(2, &meta.buffer, TensorMode::Meta),
        ],
        grid_head(shape),
    );
}

/// wide `src` -> compact `dst`
pub(crate) fn head_gather_with<B: Backend, P: FwdPhase>(
    gb: &mut GraphBuilder<'_, B, P>,
    src: &Arc<Tensor<B>>,
    dst: &Arc<Tensor<B>>,
    shape: HeadMoveMeta,
    meta: &Arc<Tensor<B>>,
) {
    move_node(gb, &shaders::HEAD_GATHER, src, dst, shape, meta);
}

pub(crate) fn head_gather<B: Backend, P: FwdPhase>(
    gb: &mut GraphBuilder<'_, B, P>,
    src: &Arc<Tensor<B>>,
    dst: &Arc<Tensor<B>>,
    shape: HeadMoveMeta,
) {
    let meta = shape.upload(&src.ctx);
    head_gather_with(gb, src, dst, shape, &meta);
}

/// compact `src` -> wide `dst`
pub(crate) fn head_scatter_with<B: Backend, P: FwdPhase>(
    gb: &mut GraphBuilder<'_, B, P>,
    src: &Arc<Tensor<B>>,
    dst: &Arc<Tensor<B>>,
    shape: HeadMoveMeta,
    meta: &Arc<Tensor<B>>,
) {
    move_node(gb, &shaders::HEAD_SCATTER, src, dst, shape, meta);
}

pub(crate) fn head_scatter<B: Backend, P: FwdPhase>(
    gb: &mut GraphBuilder<'_, B, P>,
    src: &Arc<Tensor<B>>,
    dst: &Arc<Tensor<B>>,
    shape: HeadMoveMeta,
) {
    let meta = shape.upload(&src.ctx);
    head_scatter_with(gb, src, dst, shape, &meta);
}

pub(crate) fn qkv_split<B: Backend, P: FwdPhase>(
    gb: &mut GraphBuilder<'_, B, P>,
    src: &Arc<Tensor<B>>,
    q_buf: &Arc<Tensor<B>>,
    k_buf: &Arc<Tensor<B>>,
    v_buf: &Arc<Tensor<B>>,
    shape: HeadMoveMeta,
) {
    let meta = shape.upload(&src.ctx);
    gb.graph.add_node(
        &shaders::QKV_SPLIT,
        &[
            Binding::new(0, &src.buffer, TensorMode::Input),
            Binding::new(1, &q_buf.buffer, TensorMode::Output),
            Binding::new(2, &k_buf.buffer, TensorMode::Output),
            Binding::new(3, &v_buf.buffer, TensorMode::Output),
            Binding::new(4, &meta.buffer, TensorMode::Meta),
        ],
        grid_head(shape),
    );
}

pub(crate) fn qkv_scatter<B: Backend, P: FwdPhase>(
    gb: &mut GraphBuilder<'_, B, P>,
    grad_q: &Arc<Tensor<B>>,
    grad_k: &Arc<Tensor<B>>,
    grad_v: &Arc<Tensor<B>>,
    dst: &Arc<Tensor<B>>,
    shape: HeadMoveMeta,
) {
    let meta = shape.upload(&dst.ctx);
    gb.graph.add_node(
        &shaders::QKV_SCATTER,
        &[
            Binding::new(0, &grad_q.buffer, TensorMode::Input),
            Binding::new(1, &grad_k.buffer, TensorMode::Input),
            Binding::new(2, &grad_v.buffer, TensorMode::Input),
            Binding::new(3, &dst.buffer, TensorMode::Output),
            Binding::new(4, &meta.buffer, TensorMode::Meta),
        ],
        grid_head(shape),
    );
}

// ---- attention ----

/// Fused causal-mask + scale + softmax, in place.
pub(crate) fn causal_softmax<B: Backend, P: FullSeqPhase>(
    gb: &mut GraphBuilder<'_, B, P>,
    scores: &Arc<Tensor<B>>,
    shape: AttnScaleMeta,
) {
    let meta = shape.upload(&scores.ctx);
    gb.graph.add_node(
        &shaders::CAUSAL_SOFTMAX,
        &[
            Binding::new(0, &scores.buffer, TensorMode::InOut),
            Binding::new(1, &meta.buffer, TensorMode::Meta),
        ],
        [(shape.seq_len + 255) / 256, 1, 1],
    );
}

/// Scaled softmax, in place; no mask (decode cache only contains past).
pub(crate) fn softmax_rect_with<B: Backend>(
    gb: &mut GraphBuilder<'_, B, Decode>,
    scores: &Arc<Tensor<B>>,
    shape: SoftmaxRectMeta,
    meta: &Arc<Tensor<B>>,
) {
    gb.graph.add_node(
        &shaders::SOFTMAX_RECT,
        &[
            Binding::new(0, &scores.buffer, TensorMode::InOut),
            Binding::new(1, &meta.buffer, TensorMode::Meta),
        ],
        [(shape.num_rows + 255) / 256, 1, 1],
    );
}

pub(crate) fn softmax_bwd<B: Backend>(
    gb: &mut GraphBuilder<'_, B, Train>,
    y: &Arc<Tensor<B>>,
    grad_y: &Arc<Tensor<B>>,
    grad_raw: &Arc<Tensor<B>>,
    shape: AttnScaleMeta,
) {
    let meta = shape.upload(&y.ctx);
    gb.graph.add_node(
        &shaders::SOFTMAX_BWD,
        &[
            Binding::new(0, &y.buffer, TensorMode::Input),
            Binding::new(1, &grad_y.buffer, TensorMode::Input),
            Binding::new(2, &grad_raw.buffer, TensorMode::Output),
            Binding::new(3, &meta.buffer, TensorMode::Meta),
        ],
        [(shape.seq_len + 255) / 256, 1, 1],
    );
}

/// Saved per-head activations; the train backward pass reads these.
pub(crate) struct CausalAttnBuffers<B: Backend> {
    pub q_heads: Vec<Arc<Tensor<B>>>,
    pub k_heads: Vec<Arc<Tensor<B>>>,
    pub v_heads: Vec<Arc<Tensor<B>>>,
    pub scores_heads: Vec<Arc<Tensor<B>>>,
}

/// Multi-head causal attention forward, shared verbatim by train and prefill.
pub(crate) fn causal_attention<B: Backend, P: FullSeqPhase>(
    gb: &mut GraphBuilder<'_, B, P>,
    q_buf: &Arc<Tensor<B>>,
    k_buf: &Arc<Tensor<B>>,
    v_buf: &Arc<Tensor<B>>,
    out_buffer: &Arc<Tensor<B>>,
    seq_len: u32,
    dim: u32,
    num_heads: u32,
) -> CausalAttnBuffers<B> {
    let ctx = q_buf.ctx.clone();
    assert_eq!(dim % num_heads, 0, "dim must be divisible by num_heads");
    let head_dim = dim / num_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let head_size = (seq_len * head_dim) as usize;
    let scores_size = (seq_len * seq_len) as usize;

    let mut bufs = CausalAttnBuffers {
        q_heads: Vec::with_capacity(num_heads as usize),
        k_heads: Vec::with_capacity(num_heads as usize),
        v_heads: Vec::with_capacity(num_heads as usize),
        scores_heads: Vec::with_capacity(num_heads as usize),
    };

    // Shared scratch: reused/overwritten sequentially per head
    let out_head = Arc::new(Tensor::init_from_cpu(
        ctx.clone(),
        &vec![0.0 as Real; head_size],
    ));

    for h in 0..num_heads {
        let head_move = HeadMoveMeta {
            seq_len,
            full_dim: dim,
            head_dim,
            head_offset: h * head_dim,
        };

        let q_head = Arc::new(Tensor::init_from_cpu(
            ctx.clone(),
            &vec![0.0 as Real; head_size],
        ));
        let k_head = Arc::new(Tensor::init_from_cpu(
            ctx.clone(),
            &vec![0.0 as Real; head_size],
        ));
        let v_head = Arc::new(Tensor::init_from_cpu(
            ctx.clone(),
            &vec![0.0 as Real; head_size],
        ));
        let t_scores = Arc::new(Tensor::init_from_cpu(
            ctx.clone(),
            &vec![0.0 as Real; scores_size],
        ));

        head_gather(gb, q_buf, &q_head, head_move);
        head_gather(gb, k_buf, &k_head, head_move);
        head_gather(gb, v_buf, &v_head, head_move);

        matmul_trp(
            gb,
            &q_head,
            &k_head,
            &t_scores,
            MatMulMeta {
                m: seq_len,
                n: seq_len,
                k: head_dim,
            },
        );

        causal_softmax(gb, &t_scores, AttnScaleMeta { seq_len, scale });

        matmul(
            gb,
            &t_scores,
            &v_head,
            &out_head,
            MatMulMeta {
                m: seq_len,
                n: head_dim,
                k: seq_len,
            },
        );

        head_scatter(gb, &out_head, out_buffer, head_move);

        bufs.q_heads.push(q_head);
        bufs.k_heads.push(k_head);
        bufs.v_heads.push(v_head);
        bufs.scores_heads.push(t_scores);
    }

    bufs
}

// ---- flash attention  ----

fn grid_flash(shape: FlashAttnMeta) -> [u32; 3] {
    let num_heads = shape.dim / shape.head_dim;
    [(shape.seq_len + 63) / 64, num_heads, 1]
}

pub(crate) struct FlashAttnBuffers<B: Backend> {
    pub out: Arc<Tensor<B>>,
    pub l_cache: Arc<Tensor<B>>,
}

pub(crate) fn flash_attention<B: Backend, P: FullSeqPhase>(
    gb: &mut GraphBuilder<'_, B, P>,
    q_buf: &Arc<Tensor<B>>,
    k_buf: &Arc<Tensor<B>>,
    v_buf: &Arc<Tensor<B>>,
    out_buffer: &Arc<Tensor<B>>,
    shape: FlashAttnMeta,
) -> FlashAttnBuffers<B> {
    assert!(
        shape.head_dim <= 128,
        "flash_attention: head_dim must be <= 128 (fixed kernel accumulator size)"
    );
    assert_eq!(
        shape.dim % shape.head_dim,
        0,
        "flash_attention: dim must be divisible by head_dim"
    );

    let ctx = q_buf.ctx.clone();
    let num_heads = shape.dim / shape.head_dim;
    let l_size = (shape.seq_len * num_heads) as usize;
    let l_cache = Arc::new(Tensor::init_from_cpu(ctx, &vec![0.0 as Real; l_size]));
    let meta = shape.upload(&q_buf.ctx);

    gb.graph.add_node(
        &shaders::FLASH_ATTENTION,
        &[
            Binding::new(0, &q_buf.buffer, TensorMode::Input),
            Binding::new(1, &k_buf.buffer, TensorMode::Input),
            Binding::new(2, &v_buf.buffer, TensorMode::Input),
            Binding::new(3, &out_buffer.buffer, TensorMode::Output),
            Binding::new(4, &l_cache.buffer, TensorMode::Output),
            Binding::new(5, &meta.buffer, TensorMode::Meta),
        ],
        grid_flash(shape),
    );

    FlashAttnBuffers {
        out: out_buffer.clone(),
        l_cache,
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn flash_attention_bwd<B: Backend>(
    gb: &mut GraphBuilder<'_, B, Train>,
    q_buf: &Arc<Tensor<B>>,
    k_buf: &Arc<Tensor<B>>,
    v_buf: &Arc<Tensor<B>>,
    saved: &FlashAttnBuffers<B>,
    grad_output: &Arc<Tensor<B>>,
    grad_q: &Arc<Tensor<B>>,
    grad_k: &Arc<Tensor<B>>,
    grad_v: &Arc<Tensor<B>>,
    shape: FlashAttnMeta,
) {
    let meta = shape.upload(&q_buf.ctx);
    let grid = grid_flash(shape);

    gb.graph.add_node(
        &shaders::FLASH_ATTENTION_BWD_DQ,
        &[
            Binding::new(0, &q_buf.buffer, TensorMode::Input),
            Binding::new(1, &k_buf.buffer, TensorMode::Input),
            Binding::new(2, &v_buf.buffer, TensorMode::Input),
            Binding::new(3, &saved.out.buffer, TensorMode::Input),
            Binding::new(4, &grad_output.buffer, TensorMode::Input),
            Binding::new(5, &saved.l_cache.buffer, TensorMode::Input),
            Binding::new(6, &grad_q.buffer, TensorMode::Output),
            Binding::new(7, &meta.buffer, TensorMode::Meta),
        ],
        grid,
    );

    gb.graph.add_node(
        &shaders::FLASH_ATTENTION_BWD_DKDV,
        &[
            Binding::new(0, &q_buf.buffer, TensorMode::Input),
            Binding::new(1, &k_buf.buffer, TensorMode::Input),
            Binding::new(2, &v_buf.buffer, TensorMode::Input),
            Binding::new(3, &saved.out.buffer, TensorMode::Input),
            Binding::new(4, &grad_output.buffer, TensorMode::Input),
            Binding::new(5, &saved.l_cache.buffer, TensorMode::Input),
            Binding::new(6, &grad_k.buffer, TensorMode::Output),
            Binding::new(7, &grad_v.buffer, TensorMode::Output),
            Binding::new(8, &meta.buffer, TensorMode::Meta),
        ],
        grid,
    );
}

// ---- cache ----

pub(crate) fn cache_write_with<B: Backend, P: CachedPhase>(
    gb: &mut GraphBuilder<'_, B, P>,
    src: &Arc<Tensor<B>>,
    cache: &Arc<Tensor<B>>,
    shape: CacheWriteMeta,
    meta: &Arc<Tensor<B>>,
) {
    gb.graph.add_node(
        &shaders::CACHE_WRITE,
        &[
            Binding::new(0, &src.buffer, TensorMode::Input),
            Binding::new(1, &cache.buffer, TensorMode::InOut),
            Binding::new(2, &meta.buffer, TensorMode::Meta),
        ],
        [(shape.width + 15) / 16, (shape.row_count + 15) / 16, 1],
    );
}

pub(crate) fn cache_write<B: Backend, P: CachedPhase>(
    gb: &mut GraphBuilder<'_, B, P>,
    src: &Arc<Tensor<B>>,
    cache: &Arc<Tensor<B>>,
    shape: CacheWriteMeta,
) {
    let meta = shape.upload(&src.ctx);
    cache_write_with(gb, src, cache, shape, &meta);
}

// ---- elementwise ----

fn grid256(len: u32) -> [u32; 3] {
    [(len + 255) / 256, 1, 1]
}

// For tensors that can exceed 65535 workgroups in x (embedding-sized);
// the kernel must linearize (wg.y * num_wg.x + wg.x).
fn grid256_2d(len: u32) -> [u32; 3] {
    let total = (len + 255) / 256;
    let x = total.clamp(1, 8192);
    [x, (total + x - 1) / x, 1]
}

pub(crate) fn silu<B: Backend, P: FwdPhase>(
    gb: &mut GraphBuilder<'_, B, P>,
    buf: &Arc<Tensor<B>>,
    len: u32,
) {
    gb.graph.add_node(
        &shaders::SILU,
        &[Binding::new(0, &buf.buffer, TensorMode::InOut)],
        grid256(len),
    );
}

pub(crate) fn silu_out<B: Backend, P: FwdPhase>(
    gb: &mut GraphBuilder<'_, B, P>,
    input: &Arc<Tensor<B>>,
    out: &Arc<Tensor<B>>,
    len: u32,
) {
    gb.graph.add_node(
        &shaders::SILU_OUT,
        &[
            Binding::new(0, &input.buffer, TensorMode::Input),
            Binding::new(1, &out.buffer, TensorMode::Output),
        ],
        grid256(len),
    );
}

/// `input` is the pre-activation buffer saved by the forward pass.
pub(crate) fn silu_bwd<B: Backend>(
    gb: &mut GraphBuilder<'_, B, Train>,
    input: &Arc<Tensor<B>>,
    grad_output: &Arc<Tensor<B>>,
    grad_input: &Arc<Tensor<B>>,
    len: u32,
) {
    gb.graph.add_node(
        &shaders::SILU_BWD,
        &[
            Binding::new(0, &input.buffer, TensorMode::Input),
            Binding::new(1, &grad_output.buffer, TensorMode::Input),
            Binding::new(2, &grad_input.buffer, TensorMode::Output),
        ],
        grid256(len),
    );
}

/// `target += source`
pub(crate) fn residual_add<B: Backend, P: FwdPhase>(
    gb: &mut GraphBuilder<'_, B, P>,
    target: &Arc<Tensor<B>>,
    source: &Arc<Tensor<B>>,
    len: u32,
) {
    gb.graph.add_node(
        &builtin::RESIDUAL_ADD,
        &[
            Binding::new(0, &target.buffer, TensorMode::InOut),
            Binding::new(1, &source.buffer, TensorMode::Input),
        ],
        grid256(len),
    );
}

pub(crate) fn add_out<B: Backend, P: FwdPhase>(
    gb: &mut GraphBuilder<'_, B, P>,
    a: &Arc<Tensor<B>>,
    b: &Arc<Tensor<B>>,
    out: &Arc<Tensor<B>>,
    len: u32,
) {
    gb.graph.add_node(
        &shaders::ADD,
        &[
            Binding::new(0, &a.buffer, TensorMode::Input),
            Binding::new(1, &b.buffer, TensorMode::Input),
            Binding::new(2, &out.buffer, TensorMode::Output),
        ],
        grid256(len),
    );
}

/// `target += source`, backward-gb kernel (keeps a fusion barrier).
pub(crate) fn add_inplace_bwd<B: Backend>(
    gb: &mut GraphBuilder<'_, B, Train>,
    target: &Arc<Tensor<B>>,
    source: &Arc<Tensor<B>>,
    len: u32,
) {
    gb.graph.add_node(
        &builtin::BWD_ADD_INPLACE,
        &[
            Binding::new(0, &target.buffer, TensorMode::InOut),
            Binding::new(1, &source.buffer, TensorMode::Input),
        ],
        grid256(len),
    );
}

/// On-device zeroing, as a gb node.
pub(crate) fn zero<B: Backend, P: FwdPhase>(
    gb: &mut GraphBuilder<'_, B, P>,
    buf: &Arc<Tensor<B>>,
    len: u32,
) {
    let meta = ZeroMeta { len }.upload(&buf.ctx);
    gb.graph.add_node(
        &builtin::ZERO_TENSOR,
        &[
            Binding::new(0, &buf.buffer, TensorMode::Output),
            Binding::new(1, &meta.buffer, TensorMode::Meta),
        ],
        grid256_2d(len),
    );
}

// ---- grad clip ----

pub(crate) fn grad_sumsq_wgs(len: u32) -> u32 {
    ((len + 255) / 256).clamp(1, 256)
}

pub(crate) fn grad_sumsq<B: Backend>(
    gb: &mut GraphBuilder<'_, B, Train>,
    grad: &Arc<Tensor<B>>,
    partials: &Arc<Tensor<B>>,
    shape: GradSumSqMeta,
) {
    let meta = shape.upload(&grad.ctx);
    gb.graph.add_node(
        &shaders::GRAD_SUMSQ,
        &[
            Binding::new(0, &grad.buffer, TensorMode::Input),
            Binding::new(1, &partials.buffer, TensorMode::Output),
            Binding::new(2, &meta.buffer, TensorMode::Meta),
        ],
        [grad_sumsq_wgs(shape.len), 1, 1],
    );
}

pub(crate) fn grad_norm_scale<B: Backend>(
    gb: &mut GraphBuilder<'_, B, Train>,
    partials: &Arc<Tensor<B>>,
    scale: &Arc<Tensor<B>>,
    shape: GradNormMeta,
) {
    let meta = shape.upload(&partials.ctx);
    gb.graph.add_node(
        &shaders::GRAD_NORM_SCALE,
        &[
            Binding::new(0, &partials.buffer, TensorMode::Input),
            Binding::new(1, &scale.buffer, TensorMode::Output),
            Binding::new(2, &meta.buffer, TensorMode::Meta),
        ],
        [1, 1, 1],
    );
}

pub(crate) fn grad_scale<B: Backend>(
    gb: &mut GraphBuilder<'_, B, Train>,
    grad: &Arc<Tensor<B>>,
    scale: &Arc<Tensor<B>>,
    len: u32,
) {
    let meta = ZeroMeta { len }.upload(&grad.ctx);
    gb.graph.add_node(
        &shaders::GRAD_SCALE,
        &[
            Binding::new(0, &grad.buffer, TensorMode::InOut),
            Binding::new(1, &scale.buffer, TensorMode::Input),
            Binding::new(2, &meta.buffer, TensorMode::Meta),
        ],
        grid256_2d(len),
    );
}

// ---- loss ----

pub(crate) fn cross_entropy<B: Backend>(
    gb: &mut GraphBuilder<'_, B, Train>,
    logits: &Arc<Tensor<B>>,
    target_tokens: &Arc<Tensor<B>>,
    probs: &Arc<Tensor<B>>,
    losses: &Arc<Tensor<B>>,
    shape: CrossEntropyMeta,
) {
    let meta = shape.upload(&logits.ctx);
    gb.graph.add_node(
        &shaders::CROSS_ENTROPY,
        &[
            Binding::new(0, &logits.buffer, TensorMode::Input),
            Binding::new(1, &target_tokens.buffer, TensorMode::Input),
            Binding::new(2, &probs.buffer, TensorMode::Output),
            Binding::new(3, &losses.buffer, TensorMode::Output),
            Binding::new(4, &meta.buffer, TensorMode::Meta),
        ],
        [(shape.num_rows + 255) / 256, 1, 1],
    );
}

pub(crate) fn cross_entropy_bwd<B: Backend>(
    gb: &mut GraphBuilder<'_, B, Train>,
    probs: &Arc<Tensor<B>>,
    target_tokens: &Arc<Tensor<B>>,
    d_losses: &Arc<Tensor<B>>,
    grad_logits: &Arc<Tensor<B>>,
    shape: CrossEntropyMeta,
) {
    let meta = shape.upload(&probs.ctx);
    gb.graph.add_node(
        &shaders::CROSS_ENTROPY_BWD,
        &[
            Binding::new(0, &probs.buffer, TensorMode::Input),
            Binding::new(1, &target_tokens.buffer, TensorMode::Input),
            Binding::new(2, &d_losses.buffer, TensorMode::Input),
            Binding::new(3, &grad_logits.buffer, TensorMode::Output),
            Binding::new(4, &meta.buffer, TensorMode::Meta),
        ],
        [(shape.num_rows + 255) / 256, 1, 1],
    );
}

// Flash Attention (new) vs "causal_attention" + "SelfAttention"
#[cfg(test)]
mod flash_attention_validation {
    use super::*;
    use crate::nn::layers::{Layer, SelfAttention};
    use wilupgu::{ComputeGraph, WgpuBackend};

    fn rand_vec(n: usize, seed: u64) -> Vec<Real> {
        let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let bits = ((state >> 40) as u32) & 0x00FF_FFFF;
                (bits as f32 / 0x00FF_FFFF as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn max_abs_diff(a: &[Real], b: &[Real]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn flash_attention_matches_causal_attention() {
        check(8, 2, 4);
        check(37, 3, 16);
        check(65, 12, 64);
    }

    fn check(seq_len: u32, num_heads: u32, head_dim: u32) {
        let ctx = Arc::new(pollster::block_on(WgpuBackend::new()));

        let dim: u32 = num_heads * head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let n = (seq_len * dim) as usize;

        let q_buf = Arc::new(Tensor::init_from_cpu(ctx.clone(), &rand_vec(n, 1)));
        let k_buf = Arc::new(Tensor::init_from_cpu(ctx.clone(), &rand_vec(n, 2)));
        let v_buf = Arc::new(Tensor::init_from_cpu(ctx.clone(), &rand_vec(n, 3)));
        let grad_output = Arc::new(Tensor::init_from_cpu(ctx.clone(), &rand_vec(n, 4)));

        let zeros = || Arc::new(Tensor::init_from_cpu(ctx.clone(), &vec![0.0 as Real; n]));

        // ---- reference: existing causal_attention + SelfAttention's inline backward ----
        let (old_grad_q, old_grad_k, old_grad_v) = (zeros(), zeros(), zeros());
        let old_attn = SelfAttention::new(
            ctx.clone(),
            seq_len,
            dim,
            num_heads,
            1,
            &q_buf,
            &k_buf,
            &v_buf,
            &grad_output,
            &old_grad_q,
            &old_grad_k,
            &old_grad_v,
        );
        old_attn.forward();
        old_attn.backward();
        ctx.synchronize();

        // ---- new: flash_attention + flash_attention_bwd ----
        let new_out = zeros();
        let (new_grad_q, new_grad_k, new_grad_v) = (zeros(), zeros(), zeros());
        let shape = FlashAttnMeta {
            seq_len,
            dim,
            head_dim,
            scale,
            row_offset: 0,
        };

        let mut new_fwd = ComputeGraph::new(ctx.clone());
        let mut gb = GraphBuilder::train(&mut new_fwd);
        let saved = flash_attention(&mut gb, &q_buf, &k_buf, &v_buf, &new_out, shape);
        new_fwd.execute();
        ctx.synchronize();

        let mut new_bwd = ComputeGraph::new(ctx.clone());
        let mut gb = GraphBuilder::train(&mut new_bwd);
        flash_attention_bwd(
            &mut gb,
            &q_buf,
            &k_buf,
            &v_buf,
            &saved,
            &grad_output,
            &new_grad_q,
            &new_grad_k,
            &new_grad_v,
            shape,
        );
        new_bwd.execute();
        ctx.synchronize();

        let tol = 1e-3;

        let ctx_msg = format!("seq_len={seq_len} num_heads={num_heads} head_dim={head_dim}");

        let old_out_cpu = old_attn.out_buffer.to_cpu::<Real>();
        let new_out_cpu = new_out.to_cpu::<Real>();
        let out_diff = max_abs_diff(&old_out_cpu, &new_out_cpu);
        assert!(
            out_diff < tol,
            "forward output mismatch ({ctx_msg}): max_abs_diff={out_diff}"
        );

        let dq_diff = max_abs_diff(&old_grad_q.to_cpu::<Real>(), &new_grad_q.to_cpu::<Real>());
        assert!(
            dq_diff < tol,
            "dQ mismatch ({ctx_msg}): max_abs_diff={dq_diff}"
        );

        let dk_diff = max_abs_diff(&old_grad_k.to_cpu::<Real>(), &new_grad_k.to_cpu::<Real>());
        assert!(
            dk_diff < tol,
            "dK mismatch ({ctx_msg}): max_abs_diff={dk_diff}"
        );

        let dv_diff = max_abs_diff(&old_grad_v.to_cpu::<Real>(), &new_grad_v.to_cpu::<Real>());
        assert!(
            dv_diff < tol,
            "dV mismatch ({ctx_msg}): max_abs_diff={dv_diff}"
        );
    }
}

#[cfg(test)]
mod kernel_fusion_validation {
    use super::*;
    use wilupgu::{ComputeGraph, WgpuBackend};

    fn rand_vec(n: usize, seed: u64) -> Vec<Real> {
        let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let bits = ((state >> 40) as u32) & 0x00FF_FFFF;
                (bits as f32 / 0x00FF_FFFF as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn max_abs_diff(a: &[Real], b: &[Real]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn rope_qk_matches_two_rope_calls() {
        check_rope(8, 8, 4);
        check_rope(37, 12, 4);
        check_rope(65, 48, 16);
    }

    fn check_rope(seq_len: u32, dim: u32, head_dim: u32) {
        let ctx = Arc::new(pollster::block_on(WgpuBackend::new()));
        let n = (seq_len * dim) as usize;
        let shape = RopeMeta {
            seq_len,
            dim,
            head_dim,
            row_offset: 0,
        };

        let q_data = rand_vec(n, 10);
        let k_data = rand_vec(n, 20);
        let dq_data = rand_vec(n, 30);
        let dk_data = rand_vec(n, 40);

        // ---- forward ----
        let old_q = Arc::new(Tensor::init_from_cpu(ctx.clone(), &q_data));
        let old_k = Arc::new(Tensor::init_from_cpu(ctx.clone(), &k_data));
        let mut old_graph = ComputeGraph::new(ctx.clone());
        let mut gb = GraphBuilder::train(&mut old_graph);
        rope(&mut gb, &old_q, shape);
        rope(&mut gb, &old_k, shape);
        old_graph.execute();
        ctx.synchronize();

        let new_q = Arc::new(Tensor::init_from_cpu(ctx.clone(), &q_data));
        let new_k = Arc::new(Tensor::init_from_cpu(ctx.clone(), &k_data));
        let mut new_graph = ComputeGraph::new(ctx.clone());
        let mut gb = GraphBuilder::train(&mut new_graph);
        rope_qk(&mut gb, &new_q, &new_k, shape);
        new_graph.execute();
        ctx.synchronize();

        let ctx_msg = format!("seq_len={seq_len} dim={dim} head_dim={head_dim}");
        let q_diff = max_abs_diff(&old_q.to_cpu::<Real>(), &new_q.to_cpu::<Real>());
        assert!(q_diff < 1e-4, "rope_qk Q mismatch ({ctx_msg}): {q_diff}");
        let k_diff = max_abs_diff(&old_k.to_cpu::<Real>(), &new_k.to_cpu::<Real>());
        assert!(k_diff < 1e-4, "rope_qk K mismatch ({ctx_msg}): {k_diff}");

        // ---- backward ----
        let old_dq = Arc::new(Tensor::init_from_cpu(ctx.clone(), &dq_data));
        let old_dk = Arc::new(Tensor::init_from_cpu(ctx.clone(), &dk_data));
        let mut old_bwd = ComputeGraph::new(ctx.clone());
        let mut gb = GraphBuilder::train(&mut old_bwd);
        rope_bwd(&mut gb, &old_dq, shape);
        rope_bwd(&mut gb, &old_dk, shape);
        old_bwd.execute();
        ctx.synchronize();

        let new_dq = Arc::new(Tensor::init_from_cpu(ctx.clone(), &dq_data));
        let new_dk = Arc::new(Tensor::init_from_cpu(ctx.clone(), &dk_data));
        let mut new_bwd = ComputeGraph::new(ctx.clone());
        let mut gb = GraphBuilder::train(&mut new_bwd);
        rope_bwd_qk(&mut gb, &new_dq, &new_dk, shape);
        new_bwd.execute();
        ctx.synchronize();

        let dq_diff = max_abs_diff(&old_dq.to_cpu::<Real>(), &new_dq.to_cpu::<Real>());
        assert!(
            dq_diff < 1e-4,
            "rope_bwd_qk dQ mismatch ({ctx_msg}): {dq_diff}"
        );
        let dk_diff = max_abs_diff(&old_dk.to_cpu::<Real>(), &new_dk.to_cpu::<Real>());
        assert!(
            dk_diff < 1e-4,
            "rope_bwd_qk dK mismatch ({ctx_msg}): {dk_diff}"
        );
    }

    #[test]
    fn qkv_split_matches_three_head_gathers() {
        check_qkv(8, 4);
        check_qkv(37, 12);
        check_qkv(65, 48);
    }

    fn check_qkv(seq_len: u32, dim: u32) {
        let ctx = Arc::new(pollster::block_on(WgpuBackend::new()));
        let n = (seq_len * dim) as usize;
        let src_data = rand_vec(n * 3, 50);
        let src = Arc::new(Tensor::init_from_cpu(ctx.clone(), &src_data));
        let zeros = || Arc::new(Tensor::init_from_cpu(ctx.clone(), &vec![0.0 as Real; n]));

        // ---- forward ----
        let (old_q, old_k, old_v) = (zeros(), zeros(), zeros());
        let mut old_graph = ComputeGraph::new(ctx.clone());
        let mut gb = GraphBuilder::train(&mut old_graph);
        for (buf, off) in [(&old_q, 0), (&old_k, dim), (&old_v, 2 * dim)] {
            head_gather(
                &mut gb,
                &src,
                buf,
                HeadMoveMeta::qkv_slice(seq_len, dim, off),
            );
        }
        old_graph.execute();
        ctx.synchronize();

        let (new_q, new_k, new_v) = (zeros(), zeros(), zeros());
        let mut new_graph = ComputeGraph::new(ctx.clone());
        let mut gb = GraphBuilder::train(&mut new_graph);
        qkv_split(
            &mut gb,
            &src,
            &new_q,
            &new_k,
            &new_v,
            HeadMoveMeta::qkv_slice(seq_len, dim, 0),
        );
        new_graph.execute();
        ctx.synchronize();

        let ctx_msg = format!("seq_len={seq_len} dim={dim}");
        assert!(
            max_abs_diff(&old_q.to_cpu::<Real>(), &new_q.to_cpu::<Real>()) < 1e-6,
            "qkv_split Q mismatch ({ctx_msg})"
        );
        assert!(
            max_abs_diff(&old_k.to_cpu::<Real>(), &new_k.to_cpu::<Real>()) < 1e-6,
            "qkv_split K mismatch ({ctx_msg})"
        );
        assert!(
            max_abs_diff(&old_v.to_cpu::<Real>(), &new_v.to_cpu::<Real>()) < 1e-6,
            "qkv_split V mismatch ({ctx_msg})"
        );

        // ---- backward ----
        let grad_q = Arc::new(Tensor::init_from_cpu(ctx.clone(), &rand_vec(n, 60)));
        let grad_k = Arc::new(Tensor::init_from_cpu(ctx.clone(), &rand_vec(n, 70)));
        let grad_v = Arc::new(Tensor::init_from_cpu(ctx.clone(), &rand_vec(n, 80)));

        let old_dst = Arc::new(Tensor::init_from_cpu(
            ctx.clone(),
            &vec![0.0 as Real; n * 3],
        ));
        let mut old_bwd = ComputeGraph::new(ctx.clone());
        let mut gb = GraphBuilder::train(&mut old_bwd);
        for (buf, off) in [(&grad_q, 0), (&grad_k, dim), (&grad_v, 2 * dim)] {
            head_scatter(
                &mut gb,
                buf,
                &old_dst,
                HeadMoveMeta::qkv_slice(seq_len, dim, off),
            );
        }
        old_bwd.execute();
        ctx.synchronize();

        let new_dst = Arc::new(Tensor::init_from_cpu(
            ctx.clone(),
            &vec![0.0 as Real; n * 3],
        ));
        let mut new_bwd = ComputeGraph::new(ctx.clone());
        let mut gb = GraphBuilder::train(&mut new_bwd);
        qkv_scatter(
            &mut gb,
            &grad_q,
            &grad_k,
            &grad_v,
            &new_dst,
            HeadMoveMeta::qkv_slice(seq_len, dim, 0),
        );
        new_bwd.execute();
        ctx.synchronize();

        assert!(
            max_abs_diff(&old_dst.to_cpu::<Real>(), &new_dst.to_cpu::<Real>()) < 1e-6,
            "qkv_scatter mismatch ({ctx_msg})"
        );
    }
}

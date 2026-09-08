use super::meta::{
    CacheWriteMeta, CrossEntropyMeta, EmbeddingMeta, FlashAttnMeta, GradNormMeta, GradSumSqMeta,
    HeadMoveMeta, KernelMeta, MatMulMeta, NormMeta, RopeMeta, RopeOffsetMeta, SoftmaxRectMeta,
    ZeroMeta,
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

/// `C[m,n] = A[m,k] @ B[k,n]`. m=1 (decode) routes to the flat GEMV kernel;
/// the tiled matmul would idle 15/16 of every workgroup on a single row.
/// The build-time `shape.m` decides — fine, because dynamic metas only ever
/// change n/k (decode is m=1 throughout).
pub(crate) fn matmul_with<B: Backend, P: FwdPhase>(
    gb: &mut GraphBuilder<'_, B, P>,
    a: &Arc<Tensor<B>>,
    b: &Arc<Tensor<B>>,
    c: &Arc<Tensor<B>>,
    shape: MatMulMeta,
    meta: &Arc<Tensor<B>>,
) {
    let (shader, grid) = if shape.m == 1 {
        (&builtin::GEMV, [(shape.n + 255) / 256, 1, 1])
    } else {
        (&builtin::MATMUL, grid_nm(shape))
    };
    gb.graph.add_node(
        shader,
        &[
            Binding::new(0, &a.buffer, TensorMode::Input),
            Binding::new(1, &b.buffer, TensorMode::Input),
            Binding::new(2, &c.buffer, TensorMode::Output),
            Binding::new(3, &meta.buffer, TensorMode::Meta),
        ],
        grid,
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

/// `C[m,n] += A[m,k] @ B[k,n]` (fused residual, `c` accumulates).
pub(crate) fn matmul_add_with<B: Backend, P: FwdPhase>(
    gb: &mut GraphBuilder<'_, B, P>,
    a: &Arc<Tensor<B>>,
    b: &Arc<Tensor<B>>,
    c: &Arc<Tensor<B>>,
    shape: MatMulMeta,
    meta: &Arc<Tensor<B>>,
) {
    let (shader, grid) = if shape.m == 1 {
        (&builtin::GEMV_ADD, [(shape.n + 255) / 256, 1, 1])
    } else {
        (&builtin::MATMUL_ADD, grid_nm(shape))
    };
    gb.graph.add_node(
        shader,
        &[
            Binding::new(0, &a.buffer, TensorMode::Input),
            Binding::new(1, &b.buffer, TensorMode::Input),
            Binding::new(2, &c.buffer, TensorMode::Accumulate),
            Binding::new(3, &meta.buffer, TensorMode::Meta),
        ],
        grid,
    );
}

#[cfg(test)] // reference impl: only gemv_routing_matches_cpu_reference calls it, real callers went to matmul_add_with (block_pre_attn/block_post_attn own their meta)
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
            Binding::new(2, &grad_weight.buffer, TensorMode::Accumulate),
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
            Binding::new(3, &grad_weight.buffer, TensorMode::Accumulate),
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
            Binding::new(2, &grad_table.buffer, TensorMode::Accumulate),
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

#[cfg(test)] // reference impl: only the rope_qk fusion test compares against it
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
#[cfg(test)] // reference impl: only the qkv_scatter fusion test compares against it
pub(crate) fn head_scatter<B: Backend, P: FwdPhase>(
    gb: &mut GraphBuilder<'_, B, P>,
    src: &Arc<Tensor<B>>,
    dst: &Arc<Tensor<B>>,
    shape: HeadMoveMeta,
) {
    let meta = shape.upload(&src.ctx);
    move_node(gb, &shaders::HEAD_SCATTER, src, dst, shape, &meta);
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

/// Decode QK^T for all heads in one dispatch, K read strided from the cache
pub(crate) fn attn_qk_cached_with<B: Backend>(
    gb: &mut GraphBuilder<'_, B, Decode>,
    q: &Arc<Tensor<B>>,
    k_cache: &Arc<Tensor<B>>,
    scores: &Arc<Tensor<B>>,
    num_heads: u32,
    max_attn_len: u32,
    meta: &Arc<Tensor<B>>,
) {
    gb.graph.add_node(
        &shaders::ATTN_QK_CACHED,
        &[
            Binding::new(0, &q.buffer, TensorMode::Input),
            Binding::new(1, &k_cache.buffer, TensorMode::Input),
            Binding::new(2, &scores.buffer, TensorMode::Output),
            Binding::new(3, &meta.buffer, TensorMode::Meta),
        ],
        [(max_attn_len + 255) / 256, num_heads, 1],
    );
}

/// Decode P@V for all heads in one dispatch, V read strided from the cache;
pub(crate) fn attn_av_cached_with<B: Backend>(
    gb: &mut GraphBuilder<'_, B, Decode>,
    scores: &Arc<Tensor<B>>,
    v_cache: &Arc<Tensor<B>>,
    out: &Arc<Tensor<B>>,
    dim: u32,
    meta: &Arc<Tensor<B>>,
) {
    gb.graph.add_node(
        &shaders::ATTN_AV_CACHED,
        &[
            Binding::new(0, &scores.buffer, TensorMode::Input),
            Binding::new(1, &v_cache.buffer, TensorMode::Input),
            Binding::new(2, &out.buffer, TensorMode::Output),
            Binding::new(3, &meta.buffer, TensorMode::Meta),
        ],
        [(dim + 255) / 256, 1, 1],
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

// ---- flash attention  ----

fn grid_flash(shape: FlashAttnMeta) -> [u32; 3] {
    let num_heads = shape.dim / shape.head_dim;
    [(shape.seq_len + 63) / 64, num_heads, 1]
}

/// wgsl kernel is hardcoded to head_dim=64 (register-spill fix); other
/// values silently corrupt instead of erroring, hence the assert.
fn assert_flash_head_dim(shape: FlashAttnMeta) {
    assert_eq!(
        shape.head_dim, 64,
        "flash_attention: head_dim must be 64 (wgsl kernel is hardcoded to it)"
    );
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
    assert_flash_head_dim(shape);
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
    assert_flash_head_dim(shape);
    let meta = shape.upload(&q_buf.ctx);
    let grid = grid_flash(shape);

    let ctx = q_buf.ctx.clone();
    let num_heads = shape.dim / shape.head_dim;
    let d_size = (shape.seq_len * num_heads) as usize;
    let d_sum = Arc::new(Tensor::init_from_cpu(ctx, &vec![0.0 as Real; d_size]));

    // D[i] = sum_d dO_i . O_i, precomputed once per row instead of being
    // recomputed by every (row, col) pair that visits row i below (B11b).
    gb.graph.add_node(
        &shaders::FLASH_ATTENTION_BWD_D,
        &[
            Binding::new(0, &grad_output.buffer, TensorMode::Input),
            Binding::new(1, &saved.out.buffer, TensorMode::Input),
            Binding::new(2, &d_sum.buffer, TensorMode::Output),
            Binding::new(3, &meta.buffer, TensorMode::Meta),
        ],
        grid,
    );

    gb.graph.add_node(
        &shaders::FLASH_ATTENTION_BWD_DQ,
        &[
            Binding::new(0, &q_buf.buffer, TensorMode::Input),
            Binding::new(1, &k_buf.buffer, TensorMode::Input),
            Binding::new(2, &v_buf.buffer, TensorMode::Input),
            Binding::new(3, &d_sum.buffer, TensorMode::Input),
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
            Binding::new(3, &d_sum.buffer, TensorMode::Input),
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

// ---- elementwise ----

// (wg.y * num_wg.x + wg.x).
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
        grid256_2d(len),
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
        grid256_2d(len),
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
        grid256_2d(len),
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
            Binding::new(0, &target.buffer, TensorMode::Accumulate),
            Binding::new(1, &source.buffer, TensorMode::Input),
        ],
        grid256_2d(len),
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
        grid256_2d(len),
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
            Binding::new(0, &target.buffer, TensorMode::Accumulate),
            Binding::new(1, &source.buffer, TensorMode::Input),
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
    losses: &Arc<Tensor<B>>,
    shape: CrossEntropyMeta,
) {
    let meta = shape.upload(&logits.ctx);
    gb.graph.add_node(
        &shaders::CROSS_ENTROPY,
        &[
            Binding::new(0, &logits.buffer, TensorMode::InOut),
            Binding::new(1, &target_tokens.buffer, TensorMode::Input),
            Binding::new(2, &losses.buffer, TensorMode::Output),
            Binding::new(3, &meta.buffer, TensorMode::Meta),
        ],
        [shape.num_rows, 1, 1],
    );
}

pub(crate) fn cross_entropy_bwd<B: Backend>(
    gb: &mut GraphBuilder<'_, B, Train>,
    probs: &Arc<Tensor<B>>,
    target_tokens: &Arc<Tensor<B>>,
    d_losses: &Arc<Tensor<B>>,
    shape: CrossEntropyMeta,
) {
    let meta = shape.upload(&probs.ctx);
    gb.graph.add_node(
        &shaders::CROSS_ENTROPY_BWD,
        &[
            Binding::new(0, &probs.buffer, TensorMode::InOut),
            Binding::new(1, &target_tokens.buffer, TensorMode::Input),
            Binding::new(2, &d_losses.buffer, TensorMode::Input),
            Binding::new(3, &meta.buffer, TensorMode::Meta),
        ],
        [(shape.vocab_size + 255) / 256, shape.num_rows, 1],
    );
}

// Flash Attention (GPU kernels) vs plain-Rust CPU reference

#[cfg(test)]
#[path = "../../tests/kernels_emit_tests.rs"]
mod tests;

use super::meta::{
    AttnScaleMeta, CacheWriteMeta, CrossEntropyMeta, EmbeddingMeta, HeadMoveMeta, KernelMeta,
    MatMulMeta, NormMeta, RopeMeta, RopeOffsetMeta, SoftmaxRectMeta, ZeroMeta,
};
use crate::Real;
use std::sync::Arc;
use wilupgu::{Backend, Binding, ComputeGraph, Tensor, TensorMode};

// ---- matmul ----

fn grid_nm(shape: MatMulMeta) -> [u32; 3] {
    [(shape.n + 15) / 16, (shape.m + 15) / 16, 1]
}

/// `C[m,n] = A[m,k] @ B[k,n]`
pub(crate) fn matmul_with<B: Backend>(
    graph: &mut ComputeGraph<B>,
    a: &Arc<Tensor<B>>,
    b: &Arc<Tensor<B>>,
    c: &Arc<Tensor<B>>,
    shape: MatMulMeta,
    meta: &Arc<Tensor<B>>,
) {
    graph.add_node(
        "MatMul",
        &[
            Binding::new(0, &a.buffer, TensorMode::Input),
            Binding::new(1, &b.buffer, TensorMode::Input),
            Binding::new(2, &c.buffer, TensorMode::Output),
            Binding::new(3, &meta.buffer, TensorMode::Meta),
        ],
        grid_nm(shape),
    );
}

pub(crate) fn matmul<B: Backend>(
    graph: &mut ComputeGraph<B>,
    a: &Arc<Tensor<B>>,
    b: &Arc<Tensor<B>>,
    c: &Arc<Tensor<B>>,
    shape: MatMulMeta,
) {
    let meta = shape.upload(&a.ctx);
    matmul_with(graph, a, b, c, shape, &meta);
}

/// `C[m,n] = A[m,k] @ B[n,k]^T`
pub(crate) fn matmul_trp_with<B: Backend>(
    graph: &mut ComputeGraph<B>,
    a: &Arc<Tensor<B>>,
    b: &Arc<Tensor<B>>,
    c: &Arc<Tensor<B>>,
    shape: MatMulMeta,
    meta: &Arc<Tensor<B>>,
) {
    graph.add_node(
        "MatMulTrp",
        &[
            Binding::new(0, &a.buffer, TensorMode::Input),
            Binding::new(1, &b.buffer, TensorMode::Input),
            Binding::new(2, &c.buffer, TensorMode::Output),
            Binding::new(3, &meta.buffer, TensorMode::Meta),
        ],
        grid_nm(shape),
    );
}

pub(crate) fn matmul_trp<B: Backend>(
    graph: &mut ComputeGraph<B>,
    a: &Arc<Tensor<B>>,
    b: &Arc<Tensor<B>>,
    c: &Arc<Tensor<B>>,
    shape: MatMulMeta,
) {
    let meta = shape.upload(&a.ctx);
    matmul_trp_with(graph, a, b, c, shape, &meta);
}

/// `C[m,n] += A[m,k] @ B[k,n]` (fused residual, `c` is InOut)
pub(crate) fn matmul_add_with<B: Backend>(
    graph: &mut ComputeGraph<B>,
    a: &Arc<Tensor<B>>,
    b: &Arc<Tensor<B>>,
    c: &Arc<Tensor<B>>,
    shape: MatMulMeta,
    meta: &Arc<Tensor<B>>,
) {
    graph.add_node(
        "MatMulAdd",
        &[
            Binding::new(0, &a.buffer, TensorMode::Input),
            Binding::new(1, &b.buffer, TensorMode::Input),
            Binding::new(2, &c.buffer, TensorMode::InOut),
            Binding::new(3, &meta.buffer, TensorMode::Meta),
        ],
        grid_nm(shape),
    );
}

pub(crate) fn matmul_add<B: Backend>(
    graph: &mut ComputeGraph<B>,
    a: &Arc<Tensor<B>>,
    b: &Arc<Tensor<B>>,
    c: &Arc<Tensor<B>>,
    shape: MatMulMeta,
) {
    let meta = shape.upload(&a.ctx);
    matmul_add_with(graph, a, b, c, shape, &meta);
}

/// `dW[k,n] += A[m,k]^T @ dY[m,n]` -- accumulates, zero `grad_weight` first.
pub(crate) fn matmul_weight_bwd<B: Backend>(
    graph: &mut ComputeGraph<B>,
    input: &Arc<Tensor<B>>,
    grad_output: &Arc<Tensor<B>>,
    grad_weight: &Arc<Tensor<B>>,
    shape: MatMulMeta,
) {
    let meta = shape.upload(&input.ctx);
    graph.add_node(
        "MatMulWeightBwd",
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

pub(crate) fn rmsnorm_with<B: Backend>(
    graph: &mut ComputeGraph<B>,
    input: &Arc<Tensor<B>>,
    weight: &Arc<Tensor<B>>,
    output: &Arc<Tensor<B>>,
    shape: NormMeta,
    meta: &Arc<Tensor<B>>,
) {
    graph.add_node(
        "RMSNorm",
        &[
            Binding::new(0, &input.buffer, TensorMode::Input),
            Binding::new(1, &weight.buffer, TensorMode::Input),
            Binding::new(2, &output.buffer, TensorMode::Output),
            Binding::new(3, &meta.buffer, TensorMode::Meta),
        ],
        [shape.seq_len, 1, 1],
    );
}

pub(crate) fn rmsnorm<B: Backend>(
    graph: &mut ComputeGraph<B>,
    input: &Arc<Tensor<B>>,
    weight: &Arc<Tensor<B>>,
    output: &Arc<Tensor<B>>,
    shape: NormMeta,
) {
    let meta = shape.upload(&input.ctx);
    rmsnorm_with(graph, input, weight, output, shape, &meta);
}

/// Both backward nodes (input grad + weight grad, linked by `rsqrt_cache`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn rmsnorm_bwd<B: Backend>(
    graph: &mut ComputeGraph<B>,
    grad_output: &Arc<Tensor<B>>,
    input: &Arc<Tensor<B>>,
    weight: &Arc<Tensor<B>>,
    grad_input: &Arc<Tensor<B>>,
    rsqrt_cache: &Arc<Tensor<B>>,
    grad_weight: &Arc<Tensor<B>>,
    shape: NormMeta,
) {
    let meta = shape.upload(&input.ctx);

    graph.add_node(
        "RMSNormBwd",
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

    graph.add_node(
        "RMSNormWeightBwd",
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

pub(crate) fn embedding_with<B: Backend>(
    graph: &mut ComputeGraph<B>,
    tokens: &Arc<Tensor<B>>,
    table: &Arc<Tensor<B>>,
    output: &Arc<Tensor<B>>,
    shape: EmbeddingMeta,
    meta: &Arc<Tensor<B>>,
) {
    graph.add_node(
        "Embedding",
        &[
            Binding::new(0, &tokens.buffer, TensorMode::Input),
            Binding::new(1, &table.buffer, TensorMode::Input),
            Binding::new(2, &output.buffer, TensorMode::Output),
            Binding::new(3, &meta.buffer, TensorMode::Meta),
        ],
        grid_embedding(shape),
    );
}

pub(crate) fn embedding<B: Backend>(
    graph: &mut ComputeGraph<B>,
    tokens: &Arc<Tensor<B>>,
    table: &Arc<Tensor<B>>,
    output: &Arc<Tensor<B>>,
    shape: EmbeddingMeta,
) {
    let meta = shape.upload(&tokens.ctx);
    embedding_with(graph, tokens, table, output, shape, &meta);
}

pub(crate) fn embedding_bwd<B: Backend>(
    graph: &mut ComputeGraph<B>,
    tokens: &Arc<Tensor<B>>,
    grad_output: &Arc<Tensor<B>>,
    grad_table: &Arc<Tensor<B>>,
    shape: EmbeddingMeta,
) {
    let meta = shape.upload(&tokens.ctx);
    graph.add_node(
        "EmbeddingBwd",
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

fn inout_meta_node<B: Backend>(
    graph: &mut ComputeGraph<B>,
    kernel: &str,
    buf: &Arc<Tensor<B>>,
    meta: &Arc<Tensor<B>>,
    grid: [u32; 3],
) {
    graph.add_node(
        kernel,
        &[
            Binding::new(0, &buf.buffer, TensorMode::InOut),
            Binding::new(1, &meta.buffer, TensorMode::Meta),
        ],
        grid,
    );
}

fn grid_full(shape: RopeMeta) -> [u32; 3] {
    [(shape.dim / 2 + 15) / 16, (shape.seq_len + 15) / 16, 1]
}

pub(crate) fn rope<B: Backend>(graph: &mut ComputeGraph<B>, buf: &Arc<Tensor<B>>, shape: RopeMeta) {
    let meta = shape.upload(&buf.ctx);
    inout_meta_node(graph, "RoPE", buf, &meta, grid_full(shape));
}

pub(crate) fn rope_bwd<B: Backend>(
    graph: &mut ComputeGraph<B>,
    grad: &Arc<Tensor<B>>,
    shape: RopeMeta,
) {
    let meta = shape.upload(&grad.ctx);
    inout_meta_node(graph, "RoPEBwd", grad, &meta, grid_full(shape));
}

pub(crate) fn rope_offset_with<B: Backend>(
    graph: &mut ComputeGraph<B>,
    buf: &Arc<Tensor<B>>,
    shape: RopeOffsetMeta,
    meta: &Arc<Tensor<B>>,
) {
    inout_meta_node(
        graph,
        "RoPEOffset",
        buf,
        meta,
        [(shape.head_dim / 2 + 15) / 16, 1, 1],
    );
}

// ---- head_move ----

fn grid_head(shape: HeadMoveMeta) -> [u32; 3] {
    [(shape.head_dim + 15) / 16, (shape.seq_len + 15) / 16, 1]
}

fn move_node<B: Backend>(
    graph: &mut ComputeGraph<B>,
    kernel: &str,
    src: &Arc<Tensor<B>>,
    dst: &Arc<Tensor<B>>,
    shape: HeadMoveMeta,
    meta: &Arc<Tensor<B>>,
) {
    graph.add_node(
        kernel,
        &[
            Binding::new(0, &src.buffer, TensorMode::Input),
            Binding::new(1, &dst.buffer, TensorMode::Output),
            Binding::new(2, &meta.buffer, TensorMode::Meta),
        ],
        grid_head(shape),
    );
}

/// wide `src` -> compact `dst`
pub(crate) fn head_gather_with<B: Backend>(
    graph: &mut ComputeGraph<B>,
    src: &Arc<Tensor<B>>,
    dst: &Arc<Tensor<B>>,
    shape: HeadMoveMeta,
    meta: &Arc<Tensor<B>>,
) {
    move_node(graph, "HeadGather", src, dst, shape, meta);
}

pub(crate) fn head_gather<B: Backend>(
    graph: &mut ComputeGraph<B>,
    src: &Arc<Tensor<B>>,
    dst: &Arc<Tensor<B>>,
    shape: HeadMoveMeta,
) {
    let meta = shape.upload(&src.ctx);
    head_gather_with(graph, src, dst, shape, &meta);
}

/// compact `src` -> wide `dst`
pub(crate) fn head_scatter_with<B: Backend>(
    graph: &mut ComputeGraph<B>,
    src: &Arc<Tensor<B>>,
    dst: &Arc<Tensor<B>>,
    shape: HeadMoveMeta,
    meta: &Arc<Tensor<B>>,
) {
    move_node(graph, "HeadScatter", src, dst, shape, meta);
}

pub(crate) fn head_scatter<B: Backend>(
    graph: &mut ComputeGraph<B>,
    src: &Arc<Tensor<B>>,
    dst: &Arc<Tensor<B>>,
    shape: HeadMoveMeta,
) {
    let meta = shape.upload(&src.ctx);
    head_scatter_with(graph, src, dst, shape, &meta);
}

// ---- attention ----

/// Fused causal-mask + scale + softmax, in place.
pub(crate) fn causal_softmax<B: Backend>(
    graph: &mut ComputeGraph<B>,
    scores: &Arc<Tensor<B>>,
    shape: AttnScaleMeta,
) {
    let meta = shape.upload(&scores.ctx);
    graph.add_node(
        "CausalSoftmax",
        &[
            Binding::new(0, &scores.buffer, TensorMode::InOut),
            Binding::new(1, &meta.buffer, TensorMode::Meta),
        ],
        [(shape.seq_len + 255) / 256, 1, 1],
    );
}

/// Scaled softmax, in place; no mask (decode cache only contains past).
pub(crate) fn softmax_rect_with<B: Backend>(
    graph: &mut ComputeGraph<B>,
    scores: &Arc<Tensor<B>>,
    shape: SoftmaxRectMeta,
    meta: &Arc<Tensor<B>>,
) {
    graph.add_node(
        "SoftmaxRect",
        &[
            Binding::new(0, &scores.buffer, TensorMode::InOut),
            Binding::new(1, &meta.buffer, TensorMode::Meta),
        ],
        [(shape.num_rows + 255) / 256, 1, 1],
    );
}

pub(crate) fn softmax_bwd<B: Backend>(
    graph: &mut ComputeGraph<B>,
    y: &Arc<Tensor<B>>,
    grad_y: &Arc<Tensor<B>>,
    grad_raw: &Arc<Tensor<B>>,
    shape: AttnScaleMeta,
) {
    let meta = shape.upload(&y.ctx);
    graph.add_node(
        "SoftmaxBwd",
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
pub(crate) fn causal_attention<B: Backend>(
    graph: &mut ComputeGraph<B>,
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

        head_gather(graph, q_buf, &q_head, head_move);
        head_gather(graph, k_buf, &k_head, head_move);
        head_gather(graph, v_buf, &v_head, head_move);

        matmul_trp(
            graph,
            &q_head,
            &k_head,
            &t_scores,
            MatMulMeta {
                m: seq_len,
                n: seq_len,
                k: head_dim,
            },
        );

        causal_softmax(graph, &t_scores, AttnScaleMeta { seq_len, scale });

        matmul(
            graph,
            &t_scores,
            &v_head,
            &out_head,
            MatMulMeta {
                m: seq_len,
                n: head_dim,
                k: seq_len,
            },
        );

        head_scatter(graph, &out_head, out_buffer, head_move);

        bufs.q_heads.push(q_head);
        bufs.k_heads.push(k_head);
        bufs.v_heads.push(v_head);
        bufs.scores_heads.push(t_scores);
    }

    bufs
}

// ---- cache ----

pub(crate) fn cache_write_with<B: Backend>(
    graph: &mut ComputeGraph<B>,
    src: &Arc<Tensor<B>>,
    cache: &Arc<Tensor<B>>,
    shape: CacheWriteMeta,
    meta: &Arc<Tensor<B>>,
) {
    graph.add_node(
        "CacheWrite",
        &[
            Binding::new(0, &src.buffer, TensorMode::Input),
            Binding::new(1, &cache.buffer, TensorMode::InOut),
            Binding::new(2, &meta.buffer, TensorMode::Meta),
        ],
        [(shape.width + 15) / 16, (shape.row_count + 15) / 16, 1],
    );
}

pub(crate) fn cache_write<B: Backend>(
    graph: &mut ComputeGraph<B>,
    src: &Arc<Tensor<B>>,
    cache: &Arc<Tensor<B>>,
    shape: CacheWriteMeta,
) {
    let meta = shape.upload(&src.ctx);
    cache_write_with(graph, src, cache, shape, &meta);
}

// ---- elementwise ----

fn grid256(len: u32) -> [u32; 3] {
    [(len + 255) / 256, 1, 1]
}

pub(crate) fn silu<B: Backend>(graph: &mut ComputeGraph<B>, buf: &Arc<Tensor<B>>, len: u32) {
    graph.add_node(
        "SiLU",
        &[Binding::new(0, &buf.buffer, TensorMode::InOut)],
        grid256(len),
    );
}

/// `input` is the pre-activation buffer saved by the forward pass.
pub(crate) fn silu_bwd<B: Backend>(
    graph: &mut ComputeGraph<B>,
    input: &Arc<Tensor<B>>,
    grad_output: &Arc<Tensor<B>>,
    grad_input: &Arc<Tensor<B>>,
    len: u32,
) {
    graph.add_node(
        "SiLUBwd",
        &[
            Binding::new(0, &input.buffer, TensorMode::Input),
            Binding::new(1, &grad_output.buffer, TensorMode::Input),
            Binding::new(2, &grad_input.buffer, TensorMode::Output),
        ],
        grid256(len),
    );
}

/// `target += source`
pub(crate) fn residual_add<B: Backend>(
    graph: &mut ComputeGraph<B>,
    target: &Arc<Tensor<B>>,
    source: &Arc<Tensor<B>>,
    len: u32,
) {
    graph.add_node(
        "ResidualAdd",
        &[
            Binding::new(0, &target.buffer, TensorMode::InOut),
            Binding::new(1, &source.buffer, TensorMode::Input),
        ],
        grid256(len),
    );
}

/// `target += source`, backward-graph kernel (keeps a fusion barrier).
pub(crate) fn add_inplace_bwd<B: Backend>(
    graph: &mut ComputeGraph<B>,
    target: &Arc<Tensor<B>>,
    source: &Arc<Tensor<B>>,
    len: u32,
) {
    graph.add_node(
        "BwdAddInplace",
        &[
            Binding::new(0, &target.buffer, TensorMode::InOut),
            Binding::new(1, &source.buffer, TensorMode::Input),
        ],
        grid256(len),
    );
}

/// On-device zeroing, as a graph node.
pub(crate) fn zero<B: Backend>(graph: &mut ComputeGraph<B>, buf: &Arc<Tensor<B>>, len: u32) {
    let meta = ZeroMeta { len }.upload(&buf.ctx);
    graph.add_node(
        "ZeroTensor",
        &[
            Binding::new(0, &buf.buffer, TensorMode::Output),
            Binding::new(1, &meta.buffer, TensorMode::Meta),
        ],
        grid256(len),
    );
}

// ---- loss ----

pub(crate) fn cross_entropy<B: Backend>(
    graph: &mut ComputeGraph<B>,
    logits: &Arc<Tensor<B>>,
    target_tokens: &Arc<Tensor<B>>,
    probs: &Arc<Tensor<B>>,
    losses: &Arc<Tensor<B>>,
    shape: CrossEntropyMeta,
) {
    let meta = shape.upload(&logits.ctx);
    graph.add_node(
        "CrossEntropy",
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
    graph: &mut ComputeGraph<B>,
    probs: &Arc<Tensor<B>>,
    target_tokens: &Arc<Tensor<B>>,
    d_losses: &Arc<Tensor<B>>,
    grad_logits: &Arc<Tensor<B>>,
    shape: CrossEntropyMeta,
) {
    let meta = shape.upload(&probs.ctx);
    graph.add_node(
        "CrossEntropyBwd",
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

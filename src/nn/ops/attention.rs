use super::head_move::{head_gather, head_scatter};
use super::matmul::{matmul, matmul_trp};
use super::meta::{AttnScaleMeta, HeadMoveMeta, KernelMeta, MatMulMeta, SoftmaxRectMeta};
use crate::Real;
use std::sync::Arc;
use wilupgu::{Backend, Binding, ComputeGraph, Tensor, TensorMode};

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

use super::meta::{EmbeddingMeta, KernelMeta};
use std::sync::Arc;
use wilupgu::{Backend, Binding, ComputeGraph, Tensor, TensorMode};

fn grid(shape: EmbeddingMeta) -> [u32; 3] {
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
        grid(shape),
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
        grid(shape),
    );
}

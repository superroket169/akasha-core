use super::meta::{CrossEntropyMeta, KernelMeta};
use std::sync::Arc;
use wilupgu::{Backend, Binding, ComputeGraph, Tensor, TensorMode};

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

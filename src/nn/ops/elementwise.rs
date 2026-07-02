use super::meta::{KernelMeta, ZeroMeta};
use std::sync::Arc;
use wilupgu::{Backend, Binding, ComputeGraph, Tensor, TensorMode};

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

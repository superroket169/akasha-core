use super::meta::{HeadMoveMeta, KernelMeta};
use std::sync::Arc;
use wilupgu::{Backend, Binding, ComputeGraph, Tensor, TensorMode};

fn grid(shape: HeadMoveMeta) -> [u32; 3] {
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
        grid(shape),
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

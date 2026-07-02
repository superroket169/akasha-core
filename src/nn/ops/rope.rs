use super::meta::{KernelMeta, RopeMeta, RopeOffsetMeta};
use std::sync::Arc;
use wilupgu::{Backend, Binding, ComputeGraph, Tensor, TensorMode};

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

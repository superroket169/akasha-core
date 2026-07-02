use super::meta::{CacheWriteMeta, KernelMeta};
use std::sync::Arc;
use wilupgu::{Backend, Binding, ComputeGraph, Tensor, TensorMode};

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

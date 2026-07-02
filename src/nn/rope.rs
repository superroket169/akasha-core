use super::ops;
use super::ops::meta::RopeMeta;
use super::traits::Layer;
use crate::Real;
use std::sync::Arc;
use wilupgu::{Backend, ComputeGraph, Tensor};

pub struct RoPE<B: Backend> {
    pub in_out_buffer: Arc<Tensor<B>>,
    pub grad_input: Arc<Tensor<B>>,
    pub forward_graph: ComputeGraph<B>,
    pub backward_graph: ComputeGraph<B>,
}

impl<B: Backend> RoPE<B> {
    pub fn new(
        ctx: Arc<B>,
        dim: u32,
        seq_len: u32,
        input_buffer: &Arc<Tensor<B>>,
        grad_output: &Arc<Tensor<B>>,
    ) -> Self {
        // FIXME: (stage 5): hardcoded head_dim -- this legacy struct is unused by
        // the model (pipeline.rs emits RoPE nodes directly); will be removed
        let shape = RopeMeta {
            seq_len,
            dim,
            head_dim: 64,
        };

        let mut forward_graph = ComputeGraph::new(ctx.clone());
        ops::rope::rope(&mut forward_graph, input_buffer, shape);

        let grad_zero = vec![0.0 as Real; (seq_len * dim) as usize];
        let grad_input = Arc::new(Tensor::init_from_cpu(ctx.clone(), &grad_zero));

        let mut backward_graph = ComputeGraph::new(ctx.clone());
        ops::rope::rope_bwd(&mut backward_graph, grad_output, shape);

        Self {
            in_out_buffer: input_buffer.clone(),
            grad_input,
            forward_graph,
            backward_graph,
        }
    }
}

impl<B: Backend> Layer for RoPE<B> {
    fn forward(&self) {
        self.forward_graph.execute();
    }
    fn backward(&self) {
        self.backward_graph.execute();
    }
}

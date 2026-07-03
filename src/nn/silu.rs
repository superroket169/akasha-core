use super::ops;
use super::traits::Layer;
use std::sync::Arc;
use wilupgu::{Backend, ComputeGraph, Tensor};

pub struct SiLU<B: Backend> {
    pub in_out_buffer: Arc<Tensor<B>>,
    pub grad_input: Arc<Tensor<B>>,
    pub forward_graph: ComputeGraph<B>,
    pub backward_graph: ComputeGraph<B>,
}

impl<B: Backend> SiLU<B> {
    pub fn new(
        ctx: Arc<B>,
        total_elements: u32,
        input_buffer: &Arc<Tensor<B>>,
        grad_output: &Arc<Tensor<B>>,
        grad_input: &Arc<Tensor<B>>,
    ) -> Self {
        let grad_input = grad_input.clone();

        let mut forward_graph = ComputeGraph::new(ctx.clone());
        ops::silu(&mut forward_graph, input_buffer, total_elements);

        let mut backward_graph = ComputeGraph::new(ctx.clone());
        ops::silu_bwd(
            &mut backward_graph,
            input_buffer,
            grad_output,
            &grad_input,
            total_elements,
        );

        Self {
            in_out_buffer: input_buffer.clone(),
            grad_input,
            forward_graph,
            backward_graph,
        }
    }
}

impl<B: Backend> Layer for SiLU<B> {
    fn forward(&self) {
        self.forward_graph.execute();
    }
    fn backward(&self) {
        self.backward_graph.execute();
    }
}

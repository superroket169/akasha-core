use super::traits::Layer;
use std::sync::Arc;
use wilupgu::{Backend, Binding, ComputeGraph, Tensor, TensorMode};

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
        let mut forward_graph = ComputeGraph::new(ctx.clone());

        forward_graph.add_node(
            "SiLU",
            &[Binding::new(0, &input_buffer.buffer, TensorMode::InOut)],
            [(total_elements + 255) / 256, 1, 1],
        );

        let grad_input = grad_input.clone();

        let mut backward_graph = ComputeGraph::new(ctx.clone());

        backward_graph.add_node(
            "SiLUBwd",
            &[
                Binding::new(0, &input_buffer.buffer, TensorMode::Input),
                Binding::new(1, &grad_output.buffer, TensorMode::Input),
                Binding::new(2, &grad_input.buffer, TensorMode::Output),
            ],
            [(total_elements + 255) / 256, 1, 1],
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

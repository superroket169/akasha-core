use super::traits::Layer;
use std::sync::Arc;
use wilupgu::{Backend, Binding, ComputeGraph, Tensor, TensorMode};

pub struct Add<B: Backend> {
    pub in_out_buffer: Arc<Tensor<B>>,
    pub grad_a: Arc<Tensor<B>>,
    pub grad_b: Arc<Tensor<B>>,
    pub forward_graph: ComputeGraph<B>,
    pub backward_graph: ComputeGraph<B>,
}

impl<B: Backend> Add<B> {
    pub fn new(
        ctx: Arc<B>,
        length: u32,
        buf_a: &Arc<Tensor<B>>,
        buf_b: &Arc<Tensor<B>>,
        grad_output: &Arc<Tensor<B>>,
        grad_a: &Arc<Tensor<B>>,
        grad_b: &Arc<Tensor<B>>,
    ) -> Self {
        let mut forward_graph = ComputeGraph::new(ctx.clone());

        forward_graph.add_node(
            "ResidualAdd",
            &[
                Binding::new(0, &buf_a.buffer, TensorMode::InOut),
                Binding::new(1, &buf_b.buffer, TensorMode::Input),
            ],
            [(length + 255) / 256, 1, 1],
        );

        let grad_a = grad_a.clone();
        let grad_b = grad_b.clone();

        let mut backward_graph = ComputeGraph::new(ctx.clone());

        backward_graph.add_node(
            "ResidualAdd",
            &[
                Binding::new(0, &grad_a.buffer, TensorMode::InOut),
                Binding::new(1, &grad_output.buffer, TensorMode::Input),
            ],
            [(length + 255) / 256, 1, 1],
        );

        backward_graph.add_node(
            "ResidualAdd",
            &[
                Binding::new(0, &grad_b.buffer, TensorMode::InOut),
                Binding::new(1, &grad_output.buffer, TensorMode::Input),
            ],
            [(length + 255) / 256, 1, 1],
        );

        Self {
            in_out_buffer: buf_a.clone(),
            grad_a,
            grad_b,
            forward_graph,
            backward_graph,
        }
    }
}

impl<B: Backend> Layer for Add<B> {
    fn forward(&self) {
        self.forward_graph.execute();
    }

    fn backward(&self) {
        self.backward_graph.execute();
    }
}

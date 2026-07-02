use super::ops;
use super::traits::Layer;
use std::sync::Arc;
use wilupgu::{Backend, ComputeGraph, Tensor};

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
        let grad_a = grad_a.clone();
        let grad_b = grad_b.clone();

        let mut forward_graph = ComputeGraph::new(ctx.clone());
        ops::elementwise::residual_add(&mut forward_graph, buf_a, buf_b, length);

        let mut backward_graph = ComputeGraph::new(ctx.clone());
        ops::elementwise::residual_add(&mut backward_graph, &grad_a, grad_output, length);
        ops::elementwise::residual_add(&mut backward_graph, &grad_b, grad_output, length);

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

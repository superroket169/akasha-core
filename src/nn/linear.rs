use super::traits::Layer;
use crate::Real;
use std::sync::Arc;
use wilupgu::{Backend, Binding, ComputeGraph, Tensor, TensorMode};

pub struct Linear<B: Backend> {
    pub weight: Arc<Tensor<B>>,
    pub out_buffer: Arc<Tensor<B>>,
    pub grad_weight: Arc<Tensor<B>>,
    pub grad_input: Arc<Tensor<B>>,
    pub forward_graph: ComputeGraph<B>,
    pub backward_graph: ComputeGraph<B>,
}

impl<B: Backend> Linear<B> {
    pub fn new(
        ctx: Arc<B>,
        in_features: u32,
        out_features: u32,
        seq_len: u32,
        weight_data: &[Real],
        input_buffer: &Arc<Tensor<B>>,
        grad_output: &Arc<Tensor<B>>,
        grad_input: &Arc<Tensor<B>>,
    ) -> Self {
        let weight = Arc::new(Tensor::init_from_cpu(ctx.clone(), weight_data));

        let m = seq_len;
        let meta_data = vec![m, out_features, in_features];
        let t_meta = Arc::new(Tensor::init_from_cpu(ctx.clone(), &meta_data));

        let out_size = (m * out_features) as usize;
        let zero_out = vec![0.0 as Real; out_size];
        let out_buffer = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zero_out));

        let zero_grad_w = vec![0.0 as Real; (in_features * out_features) as usize];
        let grad_weight = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zero_grad_w));

        let grad_input = grad_input.clone();

        let mut forward_graph = ComputeGraph::new(ctx.clone());
        forward_graph.add_node(
            "MatMul",
            &[
                Binding::new(0, &input_buffer.buffer, TensorMode::Input),
                Binding::new(1, &weight.buffer, TensorMode::Input),
                Binding::new(2, &out_buffer.buffer, TensorMode::Output),
                Binding::new(3, &t_meta.buffer, TensorMode::Meta),
            ],
            [(out_features + 15) / 16, (m + 15) / 16, 1],
        );

        let meta_grad_in_data = vec![m, in_features, out_features];
        let t_meta_grad_in = Arc::new(Tensor::init_from_cpu(ctx.clone(), &meta_grad_in_data));

        let mut backward_graph = ComputeGraph::new(ctx.clone());

        backward_graph.add_node(
            "MatMulWeightBwd",
            &[
                Binding::new(0, &input_buffer.buffer, TensorMode::Input),
                Binding::new(1, &grad_output.buffer, TensorMode::Input),
                Binding::new(2, &grad_weight.buffer, TensorMode::Output),
                Binding::new(3, &t_meta.buffer, TensorMode::Meta),
            ],
            [(out_features + 15) / 16, (in_features + 15) / 16, 1],
        );

        backward_graph.add_node(
            "MatMulTrp",
            &[
                Binding::new(0, &grad_output.buffer, TensorMode::Input),
                Binding::new(1, &weight.buffer, TensorMode::Input),
                Binding::new(2, &grad_input.buffer, TensorMode::Output),
                Binding::new(3, &t_meta_grad_in.buffer, TensorMode::Meta),
            ],
            [(in_features + 15) / 16, (m + 15) / 16, 1],
        );

        Self {
            weight,
            out_buffer,
            grad_weight,
            grad_input,
            forward_graph,
            backward_graph,
        }
    }
}

impl<B: Backend> Layer for Linear<B> {
    fn forward(&self) {
        self.forward_graph.execute();
    }
    fn backward(&self) {
        self.backward_graph.execute();
    }
}

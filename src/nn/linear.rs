use super::ops;
use super::ops::meta::MatMulMeta;
use super::traits::Layer;
use crate::Real;
use std::sync::Arc;
use wilupgu::{Backend, ComputeGraph, Tensor};

pub struct Linear<B: Backend> {
    pub weight: Arc<Tensor<B>>,
    pub out_buffer: Arc<Tensor<B>>,
    pub grad_weight: Arc<Tensor<B>>,
    pub grad_input: Arc<Tensor<B>>,
    pub forward_graph: ComputeGraph<B>,
    pub backward_graph: ComputeGraph<B>,
}

impl<B: Backend> Linear<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ctx: Arc<B>,
        in_features: u32,
        out_features: u32,
        seq_len: u32,
        weight: &Arc<Tensor<B>>,
        input_buffer: &Arc<Tensor<B>>,
        grad_output: &Arc<Tensor<B>>,
        grad_input: &Arc<Tensor<B>>,
    ) -> Self {
        let weight = weight.clone();

        let out_size = (seq_len * out_features) as usize;
        let zero_out = vec![0.0 as Real; out_size];
        let out_buffer = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zero_out));

        let zero_grad_w = vec![0.0 as Real; (in_features * out_features) as usize];
        let grad_weight = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zero_grad_w));

        let grad_input = grad_input.clone();

        let mut forward_graph = ComputeGraph::new(ctx.clone());
        ops::matmul(
            &mut forward_graph,
            input_buffer,
            &weight,
            &out_buffer,
            MatMulMeta {
                m: seq_len,
                n: out_features,
                k: in_features,
            },
        );

        let mut backward_graph = ComputeGraph::new(ctx.clone());
        ops::matmul_weight_bwd(
            &mut backward_graph,
            input_buffer,
            grad_output,
            &grad_weight,
            MatMulMeta {
                m: seq_len,
                n: out_features,
                k: in_features,
            },
        );
        ops::matmul_trp(
            &mut backward_graph,
            grad_output,
            &weight,
            &grad_input,
            MatMulMeta {
                m: seq_len,
                n: in_features,
                k: out_features,
            },
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

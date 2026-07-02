use super::ops;
use super::ops::meta::NormMeta;
use super::traits::Layer;
use crate::Real;
use std::sync::Arc;
use wilupgu::{Backend, ComputeGraph, Tensor};

pub struct RMSNorm<B: Backend> {
    pub weight: Arc<Tensor<B>>,
    pub out_buffer: Arc<Tensor<B>>,
    pub grad_weight: Arc<Tensor<B>>,
    pub grad_input: Arc<Tensor<B>>,
    pub rsqrt_cache: Arc<Tensor<B>>,
    pub forward_graph: ComputeGraph<B>,
    pub backward_graph: ComputeGraph<B>,
}

impl<B: Backend> RMSNorm<B> {
    pub fn new(
        ctx: Arc<B>,
        dim: u32,
        seq_len: u32,
        weight: &Arc<Tensor<B>>,
        input_buffer: &Arc<Tensor<B>>,
        grad_output: &Arc<Tensor<B>>,
        grad_input: &Arc<Tensor<B>>,
    ) -> Self {
        assert_eq!(
            weight.size,
            dim as u64 * std::mem::size_of::<Real>() as u64,
            "RMSNorm weight size mismatch!"
        );

        let shape = NormMeta {
            seq_len,
            size: dim,
            eps: 1e-5,
        };

        let weight = weight.clone();
        let zero_dim = vec![0.0 as Real; dim as usize];
        let grad_weight = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zero_dim));

        let out_size = (seq_len * dim) as usize;
        let zero_out = vec![0.0 as Real; out_size];
        let out_buffer = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zero_out));
        let grad_input = grad_input.clone();

        let rsqrt_cache = Arc::new(Tensor::init_from_cpu(
            ctx.clone(),
            &vec![0.0 as Real; seq_len as usize],
        ));

        let mut forward_graph = ComputeGraph::new(ctx.clone());
        ops::norm::rmsnorm(
            &mut forward_graph,
            input_buffer,
            &weight,
            &out_buffer,
            shape,
        );

        let mut backward_graph = ComputeGraph::new(ctx.clone());
        ops::norm::rmsnorm_bwd(
            &mut backward_graph,
            grad_output,
            input_buffer,
            &weight,
            &grad_input,
            &rsqrt_cache,
            &grad_weight,
            shape,
        );

        Self {
            weight,
            out_buffer,
            grad_weight,
            grad_input,
            rsqrt_cache,
            forward_graph,
            backward_graph,
        }
    }
}

impl<B: Backend> Layer for RMSNorm<B> {
    fn forward(&self) {
        self.forward_graph.execute();
    }
    fn backward(&self) {
        self.backward_graph.execute();
    }
}

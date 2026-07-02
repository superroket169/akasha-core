use super::ops::meta::{KernelMeta, NormMeta};
use super::traits::Layer;
use crate::Real;
use std::sync::Arc;
use wilupgu::{Backend, Binding, ComputeGraph, Tensor, TensorMode};

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
    pub(crate) fn forward_nodes(
        graph: &mut ComputeGraph<B>,
        weight: &Arc<Tensor<B>>,
        input: &Arc<Tensor<B>>,
        output: &Arc<Tensor<B>>,
        seq_len: u32,
        dim: u32,
        eps: f32,
    ) {
        let t_meta = NormMeta {
            seq_len,
            size: dim,
            eps,
        }
        .upload(&input.ctx);

        graph.add_node(
            "RMSNorm",
            &[
                Binding::new(0, &input.buffer, TensorMode::Input),
                Binding::new(1, &weight.buffer, TensorMode::Input),
                Binding::new(2, &output.buffer, TensorMode::Output),
                Binding::new(3, &t_meta.buffer, TensorMode::Meta),
            ],
            [seq_len, 1, 1],
        );
    }

    pub fn new(
        ctx: Arc<B>,
        dim: u32,
        seq_len: u32,
        weight_data: &[Real],
        input_buffer: &Arc<Tensor<B>>,
        grad_output: &Arc<Tensor<B>>,
        grad_input: &Arc<Tensor<B>>,
    ) -> Self {
        assert_eq!(
            weight_data.len(),
            dim as usize,
            "RMSNorm weight size mismatch!"
        );

        let weight = Arc::new(Tensor::init_from_cpu(ctx.clone(), weight_data));
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

        let t_meta = NormMeta {
            seq_len,
            size: dim,
            eps: 1e-5,
        }
        .upload(&ctx);

        let mut forward_graph = ComputeGraph::new(ctx.clone());
        Self::forward_nodes(
            &mut forward_graph,
            &weight,
            input_buffer,
            &out_buffer,
            seq_len,
            dim,
            1e-5,
        );

        let mut backward_graph = ComputeGraph::new(ctx.clone());
        backward_graph.add_node(
            "RMSNormBwd",
            &[
                Binding::new(0, &grad_output.buffer, TensorMode::Input),
                Binding::new(1, &input_buffer.buffer, TensorMode::Input),
                Binding::new(2, &weight.buffer, TensorMode::Input),
                Binding::new(3, &grad_input.buffer, TensorMode::Output),
                Binding::new(4, &rsqrt_cache.buffer, TensorMode::Output),
                Binding::new(5, &t_meta.buffer, TensorMode::Meta),
            ],
            [seq_len, 1, 1],
        );

        backward_graph.add_node(
            "RMSNormWeightBwd",
            &[
                Binding::new(0, &grad_output.buffer, TensorMode::Input),
                Binding::new(1, &input_buffer.buffer, TensorMode::Input),
                Binding::new(2, &rsqrt_cache.buffer, TensorMode::Input),
                Binding::new(3, &grad_weight.buffer, TensorMode::Output),
                Binding::new(4, &t_meta.buffer, TensorMode::Meta),
            ],
            [(dim + 255) / 256, 1, 1],
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

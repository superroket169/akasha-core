use super::traits::Layer;
use crate::Real;
use std::sync::Arc;
use wilupgu::context::WgpuContext;
use wilupgu::graph::{ComputeGraph, TensorBind, TensorMode};
use wilupgu::nn::shaders::BuiltInShader;
use wilupgu::tensor::Tensor;

pub struct RoPE {
    pub in_out_buffer: Arc<Tensor>,
    pub grad_input: Arc<Tensor>,
    pub forward_graph: ComputeGraph,
    pub backward_graph: ComputeGraph,
}

impl RoPE {
    pub fn new(
        ctx: Arc<WgpuContext>,
        dim: u32,
        seq_len: u32,
        input_buffer: &Arc<Tensor>,
        grad_output: &Arc<Tensor>,
    ) -> Self {
        let head_dim = 64u32;
        let meta_data = vec![seq_len, dim, head_dim];
        let t_meta = Arc::new(Tensor::init_from_cpu(ctx.clone(), &meta_data));

        let shader_fw = BuiltInShader::RoPE.get_def();
        let mut forward_graph = ComputeGraph::new(ctx.clone());

        // --- FORWARD ---
        forward_graph.add_node(
            &shader_fw,
            &[
                TensorBind {
                    binding: 0,
                    tensor: input_buffer,
                    mode: TensorMode::InOut,
                },
                TensorBind {
                    binding: 1,
                    tensor: &t_meta,
                    mode: TensorMode::Meta,
                },
            ],
            [(dim + 15) / 16, (seq_len + 15) / 16, 1],
        );

        // --- BACKWARD ---
        let grad_zero = vec![0.0 as Real; (seq_len * dim) as usize];
        let grad_input = Arc::new(Tensor::init_from_cpu(ctx.clone(), &grad_zero));

        let shader_bw = BuiltInShader::RoPEBwd.get_def();
        let mut backward_graph = ComputeGraph::new(ctx.clone());

        backward_graph.add_node(
            &shader_bw,
            &[
                TensorBind {
                    binding: 0,
                    tensor: grad_output,
                    mode: TensorMode::InOut,
                },
                TensorBind {
                    binding: 1,
                    tensor: &t_meta,
                    mode: TensorMode::Meta,
                },
            ],
            [(dim + 15) / 16, (seq_len + 15) / 16, 1],
        );

        Self {
            in_out_buffer: input_buffer.clone(),
            grad_input,
            forward_graph,
            backward_graph,
        }
    }
}

impl Layer for RoPE {
    fn forward(&self) {
        self.forward_graph.execute();
    }
    fn backward(&self) {
        self.backward_graph.execute();
    }
}

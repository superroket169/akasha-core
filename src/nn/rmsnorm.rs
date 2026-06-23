use super::traits::Layer;
use crate::Real;
use std::sync::Arc;
use wilupgu::context::WgpuContext;
use wilupgu::graph::{ComputeGraph, TensorBind, TensorMode};
use wilupgu::nn::shaders::BuiltInShader;
use wilupgu::tensor::Tensor;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Meta {
    seq_len: u32,
    size: u32,
    eps: f32,
}

pub struct RMSNorm {
    pub weight: Arc<Tensor>,
    pub out_buffer: Arc<Tensor>,
    pub grad_weight: Arc<Tensor>,
    pub grad_input: Arc<Tensor>,
    pub rsqrt_cache: Arc<Tensor>,
    pub forward_graph: ComputeGraph,
    pub backward_graph: ComputeGraph,
}

impl RMSNorm {
    pub fn new(
        ctx: Arc<WgpuContext>,
        dim: u32,
        seq_len: u32,
        weight_data: &[Real],
        input_buffer: &Arc<Tensor>,
        grad_output: &Arc<Tensor>,
        grad_input: &Arc<Tensor>,
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

        let eps = 1e-5f32;
        let meta_data = [Meta {
            seq_len,
            size: dim,
            eps,
        }];
        let t_meta = Arc::new(Tensor::init_from_cpu(ctx.clone(), &meta_data));

        // --- FORWARD ---
        let shader_fw = BuiltInShader::RMSNorm.get_def();
        let mut forward_graph = ComputeGraph::new(ctx.clone());
        forward_graph.add_node(
            &shader_fw,
            &[
                TensorBind {
                    binding: 0,
                    tensor: input_buffer,
                    mode: TensorMode::Input,
                },
                TensorBind {
                    binding: 1,
                    tensor: &weight,
                    mode: TensorMode::Input,
                },
                TensorBind {
                    binding: 2,
                    tensor: &out_buffer,
                    mode: TensorMode::Output,
                },
                TensorBind {
                    binding: 3,
                    tensor: &t_meta,
                    mode: TensorMode::Meta,
                },
            ],
            [seq_len, 1, 1],
        );

        // --- BACKWARD ---
        let shader_bw = BuiltInShader::RMSNormBwd.get_def();
        let mut backward_graph = ComputeGraph::new(ctx.clone());
        backward_graph.add_node(
            &shader_bw,
            &[
                TensorBind {
                    binding: 0,
                    tensor: grad_output,
                    mode: TensorMode::Input,
                },
                TensorBind {
                    binding: 1,
                    tensor: input_buffer,
                    mode: TensorMode::Input,
                },
                TensorBind {
                    binding: 2,
                    tensor: &weight,
                    mode: TensorMode::Input,
                },
                TensorBind {
                    binding: 3,
                    tensor: &grad_input,
                    mode: TensorMode::Output,
                },
                TensorBind {
                    binding: 4,
                    tensor: &rsqrt_cache,
                    mode: TensorMode::Output,
                },
                TensorBind {
                    binding: 5,
                    tensor: &t_meta,
                    mode: TensorMode::Meta,
                },
            ],
            [seq_len, 1, 1],
        );

        let shader_bw_weight = BuiltInShader::RMSNormWeightBwd.get_def();
        backward_graph.add_node(
            &shader_bw_weight,
            &[
                TensorBind {
                    binding: 0,
                    tensor: grad_output,
                    mode: TensorMode::Input,
                },
                TensorBind {
                    binding: 1,
                    tensor: input_buffer,
                    mode: TensorMode::Input,
                },
                TensorBind {
                    binding: 2,
                    tensor: &rsqrt_cache,
                    mode: TensorMode::Input,
                },
                TensorBind {
                    binding: 3,
                    tensor: &grad_weight,
                    mode: TensorMode::Output,
                },
                TensorBind {
                    binding: 4,
                    tensor: &t_meta,
                    mode: TensorMode::Meta,
                },
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

impl Layer for RMSNorm {
    fn forward(&self) {
        self.forward_graph.execute();
    }
    fn backward(&self) {
        self.backward_graph.execute();
    }
}

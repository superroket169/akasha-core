use super::traits::Layer;
use crate::Real;
use std::sync::Arc;
use wilupgu::context::WgpuContext;
use wilupgu::graph::{ComputeGraph, TensorBind, TensorMode};
use wilupgu::nn::shaders::BuiltInShader;
use wilupgu::tensor::Tensor;

pub struct Linear {
    pub weight: Arc<Tensor>,
    pub out_buffer: Arc<Tensor>,
    pub grad_weight: Arc<Tensor>,
    pub grad_input: Arc<Tensor>,
    pub forward_graph: ComputeGraph,
    pub backward_graph: ComputeGraph,
}

impl Linear {
    pub fn new(
        ctx: Arc<WgpuContext>,
        in_features: u32,
        out_features: u32,
        weight_data: &[Real],
        input_buffer: &Arc<Tensor>,
        grad_output: &Arc<Tensor>,
    ) -> Self {
        let weight = Arc::new(Tensor::init_from_cpu(ctx.clone(), weight_data));

        let m = 1u32; // şimdilik 1
        let meta_data = vec![m, out_features, in_features];
        let t_meta = Arc::new(Tensor::init_from_cpu(ctx.clone(), &meta_data));

        let out_size = (m * out_features) as usize;
        let zero_out = vec![0.0 as Real; out_size];
        let out_buffer = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zero_out));

        let zero_grad_w = vec![0.0 as Real; (in_features * out_features) as usize];
        let grad_weight = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zero_grad_w));

        let zero_grad_in = vec![0.0 as Real; (m * in_features) as usize];
        let grad_input = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zero_grad_in));

        // --- FORWARD ---
        let shader_fw = BuiltInShader::MatMul.get_def();
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
            [(out_features + 15) / 16, (m + 15) / 16, 1],
        );

        // --- BACKWARD ---
        let shader_bwd_w = BuiltInShader::MatMulWeightBwd.get_def();
        let shader_bwd_in = BuiltInShader::MatMulTrp.get_def(); // Input Trp B
        let mut backward_graph = ComputeGraph::new(ctx.clone());

        backward_graph.add_node(
            &shader_bwd_w,
            &[
                TensorBind {
                    binding: 0,
                    tensor: input_buffer,
                    mode: TensorMode::Input,
                },
                TensorBind {
                    binding: 1,
                    tensor: grad_output,
                    mode: TensorMode::Input,
                },
                TensorBind {
                    binding: 2,
                    tensor: &grad_weight,
                    mode: TensorMode::Output,
                },
                TensorBind {
                    binding: 3,
                    tensor: &t_meta,
                    mode: TensorMode::Meta,
                },
            ],
            [(out_features + 15) / 16, (in_features + 15) / 16, 1],
        );

        backward_graph.add_node(
            &shader_bwd_in,
            &[
                TensorBind {
                    binding: 0,
                    tensor: grad_output,
                    mode: TensorMode::Input,
                },
                TensorBind {
                    binding: 1,
                    tensor: &weight,
                    mode: TensorMode::Input,
                },
                TensorBind {
                    binding: 2,
                    tensor: &grad_input,
                    mode: TensorMode::Output,
                },
                TensorBind {
                    binding: 3,
                    tensor: &t_meta,
                    mode: TensorMode::Meta,
                },
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

impl Layer for Linear {
    fn forward(&self) {
        self.forward_graph.execute();
    }
    fn backward(&self) {
        self.backward_graph.execute();
    }
}

use super::traits::Layer;
use std::sync::Arc;
use wilupgu::context::WgpuContext;
use wilupgu::graph::{ComputeGraph, TensorBind, TensorMode};
use wilupgu::nn::shaders::BuiltInShader;
use wilupgu::tensor::Tensor;

pub struct SiLU {
    pub in_out_buffer: Arc<Tensor>,
    pub grad_input: Arc<Tensor>,
    pub forward_graph: ComputeGraph,
    pub backward_graph: ComputeGraph,
}

impl SiLU {
    pub fn new(
        ctx: Arc<WgpuContext>,
        total_elements: u32,
        input_buffer: &Arc<Tensor>,
        grad_output: &Arc<Tensor>,
        grad_input: &Arc<Tensor>,
    ) -> Self {
        let shader_fw = BuiltInShader::SiLU.get_def();
        let mut forward_graph = ComputeGraph::new(ctx.clone());

        forward_graph.add_node(
            &shader_fw,
            &[TensorBind {
                binding: 0,
                tensor: input_buffer,
                mode: TensorMode::InOut,
            }],
            [(total_elements + 255) / 256, 1, 1],
        );

        let grad_input = grad_input.clone();

        let shader_bw = BuiltInShader::SiLUBwd.get_def();
        let mut backward_graph = ComputeGraph::new(ctx.clone());

        backward_graph.add_node(
            &shader_bw,
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
                    tensor: &grad_input,
                    mode: TensorMode::Output,
                },
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

impl Layer for SiLU {
    fn forward(&self) {
        self.forward_graph.execute();
    }
    fn backward(&self) {
        self.backward_graph.execute();
    }
}

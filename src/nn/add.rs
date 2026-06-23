use super::traits::Layer;
use std::sync::Arc;
use wilupgu::context::WgpuContext;
use wilupgu::graph::{ComputeGraph, TensorBind, TensorMode};
use wilupgu::nn::shaders::BuiltInShader;
use wilupgu::tensor::Tensor;

pub struct Add {
    pub in_out_buffer: Arc<Tensor>,
    pub grad_a: Arc<Tensor>,
    pub grad_b: Arc<Tensor>,
    pub forward_graph: ComputeGraph,
    pub backward_graph: ComputeGraph,
}

impl Add {
    pub fn new(
        ctx: Arc<WgpuContext>,
        length: u32,
        buf_a: &Arc<Tensor>,
        buf_b: &Arc<Tensor>,
        grad_output: &Arc<Tensor>,
        grad_a: &Arc<Tensor>,
        grad_b: &Arc<Tensor>,
    ) -> Self {
        let mut forward_graph = ComputeGraph::new(ctx.clone());
        let shader_fw = BuiltInShader::ResidualAdd.get_def();

        forward_graph.add_node(
            &shader_fw,
            &[
                TensorBind {
                    binding: 0,
                    tensor: buf_a,
                    mode: TensorMode::InOut,
                },
                TensorBind {
                    binding: 1,
                    tensor: buf_b,
                    mode: TensorMode::Input,
                },
            ],
            [(length + 255) / 256, 1, 1],
        );

        let grad_a = grad_a.clone();
        let grad_b = grad_b.clone();

        let mut backward_graph = ComputeGraph::new(ctx.clone());
        let shader_bwd = BuiltInShader::ResidualAdd.get_def();

        backward_graph.add_node(
            &shader_bwd,
            &[
                TensorBind {
                    binding: 0,
                    tensor: &grad_a,
                    mode: TensorMode::InOut,
                },
                TensorBind {
                    binding: 1,
                    tensor: grad_output,
                    mode: TensorMode::Input,
                },
            ],
            [(length + 255) / 256, 1, 1],
        );

        backward_graph.add_node(
            &shader_bwd,
            &[
                TensorBind {
                    binding: 0,
                    tensor: &grad_b,
                    mode: TensorMode::InOut,
                },
                TensorBind {
                    binding: 1,
                    tensor: grad_output,
                    mode: TensorMode::Input,
                },
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

impl Layer for Add {
    fn forward(&self) {
        self.forward_graph.execute();
    }

    fn backward(&self) {
        self.backward_graph.execute();
    }
}

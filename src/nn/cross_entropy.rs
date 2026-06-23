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
    vocab_size: u32,
    num_rows: u32,
}

pub struct CrossEntropy {
    pub seq_len: u32,
    pub target_tokens: Arc<Tensor>,
    pub probs: Arc<Tensor>,
    pub losses: Arc<Tensor>,
    pub d_losses: Arc<Tensor>,
    pub grad_logits: Arc<Tensor>,
    pub forward_graph: ComputeGraph,
    pub backward_graph: ComputeGraph,
}

impl CrossEntropy {
    pub fn new(
        ctx: Arc<WgpuContext>,
        vocab_size: u32,
        seq_len: u32,
        logits: &Arc<Tensor>,
        grad_logits: &Arc<Tensor>,
    ) -> Self {
        let target_tokens = Arc::new(Tensor::init_from_cpu(
            ctx.clone(),
            &vec![0u32; seq_len as usize],
        ));
        let probs = Arc::new(Tensor::init_from_cpu(
            ctx.clone(),
            &vec![0.0 as Real; (seq_len * vocab_size) as usize],
        ));
        let losses = Arc::new(Tensor::init_from_cpu(
            ctx.clone(),
            &vec![0.0 as Real; seq_len as usize],
        ));
        let d_losses = Arc::new(Tensor::init_from_cpu(
            ctx.clone(),
            &vec![1.0 as Real / seq_len as Real; seq_len as usize],
        ));
        let grad_logits = grad_logits.clone();

        let meta_data = [Meta {
            vocab_size,
            num_rows: seq_len,
        }];
        let t_meta = Arc::new(Tensor::init_from_cpu(ctx.clone(), &meta_data));

        let dispatch_x = (seq_len + 255) / 256;

        // --- FORWARD ---
        let shader_fw = BuiltInShader::CrossEntropy.get_def();
        let mut forward_graph = ComputeGraph::new(ctx.clone());
        forward_graph.add_node(
            &shader_fw,
            &[
                TensorBind {
                    binding: 0,
                    tensor: logits,
                    mode: TensorMode::Input,
                },
                TensorBind {
                    binding: 1,
                    tensor: &target_tokens,
                    mode: TensorMode::Input,
                },
                TensorBind {
                    binding: 2,
                    tensor: &probs,
                    mode: TensorMode::Output,
                },
                TensorBind {
                    binding: 3,
                    tensor: &losses,
                    mode: TensorMode::Output,
                },
                TensorBind {
                    binding: 4,
                    tensor: &t_meta,
                    mode: TensorMode::Meta,
                },
            ],
            [dispatch_x, 1, 1],
        );

        // --- BACKWARD ---
        let shader_bw = BuiltInShader::CrossEntropyBwd.get_def();
        let mut backward_graph = ComputeGraph::new(ctx.clone());
        backward_graph.add_node(
            &shader_bw,
            &[
                TensorBind {
                    binding: 0,
                    tensor: &probs,
                    mode: TensorMode::Input,
                },
                TensorBind {
                    binding: 1,
                    tensor: &target_tokens,
                    mode: TensorMode::Input,
                },
                TensorBind {
                    binding: 2,
                    tensor: &d_losses,
                    mode: TensorMode::Input,
                },
                TensorBind {
                    binding: 3,
                    tensor: &grad_logits,
                    mode: TensorMode::Output,
                },
                TensorBind {
                    binding: 4,
                    tensor: &t_meta,
                    mode: TensorMode::Meta,
                },
            ],
            [dispatch_x, 1, 1],
        );

        Self {
            seq_len,
            target_tokens,
            probs,
            losses,
            d_losses,
            grad_logits,
            forward_graph,
            backward_graph,
        }
    }

    pub fn set_grad_scale(&self, scale: Real) {
        self.d_losses
            .copy_from_cpu(&vec![scale; self.seq_len as usize]);
    }

    pub fn loss(&self) -> Real {
        let losses: Vec<Real> = self.losses.to_cpu();
        losses.iter().sum::<Real>() / losses.len() as Real
    }
}

impl Layer for CrossEntropy {
    fn forward(&self) {
        self.forward_graph.execute();
    }
    fn backward(&self) {
        self.backward_graph.execute();
    }
}

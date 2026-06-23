use super::traits::Layer;
use crate::Real;
use std::sync::Arc;
use wilupgu::context::WgpuContext;
use wilupgu::graph::{ComputeGraph, TensorBind, TensorMode};
use wilupgu::nn::shaders::BuiltInShader;
use wilupgu::tensor::Tensor;

pub struct SelfAttention {
    pub ctx: Arc<WgpuContext>,
    pub out_buffer: Arc<Tensor>,
    pub grad_q: Arc<Tensor>,
    pub grad_k: Arc<Tensor>,
    pub grad_v: Arc<Tensor>,
    pub grad_scores: Arc<Tensor>,

    pub forward_graph: ComputeGraph,
    pub backward_graph: ComputeGraph,
}

impl SelfAttention {
    pub fn new(
        ctx: Arc<WgpuContext>,
        seq_len: u32,
        dim: u32,
        q_buf: &Arc<Tensor>,
        k_buf: &Arc<Tensor>,
        v_buf: &Arc<Tensor>,
        grad_output: &Arc<Tensor>,
        grad_q: &Arc<Tensor>,
        grad_k: &Arc<Tensor>,
        grad_v: &Arc<Tensor>,
    ) -> Self {
        let scores_size = (seq_len * seq_len) as usize;
        let out_size = (seq_len * dim) as usize;

        let scores_data = vec![0.0 as Real; scores_size];
        let t_scores = Arc::new(Tensor::init_from_cpu(ctx.clone(), &scores_data));

        let out_data = vec![0.0 as Real; out_size];
        let out_buffer = Arc::new(Tensor::init_from_cpu(ctx.clone(), &out_data));

        let grad_q = grad_q.clone();
        let grad_k = grad_k.clone();
        let grad_v = grad_v.clone();
        let grad_scores = Arc::new(Tensor::init_from_cpu(ctx.clone(), &scores_data));
        let grad_scores_dx = Arc::new(Tensor::init_from_cpu(ctx.clone(), &scores_data));

        // Meta Tensors
        let meta_qkt_data = vec![seq_len as u32, dim as u32, seq_len as u32];
        let t_meta_qkt = Arc::new(Tensor::init_from_cpu(ctx.clone(), &meta_qkt_data));

        let meta_seq_data = vec![seq_len as u32];
        let t_meta_seq = Arc::new(Tensor::init_from_cpu(ctx.clone(), &meta_seq_data));

        let meta_out_data = vec![seq_len as u32, seq_len as u32, dim as u32];
        let t_meta_out = Arc::new(Tensor::init_from_cpu(ctx.clone(), &meta_out_data));

        // Shaders
        let shader_qkt = BuiltInShader::MatMulTrp.get_def();
        let shader_mask = BuiltInShader::CausalMask.get_def();
        let shader_softmax = BuiltInShader::Softmax.get_def();
        let shader_out = BuiltInShader::MatMul.get_def();

        // ==========================================
        //                  FORWARD
        // ==========================================
        let mut forward_graph = ComputeGraph::new(ctx.clone());

        forward_graph.add_node(
            &shader_qkt,
            &[
                TensorBind {
                    binding: 0,
                    tensor: q_buf,
                    mode: TensorMode::Input,
                },
                TensorBind {
                    binding: 1,
                    tensor: k_buf,
                    mode: TensorMode::Input,
                },
                TensorBind {
                    binding: 2,
                    tensor: &t_scores,
                    mode: TensorMode::Output,
                },
                TensorBind {
                    binding: 3,
                    tensor: &t_meta_qkt,
                    mode: TensorMode::Meta,
                },
            ],
            [(seq_len + 15) / 16, (seq_len + 15) / 16, 1],
        );

        forward_graph.add_node(
            &shader_mask,
            &[
                TensorBind {
                    binding: 0,
                    tensor: &t_scores,
                    mode: TensorMode::InOut,
                },
                TensorBind {
                    binding: 1,
                    tensor: &t_meta_seq,
                    mode: TensorMode::Meta,
                },
            ],
            [(seq_len + 15) / 16, (seq_len + 15) / 16, 1],
        );

        forward_graph.add_node(
            &shader_softmax,
            &[
                TensorBind {
                    binding: 0,
                    tensor: &t_scores,
                    mode: TensorMode::InOut,
                },
                TensorBind {
                    binding: 1,
                    tensor: &t_meta_seq,
                    mode: TensorMode::Meta,
                },
            ],
            [(seq_len + 255) / 256, 1, 1],
        );

        forward_graph.add_node(
            &shader_out,
            &[
                TensorBind {
                    binding: 0,
                    tensor: &t_scores,
                    mode: TensorMode::Input,
                },
                TensorBind {
                    binding: 1,
                    tensor: v_buf,
                    mode: TensorMode::Input,
                },
                TensorBind {
                    binding: 2,
                    tensor: &out_buffer,
                    mode: TensorMode::Output,
                },
                TensorBind {
                    binding: 3,
                    tensor: &t_meta_out,
                    mode: TensorMode::Meta,
                },
            ],
            [(dim + 15) / 16, (seq_len + 15) / 16, 1],
        );

        // ==========================================
        //                  BACKWARD
        // ==========================================
        let mut backward_graph = ComputeGraph::new(ctx.clone());
        let shader_matmul_bwd = BuiltInShader::MatMul.get_def();
        let shader_matmul_bwd_trp = BuiltInShader::MatMulTrp.get_def();
        let shader_softmax_bwd = BuiltInShader::SoftmaxBwd.get_def();

        backward_graph.add_node(
            &shader_matmul_bwd_trp,
            &[
                TensorBind {
                    binding: 0,
                    tensor: &t_scores,
                    mode: TensorMode::Input,
                },
                TensorBind {
                    binding: 1,
                    tensor: grad_output,
                    mode: TensorMode::Input,
                },
                TensorBind {
                    binding: 2,
                    tensor: &grad_v,
                    mode: TensorMode::Output,
                },
                TensorBind {
                    binding: 3,
                    tensor: &t_meta_qkt,
                    mode: TensorMode::Meta,
                },
            ],
            [(dim + 15) / 16, (seq_len + 15) / 16, 1],
        );

        backward_graph.add_node(
            &shader_matmul_bwd_trp,
            &[
                TensorBind {
                    binding: 0,
                    tensor: grad_output,
                    mode: TensorMode::Input,
                },
                TensorBind {
                    binding: 1,
                    tensor: v_buf,
                    mode: TensorMode::Input,
                },
                TensorBind {
                    binding: 2,
                    tensor: &grad_scores,
                    mode: TensorMode::Output,
                },
                TensorBind {
                    binding: 3,
                    tensor: &t_meta_out,
                    mode: TensorMode::Meta,
                },
            ],
            [(seq_len + 15) / 16, (seq_len + 15) / 16, 1],
        );

        backward_graph.add_node(
            &shader_softmax_bwd,
            &[
                TensorBind {
                    binding: 0,
                    tensor: &t_scores,
                    mode: TensorMode::Input,
                },
                TensorBind {
                    binding: 1,
                    tensor: &grad_scores,
                    mode: TensorMode::Input,
                },
                TensorBind {
                    binding: 2,
                    tensor: &grad_scores_dx,
                    mode: TensorMode::Output,
                },
                TensorBind {
                    binding: 3,
                    tensor: &t_meta_seq,
                    mode: TensorMode::Meta,
                }, // Config
            ],
            [(seq_len + 255) / 256, 1, 1],
        );

        backward_graph.add_node(
            &shader_matmul_bwd,
            &[
                TensorBind {
                    binding: 0,
                    tensor: &grad_scores_dx,
                    mode: TensorMode::Input,
                },
                TensorBind {
                    binding: 1,
                    tensor: k_buf,
                    mode: TensorMode::Input,
                },
                TensorBind {
                    binding: 2,
                    tensor: &grad_q,
                    mode: TensorMode::Output,
                },
                TensorBind {
                    binding: 3,
                    tensor: &t_meta_qkt,
                    mode: TensorMode::Meta,
                },
            ],
            [(dim + 15) / 16, (seq_len + 15) / 16, 1],
        );

        backward_graph.add_node(
            &shader_matmul_bwd_trp,
            &[
                TensorBind {
                    binding: 0,
                    tensor: &grad_scores_dx,
                    mode: TensorMode::Input,
                },
                TensorBind {
                    binding: 1,
                    tensor: q_buf,
                    mode: TensorMode::Input,
                },
                TensorBind {
                    binding: 2,
                    tensor: &grad_k,
                    mode: TensorMode::Output,
                },
                TensorBind {
                    binding: 3,
                    tensor: &t_meta_qkt,
                    mode: TensorMode::Meta,
                },
            ],
            [(dim + 15) / 16, (seq_len + 15) / 16, 1],
        );

        Self {
            ctx,
            out_buffer,
            grad_q,
            grad_k,
            grad_v,
            grad_scores,
            forward_graph,
            backward_graph,
        }
    }
}

impl Layer for SelfAttention {
    fn forward(&self) {
        self.forward_graph.execute();
    }
    fn backward(&self) {
        self.backward_graph.execute();
    }
}

use super::traits::Layer;
use crate::Real;
use std::sync::Arc;
use wilupgu::{Backend, Binding, ComputeGraph, Tensor, TensorMode};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AttnScaleMeta {
    seq_len: u32,
    scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct HeadMoveMeta {
    seq_len: u32,
    full_dim: u32,
    head_dim: u32,
    head_offset: u32,
}

pub struct SelfAttention<B: Backend> {
    pub out_buffer: Arc<Tensor<B>>,
    pub forward_graph: ComputeGraph<B>,
    pub backward_graph: ComputeGraph<B>,
}

impl<B: Backend> SelfAttention<B> {
    pub(crate) fn forward_nodes(
        graph: &mut ComputeGraph<B>,
        seq_len: u32,
        dim: u32,
        num_heads: u32,
        q_buf: &Arc<Tensor<B>>,
        k_buf: &Arc<Tensor<B>>,
        v_buf: &Arc<Tensor<B>>,
        out_buffer: &Arc<Tensor<B>>,
        q_heads: &mut Vec<Arc<Tensor<B>>>,
        k_heads: &mut Vec<Arc<Tensor<B>>>,
        v_heads: &mut Vec<Arc<Tensor<B>>>,
        t_scores_heads: &mut Vec<Arc<Tensor<B>>>,
    ) {
        let ctx = q_buf.ctx.clone();
        assert_eq!(dim % num_heads, 0, "dim must be divisible by num_heads");
        let head_dim = dim / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let head_size = (seq_len * head_dim) as usize;
        let scores_size = (seq_len * seq_len) as usize;

        let t_meta_seq = Arc::new(Tensor::init_from_cpu(
            ctx.clone(),
            &[AttnScaleMeta { seq_len, scale }],
        ));

        let grid_seq16 = (seq_len + 15) / 16;
        let grid_hd16 = (head_dim + 15) / 16;
        let grid_softmax = (seq_len + 255) / 256;

        // Shared scratch: reused/overwritten sequentially per head
        let out_head = Arc::new(Tensor::init_from_cpu(
            ctx.clone(),
            &vec![0.0 as Real; head_size],
        ));

        for h in 0..num_heads {
            let head_offset = h * head_dim;
            let t_meta_head = Arc::new(Tensor::init_from_cpu(
                ctx.clone(),
                &[HeadMoveMeta {
                    seq_len,
                    full_dim: dim,
                    head_dim,
                    head_offset,
                }],
            ));

            let q_head = Arc::new(Tensor::init_from_cpu(
                ctx.clone(),
                &vec![0.0 as Real; head_size],
            ));
            let k_head = Arc::new(Tensor::init_from_cpu(
                ctx.clone(),
                &vec![0.0 as Real; head_size],
            ));
            let v_head = Arc::new(Tensor::init_from_cpu(
                ctx.clone(),
                &vec![0.0 as Real; head_size],
            ));
            let t_scores = Arc::new(Tensor::init_from_cpu(
                ctx.clone(),
                &vec![0.0 as Real; scores_size],
            ));

            // ---- gather Q/K/V columns for this head ----
            for (src, dst) in [(q_buf, &q_head), (k_buf, &k_head), (v_buf, &v_head)] {
                graph.add_node(
                    "HeadGather",
                    &[
                        Binding::new(0, &src.buffer, TensorMode::Input),
                        Binding::new(1, &dst.buffer, TensorMode::Output),
                        Binding::new(2, &t_meta_head.buffer, TensorMode::Meta),
                    ],
                    [grid_hd16, grid_seq16, 1],
                );
            }

            // ---- scores = Q_h @ K_h^T, meta {M=seq,N=seq,K=head_dim} ----
            let meta_qkt = Arc::new(Tensor::init_from_cpu(
                ctx.clone(),
                &[seq_len, seq_len, head_dim],
            ));
            graph.add_node(
                "MatMulTrp",
                &[
                    Binding::new(0, &q_head.buffer, TensorMode::Input),
                    Binding::new(1, &k_head.buffer, TensorMode::Input),
                    Binding::new(2, &t_scores.buffer, TensorMode::Output),
                    Binding::new(3, &meta_qkt.buffer, TensorMode::Meta),
                ],
                [grid_seq16, grid_seq16, 1],
            );

            graph.add_node(
                "CausalSoftmax",
                &[
                    Binding::new(0, &t_scores.buffer, TensorMode::InOut),
                    Binding::new(1, &t_meta_seq.buffer, TensorMode::Meta),
                ],
                [grid_softmax, 1, 1],
            );

            // ---- out_h = scores @ V_h, meta {M=seq,N=head_dim,K=seq} ----
            let meta_out = Arc::new(Tensor::init_from_cpu(
                ctx.clone(),
                &[seq_len, head_dim, seq_len],
            ));
            graph.add_node(
                "MatMul",
                &[
                    Binding::new(0, &t_scores.buffer, TensorMode::Input),
                    Binding::new(1, &v_head.buffer, TensorMode::Input),
                    Binding::new(2, &out_head.buffer, TensorMode::Output),
                    Binding::new(3, &meta_out.buffer, TensorMode::Meta),
                ],
                [grid_hd16, grid_seq16, 1],
            );

            graph.add_node(
                "HeadScatter",
                &[
                    Binding::new(0, &out_head.buffer, TensorMode::Input),
                    Binding::new(1, &out_buffer.buffer, TensorMode::Output),
                    Binding::new(2, &t_meta_head.buffer, TensorMode::Meta),
                ],
                [grid_hd16, grid_seq16, 1],
            );

            t_scores_heads.push(t_scores);
            q_heads.push(q_head);
            k_heads.push(k_head);
            v_heads.push(v_head);
        }
    }

    pub fn new(
        ctx: Arc<B>,
        seq_len: u32,
        dim: u32,
        num_heads: u32,
        q_buf: &Arc<Tensor<B>>,
        k_buf: &Arc<Tensor<B>>,
        v_buf: &Arc<Tensor<B>>,
        grad_output: &Arc<Tensor<B>>,
        grad_q: &Arc<Tensor<B>>,
        grad_k: &Arc<Tensor<B>>,
        grad_v: &Arc<Tensor<B>>,
    ) -> Self {
        assert_eq!(dim % num_heads, 0, "dim must be divisible by num_heads");
        let head_dim = dim / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let out_size = (seq_len * dim) as usize;
        let out_buffer = Arc::new(Tensor::init_from_cpu(
            ctx.clone(),
            &vec![0.0 as Real; out_size],
        ));

        let head_size = (seq_len * head_dim) as usize;
        let scores_size = (seq_len * seq_len) as usize;

        let t_meta_seq = Arc::new(Tensor::init_from_cpu(
            ctx.clone(),
            &[AttnScaleMeta { seq_len, scale }],
        ));

        let grid_seq16 = (seq_len + 15) / 16;
        let grid_hd16 = (head_dim + 15) / 16;
        let grid_softmax = (seq_len + 255) / 256;
        let grid_zero_head = ((seq_len * head_dim) + 255) / 256;

        let mut forward_graph = ComputeGraph::new(ctx.clone());
        let mut backward_graph = ComputeGraph::new(ctx.clone());

        let mut t_scores_heads: Vec<Arc<Tensor<B>>> = Vec::with_capacity(num_heads as usize);
        let mut q_heads: Vec<Arc<Tensor<B>>> = Vec::with_capacity(num_heads as usize);
        let mut k_heads: Vec<Arc<Tensor<B>>> = Vec::with_capacity(num_heads as usize);
        let mut v_heads: Vec<Arc<Tensor<B>>> = Vec::with_capacity(num_heads as usize);

        Self::forward_nodes(
            &mut forward_graph,
            seq_len,
            dim,
            num_heads,
            q_buf,
            k_buf,
            v_buf,
            &out_buffer,
            &mut q_heads,
            &mut k_heads,
            &mut v_heads,
            &mut t_scores_heads,
        );

        let grad_out_head = Arc::new(Tensor::init_from_cpu(
            ctx.clone(),
            &vec![0.0 as Real; head_size],
        ));
        let grad_q_head = Arc::new(Tensor::init_from_cpu(
            ctx.clone(),
            &vec![0.0 as Real; head_size],
        ));
        let grad_k_head = Arc::new(Tensor::init_from_cpu(
            ctx.clone(),
            &vec![0.0 as Real; head_size],
        ));
        let grad_v_head = Arc::new(Tensor::init_from_cpu(
            ctx.clone(),
            &vec![0.0 as Real; head_size],
        ));
        let grad_y = Arc::new(Tensor::init_from_cpu(
            ctx.clone(),
            &vec![0.0 as Real; scores_size],
        ));
        let grad_raw = Arc::new(Tensor::init_from_cpu(
            ctx.clone(),
            &vec![0.0 as Real; scores_size],
        ));
        let zero_meta = Arc::new(Tensor::init_from_cpu(ctx.clone(), &[(seq_len * head_dim)]));

        for h in 0..num_heads {
            let head_offset = h * head_dim;
            let t_meta_head = Arc::new(Tensor::init_from_cpu(
                ctx.clone(),
                &[HeadMoveMeta {
                    seq_len,
                    full_dim: dim,
                    head_dim,
                    head_offset,
                }],
            ));

            backward_graph.add_node(
                "HeadGather",
                &[
                    Binding::new(0, &grad_output.buffer, TensorMode::Input),
                    Binding::new(1, &grad_out_head.buffer, TensorMode::Output),
                    Binding::new(2, &t_meta_head.buffer, TensorMode::Meta),
                ],
                [grid_hd16, grid_seq16, 1],
            );

            // dV_h = Y^T @ dOut_h  (accumulating MatMulWeightBwd -- zero first)
            backward_graph.add_node(
                "ZeroTensor",
                &[
                    Binding::new(0, &grad_v_head.buffer, TensorMode::Output),
                    Binding::new(1, &zero_meta.buffer, TensorMode::Meta),
                ],
                [grid_zero_head, 1, 1],
            );
            let meta_dv = Arc::new(Tensor::init_from_cpu(
                ctx.clone(),
                &[seq_len, head_dim, seq_len],
            ));
            backward_graph.add_node(
                "MatMulWeightBwd",
                &[
                    Binding::new(0, &t_scores_heads[h as usize].buffer, TensorMode::Input),
                    Binding::new(1, &grad_out_head.buffer, TensorMode::Input),
                    Binding::new(2, &grad_v_head.buffer, TensorMode::Output),
                    Binding::new(3, &meta_dv.buffer, TensorMode::Meta),
                ],
                [(head_dim + 15) / 16, grid_seq16, 1],
            );

            // dY_h = dOut_h @ V_h^T, meta {M=seq,N=seq,K=head_dim}
            let meta_dy = Arc::new(Tensor::init_from_cpu(
                ctx.clone(),
                &[seq_len, seq_len, head_dim],
            ));
            backward_graph.add_node(
                "MatMulTrp",
                &[
                    Binding::new(0, &grad_out_head.buffer, TensorMode::Input),
                    Binding::new(1, &v_heads[h as usize].buffer, TensorMode::Input),
                    Binding::new(2, &grad_y.buffer, TensorMode::Output),
                    Binding::new(3, &meta_dy.buffer, TensorMode::Meta),
                ],
                [grid_seq16, grid_seq16, 1],
            );

            backward_graph.add_node(
                "SoftmaxBwd",
                &[
                    Binding::new(0, &t_scores_heads[h as usize].buffer, TensorMode::Input),
                    Binding::new(1, &grad_y.buffer, TensorMode::Input),
                    Binding::new(2, &grad_raw.buffer, TensorMode::Output),
                    Binding::new(3, &t_meta_seq.buffer, TensorMode::Meta),
                ],
                [grid_softmax, 1, 1],
            );

            let meta_dq = Arc::new(Tensor::init_from_cpu(
                ctx.clone(),
                &[seq_len, head_dim, seq_len],
            ));
            backward_graph.add_node(
                "MatMul",
                &[
                    Binding::new(0, &grad_raw.buffer, TensorMode::Input),
                    Binding::new(1, &k_heads[h as usize].buffer, TensorMode::Input),
                    Binding::new(2, &grad_q_head.buffer, TensorMode::Output),
                    Binding::new(3, &meta_dq.buffer, TensorMode::Meta),
                ],
                [grid_hd16, grid_seq16, 1],
            );

            backward_graph.add_node(
                "ZeroTensor",
                &[
                    Binding::new(0, &grad_k_head.buffer, TensorMode::Output),
                    Binding::new(1, &zero_meta.buffer, TensorMode::Meta),
                ],
                [grid_zero_head, 1, 1],
            );
            let meta_dk = Arc::new(Tensor::init_from_cpu(
                ctx.clone(),
                &[seq_len, head_dim, seq_len],
            ));
            backward_graph.add_node(
                "MatMulWeightBwd",
                &[
                    Binding::new(0, &grad_raw.buffer, TensorMode::Input),
                    Binding::new(1, &q_heads[h as usize].buffer, TensorMode::Input),
                    Binding::new(2, &grad_k_head.buffer, TensorMode::Output),
                    Binding::new(3, &meta_dk.buffer, TensorMode::Meta),
                ],
                [(head_dim + 15) / 16, grid_seq16, 1],
            );

            for (src, dst) in [
                (&grad_q_head, grad_q),
                (&grad_k_head, grad_k),
                (&grad_v_head, grad_v),
            ] {
                backward_graph.add_node(
                    "HeadScatter",
                    &[
                        Binding::new(0, &src.buffer, TensorMode::Input),
                        Binding::new(1, &dst.buffer, TensorMode::Output),
                        Binding::new(2, &t_meta_head.buffer, TensorMode::Meta),
                    ],
                    [grid_hd16, grid_seq16, 1],
                );
            }
        }

        Self {
            out_buffer,
            forward_graph,
            backward_graph,
        }
    }
}

impl<B: Backend> Layer for SelfAttention<B> {
    fn forward(&self) {
        self.forward_graph.execute();
    }
    fn backward(&self) {
        self.backward_graph.execute();
    }
}

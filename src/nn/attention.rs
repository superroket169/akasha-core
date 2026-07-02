use super::ops;
use super::ops::meta::{AttnScaleMeta, HeadMoveMeta, MatMulMeta};
use super::traits::Layer;
use crate::Real;
use std::sync::Arc;
use wilupgu::{Backend, ComputeGraph, Tensor};

pub struct SelfAttention<B: Backend> {
    pub out_buffer: Arc<Tensor<B>>,
    pub forward_graph: ComputeGraph<B>,
    pub backward_graph: ComputeGraph<B>,
}

impl<B: Backend> SelfAttention<B> {
    #[allow(clippy::too_many_arguments)]
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

        let mut forward_graph = ComputeGraph::new(ctx.clone());
        let mut backward_graph = ComputeGraph::new(ctx.clone());

        let saved = ops::attention::causal_attention(
            &mut forward_graph,
            q_buf,
            k_buf,
            v_buf,
            &out_buffer,
            seq_len,
            dim,
            num_heads,
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

        let attn_shape = AttnScaleMeta { seq_len, scale };

        for h in 0..num_heads {
            let head_move = HeadMoveMeta {
                seq_len,
                full_dim: dim,
                head_dim,
                head_offset: h * head_dim,
            };
            let scores_h = &saved.scores_heads[h as usize];

            ops::head_move::head_gather(
                &mut backward_graph,
                grad_output,
                &grad_out_head,
                head_move,
            );

            // dV_h = Y^T @ dOut_h  (accumulating MatMulWeightBwd -- zero first)
            ops::elementwise::zero(&mut backward_graph, &grad_v_head, seq_len * head_dim);
            ops::matmul::matmul_weight_bwd(
                &mut backward_graph,
                scores_h,
                &grad_out_head,
                &grad_v_head,
                MatMulMeta {
                    m: seq_len,
                    n: head_dim,
                    k: seq_len,
                },
            );

            // dY_h = dOut_h @ V_h^T
            ops::matmul::matmul_trp(
                &mut backward_graph,
                &grad_out_head,
                &saved.v_heads[h as usize],
                &grad_y,
                MatMulMeta {
                    m: seq_len,
                    n: seq_len,
                    k: head_dim,
                },
            );

            ops::attention::softmax_bwd(
                &mut backward_graph,
                scores_h,
                &grad_y,
                &grad_raw,
                attn_shape,
            );

            // dQ_h = dRaw @ K_h
            ops::matmul::matmul(
                &mut backward_graph,
                &grad_raw,
                &saved.k_heads[h as usize],
                &grad_q_head,
                MatMulMeta {
                    m: seq_len,
                    n: head_dim,
                    k: seq_len,
                },
            );

            // dK_h = dRaw^T @ Q_h  (accumulating -- zero first)
            ops::elementwise::zero(&mut backward_graph, &grad_k_head, seq_len * head_dim);
            ops::matmul::matmul_weight_bwd(
                &mut backward_graph,
                &grad_raw,
                &saved.q_heads[h as usize],
                &grad_k_head,
                MatMulMeta {
                    m: seq_len,
                    n: head_dim,
                    k: seq_len,
                },
            );

            for (src, dst) in [
                (&grad_q_head, grad_q),
                (&grad_k_head, grad_k),
                (&grad_v_head, grad_v),
            ] {
                ops::head_move::head_scatter(&mut backward_graph, src, dst, head_move);
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

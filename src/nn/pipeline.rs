use super::add::Add;
use super::attention::SelfAttention;
use super::linear::Linear;
use super::ops;
use super::ops::meta::{HeadMoveMeta, RopeMeta};
use super::rmsnorm::RMSNorm;
use super::silu::SiLU;
use super::traits::Layer;
use super::weights::BlockWeights;
use crate::Real;
use crate::config::ModelConfig;
use std::sync::Arc;
use wilupgu::{Backend, ComputeGraph, Tensor, fuse_compute_graphs};

/// One dim-wide Q/K/V role slice of a fused [seq_len, 3*dim] QKV buffer.
pub(crate) fn qkv_slice(seq_len: u32, dim: u32, role_offset: u32) -> HeadMoveMeta {
    HeadMoveMeta {
        seq_len,
        full_dim: 3 * dim,
        head_dim: dim,
        head_offset: role_offset,
    }
}

pub struct TransformerBlock<B: Backend> {
    pub norm_1: RMSNorm<B>,
    pub qkv_proj: Linear<B>,
    pub q_buf: Arc<Tensor<B>>,
    pub k_buf: Arc<Tensor<B>>,
    pub v_buf: Arc<Tensor<B>>,
    pub qkv_split_forward: ComputeGraph<B>,
    pub out_proj: Linear<B>,
    pub attention: SelfAttention<B>,
    pub add_1: Add<B>,
    pub norm_2: RMSNorm<B>,
    pub ffn_up: Linear<B>,
    pub silu: SiLU<B>,
    pub ffn_down: Linear<B>,
    pub add_2: Add<B>,
    pub grad_input: Arc<Tensor<B>>,
    pub rope_forward: ComputeGraph<B>,
    pub backward_graph: ComputeGraph<B>,
}

impl<B: Backend> TransformerBlock<B> {
    pub fn new(
        ctx: Arc<B>,
        cfg: &ModelConfig,
        bw: &BlockWeights<B>,
        input_tensor: &Arc<Tensor<B>>,
        grad_output: &Arc<Tensor<B>>,
        grad_input: &Arc<Tensor<B>>,
    ) -> Self {
        let ModelConfig {
            dim,
            seq_len,
            num_heads,
            ffn_hidden: hidden_dim,
            ..
        } = *cfg;
        let dim_size = (seq_len * dim) as usize;
        let hidden_size = (seq_len * hidden_dim) as usize;

        let zeros_dim = vec![0.0 as Real; dim_size];
        let zeros_hidden = vec![0.0 as Real; hidden_size];
        let zeros_qkv = vec![0.0 as Real; dim_size * 3];

        let g_add2_a = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros_dim));
        let g_add2_b = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros_dim));
        let g_ffndown_in = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros_hidden));
        let g_silu_in = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros_hidden));
        let g_ffnup_in = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros_dim));
        let g_norm2_in = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros_dim));
        let g_add1_b = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros_dim));
        let g_outproj_in = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros_dim));
        let g_attn_q = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros_dim));
        let g_attn_k = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros_dim));
        let g_attn_v = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros_dim));
        let g_attn_qkv = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros_qkv));
        let g_qkvproj_in = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros_dim));
        let g_norm1_in = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros_dim));
        let grad_input = grad_input.clone();
        let elems = seq_len * dim;

        let norm_1 = RMSNorm::new(
            ctx.clone(),
            dim,
            seq_len,
            &bw.norm_1,
            input_tensor,
            &g_qkvproj_in,
            &g_norm1_in,
        );

        // one [dim, 3*dim] matmul instead of three [dim,dim] matmuls
        let qkv_proj = Linear::new(
            ctx.clone(),
            dim,
            dim * 3,
            seq_len,
            &bw.qkv_proj,
            &norm_1.out_buffer,
            &g_attn_qkv,
            &g_qkvproj_in,
        );

        let q_buf = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros_dim));
        let k_buf = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros_dim));
        let v_buf = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros_dim));
        let mut qkv_split_forward = ComputeGraph::new(ctx.clone());
        for (buf, off) in [(&q_buf, 0), (&k_buf, dim), (&v_buf, 2 * dim)] {
            ops::head_move::head_gather(
                &mut qkv_split_forward,
                &qkv_proj.out_buffer,
                buf,
                qkv_slice(seq_len, dim, off),
            );
        }

        let rope_shape = RopeMeta {
            seq_len,
            dim,
            head_dim: cfg.head_dim(),
        };

        let mut rope_forward = ComputeGraph::new(ctx.clone());
        ops::rope::rope(&mut rope_forward, &q_buf, rope_shape);
        ops::rope::rope(&mut rope_forward, &k_buf, rope_shape);

        let attention = SelfAttention::new(
            ctx.clone(),
            seq_len,
            dim,
            num_heads,
            &q_buf,
            &k_buf,
            &v_buf,
            &g_outproj_in,
            &g_attn_q,
            &g_attn_k,
            &g_attn_v,
        );

        let out_proj = Linear::new(
            ctx.clone(),
            dim,
            dim,
            seq_len,
            &bw.out_proj,
            &attention.out_buffer,
            &g_add1_b,
            &g_outproj_in,
        );

        let add_1 = Add::new(
            ctx.clone(),
            dim * seq_len,
            input_tensor,
            &out_proj.out_buffer,
            &g_add2_a,
            &grad_input,
            &g_add1_b,
        );

        let norm_2 = RMSNorm::new(
            ctx.clone(),
            dim,
            seq_len,
            &bw.norm_2,
            &add_1.in_out_buffer,
            &g_ffnup_in,
            &g_norm2_in,
        );

        let ffn_up = Linear::new(
            ctx.clone(),
            dim,
            hidden_dim,
            seq_len,
            &bw.ffn_up,
            &norm_2.out_buffer,
            &g_silu_in,
            &g_ffnup_in,
        );

        let silu = SiLU::new(
            ctx.clone(),
            (seq_len * hidden_dim) as u32,
            &ffn_up.out_buffer,
            &g_ffndown_in,
            &g_silu_in,
        );

        let ffn_down = Linear::new(
            ctx.clone(),
            hidden_dim,
            dim,
            seq_len,
            &bw.ffn_down,
            &silu.in_out_buffer,
            &g_add2_b,
            &g_ffndown_in,
        );

        let add_2 = Add::new(
            ctx.clone(),
            dim * seq_len,
            &add_1.in_out_buffer,
            &ffn_down.out_buffer,
            grad_output,
            &g_add2_a,
            &g_add2_b,
        );

        let mut barrier_1 = ComputeGraph::new(ctx.clone());
        ops::elementwise::add_inplace_bwd(&mut barrier_1, &add_2.grad_a, &norm_2.grad_input, elems);

        let mut barrier_3 = ComputeGraph::new(ctx.clone());
        ops::elementwise::add_inplace_bwd(&mut barrier_3, &grad_input, &norm_1.grad_input, elems);

        let mut rope_backward = ComputeGraph::new(ctx.clone());
        ops::rope::rope_bwd(&mut rope_backward, &g_attn_q, rope_shape);
        ops::rope::rope_bwd(&mut rope_backward, &g_attn_k, rope_shape);

        // dL/dQ + dL/dK + dL/dV -> one fused grad_output for qkv_proj's backward
        let mut qkv_gather_backward = ComputeGraph::new(ctx.clone());
        for (buf, off) in [(&g_attn_q, 0), (&g_attn_k, dim), (&g_attn_v, 2 * dim)] {
            ops::head_move::head_scatter(
                &mut qkv_gather_backward,
                buf,
                &g_attn_qkv,
                qkv_slice(seq_len, dim, off),
            );
        }

        let backward_graph = fuse_compute_graphs(
            ctx.clone(),
            &[
                &add_2.backward_graph,
                &ffn_down.backward_graph,
                &silu.backward_graph,
                &ffn_up.backward_graph,
                &norm_2.backward_graph,
                &barrier_1,
                &add_1.backward_graph,
                &out_proj.backward_graph,
                &attention.backward_graph,
                &rope_backward,
                &qkv_gather_backward,
                &qkv_proj.backward_graph,
                &norm_1.backward_graph,
                &barrier_3,
            ],
        );

        Self {
            norm_1,
            qkv_proj,
            q_buf,
            k_buf,
            v_buf,
            qkv_split_forward,
            out_proj,
            attention,
            add_1,
            norm_2,
            ffn_up,
            silu,
            ffn_down,
            add_2,
            grad_input,
            rope_forward,
            backward_graph,
        }
    }

    pub fn zero_grad(&self) {
        ops::zero_tensor(&self.norm_1.grad_weight);
        ops::zero_tensor(&self.qkv_proj.grad_weight);
        ops::zero_tensor(&self.out_proj.grad_weight);
        ops::zero_tensor(&self.norm_2.grad_weight);
        ops::zero_tensor(&self.ffn_up.grad_weight);
        ops::zero_tensor(&self.ffn_down.grad_weight);
        self.zero_transient_grads();
    }

    pub fn zero_transient_grads(&self) {
        ops::zero_tensor(&self.add_1.grad_a);
        ops::zero_tensor(&self.add_1.grad_b);
        ops::zero_tensor(&self.add_2.grad_a);
        ops::zero_tensor(&self.add_2.grad_b);
    }
}

impl<B: Backend> Layer for TransformerBlock<B> {
    fn forward(&self) {
        self.norm_1.forward();
        self.qkv_proj.forward();
        self.qkv_split_forward.execute();

        self.rope_forward.execute();
        self.attention.forward();

        self.out_proj.forward();
        self.add_1.forward();

        self.norm_2.forward();
        self.ffn_up.forward();
        self.silu.forward();
        self.ffn_down.forward();

        self.add_2.forward();
    }

    fn backward(&self) {
        self.backward_graph.execute();
    }
}

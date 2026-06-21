use super::add::Add;
use super::attention::SelfAttention;
use super::linear::Linear;
use super::rmsnorm::RMSNorm;
use super::rope::RoPE;
use super::silu::SiLU;
use super::traits::Layer;
use crate::Real;
use filuplex::context::Context;
use filuplex::ops::GpuBuffer;
use std::sync::Arc;

pub struct TransformerBlock {
    // Attention
    pub norm_1: RMSNorm,
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub rope_q: RoPE,
    pub rope_k: RoPE,
    pub attention: SelfAttention,
    pub out_proj: Linear,
    pub add_1: Add,

    // Feed Forward
    pub norm_2: RMSNorm,
    pub ffn_up: Linear,
    pub silu: SiLU,
    pub ffn_down: Linear,
    pub add_2: Add,
}

impl TransformerBlock {
    pub fn new(ctx: Arc<Context>, dim: u32, seq_len: u32, input_buffer: &GpuBuffer) -> Self {
        let dummy_norm_w = vec![1.0 as Real; dim as usize];
        let dummy_w_proj = vec![0.01 as Real; (dim * dim) as usize];
        let dummy_w_up = vec![0.01 as Real; (dim * (dim * 4)) as usize];
        let dummy_w_down = vec![0.01 as Real; ((dim * 4) * dim) as usize];
        let m = seq_len;

        let dummy_grad_dim = GpuBuffer::from_cpu(&vec![0.0 as Real; (m * dim) as usize], &ctx);
        let dummy_grad_4dim = GpuBuffer::from_cpu(&vec![0.0 as Real; (m * dim * 4) as usize], &ctx);
        // ------ Attention --------
        let norm_1 = RMSNorm::new(
            ctx.clone(),
            dim,
            1,
            &dummy_norm_w,
            input_buffer,
            &dummy_grad_dim,
        );

        // Q, K, V projs
        let q_proj = Linear::new(
            ctx.clone(),
            dim,
            dim,
            &dummy_w_proj,
            &norm_1.out_buffer,
            &dummy_grad_dim,
        );
        let k_proj = Linear::new(
            ctx.clone(),
            dim,
            dim,
            &dummy_w_proj,
            &norm_1.out_buffer,
            &dummy_grad_dim,
        );
        let v_proj = Linear::new(
            ctx.clone(),
            dim,
            dim,
            &dummy_w_proj,
            &norm_1.out_buffer,
            &dummy_grad_dim,
        );

        // RoPE
        let rope_q = RoPE::new(ctx.clone(), dim, &q_proj.out_buffer, &dummy_grad_dim);
        let rope_k = RoPE::new(ctx.clone(), dim, &k_proj.out_buffer, &dummy_grad_dim);

        // Self Attention
        let attention = SelfAttention::new(
            ctx.clone(),
            seq_len,
            dim,
            &q_proj.out_buffer,
            &k_proj.out_buffer,
            &v_proj.out_buffer,
            &dummy_grad_dim,
        );

        let out_proj = Linear::new(
            ctx.clone(),
            dim,
            dim,
            &dummy_w_proj,
            &attention.out_buffer,
            &dummy_grad_dim,
        );

        // Residual Add
        let add_1 = Add::new(
            ctx.clone(),
            dim * seq_len,
            input_buffer,
            &out_proj.out_buffer,
            &dummy_grad_dim,
        );

        // FNN
        let norm_2 = RMSNorm::new(
            ctx.clone(),
            dim,
            1,
            &dummy_norm_w,
            &add_1.in_out_buffer,
            &dummy_grad_dim,
        );
        let ffn_up = Linear::new(
            ctx.clone(),
            dim,
            dim * 4,
            &dummy_w_up,
            &norm_2.out_buffer,
            &dummy_grad_4dim,
        );
        let silu = SiLU::new(
            ctx.clone(),
            dim * 4 * seq_len,
            &ffn_up.out_buffer,
            &dummy_grad_4dim,
        );
        let ffn_down = Linear::new(
            ctx.clone(),
            dim * 4,
            dim,
            &dummy_w_down,
            &silu.in_out_buffer,
            &dummy_grad_dim,
        );

        let add_2 = Add::new(
            ctx.clone(),
            dim * seq_len,
            &add_1.in_out_buffer,
            &ffn_down.out_buffer,
            &dummy_grad_dim,
        );

        Self {
            norm_1,
            q_proj,
            k_proj,
            v_proj,
            rope_q,
            rope_k,
            attention,
            out_proj,
            add_1,
            norm_2,
            ffn_up,
            silu,
            ffn_down,
            add_2,
        }
    }
}

impl Layer for TransformerBlock {
    fn forward(&self) {
        self.norm_1.forward();
        self.q_proj.forward();
        self.k_proj.forward();
        self.v_proj.forward();

        self.rope_q.forward();
        self.rope_k.forward();
        self.attention.forward();
        self.out_proj.forward();
        self.add_1.forward();

        self.norm_2.forward();
        self.ffn_up.forward();
        self.silu.forward();
        self.ffn_down.forward();
    }

    fn backward(&self) {
        self.add_2.backward();
        self.ffn_down.backward();
        self.silu.backward();
        self.ffn_up.backward();
        self.norm_2.backward();

        self.add_1.backward();
        self.out_proj.backward();
        self.attention.backward();
        self.rope_k.backward();
        self.rope_q.backward();
        self.v_proj.backward();
        self.k_proj.backward();
        self.q_proj.backward();
        self.norm_1.backward();
    }
}

use super::add::Add;
use super::attention::SelfAttention;
use super::linear::Linear;
use super::rmsnorm::RMSNorm;
use super::rope::RoPE;
use super::silu::SiLU;
use super::traits::Layer;
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
    // NOTE ağırlıklar ileride weights.bin den okunacak. şimdilik otomatik dummy oluşturup onun
    // üzerinden gidiyoruz
    pub fn new(ctx: Arc<Context>, dim: u32, seq_len: u32, input_buffer: &GpuBuffer) -> Self {
        let dummy_norm_w = vec![1.0f32; dim as usize];
        let dummy_w_proj = vec![0.01f32; (dim * dim) as usize];
        let dummy_w_up = vec![0.01f32; (dim * (dim * 4)) as usize];
        let dummy_w_down = vec![0.01f32; ((dim * 4) * dim) as usize];

        // ------ Attention --------
        let norm_1 = RMSNorm::new(ctx.clone(), dim, &dummy_norm_w, input_buffer);

        // Q, K, V projs
        let q_proj = Linear::new(ctx.clone(), dim, dim, &dummy_w_proj, &norm_1.out_buffer);
        let k_proj = Linear::new(ctx.clone(), dim, dim, &dummy_w_proj, &norm_1.out_buffer);
        let v_proj = Linear::new(ctx.clone(), dim, dim, &dummy_w_proj, &norm_1.out_buffer);

        // RoPE
        let rope_q = RoPE::new(ctx.clone(), dim, &q_proj.out_buffer);
        let rope_k = RoPE::new(ctx.clone(), dim, &k_proj.out_buffer);

        // Self Attention
        let attention = SelfAttention::new(
            ctx.clone(),
            seq_len,
            dim,
            &q_proj.out_buffer,
            &k_proj.out_buffer,
            &v_proj.out_buffer,
        );

        let out_proj = Linear::new(ctx.clone(), dim, dim, &dummy_w_proj, &attention.out_buffer);

        // Residual Add
        let add_1 = Add::new(
            ctx.clone(),
            dim * seq_len,
            input_buffer,
            &out_proj.out_buffer,
        );

        // FNN
        let norm_2 = RMSNorm::new(ctx.clone(), dim, &dummy_norm_w, input_buffer);
        let ffn_up = Linear::new(ctx.clone(), dim, dim * 4, &dummy_w_up, &norm_2.out_buffer);
        let silu = SiLU::new(ctx.clone(), dim * 4 * seq_len, &ffn_up.out_buffer);
        let ffn_down = Linear::new(ctx.clone(), dim * 4, dim, &dummy_w_down, &ffn_up.out_buffer);

        let add_2 = Add::new(
            ctx.clone(),
            dim * seq_len,
            input_buffer,
            &ffn_down.out_buffer,
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
    fn forward(&self, _input: &GpuBuffer) -> GpuBuffer {
        // Attention
        self.norm_1.forward(&self.norm_1.out_buffer);
        self.q_proj.forward(&self.q_proj.out_buffer);
        self.k_proj.forward(&self.k_proj.out_buffer);
        self.v_proj.forward(&self.v_proj.out_buffer);

        self.rope_q.forward(&self.rope_q.in_out_buffer);
        self.rope_k.forward(&self.rope_k.in_out_buffer);

        self.attention.forward();
        self.out_proj.forward(&self.out_proj.out_buffer);

        self.add_1.forward();

        // Feed Forward
        self.norm_2.forward(&self.norm_2.out_buffer);
        self.ffn_up.forward(&self.ffn_up.out_buffer);
        self.silu.forward(&self.silu.in_out_buffer);
        self.ffn_down.forward(&self.ffn_down.out_buffer);

        let final_output = self.add_2.forward();

        final_output
    }
}

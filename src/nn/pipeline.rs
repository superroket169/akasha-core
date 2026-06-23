use super::add::Add;
use super::attention::SelfAttention;
use super::linear::Linear;
use super::rmsnorm::RMSNorm;
use super::traits::Layer;
use crate::Real;
use std::sync::Arc;
use wilupgu::context::WgpuContext;
use wilupgu::tensor::Tensor;

pub struct TransformerBlock {
    pub norm_1: RMSNorm,
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub out_proj: Linear,
    pub attention: SelfAttention,
    pub add_1: Add,
    pub norm_2: RMSNorm,
    pub ffn_up: Linear,
    pub ffn_down: Linear,
    pub add_2: Add,
}

impl TransformerBlock {
    pub fn new(ctx: Arc<WgpuContext>, dim: u32, seq_len: u32, input_tensor: &Arc<Tensor>) -> Self {
        let out_size = (seq_len * dim) as usize;
        let dummy_grad = Arc::new(Tensor::init_from_cpu(
            ctx.clone(),
            &vec![0.0 as Real; out_size],
        ));
        let dummy_w = vec![0.01 as Real; (dim * dim) as usize];
        let dummy_norm_w = vec![1.0 as Real; dim as usize];

        let norm_1 = RMSNorm::new(
            ctx.clone(),
            dim,
            seq_len,
            &dummy_norm_w,
            input_tensor,
            &dummy_grad,
        );

        let q_proj = Linear::new(
            ctx.clone(),
            dim,
            dim,
            seq_len,
            &dummy_w,
            &norm_1.out_buffer,
            &dummy_grad,
        );
        let k_proj = Linear::new(
            ctx.clone(),
            dim,
            dim,
            seq_len,
            &dummy_w,
            &norm_1.out_buffer,
            &dummy_grad,
        );
        let v_proj = Linear::new(
            ctx.clone(),
            dim,
            dim,
            seq_len,
            &dummy_w,
            &norm_1.out_buffer,
            &dummy_grad,
        );

        let attention = SelfAttention::new(
            ctx.clone(),
            seq_len,
            dim,
            &q_proj.out_buffer,
            &k_proj.out_buffer,
            &v_proj.out_buffer,
            &dummy_grad,
        );

        let out_proj = Linear::new(
            ctx.clone(),
            dim,
            dim,
            seq_len,
            &dummy_w,
            &attention.out_buffer,
            &dummy_grad,
        );

        let add_1 = Add::new(
            ctx.clone(),
            dim * seq_len,
            input_tensor,
            &out_proj.out_buffer,
            &dummy_grad,
        );

        let norm_2 = RMSNorm::new(
            ctx.clone(),
            dim,
            seq_len,
            &dummy_norm_w,
            &add_1.in_out_buffer,
            &dummy_grad,
        );

        let hidden_dim = dim * 4;
        let dummy_ffn_w = vec![0.01 as Real; (dim * hidden_dim) as usize];
        let dummy_ffn_down_w = vec![0.01 as Real; (hidden_dim * dim) as usize];

        let ffn_up = Linear::new(
            ctx.clone(),
            dim,
            hidden_dim,
            seq_len,
            &dummy_ffn_w,
            &norm_2.out_buffer,
            &dummy_grad,
        );
        let ffn_down = Linear::new(
            ctx.clone(),
            hidden_dim,
            dim,
            seq_len,
            &dummy_ffn_down_w,
            &ffn_up.out_buffer,
            &dummy_grad,
        );

        let add_2 = Add::new(
            ctx.clone(),
            dim * seq_len,
            &add_1.in_out_buffer,
            &ffn_down.out_buffer,
            &dummy_grad,
        );

        Self {
            norm_1,
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            attention,
            add_1,
            norm_2,
            ffn_up,
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

        self.attention.forward();

        self.out_proj.forward();
        self.add_1.forward();

        self.norm_2.forward();
        self.ffn_up.forward();
        self.ffn_down.forward();

        self.add_2.forward();
    }

    fn backward(&self) {
        self.add_2.backward();
        self.ffn_down.backward();
        self.ffn_up.backward();
        self.norm_2.backward();
        self.add_1.backward();
        self.out_proj.backward();
        self.attention.backward();
        self.v_proj.backward();
        self.k_proj.backward();
        self.q_proj.backward();
        self.norm_1.backward();
    }
}

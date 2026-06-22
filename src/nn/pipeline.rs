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
    pub attention: SelfAttention,
    pub norm_2: RMSNorm,
    pub ffn_up: Linear,
    pub ffn_down: Linear,
}

impl TransformerBlock {
    pub fn new(ctx: Arc<WgpuContext>, dim: u32, seq_len: u32, input_tensor: &Arc<Tensor>) -> Self {
        let out_size = (seq_len * dim) as usize;
        let dummy_grad = vec![0.0 as Real; out_size];
        let t_dummy_grad = Arc::new(Tensor::init_from_cpu(ctx.clone(), &dummy_grad));

        // 1. Pre-Norm
        let dummy_norm_w = vec![1.0 as Real; dim as usize];
        let norm_1 = RMSNorm::new(
            ctx.clone(),
            dim,
            seq_len,
            &dummy_norm_w,
            input_tensor,
            &t_dummy_grad,
        );

        let qkv_data = vec![0.0 as Real; out_size];
        let t_q = Arc::new(Tensor::init_from_cpu(ctx.clone(), &qkv_data));
        let t_k = Arc::new(Tensor::init_from_cpu(ctx.clone(), &qkv_data));
        let t_v = Arc::new(Tensor::init_from_cpu(ctx.clone(), &qkv_data));

        let attention =
            SelfAttention::new(ctx.clone(), seq_len, dim, &t_q, &t_k, &t_v, &t_dummy_grad);

        let norm_2 = RMSNorm::new(
            ctx.clone(),
            dim,
            seq_len,
            &dummy_norm_w,
            &attention.out_buffer,
            &t_dummy_grad,
        );

        let hidden_dim = dim * 4;
        let dummy_ffn_w = vec![0.01 as Real; (dim * hidden_dim) as usize];
        let ffn_up = Linear::new(
            ctx.clone(),
            dim,
            hidden_dim,
            &dummy_ffn_w,
            &norm_2.out_buffer,
            &t_dummy_grad,
        );

        let dummy_ffn_down_w = vec![0.01 as Real; (hidden_dim * dim) as usize];
        let ffn_down = Linear::new(
            ctx.clone(),
            hidden_dim,
            dim,
            &dummy_ffn_down_w,
            &ffn_up.out_buffer,
            &t_dummy_grad,
        );

        Self {
            norm_1,
            attention,
            norm_2,
            ffn_up,
            ffn_down,
        }
    }
}

impl Layer for TransformerBlock {
    fn forward(&self) {
        self.norm_1.forward();
        self.attention.forward();
        self.norm_2.forward();
        self.ffn_up.forward();
        self.ffn_down.forward();
    }

    fn backward(&self) {
        self.ffn_down.backward();
        self.ffn_up.backward();
        self.norm_2.backward();
        self.attention.backward();
        self.norm_1.backward();
    }
}

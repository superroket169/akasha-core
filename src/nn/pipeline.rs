use super::add::Add;
use super::attention::SelfAttention;
use super::linear::Linear;
use super::rmsnorm::RMSNorm;
use super::silu::SiLU;
use super::traits::Layer;
use crate::Real;
use std::sync::Arc;
use wilupgu::context::WgpuContext;
use wilupgu::tensor::Tensor;

// Helpers

fn zero_tensor(t: &Tensor) {
    let len = (t.size / std::mem::size_of::<Real>() as u64) as usize;
    t.copy_from_cpu(&vec![0.0 as Real; len]);
}

fn cpu_sum2_into(dest: &Tensor, a: &Tensor, b: &Tensor) {
    let a_cpu: Vec<Real> = a.to_cpu();
    let b_cpu: Vec<Real> = b.to_cpu();
    let summed: Vec<Real> = a_cpu.iter().zip(b_cpu.iter()).map(|(x, y)| x + y).collect();
    dest.copy_from_cpu(&summed);
}

fn cpu_sum3_into(dest: &Tensor, a: &Tensor, b: &Tensor, c: &Tensor) {
    let a_cpu: Vec<Real> = a.to_cpu();
    let b_cpu: Vec<Real> = b.to_cpu();
    let c_cpu: Vec<Real> = c.to_cpu();
    let summed: Vec<Real> = a_cpu
        .iter()
        .zip(b_cpu.iter())
        .zip(c_cpu.iter())
        .map(|((x, y), z)| x + y + z)
        .collect();
    dest.copy_from_cpu(&summed);
}

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
    pub silu: SiLU,
    pub ffn_down: Linear,
    pub add_2: Add,
    pub grad_input: Arc<Tensor>,
    g_add1_out: Arc<Tensor>,
    g_norm1_out: Arc<Tensor>,
}

impl TransformerBlock {
    pub fn new(
        ctx: Arc<WgpuContext>,
        dim: u32,
        seq_len: u32,
        input_tensor: &Arc<Tensor>,
        grad_output: &Arc<Tensor>,
        grad_input: &Arc<Tensor>,
    ) -> Self {
        let dim_size = (seq_len * dim) as usize;
        let hidden_dim = dim * 4;
        let hidden_size = (seq_len * hidden_dim) as usize;

        let zeros_dim = vec![0.0 as Real; dim_size];
        let zeros_hidden = vec![0.0 as Real; hidden_size];

        let g_add2_a = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros_dim));
        let g_add2_b = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros_dim));
        let g_ffndown_in = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros_hidden));
        let g_silu_in = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros_hidden));
        let g_ffnup_in = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros_dim));
        let g_norm2_in = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros_dim));
        let g_add1_out = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros_dim));
        let g_add1_a = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros_dim));
        let g_add1_b = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros_dim));
        let g_outproj_in = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros_dim));
        let g_attn_q = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros_dim));
        let g_attn_k = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros_dim));
        let g_attn_v = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros_dim));
        let g_qproj_in = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros_dim));
        let g_kproj_in = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros_dim));
        let g_vproj_in = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros_dim));
        let g_norm1_out = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros_dim));
        let g_norm1_in = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros_dim));
        let grad_input = grad_input.clone();

        let dummy_w = vec![0.01 as Real; (dim * dim) as usize];
        let dummy_norm_w = vec![1.0 as Real; dim as usize];

        let norm_1 = RMSNorm::new(
            ctx.clone(),
            dim,
            seq_len,
            &dummy_norm_w,
            input_tensor,
            &g_norm1_out,
            &g_norm1_in,
        );

        let q_proj = Linear::new(
            ctx.clone(),
            dim,
            dim,
            seq_len,
            &dummy_w,
            &norm_1.out_buffer,
            &g_attn_q,
            &g_qproj_in,
        );
        let k_proj = Linear::new(
            ctx.clone(),
            dim,
            dim,
            seq_len,
            &dummy_w,
            &norm_1.out_buffer,
            &g_attn_k,
            &g_kproj_in,
        );
        let v_proj = Linear::new(
            ctx.clone(),
            dim,
            dim,
            seq_len,
            &dummy_w,
            &norm_1.out_buffer,
            &g_attn_v,
            &g_vproj_in,
        );

        let attention = SelfAttention::new(
            ctx.clone(),
            seq_len,
            dim,
            &q_proj.out_buffer,
            &k_proj.out_buffer,
            &v_proj.out_buffer,
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
            &dummy_w,
            &attention.out_buffer,
            &g_add1_b,
            &g_outproj_in,
        );

        let add_1 = Add::new(
            ctx.clone(),
            dim * seq_len,
            input_tensor,
            &out_proj.out_buffer,
            &g_add1_out,
            &g_add1_a,
            &g_add1_b,
        );

        let norm_2 = RMSNorm::new(
            ctx.clone(),
            dim,
            seq_len,
            &dummy_norm_w,
            &add_1.in_out_buffer,
            &g_ffnup_in,
            &g_norm2_in,
        );

        let dummy_ffn_w = vec![0.01 as Real; (dim * hidden_dim) as usize];
        let dummy_ffn_down_w = vec![0.01 as Real; (hidden_dim * dim) as usize];

        let ffn_up = Linear::new(
            ctx.clone(),
            dim,
            hidden_dim,
            seq_len,
            &dummy_ffn_w,
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
            &dummy_ffn_down_w,
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
            silu,
            ffn_down,
            add_2,
            grad_input,
            g_add1_out,
            g_norm1_out,
        }
    }

    pub fn zero_grad(&self) {
        zero_tensor(&self.norm_1.grad_weight);
        zero_tensor(&self.norm_2.grad_weight);
        zero_tensor(&self.add_1.grad_a);
        zero_tensor(&self.add_1.grad_b);
        zero_tensor(&self.add_2.grad_a);
        zero_tensor(&self.add_2.grad_b);
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
        self.silu.forward();
        self.ffn_down.forward();

        self.add_2.forward();
    }

    fn backward(&self) {
        self.add_2.backward();
        self.ffn_down.backward();
        self.silu.backward();
        self.ffn_up.backward();
        self.norm_2.backward();

        cpu_sum2_into(
            &self.g_add1_out,
            &self.add_2.grad_a,
            &self.norm_2.grad_input,
        );

        self.add_1.backward();
        self.out_proj.backward();
        self.attention.backward();
        self.v_proj.backward();
        self.k_proj.backward();
        self.q_proj.backward();

        cpu_sum3_into(
            &self.g_norm1_out,
            &self.q_proj.grad_input,
            &self.k_proj.grad_input,
            &self.v_proj.grad_input,
        );

        self.norm_1.backward();

        cpu_sum2_into(
            &self.grad_input,
            &self.add_1.grad_a,
            &self.norm_1.grad_input,
        );
    }
}

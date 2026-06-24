use super::add::Add;
use super::attention::SelfAttention;
use super::init::{random_normal_vec, xavier_std};
use super::linear::Linear;
use super::rmsnorm::RMSNorm;
use super::silu::SiLU;
use super::traits::Layer;
use crate::Real;
use std::sync::Arc;
use wilupgu::context::WgpuContext;
use wilupgu::graph::{ComputeGraph, TensorBind, TensorMode, fuse_compute_graphs};
use wilupgu::nn::shaders::BuiltInShader;
use wilupgu::tensor::Tensor;

// Helpers

fn zero_tensor(t: &Tensor) {
    let len = (t.size / std::mem::size_of::<Real>() as u64) as usize;
    t.copy_from_cpu(&vec![0.0 as Real; len]);
}

fn add_inplace_node(
    graph: &mut ComputeGraph,
    target: &Arc<Tensor>,
    source: &Arc<Tensor>,
    elems: u32,
) {
    let shader = BuiltInShader::BwdAddInplace.get_def();
    graph.add_node(
        &shader,
        &[
            TensorBind {
                binding: 0,
                tensor: target,
                mode: TensorMode::InOut,
            },
            TensorBind {
                binding: 1,
                tensor: source,
                mode: TensorMode::Input,
            },
        ],
        [(elems + 255) / 256, 1, 1],
    );
}

fn add_rope_node(
    graph: &mut ComputeGraph,
    shader: BuiltInShader,
    vec: &Arc<Tensor>,
    meta: &Arc<Tensor>,
    dim: u32,
    seq_len: u32,
) {
    let shader_def = shader.get_def();
    graph.add_node(
        &shader_def,
        &[
            TensorBind {
                binding: 0,
                tensor: vec,
                mode: TensorMode::InOut,
            },
            TensorBind {
                binding: 1,
                tensor: meta,
                mode: TensorMode::Meta,
            },
        ],
        [(dim / 2 + 15) / 16, (seq_len + 15) / 16, 1],
    );
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
    pub rope_forward: ComputeGraph,
    pub backward_graph: ComputeGraph,
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
        let g_add1_b = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros_dim));
        let g_outproj_in = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros_dim));
        let g_attn_q = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros_dim));
        let g_attn_k = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros_dim));
        let g_attn_v = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros_dim));
        let g_qproj_in = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros_dim));
        let g_kproj_in = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros_dim));
        let g_vproj_in = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros_dim));
        let g_norm1_in = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros_dim));
        let grad_input = grad_input.clone();
        let elems = seq_len * dim;

        let norm_1_w = random_normal_vec(dim as usize, 1.0, 0.02);

        let norm_1 = RMSNorm::new(
            ctx.clone(),
            dim,
            seq_len,
            &norm_1_w,
            input_tensor,
            &g_qproj_in,
            &g_norm1_in,
        );

        let proj_std = xavier_std(dim);
        let q_w = random_normal_vec((dim * dim) as usize, 0.0, proj_std);
        let k_w = random_normal_vec((dim * dim) as usize, 0.0, proj_std);
        let v_w = random_normal_vec((dim * dim) as usize, 0.0, proj_std);
        let out_w = random_normal_vec((dim * dim) as usize, 0.0, proj_std);

        let q_proj = Linear::new(
            ctx.clone(),
            dim,
            dim,
            seq_len,
            &q_w,
            &norm_1.out_buffer,
            &g_attn_q,
            &g_qproj_in,
        );
        let k_proj = Linear::new(
            ctx.clone(),
            dim,
            dim,
            seq_len,
            &k_w,
            &norm_1.out_buffer,
            &g_attn_k,
            &g_kproj_in,
        );
        let v_proj = Linear::new(
            ctx.clone(),
            dim,
            dim,
            seq_len,
            &v_w,
            &norm_1.out_buffer,
            &g_attn_v,
            &g_vproj_in,
        );

        let rope_meta_data = vec![seq_len, dim, dim];
        let t_meta_rope = Arc::new(Tensor::init_from_cpu(ctx.clone(), &rope_meta_data));

        let mut rope_forward = ComputeGraph::new(ctx.clone());
        add_rope_node(
            &mut rope_forward,
            BuiltInShader::RoPE,
            &q_proj.out_buffer,
            &t_meta_rope,
            dim,
            seq_len,
        );
        add_rope_node(
            &mut rope_forward,
            BuiltInShader::RoPE,
            &k_proj.out_buffer,
            &t_meta_rope,
            dim,
            seq_len,
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
            &out_w,
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

        let norm_2_w = random_normal_vec(dim as usize, 1.0, 0.02);
        let norm_2 = RMSNorm::new(
            ctx.clone(),
            dim,
            seq_len,
            &norm_2_w,
            &add_1.in_out_buffer,
            &g_ffnup_in,
            &g_norm2_in,
        );

        let ffn_up_w = random_normal_vec((dim * hidden_dim) as usize, 0.0, xavier_std(dim));
        let ffn_down_w =
            random_normal_vec((hidden_dim * dim) as usize, 0.0, xavier_std(hidden_dim));

        let ffn_up = Linear::new(
            ctx.clone(),
            dim,
            hidden_dim,
            seq_len,
            &ffn_up_w,
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
            &ffn_down_w,
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
        add_inplace_node(&mut barrier_1, &add_2.grad_a, &norm_2.grad_input, elems);

        let mut barrier_2 = ComputeGraph::new(ctx.clone());
        add_inplace_node(
            &mut barrier_2,
            &q_proj.grad_input,
            &k_proj.grad_input,
            elems,
        );
        add_inplace_node(
            &mut barrier_2,
            &q_proj.grad_input,
            &v_proj.grad_input,
            elems,
        );

        let mut barrier_3 = ComputeGraph::new(ctx.clone());
        add_inplace_node(&mut barrier_3, &grad_input, &norm_1.grad_input, elems);

        let mut rope_backward = ComputeGraph::new(ctx.clone());
        add_rope_node(
            &mut rope_backward,
            BuiltInShader::RoPEBwd,
            &g_attn_q,
            &t_meta_rope,
            dim,
            seq_len,
        );
        add_rope_node(
            &mut rope_backward,
            BuiltInShader::RoPEBwd,
            &g_attn_k,
            &t_meta_rope,
            dim,
            seq_len,
        );

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
                &v_proj.backward_graph,
                &k_proj.backward_graph,
                &q_proj.backward_graph,
                &barrier_2,
                &norm_1.backward_graph,
                &barrier_3,
            ],
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
            rope_forward,
            backward_graph,
        }
    }

    pub fn zero_grad(&self) {
        zero_tensor(&self.norm_1.grad_weight);
        zero_tensor(&self.q_proj.grad_weight);
        zero_tensor(&self.k_proj.grad_weight);
        zero_tensor(&self.v_proj.grad_weight);
        zero_tensor(&self.out_proj.grad_weight);
        zero_tensor(&self.norm_2.grad_weight);
        zero_tensor(&self.ffn_up.grad_weight);
        zero_tensor(&self.ffn_down.grad_weight);
        self.zero_transient_grads();
    }

    pub fn zero_transient_grads(&self) {
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

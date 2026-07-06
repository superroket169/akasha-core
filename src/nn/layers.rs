//! Trainer-only layer wrappers: each struct owns its grad buffers and its
//! fused-into-`Trainer` forward/backward graphs. Inference never touches
//! these -- it reads `ModelWeights` directly.

use super::ops;
use super::ops::GraphBuilder;
use super::ops::meta::{
    CrossEntropyMeta, EmbeddingMeta, FlashAttnMeta, HeadMoveMeta, MatMulMeta, NormMeta, RopeMeta,
};
use super::weights::BlockWeights;
use crate::Real;
use crate::config::ModelConfig;
use std::sync::Arc;
use wilupgu::{Backend, ComputeGraph, Tensor, fuse_compute_graphs};

pub trait Layer {
    fn forward(&self);
    fn backward(&self);
}

pub struct Linear<B: Backend> {
    pub weight: Arc<Tensor<B>>,
    pub out_buffer: Arc<Tensor<B>>,
    pub grad_weight: Arc<Tensor<B>>,
    pub grad_input: Arc<Tensor<B>>,
    pub forward_graph: ComputeGraph<B>,
    pub backward_graph: ComputeGraph<B>,
}

impl<B: Backend> Linear<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ctx: Arc<B>,
        in_features: u32,
        out_features: u32,
        seq_len: u32,
        weight: &Arc<Tensor<B>>,
        input_buffer: &Arc<Tensor<B>>,
        grad_output: &Arc<Tensor<B>>,
        grad_input: &Arc<Tensor<B>>,
    ) -> Self {
        let weight = weight.clone();

        let out_size = (seq_len * out_features) as usize;
        let zero_out = vec![0.0 as Real; out_size];
        let out_buffer = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zero_out));

        let zero_grad_w = vec![0.0 as Real; (in_features * out_features) as usize];
        let grad_weight = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zero_grad_w));

        let grad_input = grad_input.clone();

        let mut forward_graph = ComputeGraph::new(ctx.clone());
        let mut gb = GraphBuilder::train(&mut forward_graph);
        ops::matmul(
            &mut gb,
            input_buffer,
            &weight,
            &out_buffer,
            MatMulMeta {
                m: seq_len,
                n: out_features,
                k: in_features,
            },
        );

        let mut backward_graph = ComputeGraph::new(ctx.clone());
        let mut gb = GraphBuilder::train(&mut backward_graph);
        ops::matmul_weight_bwd(
            &mut gb,
            input_buffer,
            grad_output,
            &grad_weight,
            MatMulMeta {
                m: seq_len,
                n: out_features,
                k: in_features,
            },
        );
        ops::matmul_trp(
            &mut gb,
            grad_output,
            &weight,
            &grad_input,
            MatMulMeta {
                m: seq_len,
                n: in_features,
                k: out_features,
            },
        );

        Self {
            weight,
            out_buffer,
            grad_weight,
            grad_input,
            forward_graph,
            backward_graph,
        }
    }
}

impl<B: Backend> Layer for Linear<B> {
    fn forward(&self) {
        self.forward_graph.execute();
    }
    fn backward(&self) {
        self.backward_graph.execute();
    }
}

pub struct RMSNorm<B: Backend> {
    pub weight: Arc<Tensor<B>>,
    pub out_buffer: Arc<Tensor<B>>,
    pub grad_weight: Arc<Tensor<B>>,
    pub grad_input: Arc<Tensor<B>>,
    pub rsqrt_cache: Arc<Tensor<B>>,
    pub forward_graph: ComputeGraph<B>,
    pub backward_graph: ComputeGraph<B>,
}

impl<B: Backend> RMSNorm<B> {
    pub fn new(
        ctx: Arc<B>,
        dim: u32,
        seq_len: u32,
        weight: &Arc<Tensor<B>>,
        input_buffer: &Arc<Tensor<B>>,
        grad_output: &Arc<Tensor<B>>,
        grad_input: &Arc<Tensor<B>>,
    ) -> Self {
        assert_eq!(
            weight.size,
            dim as u64 * std::mem::size_of::<Real>() as u64,
            "RMSNorm weight size mismatch!"
        );

        let shape = NormMeta {
            seq_len,
            size: dim,
            eps: 1e-5,
        };

        let weight = weight.clone();
        let zero_dim = vec![0.0 as Real; dim as usize];
        let grad_weight = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zero_dim));

        let out_size = (seq_len * dim) as usize;
        let zero_out = vec![0.0 as Real; out_size];
        let out_buffer = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zero_out));
        let grad_input = grad_input.clone();

        let rsqrt_cache = Arc::new(Tensor::init_from_cpu(
            ctx.clone(),
            &vec![0.0 as Real; seq_len as usize],
        ));

        let mut forward_graph = ComputeGraph::new(ctx.clone());
        let mut gb = GraphBuilder::train(&mut forward_graph);
        ops::rmsnorm(&mut gb, input_buffer, &weight, &out_buffer, shape);

        let mut backward_graph = ComputeGraph::new(ctx.clone());
        let mut gb = GraphBuilder::train(&mut backward_graph);
        ops::rmsnorm_bwd(
            &mut gb,
            grad_output,
            input_buffer,
            &weight,
            &grad_input,
            &rsqrt_cache,
            &grad_weight,
            shape,
        );

        Self {
            weight,
            out_buffer,
            grad_weight,
            grad_input,
            rsqrt_cache,
            forward_graph,
            backward_graph,
        }
    }
}

impl<B: Backend> Layer for RMSNorm<B> {
    fn forward(&self) {
        self.forward_graph.execute();
    }
    fn backward(&self) {
        self.backward_graph.execute();
    }
}

pub struct Embedding<B: Backend> {
    pub table: Arc<Tensor<B>>,
    pub grad_table: Arc<Tensor<B>>,
    pub out_buffer: Arc<Tensor<B>>,
    pub forward_graph: ComputeGraph<B>,
    pub backward_graph: ComputeGraph<B>,
}

impl<B: Backend> Embedding<B> {
    pub fn new(
        ctx: Arc<B>,
        vocab_size: u32,
        dim: u32,
        seq_len: u32,
        table: &Arc<Tensor<B>>,
        tokens_buffer: &Arc<Tensor<B>>,
        grad_output: &Arc<Tensor<B>>,
    ) -> Self {
        assert_eq!(
            table.size,
            (vocab_size * dim) as u64 * std::mem::size_of::<Real>() as u64,
            "Dict size mismatch!"
        );

        let shape = EmbeddingMeta {
            vocab_size,
            dim,
            seq_len,
        };

        let table = table.clone();
        let zero_table = vec![0.0 as Real; (vocab_size * dim) as usize];
        let grad_table = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zero_table));

        let out_size = (seq_len * dim) as usize;
        let zero_out = vec![0.0 as Real; out_size];
        let out_buffer = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zero_out));

        let mut forward_graph = ComputeGraph::new(ctx.clone());
        let mut gb = GraphBuilder::train(&mut forward_graph);
        ops::embedding(&mut gb, tokens_buffer, &table, &out_buffer, shape);

        let mut backward_graph = ComputeGraph::new(ctx.clone());
        let mut gb = GraphBuilder::train(&mut backward_graph);
        ops::embedding_bwd(&mut gb, tokens_buffer, grad_output, &grad_table, shape);

        Self {
            table,
            grad_table,
            out_buffer,
            forward_graph,
            backward_graph,
        }
    }
}

impl<B: Backend> Layer for Embedding<B> {
    fn forward(&self) {
        self.forward_graph.execute();
    }
    fn backward(&self) {
        self.backward_graph.execute();
    }
}

pub struct Add<B: Backend> {
    pub in_out_buffer: Arc<Tensor<B>>,
    pub grad_a: Arc<Tensor<B>>,
    pub grad_b: Arc<Tensor<B>>,
    pub forward_graph: ComputeGraph<B>,
    pub backward_graph: ComputeGraph<B>,
}

impl<B: Backend> Add<B> {
    pub fn new(
        ctx: Arc<B>,
        length: u32,
        buf_a: &Arc<Tensor<B>>,
        buf_b: &Arc<Tensor<B>>,
        grad_output: &Arc<Tensor<B>>,
        grad_a: &Arc<Tensor<B>>,
        grad_b: &Arc<Tensor<B>>,
    ) -> Self {
        let grad_a = grad_a.clone();
        let grad_b = grad_b.clone();

        let mut forward_graph = ComputeGraph::new(ctx.clone());
        let mut gb = GraphBuilder::train(&mut forward_graph);
        ops::residual_add(&mut gb, buf_a, buf_b, length);

        let mut backward_graph = ComputeGraph::new(ctx.clone());
        let mut gb = GraphBuilder::train(&mut backward_graph);
        ops::residual_add(&mut gb, &grad_a, grad_output, length);
        ops::residual_add(&mut gb, &grad_b, grad_output, length);

        Self {
            in_out_buffer: buf_a.clone(),
            grad_a,
            grad_b,
            forward_graph,
            backward_graph,
        }
    }
}

impl<B: Backend> Layer for Add<B> {
    fn forward(&self) {
        self.forward_graph.execute();
    }

    fn backward(&self) {
        self.backward_graph.execute();
    }
}

pub struct SiLU<B: Backend> {
    pub in_out_buffer: Arc<Tensor<B>>,
    pub grad_input: Arc<Tensor<B>>,
    pub forward_graph: ComputeGraph<B>,
    pub backward_graph: ComputeGraph<B>,
}

impl<B: Backend> SiLU<B> {
    pub fn new(
        ctx: Arc<B>,
        total_elements: u32,
        input_buffer: &Arc<Tensor<B>>,
        grad_output: &Arc<Tensor<B>>,
        grad_input: &Arc<Tensor<B>>,
    ) -> Self {
        let grad_input = grad_input.clone();

        let mut forward_graph = ComputeGraph::new(ctx.clone());
        let mut gb = GraphBuilder::train(&mut forward_graph);
        ops::silu(&mut gb, input_buffer, total_elements);

        let mut backward_graph = ComputeGraph::new(ctx.clone());
        let mut gb = GraphBuilder::train(&mut backward_graph);
        ops::silu_bwd(
            &mut gb,
            input_buffer,
            grad_output,
            &grad_input,
            total_elements,
        );

        Self {
            in_out_buffer: input_buffer.clone(),
            grad_input,
            forward_graph,
            backward_graph,
        }
    }
}

impl<B: Backend> Layer for SiLU<B> {
    fn forward(&self) {
        self.forward_graph.execute();
    }
    fn backward(&self) {
        self.backward_graph.execute();
    }
}

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
        batch_size: u32,
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

        let out_size = (batch_size * seq_len * dim) as usize;
        let out_buffer = Arc::new(Tensor::init_from_cpu(
            ctx.clone(),
            &vec![0.0 as Real; out_size],
        ));

        let mut forward_graph = ComputeGraph::new(ctx.clone());
        let mut backward_graph = ComputeGraph::new(ctx.clone());

        let mut gb = GraphBuilder::train(&mut forward_graph);
        let saved: Vec<_> = (0..batch_size)
            .map(|b| {
                let shape = FlashAttnMeta {
                    seq_len,
                    dim,
                    head_dim,
                    scale,
                    row_offset: b * seq_len,
                };
                ops::flash_attention(&mut gb, q_buf, k_buf, v_buf, &out_buffer, shape)
            })
            .collect();

        let mut gb = GraphBuilder::train(&mut backward_graph);
        for (b, saved_b) in saved.iter().enumerate() {
            let shape = FlashAttnMeta {
                seq_len,
                dim,
                head_dim,
                scale,
                row_offset: b as u32 * seq_len,
            };
            ops::flash_attention_bwd(
                &mut gb,
                q_buf,
                k_buf,
                v_buf,
                saved_b,
                grad_output,
                grad_q,
                grad_k,
                grad_v,
                shape,
            );
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

pub struct CrossEntropy<B: Backend> {
    pub seq_len: u32,
    pub target_tokens: Arc<Tensor<B>>,
    pub probs: Arc<Tensor<B>>,
    pub losses: Arc<Tensor<B>>,
    pub d_losses: Arc<Tensor<B>>,
    pub grad_logits: Arc<Tensor<B>>,
    pub forward_graph: ComputeGraph<B>,
    pub backward_graph: ComputeGraph<B>,
}

impl<B: Backend> CrossEntropy<B> {
    pub fn new(
        ctx: Arc<B>,
        vocab_size: u32,
        seq_len: u32,
        logits: &Arc<Tensor<B>>,
        grad_logits: &Arc<Tensor<B>>,
    ) -> Self {
        let target_tokens = Arc::new(Tensor::init_from_cpu(
            ctx.clone(),
            &vec![0u32; seq_len as usize],
        ));
        let probs = Arc::new(Tensor::init_from_cpu(
            ctx.clone(),
            &vec![0.0 as Real; (seq_len * vocab_size) as usize],
        ));
        let losses = Arc::new(Tensor::init_from_cpu(
            ctx.clone(),
            &vec![0.0 as Real; seq_len as usize],
        ));
        let d_losses = Arc::new(Tensor::init_from_cpu(
            ctx.clone(),
            &vec![1.0 as Real / seq_len as Real; seq_len as usize],
        ));
        let grad_logits = grad_logits.clone();

        let shape = CrossEntropyMeta {
            vocab_size,
            num_rows: seq_len,
        };

        let mut forward_graph = ComputeGraph::new(ctx.clone());
        let mut gb = GraphBuilder::train(&mut forward_graph);
        ops::cross_entropy(&mut gb, logits, &target_tokens, &probs, &losses, shape);

        let mut backward_graph = ComputeGraph::new(ctx.clone());
        let mut gb = GraphBuilder::train(&mut backward_graph);
        ops::cross_entropy_bwd(
            &mut gb,
            &probs,
            &target_tokens,
            &d_losses,
            &grad_logits,
            shape,
        );

        Self {
            seq_len,
            target_tokens,
            probs,
            losses,
            d_losses,
            grad_logits,
            forward_graph,
            backward_graph,
        }
    }

    pub fn set_grad_scale(&self, scale: Real) {
        self.d_losses
            .copy_from_cpu(&vec![scale; self.seq_len as usize]);
    }

    pub fn loss(&self) -> Real {
        let losses: Vec<Real> = self.losses.to_cpu();
        losses.iter().sum::<Real>() / losses.len() as Real
    }
}

impl<B: Backend> Layer for CrossEntropy<B> {
    fn forward(&self) {
        self.forward_graph.execute();
    }
    fn backward(&self) {
        self.backward_graph.execute();
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
            batch_size,
            ..
        } = *cfg;

        let rows = batch_size * seq_len;
        let dim_size = (rows * dim) as usize;
        let hidden_size = (rows * hidden_dim) as usize;

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
        let elems = rows * dim;

        let norm_1 = RMSNorm::new(
            ctx.clone(),
            dim,
            rows,
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
            rows,
            &bw.qkv_proj,
            &norm_1.out_buffer,
            &g_attn_qkv,
            &g_qkvproj_in,
        );

        let q_buf = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros_dim));
        let k_buf = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros_dim));
        let v_buf = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros_dim));
        let mut qkv_split_forward = ComputeGraph::new(ctx.clone());
        let mut gb = GraphBuilder::train(&mut qkv_split_forward);
        ops::qkv_split(
            &mut gb,
            &qkv_proj.out_buffer,
            &q_buf,
            &k_buf,
            &v_buf,
            HeadMoveMeta::qkv_slice(rows, dim, 0),
        );

        let head_dim = cfg.head_dim();

        let mut rope_forward = ComputeGraph::new(ctx.clone());
        let mut gb = GraphBuilder::train(&mut rope_forward);
        for b in 0..batch_size {
            let rope_shape = RopeMeta {
                seq_len,
                dim,
                head_dim,
                row_offset: b * seq_len,
            };
            ops::rope_qk(&mut gb, &q_buf, &k_buf, rope_shape);
        }

        let attention = SelfAttention::new(
            ctx.clone(),
            seq_len,
            dim,
            num_heads,
            batch_size,
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
            rows,
            &bw.out_proj,
            &attention.out_buffer,
            &g_add1_b,
            &g_outproj_in,
        );

        let add_1 = Add::new(
            ctx.clone(),
            dim * rows,
            input_tensor,
            &out_proj.out_buffer,
            &g_add2_a,
            &grad_input,
            &g_add1_b,
        );

        let norm_2 = RMSNorm::new(
            ctx.clone(),
            dim,
            rows,
            &bw.norm_2,
            &add_1.in_out_buffer,
            &g_ffnup_in,
            &g_norm2_in,
        );

        let ffn_up = Linear::new(
            ctx.clone(),
            dim,
            hidden_dim,
            rows,
            &bw.ffn_up,
            &norm_2.out_buffer,
            &g_silu_in,
            &g_ffnup_in,
        );

        let silu = SiLU::new(
            ctx.clone(),
            (rows * hidden_dim) as u32,
            &ffn_up.out_buffer,
            &g_ffndown_in,
            &g_silu_in,
        );

        let ffn_down = Linear::new(
            ctx.clone(),
            hidden_dim,
            dim,
            rows,
            &bw.ffn_down,
            &silu.in_out_buffer,
            &g_add2_b,
            &g_ffndown_in,
        );

        let add_2 = Add::new(
            ctx.clone(),
            dim * rows,
            &add_1.in_out_buffer,
            &ffn_down.out_buffer,
            grad_output,
            &g_add2_a,
            &g_add2_b,
        );

        let mut barrier_1 = ComputeGraph::new(ctx.clone());
        let mut gb = GraphBuilder::train(&mut barrier_1);
        ops::add_inplace_bwd(&mut gb, &add_2.grad_a, &norm_2.grad_input, elems);

        let mut barrier_3 = ComputeGraph::new(ctx.clone());
        let mut gb = GraphBuilder::train(&mut barrier_3);
        ops::add_inplace_bwd(&mut gb, &grad_input, &norm_1.grad_input, elems);

        let mut rope_backward = ComputeGraph::new(ctx.clone());
        let mut gb = GraphBuilder::train(&mut rope_backward);
        for b in 0..batch_size {
            let rope_shape = RopeMeta {
                seq_len,
                dim,
                head_dim,
                row_offset: b * seq_len,
            };
            ops::rope_bwd_qk(&mut gb, &g_attn_q, &g_attn_k, rope_shape);
        }

        // dL/dQ + dL/dK + dL/dV -> one fused grad_output for qkv_proj's backward
        let mut qkv_gather_backward = ComputeGraph::new(ctx.clone());
        let mut gb = GraphBuilder::train(&mut qkv_gather_backward);
        ops::qkv_scatter(
            &mut gb,
            &g_attn_q,
            &g_attn_k,
            &g_attn_v,
            &g_attn_qkv,
            HeadMoveMeta::qkv_slice(rows, dim, 0),
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

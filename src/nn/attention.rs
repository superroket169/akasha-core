use super::traits::Layer;
use crate::Real;
use crate::nn::shader_paths::{CAUSAL_MASK, MATMUL, MATMUL_TRP, SOFTMAX, SOFTMAX_BWD};
use filuplex::context::Context;
use filuplex::graph::{ComputeGraphBuilder, ExecutableGraph};
use filuplex::ops::{BuiltInShader, GpuBuffer};
use std::sync::Arc;

pub struct SelfAttention {
    pub ctx: Arc<Context>,
    pub out_buffer: GpuBuffer,
    pub grad_q: GpuBuffer,
    pub grad_k: GpuBuffer,
    pub grad_v: GpuBuffer,
    pub grad_scores: GpuBuffer,
    pub meta_qkt: GpuBuffer,
    pub meta_seq: GpuBuffer,
    pub meta_out: GpuBuffer,

    pub graph_qkt: ExecutableGraph,
    pub graph_mask: ExecutableGraph,
    pub graph_softmax: ExecutableGraph,
    pub graph_out: ExecutableGraph,

    pub bwd_graph_grad_v: ExecutableGraph,
    pub bwd_graph_grad_scores: ExecutableGraph,
    pub bwd_graph_softmax: ExecutableGraph,
    pub bwd_graph_grad_q: ExecutableGraph,
    pub bwd_graph_grad_k: ExecutableGraph,
}

impl SelfAttention {
    pub fn new(
        ctx: Arc<Context>,
        seq_len: u32,
        dim: u32,
        q_buf: &GpuBuffer,
        k_buf: &GpuBuffer,
        v_buf: &GpuBuffer,
        grad_output: &GpuBuffer,
    ) -> Self {
        let scores_size = (seq_len * seq_len) as usize;
        let out_size = (seq_len * dim) as usize;
        let scores_buf = GpuBuffer::from_cpu(&vec![0.0 as Real; scores_size], &ctx);
        let out_buffer = GpuBuffer::from_cpu(&vec![0.0 as Real; out_size], &ctx);

        let grad_q = GpuBuffer::from_cpu(&vec![0.0 as Real; out_size], &ctx);
        let grad_k = GpuBuffer::from_cpu(&vec![0.0 as Real; out_size], &ctx);
        let grad_v = GpuBuffer::from_cpu(&vec![0.0 as Real; out_size], &ctx);
        let grad_scores = GpuBuffer::from_cpu(&vec![0.0 as Real; scores_size], &ctx);

        let meta_qkt =
            GpuBuffer::from_cpu(&vec![seq_len as Real, dim as Real, seq_len as Real], &ctx);
        let meta_seq = GpuBuffer::from_cpu(&vec![seq_len as Real], &ctx);
        let meta_out =
            GpuBuffer::from_cpu(&vec![seq_len as Real, seq_len as Real, dim as Real], &ctx);

        let shader_qkt = BuiltInShader::load_from_file(&ctx, MATMUL_TRP).load(&ctx);
        let shader_mask = BuiltInShader::load_from_file(&ctx, CAUSAL_MASK).load(&ctx);
        let shader_softmax = BuiltInShader::load_from_file(&ctx, SOFTMAX).load(&ctx);
        let shader_out = BuiltInShader::load_from_file(&ctx, MATMUL).load(&ctx);

        // ==========================================
        //                  FORWARD
        // ==========================================
        let mut b_qkt = ComputeGraphBuilder::new(ctx.clone());
        b_qkt.add_operation(
            shader_qkt,
            vec![(0, q_buf), (1, k_buf), (2, &scores_buf), (3, &meta_qkt)],
            [(seq_len + 15) / 16, (seq_len + 15) / 16, 1],
        );

        let mut b_mask = ComputeGraphBuilder::new(ctx.clone());
        b_mask.add_operation(
            shader_mask,
            vec![(0, &scores_buf), (1, &meta_seq)],
            [(seq_len + 15) / 16, (seq_len + 15) / 16, 1],
        );

        let mut b_softmax = ComputeGraphBuilder::new(ctx.clone());
        b_softmax.add_operation(
            shader_softmax,
            vec![(0, &scores_buf), (1, &meta_seq)],
            [(seq_len + 255) / 256, 1, 1],
        );

        let mut b_out = ComputeGraphBuilder::new(ctx.clone());
        b_out.add_operation(
            shader_out,
            vec![
                (0, &scores_buf),
                (1, v_buf),
                (2, &out_buffer),
                (3, &meta_out),
            ],
            [(dim + 15) / 16, (seq_len + 15) / 16, 1],
        );

        // ==========================================
        //                  BACKWARD
        // ==========================================
        let shader_matmul_bwd = BuiltInShader::load_from_file(&ctx, MATMUL).load(&ctx);
        let shader_matmul_bwd_trp = BuiltInShader::load_from_file(&ctx, MATMUL_TRP).load(&ctx);
        let shader_softmax_bwd = BuiltInShader::load_from_file(&ctx, SOFTMAX_BWD).load(&ctx);

        let mut bb_grad_v = ComputeGraphBuilder::new(ctx.clone());
        bb_grad_v.add_operation(
            shader_matmul_bwd_trp.clone(),
            vec![
                (0, &scores_buf),
                (1, grad_output),
                (2, &grad_v),
                (3, &meta_qkt),
            ],
            [(dim + 15) / 16, (seq_len + 15) / 16, 1],
        );

        let mut bb_grad_scores = ComputeGraphBuilder::new(ctx.clone());
        bb_grad_scores.add_operation(
            shader_matmul_bwd_trp.clone(),
            vec![
                (0, grad_output),
                (1, v_buf),
                (2, &grad_scores),
                (3, &meta_out),
            ],
            [(seq_len + 15) / 16, (seq_len + 15) / 16, 1],
        );

        let mut bb_softmax = ComputeGraphBuilder::new(ctx.clone());
        bb_softmax.add_operation(
            shader_softmax_bwd,
            vec![(0, &grad_scores), (1, &scores_buf), (2, &meta_seq)],
            [(seq_len + 255) / 256, 1, 1],
        );

        let mut bb_grad_q = ComputeGraphBuilder::new(ctx.clone());
        bb_grad_q.add_operation(
            shader_matmul_bwd,
            vec![(0, &grad_scores), (1, k_buf), (2, &grad_q), (3, &meta_qkt)],
            [(dim + 15) / 16, (seq_len + 15) / 16, 1],
        );

        let mut bb_grad_k = ComputeGraphBuilder::new(ctx.clone());
        bb_grad_k.add_operation(
            shader_matmul_bwd_trp,
            vec![(0, &grad_scores), (1, q_buf), (2, &grad_k), (3, &meta_qkt)],
            [(dim + 15) / 16, (seq_len + 15) / 16, 1],
        );

        Self {
            ctx,
            out_buffer,
            grad_q,
            grad_k,
            grad_v,
            grad_scores,
            meta_qkt,
            meta_seq,
            meta_out,
            graph_qkt: b_qkt.build(),
            graph_mask: b_mask.build(),
            graph_softmax: b_softmax.build(),
            graph_out: b_out.build(),
            bwd_graph_grad_v: bb_grad_v.build(),
            bwd_graph_grad_scores: bb_grad_scores.build(),
            bwd_graph_softmax: bb_softmax.build(),
            bwd_graph_grad_q: bb_grad_q.build(),
            bwd_graph_grad_k: bb_grad_k.build(),
        }
    }
}

impl Layer for SelfAttention {
    fn forward(&self) {
        self.graph_qkt.execute();
        unsafe {
            self.ctx.device.wait_idle().unwrap();
        }

        self.graph_mask.execute();
        unsafe {
            self.ctx.device.wait_idle().unwrap();
        }

        self.graph_softmax.execute();
        unsafe {
            self.ctx.device.wait_idle().unwrap();
        }

        self.graph_out.execute();
        unsafe {
            self.ctx.device.wait_idle().unwrap();
        }
    }

    fn backward(&self) {
        self.bwd_graph_grad_v.execute();
        unsafe {
            self.ctx.device.wait_idle().unwrap();
        }

        self.bwd_graph_grad_scores.execute();
        unsafe {
            self.ctx.device.wait_idle().unwrap();
        }

        self.bwd_graph_softmax.execute();
        unsafe {
            self.ctx.device.wait_idle().unwrap();
        }

        self.bwd_graph_grad_q.execute();
        unsafe {
            self.ctx.device.wait_idle().unwrap();
        }

        self.bwd_graph_grad_k.execute();
        unsafe {
            self.ctx.device.wait_idle().unwrap();
        }
    }
}

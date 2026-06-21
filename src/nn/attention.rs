use super::traits::Layer;
use crate::Real;
use crate::nn::shader_paths::{CAUSAL_MASK, MATMUL, MATMUL_TRP, SOFTMAX, SOFTMAX_BWD};
use filuplex::context::Context;
use filuplex::graph::{ComputeGraphBuilder, ExecutableGraphQueue};
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

    pub forward_queue: ExecutableGraphQueue,
    pub backward_queue: ExecutableGraphQueue,
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
        let mut forward_queue = ExecutableGraphQueue::new();

        let mut b_qkt = ComputeGraphBuilder::new(ctx.clone());
        b_qkt.add_operation(
            shader_qkt,
            vec![(0, q_buf), (1, k_buf), (2, &scores_buf), (3, &meta_qkt)],
            [(seq_len + 15) / 16, (seq_len + 15) / 16, 1],
        );
        forward_queue.push(b_qkt.build()); // Sıraya gir koçum!

        let mut b_mask = ComputeGraphBuilder::new(ctx.clone());
        b_mask.add_operation(
            shader_mask,
            vec![(0, &scores_buf), (1, &meta_seq)],
            [(seq_len + 15) / 16, (seq_len + 15) / 16, 1],
        );
        forward_queue.push(b_mask.build()); // Arkasına geç!

        let mut b_softmax = ComputeGraphBuilder::new(ctx.clone());
        b_softmax.add_operation(
            shader_softmax,
            vec![(0, &scores_buf), (1, &meta_seq)],
            [(seq_len + 255) / 256, 1, 1],
        );
        forward_queue.push(b_softmax.build()); // Sen de geç!

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
        forward_queue.push(b_out.build()); // Ve forward kapanışı!

        // ==========================================
        //                  BACKWARD
        // ==========================================
        let mut backward_queue = ExecutableGraphQueue::new();

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
        backward_queue.push(bb_grad_v.build());

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
        backward_queue.push(bb_grad_scores.build());

        let mut bb_softmax = ComputeGraphBuilder::new(ctx.clone());
        bb_softmax.add_operation(
            shader_softmax_bwd,
            vec![(0, &grad_scores), (1, &scores_buf), (2, &meta_seq)],
            [(seq_len + 255) / 256, 1, 1],
        );
        backward_queue.push(bb_softmax.build());

        let mut bb_grad_q = ComputeGraphBuilder::new(ctx.clone());
        bb_grad_q.add_operation(
            shader_matmul_bwd,
            vec![(0, &grad_scores), (1, k_buf), (2, &grad_q), (3, &meta_qkt)],
            [(dim + 15) / 16, (seq_len + 15) / 16, 1],
        );
        backward_queue.push(bb_grad_q.build());

        let mut bb_grad_k = ComputeGraphBuilder::new(ctx.clone());
        bb_grad_k.add_operation(
            shader_matmul_bwd_trp,
            vec![(0, &grad_scores), (1, q_buf), (2, &grad_k), (3, &meta_qkt)],
            [(dim + 15) / 16, (seq_len + 15) / 16, 1],
        );
        backward_queue.push(bb_grad_k.build());

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
            forward_queue,
            backward_queue,
        }
    }
}

impl Layer for SelfAttention {
    fn forward(&self) {
        self.forward_queue.execute();
    }
    fn backward(&self) {
        self.backward_queue.execute();
    }
}

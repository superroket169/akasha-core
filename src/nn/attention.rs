use super::traits::Layer;
use filuplex::context::Context;
use filuplex::graph::{ComputeGraphBuilder, ExecutableGraph};
use filuplex::ops::{BuiltInShader, GpuBuffer};
use std::sync::Arc;

pub struct SelfAttention {
    pub out_buffer: GpuBuffer,
    pub graph: ExecutableGraph,
}

impl SelfAttention {
    pub fn new(
        ctx: Arc<Context>,
        seq_len: u32,
        dim: u32,
        // heads: u32, // şimdilik bir kafa
        q_buf: &GpuBuffer,
        k_buf: &GpuBuffer,
        v_buf: &GpuBuffer,
    ) -> Self {
        // side buffs
        let scores_size = (seq_len * seq_len) as usize;
        let out_size = (seq_len * dim) as usize;
        let scores_buf = GpuBuffer::from_cpu(&vec![0.0f32; scores_size], &ctx);
        let out_buffer = GpuBuffer::from_cpu(&vec![0.0f32; out_size], &ctx);

        // meta
        let meta_qkt = GpuBuffer::from_cpu(&vec![seq_len as f32, dim as f32, seq_len as f32], &ctx);
        let meta_seq = GpuBuffer::from_cpu(&vec![seq_len as f32], &ctx);
        let meta_out = GpuBuffer::from_cpu(&vec![seq_len as f32, seq_len as f32, dim as f32], &ctx);

        // load shaders
        let shader_qkt =
            BuiltInShader::load_from_file(&ctx, "src/shaders/matmul_trp.spv").load(&ctx);
        let shader_mask =
            BuiltInShader::load_from_file(&ctx, "src/shaders/causal_mask.spv").load(&ctx);
        let shader_softmax =
            BuiltInShader::load_from_file(&ctx, "src/shaders/softmax.spv").load(&ctx);
        let shader_out = BuiltInShader::load_from_file(&ctx, "src/shaders/matmul.spv").load(&ctx);

        // build
        let mut builder = ComputeGraphBuilder::new(ctx.clone());

        // Operation Pipeline =>
        builder.add_operation(
            shader_qkt,
            vec![(0, q_buf), (1, k_buf), (2, &scores_buf), (3, &meta_qkt)],
            [(seq_len + 15) / 16, (seq_len + 15) / 16, 1],
        );
        builder.add_operation(
            shader_mask,
            vec![(0, &scores_buf), (1, &meta_seq)],
            [(seq_len + 15) / 16, (seq_len + 15) / 16, 1],
        );
        builder.add_operation(
            shader_softmax,
            vec![(0, &scores_buf), (1, &meta_seq)],
            [(seq_len + 255) / 256, 1, 1],
        );
        builder.add_operation(
            shader_out,
            vec![
                (0, &scores_buf),
                (1, v_buf),
                (2, &out_buffer),
                (3, &meta_out),
            ],
            [(dim + 15) / 16, (seq_len + 15) / 16, 1],
        );

        Self {
            out_buffer,
            graph: builder.build(),
        }
    }
}

impl Layer for SelfAttention {
    fn forward(&self) {
        self.graph.execute();
        // self.out_buffer.clone()
    }

    fn backward(&self) {
        // TODO
    }
}

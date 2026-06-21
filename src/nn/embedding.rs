use filuplex::context::Context;
use filuplex::graph::{ComputeGraphBuilder, ExecutableGraph};
use filuplex::ops::{BuiltInShader, GpuBuffer};
use std::sync::Arc;

pub struct Embedding {
    pub table: GpuBuffer,
    pub out_buffer: GpuBuffer,
    pub graph: ExecutableGraph,
}

impl Embedding {
    // tokens_buffer comes from main, at tokenizer.rs
    pub fn new(
        ctx: Arc<Context>,
        vocab_size: u32,
        dim: u32,
        seq_len: u32,
        table_data: &[f32],
        tokens_buffer: &GpuBuffer,
    ) -> Self {
        assert_eq!(
            table_data.len(),
            (vocab_size * dim) as usize,
            "Dict size doesnt match!"
        );
        let table = GpuBuffer::from_cpu(table_data, &ctx);

        let meta_data = vec![dim as f32, seq_len as f32];
        let meta = GpuBuffer::from_cpu(&meta_data, &ctx);

        let out_size = (seq_len * dim) as usize;
        let dummy_out = vec![0.0f32; out_size];
        let out_buffer = GpuBuffer::from_cpu(&dummy_out, &ctx);

        let shader = BuiltInShader::load_from_file(&ctx, "src/shaders/embedding.spv").load(&ctx);
        let total_threads = seq_len * dim;

        let mut builder = ComputeGraphBuilder::new(ctx.clone());
        builder.add_operation(
            shader,
            vec![
                (0, tokens_buffer),
                (1, &table),
                (2, &out_buffer),
                (3, &meta),
            ],
            [(total_threads + 255) / 256, 1, 1],
        );

        Self {
            table,
            out_buffer,
            graph: builder.build(),
        }
    }

    pub fn forward(&self) {
        self.graph.execute();
    }
}

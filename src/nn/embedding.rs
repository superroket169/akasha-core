use crate::Real;
use crate::nn::Layer;
use crate::nn::shader_paths::EMBEDDING;
use filuplex::context::Context;
use filuplex::graph::{ComputeGraphBuilder, ExecutableGraph};
use filuplex::ops::{BuiltInShader, GpuBuffer};
use std::sync::Arc;

pub struct Embedding {
    pub table: GpuBuffer,
    pub grad_table: GpuBuffer,
    pub out_buffer: GpuBuffer,
    pub graph: ExecutableGraph,
    pub backward_graph: ExecutableGraph,
}

impl Embedding {
    // tokens_buffer comes from main, at tokenizer.rs
    pub fn new(
        ctx: Arc<Context>,
        vocab_size: u32,
        dim: u32,
        seq_len: u32,
        table_data: &[Real],
        tokens_buffer: &GpuBuffer,
        grad_output: &GpuBuffer,
    ) -> Self {
        assert_eq!(
            table_data.len(),
            (vocab_size * dim) as usize,
            "Dict size doesnt match!"
        );
        let table = GpuBuffer::from_cpu(table_data, &ctx);
        let grad_table = GpuBuffer::from_cpu(&vec![0.0 as Real; (vocab_size * dim) as usize], &ctx);

        let meta_data = vec![dim as Real, seq_len as Real];
        let meta = GpuBuffer::from_cpu(&meta_data, &ctx);

        let out_size = (seq_len * dim) as usize;
        let dummy_out = vec![0.0 as Real; out_size];
        let out_buffer = GpuBuffer::from_cpu(&dummy_out, &ctx);

        let shader = BuiltInShader::load_from_file(&ctx, EMBEDDING).load(&ctx);
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

        // Backward

        let shader_bwd = BuiltInShader::load_from_file(&ctx, EMBEDDING).load(&ctx);
        let mut bw_builder = ComputeGraphBuilder::new(ctx.clone());
        bw_builder.add_operation(
            shader_bwd,
            vec![
                (0, tokens_buffer),
                (1, grad_output),
                (2, &grad_table),
                (3, &meta),
            ],
            [(seq_len * dim + 255) / 256, 1, 1],
        );

        Self {
            table,
            grad_table,
            out_buffer,
            graph: builder.build(),
            backward_graph: bw_builder.build(),
        }
    }
}

impl Layer for Embedding {
    fn forward(&self) {
        self.graph.execute();
    }

    fn backward(&self) {
        self.backward_graph.execute();
    }
}

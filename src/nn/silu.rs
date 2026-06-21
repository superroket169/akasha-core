use super::traits::Layer;
use filuplex::context::Context;
use filuplex::graph::{ComputeGraphBuilder, ExecutableGraph};
use filuplex::ops::{BuiltInShader, GpuBuffer};
use std::sync::Arc;

pub struct SiLU {
    pub in_out_buffer: GpuBuffer,
    pub grad_input: GpuBuffer,
    pub graph: ExecutableGraph,
    pub backward_graph: ExecutableGraph,
}

impl SiLU {
    pub fn new(
        ctx: Arc<Context>,
        length: u32,
        input_buffer: &GpuBuffer,
        grad_output: &GpuBuffer,
    ) -> Self {
        let meta_data = vec![length as f32];
        let meta = GpuBuffer::from_cpu(&meta_data, &ctx);

        let shader = BuiltInShader::load_from_file(&ctx, "src/shaders/silu.spv").load(&ctx);
        let mut builder = ComputeGraphBuilder::new(ctx.clone());
        builder.add_operation(
            shader,
            vec![(0, input_buffer), (1, &meta)],
            [(length + 255) / 256, 1, 1],
        );

        // --- BACKWARD ---
        let grad_input = GpuBuffer::from_cpu(&vec![0.0f32; length as usize], &ctx);
        let mut bw_builder = ComputeGraphBuilder::new(ctx.clone());
        let shader_bwd = BuiltInShader::load_from_file(&ctx, "src/shaders/silu_bwd.spv").load(&ctx);

        bw_builder.add_operation(
            shader_bwd,
            vec![(0, input_buffer), (1, grad_output), (2, &grad_input)],
            [(length + 255) / 256, 1, 1],
        );

        Self {
            in_out_buffer: input_buffer.clone(),
            grad_input,
            graph: builder.build(),
            backward_graph: bw_builder.build(),
        }
    }
}

impl Layer for SiLU {
    fn forward(&self) {
        self.graph.execute();
    }
    fn backward(&self) {
        self.backward_graph.execute();
    }
}

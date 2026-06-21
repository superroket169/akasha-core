use super::traits::Layer;
use filuplex::context::Context;
use filuplex::graph::{ComputeGraphBuilder, ExecutableGraph};
use filuplex::ops::{BuiltInShader, GpuBuffer};
use std::sync::Arc;

pub struct RoPE {
    pub in_out_buffer: GpuBuffer,
    pub graph: ExecutableGraph,
}

impl RoPE {
    pub fn new(ctx: Arc<Context>, dim: u32, input_buffer: &GpuBuffer) -> Self {
        let pos = 0;

        let meta_data = vec![dim as f32, pos as f32];
        let meta = GpuBuffer::from_cpu(&meta_data, &ctx);

        let shader = BuiltInShader::load_from_file(&ctx, "src/shaders/rope.spv").load(&ctx);
        let mut builder = ComputeGraphBuilder::new(ctx.clone());
        builder.add_operation(
            shader,
            vec![(0, input_buffer), (1, &meta)],
            // X ekseni: 1024 / 64 (local_size_x) = 16 workgroup
            [(dim + 63) / 64, 1, 1],
        );

        Self {
            in_out_buffer: input_buffer.clone(),
            graph: builder.build(),
        }
    }
}

impl Layer for RoPE {
    fn forward(&self) {
        self.graph.execute();
        // self.in_out_buffer.clone()
    }

    fn backward(&self) {
        // TODO
    }
}

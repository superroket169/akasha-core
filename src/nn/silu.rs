use super::traits::Layer;
use filuplex::context::Context;
use filuplex::graph::{ComputeGraphBuilder, ExecutableGraph};
use filuplex::ops::{BuiltInShader, GpuBuffer};
use std::sync::Arc;

pub struct SiLU {
    pub in_out_buffer: GpuBuffer,
    pub graph: ExecutableGraph,
}

impl SiLU {
    pub fn new(ctx: Arc<Context>, length: u32, input_buffer: &GpuBuffer) -> Self {
        let meta_data = vec![length as f32];
        let meta = GpuBuffer::from_cpu(&meta_data, &ctx);

        let shader = BuiltInShader::load_from_file(&ctx, "src/shaders/silu.spv").load(&ctx);
        let mut builder = ComputeGraphBuilder::new(ctx.clone());
        builder.add_operation(
            shader,
            vec![(0, input_buffer), (1, &meta)],
            [(length + 255) / 256, 1, 1],
        );

        Self {
            in_out_buffer: input_buffer.clone(),
            graph: builder.build(),
        }
    }
}

impl Layer for SiLU {
    fn forward(&self) {
        self.graph.execute();
        // self.in_out_buffer.clone()
    }

    fn backward(&self) {
        // TODO
    }
}

// NOTE all thigs is seems so same. this should a soulution

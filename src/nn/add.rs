use filuplex::context::Context;
use filuplex::graph::{ComputeGraphBuilder, ExecutableGraph};
use filuplex::ops::{BuiltInShader, GpuBuffer};
use std::sync::Arc;

pub struct Add {
    pub in_out_buffer: GpuBuffer,
    pub graph: ExecutableGraph,
}

impl Add {
    pub fn new(ctx: Arc<Context>, length: u32, buf_a: &GpuBuffer, buf_b: &GpuBuffer) -> Self {
        let meta_data = vec![length as f32];
        let meta = GpuBuffer::from_cpu(&meta_data, &ctx);

        let shader = BuiltInShader::load_from_file(&ctx, "src/shaders/add_inplace.spv").load(&ctx);
        let mut builder = ComputeGraphBuilder::new(ctx.clone());
        builder.add_operation(
            shader,
            vec![(0, buf_a), (1, buf_b), (2, &meta)],
            [(length + 255) / 256, 1, 1],
        );

        Self {
            in_out_buffer: buf_a.clone(),
            graph: builder.build(),
        }
    }
}

impl Add {
    pub fn forward(&self) -> GpuBuffer {
        self.graph.execute();
        self.in_out_buffer.clone()
    }
}

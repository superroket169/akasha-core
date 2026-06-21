use super::traits::Layer;
use filuplex::context::Context;
use filuplex::graph::{ComputeGraphBuilder, ExecutableGraph};
use filuplex::ops::{BuiltInShader, GpuBuffer};
use std::sync::Arc;

pub struct Linear {
    pub weight: GpuBuffer,
    pub out_buffer: GpuBuffer,
    pub graph: ExecutableGraph,
}

impl Linear {
    pub fn new(
        ctx: Arc<Context>,
        in_features: u32,
        out_features: u32,
        weight_data: &[f32],
        input_buffer: &GpuBuffer,
    ) -> Self {
        let weight = GpuBuffer::from_cpu(weight_data, &ctx);

        let seq_len = 1;
        let m = seq_len as u32;

        let meta_data = vec![m as f32, in_features as f32, out_features as f32];
        let meta = GpuBuffer::from_cpu(&meta_data, &ctx);

        let dummy_out = vec![0.0f32; (m * out_features) as usize];
        let out_buffer = GpuBuffer::from_cpu(&dummy_out, &ctx);

        let shader = BuiltInShader::load_from_file(&ctx, "src/shaders/matmul.spv").load(&ctx);

        let mut builder = ComputeGraphBuilder::new(ctx.clone());
        builder.add_operation(
            shader,
            vec![
                (0, input_buffer),
                (1, &weight),
                (2, &out_buffer),
                (3, &meta),
            ],
            [(out_features + 15) / 16, (m + 15) / 16, 1],
        );
        let graph = builder.build();

        Self {
            weight,
            out_buffer,
            graph,
        }
    }
}

impl Layer for Linear {
    fn forward(&self) {
        self.graph.execute();
        // self.out_buffer.clone()
    }

    fn backward(&self) {
        // TODO
    }
}

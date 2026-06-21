use super::traits::Layer;
use filuplex::context::Context;
use filuplex::graph::{ComputeGraphBuilder, ExecutableGraph};
use filuplex::ops::{BuiltInShader, GpuBuffer};
use std::sync::Arc;

pub struct RMSNorm {
    pub weight: GpuBuffer,
    pub out_buffer: GpuBuffer,
    pub grad_weight: GpuBuffer,
    pub grad_input: GpuBuffer,
    pub graph: ExecutableGraph,
    pub backward_graph: ExecutableGraph,
}

impl RMSNorm {
    pub fn new(
        ctx: Arc<Context>,
        dim: u32,
        weight_data: &[f32],
        input_buffer: &GpuBuffer,
        grad_output: &GpuBuffer,
    ) -> Self {
        let weight = GpuBuffer::from_cpu(weight_data, &ctx);
        let grad_weight = GpuBuffer::from_cpu(&vec![0.0f32; dim as usize], &ctx);
        let grad_input = GpuBuffer::from_cpu(&vec![0.0f32; dim as usize], &ctx);
        let seq_len = 1;

        let meta_data = vec![dim as f32];
        let meta = GpuBuffer::from_cpu(&meta_data, &ctx);

        let out_buffer = GpuBuffer::from_cpu(&vec![0.0f32; (seq_len * dim) as usize], &ctx);
        let shader = BuiltInShader::load_from_file(&ctx, "src/shaders/rmsnorm.spv").load(&ctx);
        let mut builder = ComputeGraphBuilder::new(ctx.clone());
        builder.add_operation(
            shader,
            vec![
                (0, input_buffer),
                (1, &out_buffer),
                (2, &weight),
                (3, &meta),
            ],
            [seq_len, 1, 1],
        );

        // Backward

        let mut bw_builder = ComputeGraphBuilder::new(ctx.clone());
        let shader_bwd =
            BuiltInShader::load_from_file(&ctx, "src/shaders/rmsnorm_bwd.spv").load(&ctx);

        bw_builder.add_operation(
            shader_bwd,
            vec![
                (0, input_buffer),
                (1, grad_output),
                (2, &weight),
                (3, &grad_weight),
                (4, &grad_input),
            ],
            [1, 1, 1],
        );

        Self {
            weight,
            out_buffer,
            grad_weight,
            grad_input,
            graph: builder.build(),
            backward_graph: bw_builder.build(),
        }
    }
}

impl Layer for RMSNorm {
    fn forward(&self) {
        self.graph.execute();
    }
    fn backward(&self) {
        self.backward_graph.execute();
    }
}

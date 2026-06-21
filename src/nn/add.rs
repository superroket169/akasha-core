use super::traits::Layer;
use filuplex::context::Context;
use filuplex::graph::{ComputeGraphBuilder, ExecutableGraph};
use filuplex::ops::{BuiltInShader, GpuBuffer};
use std::sync::Arc;

pub struct Add {
    pub in_out_buffer: GpuBuffer,
    pub grad_a: GpuBuffer,
    pub grad_b: GpuBuffer,
    pub forward_graph: ExecutableGraph,
    pub backward_graph: ExecutableGraph,
}

impl Add {
    pub fn new(
        ctx: Arc<Context>,
        length: u32,
        buf_a: &GpuBuffer,
        buf_b: &GpuBuffer,
        grad_output: &GpuBuffer,
    ) -> Self {
        let meta_data = vec![length as f32];
        let meta = GpuBuffer::from_cpu(&meta_data, &ctx);

        let shader_fw =
            BuiltInShader::load_from_file(&ctx, "src/shaders/add_inplace.spv").load(&ctx);
        let mut fw_builder = ComputeGraphBuilder::new(ctx.clone());
        fw_builder.add_operation(
            shader_fw,
            vec![(0, buf_a), (1, buf_b), (2, &meta)],
            [(length + 255) / 256, 1, 1],
        );

        let grad_a = GpuBuffer::from_cpu(&vec![0.0f32; length as usize], &ctx);
        let grad_b = GpuBuffer::from_cpu(&vec![0.0f32; length as usize], &ctx);

        let shader_bwd = BuiltInShader::load_from_file(&ctx, "src/shaders/add.spv").load(&ctx);
        let mut bw_builder = ComputeGraphBuilder::new(ctx.clone());

        bw_builder.add_operation(
            shader_bwd.clone(),
            vec![(0, &grad_a), (1, grad_output), (2, &meta)],
            [(length + 255) / 256, 1, 1],
        );
        bw_builder.add_operation(
            shader_bwd.clone(),
            vec![(0, &grad_b), (1, grad_output), (2, &meta)],
            [(length + 255) / 256, 1, 1],
        );

        Self {
            in_out_buffer: buf_a.clone(),
            grad_a,
            grad_b,
            forward_graph: fw_builder.build(),
            backward_graph: bw_builder.build(),
        }
    }
}

impl Layer for Add {
    fn forward(&self) {
        self.forward_graph.execute();
    }

    fn backward(&self) {
        self.backward_graph.execute();
    }
}

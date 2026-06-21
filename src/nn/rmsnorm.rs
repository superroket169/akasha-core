use super::traits::Layer;
use crate::Real;
use crate::nn::shader_paths::{RMSNORM, RMSNORM_BWD};
use filuplex::context::Context;
use filuplex::graph::{ComputeGraphBuilder, ExecutableGraph};
use filuplex::ops::{BuiltInShader, GpuBuffer};
use std::sync::Arc;

pub struct RMSNorm {
    pub weight: GpuBuffer,
    pub out_buffer: GpuBuffer,
    pub grad_weight: GpuBuffer,
    pub grad_input: GpuBuffer,
    pub meta: GpuBuffer,
    pub graph: ExecutableGraph,
    pub backward_graph: ExecutableGraph,
}

impl RMSNorm {
    pub fn new(
        ctx: Arc<Context>,
        dim: u32,
        seq_len: u32,
        weight_data: &[Real],
        input_buffer: &GpuBuffer,
        grad_output: &GpuBuffer,
    ) -> Self {
        let weight = GpuBuffer::from_cpu(weight_data, &ctx);
        let grad_weight = GpuBuffer::from_cpu(&vec![0.0 as Real; (seq_len * dim) as usize], &ctx);
        let grad_input = GpuBuffer::from_cpu(&vec![0.0 as Real; (seq_len * dim) as usize], &ctx);

        let meta_data = vec![dim as Real];
        let meta = GpuBuffer::from_cpu(&meta_data, &ctx);

        let out_buffer = GpuBuffer::from_cpu(&vec![0.0 as Real; (seq_len * dim) as usize], &ctx);
        let shader = BuiltInShader::load_from_file(&ctx, RMSNORM).load(&ctx);
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

        // Backward Tesisatı
        let mut bw_builder = ComputeGraphBuilder::new(ctx.clone());
        let shader_bwd = BuiltInShader::load_from_file(&ctx, RMSNORM_BWD).load(&ctx);

        bw_builder.add_operation(
            shader_bwd,
            vec![
                (0, input_buffer),
                (1, grad_output),
                (2, &weight),
                (3, &grad_input),
                (4, &grad_weight),
                (5, &meta),
            ],
            [seq_len, 1, 1],
        );

        Self {
            weight,
            out_buffer,
            grad_weight,
            grad_input,
            meta,
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

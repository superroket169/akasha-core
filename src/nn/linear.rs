use super::traits::Layer;
use crate::Real;
use crate::nn::shader_paths::{MATMUL, MATMUL_BWD_INPUT_TRP_B, MATMUL_BWD_WEIGHT_TRP_A};
use filuplex::context::Context;
use filuplex::graph::{ComputeGraphBuilder, ExecutableGraph};
use filuplex::ops::{BuiltInShader, GpuBuffer};
use std::sync::Arc;

pub struct Linear {
    pub weight: GpuBuffer,
    pub out_buffer: GpuBuffer,

    pub grad_weight: GpuBuffer,
    pub grad_input: GpuBuffer,

    pub forward_graph: ExecutableGraph,
    pub backward_graph: ExecutableGraph,
}

impl Linear {
    pub fn new(
        ctx: Arc<Context>,
        in_features: u32,
        out_features: u32,
        weight_data: &[Real],
        input_buffer: &GpuBuffer,
        grad_output: &GpuBuffer,
    ) -> Self {
        // --- FORWARD ---
        let weight = GpuBuffer::from_cpu(weight_data, &ctx);

        let seq_len = 1;
        let m = seq_len as u32;

        let meta_data = vec![m as Real, in_features as Real, out_features as Real];
        let meta = GpuBuffer::from_cpu(&meta_data, &ctx);

        let dummy_out = vec![0.0 as Real; (m * out_features) as usize];
        let out_buffer = GpuBuffer::from_cpu(&dummy_out, &ctx);

        let shader = BuiltInShader::load_from_file(&ctx, MATMUL).load(&ctx);

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
        let forward_graph = builder.build();

        // --- BACKWARD ---
        let dummy_grad_w = vec![0.0 as Real; (in_features * out_features) as usize];
        let grad_weight = GpuBuffer::from_cpu(&dummy_grad_w, &ctx);

        let dummy_grad_in = vec![0.0 as Real; (m * in_features) as usize];
        let grad_input = GpuBuffer::from_cpu(&dummy_grad_in, &ctx);

        let shader_bwd_w = BuiltInShader::load_from_file(&ctx, MATMUL_BWD_WEIGHT_TRP_A).load(&ctx);
        let shader_bwd_in = BuiltInShader::load_from_file(&ctx, MATMUL_BWD_INPUT_TRP_B).load(&ctx);

        let mut bw_builder = ComputeGraphBuilder::new(ctx.clone());

        bw_builder.add_operation(
            shader_bwd_w,
            vec![
                (0, input_buffer),
                (1, grad_output),
                (2, &grad_weight),
                (3, &meta),
            ],
            [(out_features + 15) / 16, (in_features + 15) / 16, 1],
        );

        bw_builder.add_operation(
            shader_bwd_in,
            vec![(0, grad_output), (1, &weight), (2, &grad_input), (3, &meta)],
            [(in_features + 15) / 16, (m + 15) / 16, 1],
        );

        let backward_graph = bw_builder.build();

        Self {
            weight,
            out_buffer,
            grad_weight,
            grad_input,
            forward_graph,
            backward_graph,
        }
    }
}

impl Layer for Linear {
    fn forward(&self) {
        self.forward_graph.execute();
    }

    fn backward(&self) {
        self.backward_graph.execute();
    }
}

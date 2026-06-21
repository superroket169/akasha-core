use super::traits::Layer;
use crate::Real;
use crate::nn::shader_paths::{ROPE, ROPE_BWD};
use filuplex::context::Context;
use filuplex::graph::{ComputeGraphBuilder, ExecutableGraph};
use filuplex::ops::{BuiltInShader, GpuBuffer};
use std::sync::Arc;

pub struct RoPE {
    pub in_out_buffer: GpuBuffer,
    pub grad_input: GpuBuffer,
    pub inv_freq_buffer: GpuBuffer,
    pub graph: ExecutableGraph,
    pub backward_graph: ExecutableGraph,
}

impl RoPE {
    pub fn new(
        ctx: Arc<Context>,
        dim: u32,
        input_buffer: &GpuBuffer,
        grad_output: &GpuBuffer,
    ) -> Self {
        let pos = 0;
        let inv_freq_buffer = inv_freq_init(dim, ctx.clone());

        let meta_data = vec![dim as Real, pos as Real];
        let meta = GpuBuffer::from_cpu(&meta_data, &ctx);

        let shader = BuiltInShader::load_from_file(&ctx, ROPE).load(&ctx);
        let mut builder = ComputeGraphBuilder::new(ctx.clone());

        // --- FORWARD ---
        builder.add_operation(
            shader,
            vec![
                (0, input_buffer),
                (1, input_buffer),
                (2, &inv_freq_buffer),
                (3, &meta),
            ],
            [(dim + 63) / 64, 1, 1],
        );

        // --- BACKWARD ---
        let grad_input = GpuBuffer::from_cpu(&vec![0.0 as Real; dim as usize], &ctx);
        let mut bw_builder = ComputeGraphBuilder::new(ctx.clone());
        let shader_bwd = BuiltInShader::load_from_file(&ctx, ROPE_BWD).load(&ctx);

        bw_builder.add_operation(
            shader_bwd,
            vec![(0, grad_output), (1, &grad_input)],
            [(dim + 63) / 64, 1, 1],
        );

        Self {
            in_out_buffer: input_buffer.clone(),
            grad_input,
            inv_freq_buffer,
            graph: builder.build(),
            backward_graph: bw_builder.build(),
        }
    }
}

impl Layer for RoPE {
    fn forward(&self) {
        self.graph.execute();
    }
    fn backward(&self) {
        self.backward_graph.execute();
    }
}

fn inv_freq_init(dim: u32, ctx: Arc<Context>) -> GpuBuffer {
    let half_dim = dim / 2;
    let mut inv_freq_data = Vec::with_capacity(half_dim as usize);
    for i in 0..half_dim {
        let freq = (1.0 / 10000.0 as Real).powf((2 * i) as f32 / dim as f32);
        inv_freq_data.push(freq);
    }
    let inv_freq_buffer = GpuBuffer::from_cpu(&inv_freq_data, &ctx);
    inv_freq_buffer
}

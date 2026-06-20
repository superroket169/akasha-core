use super::traits::Layer;
use filuplex::context::Context;
use filuplex::ops::{BuiltInShader, BuiltInShaderType, GpuBuffer};
use std::sync::Arc;

pub struct Linear {
    pub weight: GpuBuffer,
    pub shader: BuiltInShader,
    pub ctx: Arc<Context>,
}

impl Linear {
    pub fn new(ctx: Arc<Context>, weight_data: &[f32]) -> Self {
        let weight_buf = GpuBuffer::from_cpu(weight_data, &ctx);
        let shader = BuiltInShader::new(BuiltInShaderType::MatrisMul);

        Self {
            weight: weight_buf,
            shader,
            ctx,
        }
    }
}

impl Layer for Linear {
    fn forward(&self, input: &GpuBuffer) -> GpuBuffer {
        todo!("theres the actualy part")
    }
}

use filuplex::context::Context;
use filuplex::ops::GpuBuffer;
use std::sync::Arc;

pub struct SelfAttention {}

impl SelfAttention {
    pub fn new(_ctx: Arc<Context>, _dim: u32, _heads: u32) -> Self {
        todo!("Attention compute grafiği")
    }

    // 3 farklı bufferı var. Layer traitini kullanmıyoruz
    pub fn forward(&self, _q: &GpuBuffer, _k: &GpuBuffer, _v: &GpuBuffer) -> GpuBuffer {
        // Q x K^T -> Mask -> Softmax -> x V
        todo!("Attention GPU tetikleme")
    }
}

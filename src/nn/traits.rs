use filuplex::ops::GpuBuffer;

pub trait Layer {
    fn forward(&self, input: &GpuBuffer) -> GpuBuffer;
}

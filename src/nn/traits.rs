use filuplex::ops::GpuBuffer;
use std::path::Path;

pub trait Layer {
    fn forward(&self) -> GpuBuffer;
    // fn backward(&self, grad_output: &GpuBuffer) -> GpuBuffer;
}

pub trait Serializable {
    fn save_to_file(&self, path: &Path) -> Result<(), Box<dyn std::error::Error>>;
    fn load_from_file(&mut self, path: &Path) -> Result<(), Box<dyn std::error::Error>>;
}

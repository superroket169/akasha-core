use super::linear::Linear;
use super::traits::Layer;
use filuplex::ops::GpuBuffer;

pub struct TransformerBlock {
    pub attention_qkv: Linear,
    pub attention_out: Linear,
    pub ffn_up: Linear,
    pub ffn_down: Linear,
}

impl Layer for TransformerBlock {
    fn forward(&self, input: &GpuBuffer) -> GpuBuffer {
        // MSNorm -> MatMul (QKV) -> RoPE -> Causal Mask -> Softmax -> MatMul (Out) -> sum
        // RMSNorm -> MatMul (Up) -> SiLU -> MatMul (Down) -> sum
        todo!("pipeline")
    }
}

pub struct AkashaModel {
    pub layers: Vec<TransformerBlock>, // 24 Transform block
    pub lm_head: Linear,               // word founder
}

impl AkashaModel {
    pub fn forward(&self, tokens: &GpuBuffer) -> GpuBuffer {
        // tokens GpuBuffer olacak
        println!("Akasha düşünmeye başladı...");

        let mut x: GpuBuffer = tokens.clone();

        for layer in self.layers.iter() {
            x = layer.forward(&x);
        }

        self.lm_head.forward(&x)
    }
}

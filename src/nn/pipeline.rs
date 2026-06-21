use super::attention::SelfAttention;
use super::linear::Linear;
use super::rmsnorm::RMSNorm;
use super::rope::RoPE;
use super::traits::Layer;
use filuplex::ops::GpuBuffer;

pub struct TransformerBlock {
    // Attention
    pub norm_1: RMSNorm,
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub rope: RoPE,
    pub attention: SelfAttention,
    pub out_proj: Linear,

    // Feed Forward
    pub norm_2: RMSNorm,
    pub ffn_up: Linear,
    pub ffn_down: Linear,
}

impl Layer for TransformerBlock {
    fn forward(&self, _input: &GpuBuffer) -> GpuBuffer {
        // like this =>
        // let x = self.norm_1.forward(input);
        // let q = self.rope.forward(&self.q_proj.forward(&x));
        // let k = self.rope.forward(&self.k_proj.forward(&x));
        // let v = self.v_proj.forward(&x);
        // let attn_out = self.attention.forward(&q, &k, &v);
        // ... (ve FFN kısmı)

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

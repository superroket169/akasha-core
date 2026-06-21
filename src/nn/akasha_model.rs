use filuplex::context::Context;
use filuplex::ops::GpuBuffer;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::sync::Arc;

use super::linear::Linear;
use super::pipeline::TransformerBlock;
use super::rmsnorm::RMSNorm;
use super::traits::Layer;
// use super::embedding::Embedding;

pub struct AkashaModel {
    pub ctx: Arc<Context>,
    // pub embedding: Embedding,
    pub layers: Vec<TransformerBlock>,
    pub final_norm: RMSNorm,
    pub lm_head: Linear,
}

impl AkashaModel {
    // dummy start
    pub fn new(
        ctx: Arc<Context>,
        vocab_size: u32,
        dim: u32,
        seq_len: u32,
        num_layers: usize,
    ) -> Self {
        println!("Akasha {} katman ile VRAM'e inşa ediliyor...", num_layers);

        let mut current_input = GpuBuffer::from_cpu(&vec![0.01f32; (seq_len * dim) as usize], &ctx);
        let mut layers = Vec::with_capacity(num_layers);

        for _ in 0..num_layers {
            let block = TransformerBlock::new(ctx.clone(), dim, seq_len, &current_input);
            current_input = block.add_2.in_out_buffer.clone();
            layers.push(block);
        }

        let dummy_norm_w = vec![1.0f32; dim as usize];
        let final_norm = RMSNorm::new(ctx.clone(), dim, &dummy_norm_w, &current_input);

        let dummy_head_w = vec![0.01f32; (dim * vocab_size) as usize];
        let lm_head = Linear::new(
            ctx.clone(),
            dim,
            vocab_size,
            &dummy_head_w,
            &final_norm.out_buffer,
        );

        Self {
            ctx,
            layers,
            final_norm,
            lm_head,
        }
    }

    pub fn forward(&self) {
        // self.embedding.forward();

        for layer in self.layers.iter() {
            layer.forward();
        }

        self.final_norm.forward();
        self.lm_head.forward();

        // Sonuç lm_head.out_buffer içinde
    }

    pub fn save_to_file(&self, path: &str) -> bincode::Result<()> {
        println!("Ağırlıklar diske yazılıyor: {}", path);
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // tek struct (örn: AkashaWeights) ile save implemente edliecek
        // bincode::serialize_into(&mut writer, &all_weights)

        // bincode::serialize_into(&mut writer, &self.lm_head.weight_cpu_copy)?;

        writer.flush()?;
        println!("Kayıt başarılı!");
        Ok(())
    }

    pub fn load_from_file(&mut self, path: &str) -> bincode::Result<()> {
        println!("Ağırlıklar diskten okunuyor: {}", path);
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // let weights: AkashaWeights = bincode::deserialize_from(&mut reader)?;
        // GpuBuffer::from_cpu // ile vram e yükle

        Ok(())
    }
}

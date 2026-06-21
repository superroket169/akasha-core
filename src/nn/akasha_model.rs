use filuplex::context::Context;
use filuplex::ops::GpuBuffer;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::sync::Arc;

use super::embedding::Embedding;
use super::linear::Linear;
use super::pipeline::TransformerBlock;
use super::rmsnorm::RMSNorm;
use super::traits::Layer;
use super::weights::{AkashaWeights, TransformerBlockWeights};

pub struct AkashaModel {
    pub ctx: Arc<Context>,
    pub embedding: Embedding,
    pub layers: Vec<TransformerBlock>,
    pub final_norm: RMSNorm,
    pub lm_head: Linear,
}

impl AkashaModel {
    pub fn new(
        ctx: Arc<Context>,
        vocab_size: u32,
        dim: u32,
        seq_len: u32,
        num_layers: usize,
        input_tokens: &GpuBuffer,
    ) -> Self {
        let dummy_emb_w = vec![0.01f32; (vocab_size * dim) as usize];

        let embedding = Embedding::new(
            ctx.clone(),
            vocab_size,
            dim,
            seq_len,
            &dummy_emb_w,
            input_tokens,
        );

        let mut current_input = embedding.out_buffer.clone();

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
            embedding,
            layers,
            final_norm,
            lm_head,
        }
    }

    pub fn forward(&self) {
        self.embedding.forward();

        for layer in self.layers.iter() {
            layer.forward();
        }

        self.final_norm.forward();
        self.lm_head.forward();

        // Sonuç lm_head.out_buffer içinde
    }

    pub fn save_to_file(&self, path: &str) -> bincode::Result<()> {
        println!("The weights in VRAM are being shifting to the CPU...");

        let mut blocks_weights = Vec::new();
        for block in &self.layers {
            blocks_weights.push(TransformerBlockWeights {
                norm_1: block.norm_1.weight.to_cpu(&self.ctx),
                q_proj: block.q_proj.weight.to_cpu(&self.ctx),
                k_proj: block.k_proj.weight.to_cpu(&self.ctx),
                v_proj: block.v_proj.weight.to_cpu(&self.ctx),
                out_proj: block.out_proj.weight.to_cpu(&self.ctx),
                norm_2: block.norm_2.weight.to_cpu(&self.ctx),
                ffn_up: block.ffn_up.weight.to_cpu(&self.ctx),
                ffn_down: block.ffn_down.weight.to_cpu(&self.ctx),
            });
        }

        let all_weights = AkashaWeights {
            embedding_table: self.embedding.table.to_cpu(&self.ctx),
            blocks: blocks_weights,
            final_norm: self.final_norm.weight.to_cpu(&self.ctx),
            lm_head: self.lm_head.weight.to_cpu(&self.ctx),
        };

        println!("Weights writing to disk: {}", path);
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        bincode::serialize_into(&mut writer, &all_weights)?;

        Ok(())
    }

    pub fn load_from_file(&mut self, path: &str) -> bincode::Result<()> {
        println!("Weights reading from disk: {}", path);
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let all_weights: AkashaWeights = bincode::deserialize_from(&mut reader)?;

        println!("Datas goes to Vram...");

        self.embedding.table = GpuBuffer::from_cpu(&all_weights.embedding_table, &self.ctx);

        for (i, block_weights) in all_weights.blocks.into_iter().enumerate() {
            self.layers[i].norm_1.weight = GpuBuffer::from_cpu(&block_weights.norm_1, &self.ctx);
            self.layers[i].q_proj.weight = GpuBuffer::from_cpu(&block_weights.q_proj, &self.ctx);
            self.layers[i].k_proj.weight = GpuBuffer::from_cpu(&block_weights.k_proj, &self.ctx);
            self.layers[i].v_proj.weight = GpuBuffer::from_cpu(&block_weights.v_proj, &self.ctx);
            self.layers[i].out_proj.weight =
                GpuBuffer::from_cpu(&block_weights.out_proj, &self.ctx);
            self.layers[i].norm_2.weight = GpuBuffer::from_cpu(&block_weights.norm_2, &self.ctx);
            self.layers[i].ffn_up.weight = GpuBuffer::from_cpu(&block_weights.ffn_up, &self.ctx);
            self.layers[i].ffn_down.weight =
                GpuBuffer::from_cpu(&block_weights.ffn_down, &self.ctx);
        }

        self.final_norm.weight = GpuBuffer::from_cpu(&all_weights.final_norm, &self.ctx);
        self.lm_head.weight = GpuBuffer::from_cpu(&all_weights.lm_head, &self.ctx);

        println!("Loading is completed");

        Ok(())
    }
}

use crate::Real;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::sync::Arc;
use wilupgu::context::WgpuContext;
use wilupgu::tensor::Tensor;

use super::embedding::Embedding;
use super::linear::Linear;
use super::pipeline::TransformerBlock;
use super::rmsnorm::RMSNorm;
use super::traits::Layer;
use super::weights::{AkashaWeights, TransformerBlockWeights};

pub struct AkashaModel {
    pub ctx: Arc<WgpuContext>,
    pub embedding: Embedding,
    pub layers: Vec<TransformerBlock>,
    pub final_norm: RMSNorm,
    pub lm_head: Linear,
}

impl AkashaModel {
    pub fn new(
        ctx: Arc<WgpuContext>,
        vocab_size: u32,
        dim: u32,
        seq_len: u32,
        num_layers: usize,
        input_tokens: &Arc<Tensor>,
    ) -> Self {
        let dummy_emb_w = vec![0.01 as Real; (vocab_size * dim) as usize];
        let dummy_grad_emb = vec![0.0 as Real; (seq_len * dim) as usize];
        let t_dummy_grad_emb = Arc::new(Tensor::init_from_cpu(ctx.clone(), &dummy_grad_emb));

        let embedding = Embedding::new(
            ctx.clone(),
            vocab_size,
            dim,
            seq_len,
            &dummy_emb_w,
            input_tokens,
            &t_dummy_grad_emb,
        );

        let mut current_input = embedding.out_buffer.clone();
        let mut layers = Vec::with_capacity(num_layers);

        for _ in 0..num_layers {
            let block = TransformerBlock::new(ctx.clone(), dim, seq_len, &current_input);
            current_input = block.add_2.in_out_buffer.clone();
            layers.push(block);
        }

        let last_block = layers.last().expect("At least should be one layer!");

        let dummy_grad_dim = vec![0.0 as Real; (seq_len * dim) as usize];
        let t_dummy_grad_dim = Arc::new(Tensor::init_from_cpu(ctx.clone(), &dummy_grad_dim));

        let dummy_grad_vocab = vec![0.0 as Real; (seq_len * vocab_size) as usize];
        let t_dummy_grad_vocab = Arc::new(Tensor::init_from_cpu(ctx.clone(), &dummy_grad_vocab));

        let dummy_norm_w = vec![1.0 as Real; dim as usize];
        let final_norm = RMSNorm::new(
            ctx.clone(),
            dim,
            1,
            &dummy_norm_w,
            &last_block.add_2.in_out_buffer,
            &t_dummy_grad_dim,
        );

        let dummy_head_w = vec![0.01 as Real; (dim * vocab_size) as usize];
        let lm_head = Linear::new(
            ctx.clone(),
            dim,
            vocab_size,
            &dummy_head_w,
            &final_norm.out_buffer,
            &t_dummy_grad_vocab,
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
    }

    pub fn backward(&self) {
        self.lm_head.backward();
        self.final_norm.backward();
        for layer in self.layers.iter().rev() {
            layer.backward();
        }
        self.embedding.backward();
    }

    pub fn save_to_file(&self, path: &str) -> bincode::Result<()> {
        println!("The weights in VRAM are being shifting to the CPU...");

        let mut blocks_weights = Vec::new();
        for block in &self.layers {
            blocks_weights.push(TransformerBlockWeights {
                norm_1: block.norm_1.weight.to_cpu(),
                q_proj: block.q_proj.weight.to_cpu(),
                k_proj: block.k_proj.weight.to_cpu(),
                v_proj: block.v_proj.weight.to_cpu(),
                out_proj: block.out_proj.weight.to_cpu(),
                norm_2: block.norm_2.weight.to_cpu(),
                ffn_up: block.ffn_up.weight.to_cpu(),
                ffn_down: block.ffn_down.weight.to_cpu(),
            });
        }

        let all_weights = AkashaWeights {
            embedding_table: self.embedding.table.to_cpu(),
            blocks: blocks_weights,
            final_norm: self.final_norm.weight.to_cpu(),
            lm_head: self.lm_head.weight.to_cpu(),
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

        self.embedding
            .table
            .copy_from_cpu(&all_weights.embedding_table);

        for (i, block_weights) in all_weights.blocks.iter().enumerate() {
            self.layers[i]
                .norm_1
                .weight
                .copy_from_cpu(&block_weights.norm_1);
            self.layers[i]
                .q_proj
                .weight
                .copy_from_cpu(&block_weights.q_proj);
            self.layers[i]
                .k_proj
                .weight
                .copy_from_cpu(&block_weights.k_proj);
            self.layers[i]
                .v_proj
                .weight
                .copy_from_cpu(&block_weights.v_proj);
            self.layers[i]
                .out_proj
                .weight
                .copy_from_cpu(&block_weights.out_proj);
            self.layers[i]
                .norm_2
                .weight
                .copy_from_cpu(&block_weights.norm_2);
            self.layers[i]
                .ffn_up
                .weight
                .copy_from_cpu(&block_weights.ffn_up);
            self.layers[i]
                .ffn_down
                .weight
                .copy_from_cpu(&block_weights.ffn_down);
        }

        self.final_norm
            .weight
            .copy_from_cpu(&all_weights.final_norm);
        self.lm_head.weight.copy_from_cpu(&all_weights.lm_head);

        println!("Loading is completed");

        Ok(())
    }
}

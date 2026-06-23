use crate::Real;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::sync::Arc;
use wilupgu::context::WgpuContext;
use wilupgu::tensor::Tensor;

use super::embedding::Embedding;
use super::init::random_normal_vec;
use super::linear::Linear;
use super::pipeline::TransformerBlock;
use super::rmsnorm::RMSNorm;
use super::traits::Layer;
use super::weights::{AkashaWeights, TransformerBlockWeights};

fn zero_tensor(t: &Tensor) {
    let len = (t.size / std::mem::size_of::<Real>() as u64) as usize;
    t.copy_from_cpu(&vec![0.0 as Real; len]);
}

pub struct AkashaModel {
    pub ctx: Arc<WgpuContext>,
    pub embedding: Embedding,
    pub layers: Vec<TransformerBlock>,
    pub final_norm: RMSNorm,
    pub lm_head: Linear,
    pub grad_logits: Arc<Tensor>,
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
        assert!(num_layers >= 1, "At least one layer is required!");

        let dim_size = (seq_len * dim) as usize;
        let vocab_out_size = (seq_len * vocab_size) as usize;
        let zeros_dim = vec![0.0 as Real; dim_size];

        let grad_logits = Arc::new(Tensor::init_from_cpu(
            ctx.clone(),
            &vec![0.0 as Real; vocab_out_size],
        ));
        let g_lmhead_in = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros_dim));

        let edges: Vec<Arc<Tensor>> = (0..=num_layers)
            .map(|_| Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros_dim)))
            .collect();

        let emb_w = random_normal_vec((vocab_size * dim) as usize, 0.0, 0.02);
        let embedding = Embedding::new(
            ctx.clone(),
            vocab_size,
            dim,
            seq_len,
            &emb_w,
            input_tokens,
            &edges[0],
        );

        let mut current_input = embedding.out_buffer.clone();
        let mut layers = Vec::with_capacity(num_layers);

        for i in 0..num_layers {
            let block = TransformerBlock::new(
                ctx.clone(),
                dim,
                seq_len,
                &current_input,
                &edges[i + 1],
                &edges[i],
            );
            current_input = block.add_2.in_out_buffer.clone();
            layers.push(block);
        }

        let last_block = layers.last().expect("At least should be one layer!");

        let final_norm_w = random_normal_vec(dim as usize, 1.0, 0.02);
        let final_norm = RMSNorm::new(
            ctx.clone(),
            dim,
            seq_len,
            &final_norm_w,
            &last_block.add_2.in_out_buffer,
            &g_lmhead_in,
            &edges[num_layers],
        );

        let head_w = random_normal_vec((dim * vocab_size) as usize, 0.0, 0.02);
        let lm_head = Linear::new(
            ctx.clone(),
            dim,
            vocab_size,
            seq_len,
            &head_w,
            &final_norm.out_buffer,
            &grad_logits,
            &g_lmhead_in,
        );

        Self {
            ctx,
            embedding,
            layers,
            final_norm,
            lm_head,
            grad_logits,
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

    pub fn zero_grad(&self) {
        zero_tensor(&self.final_norm.grad_weight);
        for layer in self.layers.iter() {
            layer.zero_grad();
        }
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

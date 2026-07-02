use crate::READ_LOSS;
use crate::Real;
use std::error::Error;
use std::sync::Arc;
use wilupgu::{Backend, ComputeGraph, Tensor, fuse_compute_graphs};

use super::checkpoint;
use super::cross_entropy::CrossEntropy;
use super::embedding::Embedding;
use super::linear::Linear;
use super::ops::zero_tensor;
use super::pipeline::TransformerBlock;
use super::rmsnorm::RMSNorm;
use super::traits::Layer;
use super::weights::ModelWeights;
use crate::config::{ADAM_WEIGHT_DECAY, GRAD_CLIP_NORM, ModelConfig};
use crate::optim::AdamW;

const ADAM_BETA1: Real = 0.9;
const ADAM_BETA2: Real = 0.95;

fn collect_trainable_params<B: Backend>(
    embedding: &Embedding<B>,
    layers: &[TransformerBlock<B>],
    final_norm: &RMSNorm<B>,
    lm_head: &Linear<B>,
) -> Vec<(Arc<Tensor<B>>, Arc<Tensor<B>>)> {
    let mut params: Vec<(Arc<Tensor<B>>, Arc<Tensor<B>>)> =
        vec![(embedding.table.clone(), embedding.grad_table.clone())];
    for layer in layers.iter() {
        params.push((
            layer.norm_1.weight.clone(),
            layer.norm_1.grad_weight.clone(),
        ));
        params.push((
            layer.qkv_proj.weight.clone(),
            layer.qkv_proj.grad_weight.clone(),
        ));
        params.push((
            layer.out_proj.weight.clone(),
            layer.out_proj.grad_weight.clone(),
        ));
        params.push((
            layer.norm_2.weight.clone(),
            layer.norm_2.grad_weight.clone(),
        ));
        params.push((
            layer.ffn_up.weight.clone(),
            layer.ffn_up.grad_weight.clone(),
        ));
        params.push((
            layer.ffn_down.weight.clone(),
            layer.ffn_down.grad_weight.clone(),
        ));
    }
    params.push((final_norm.weight.clone(), final_norm.grad_weight.clone()));
    params.push((lm_head.weight.clone(), lm_head.grad_weight.clone()));
    params
}

pub struct Trainer<B: Backend> {
    pub ctx: Arc<B>,
    pub cfg: ModelConfig,
    pub weights: Arc<ModelWeights<B>>,
    pub input_tokens: Arc<Tensor<B>>,
    pub embedding: Embedding<B>,
    pub layers: Vec<TransformerBlock<B>>,
    pub final_norm: RMSNorm<B>,
    pub lm_head: Linear<B>,
    pub grad_logits: Arc<Tensor<B>>,
    pub cross_entropy: CrossEntropy<B>,
    pub optimizer: AdamW<B>,
    pub fused_forward_graph: ComputeGraph<B>,
    pub fused_backward_graph: ComputeGraph<B>,
}

impl<B: Backend> Trainer<B> {
    pub fn new(ctx: Arc<B>, weights: Arc<ModelWeights<B>>, input_tokens: &Arc<Tensor<B>>) -> Self {
        let cfg = weights.cfg;
        let ModelConfig {
            vocab_size,
            dim,
            seq_len,
            num_layers,
            ..
        } = cfg;

        let dim_size = (seq_len * dim) as usize;
        let vocab_out_size = (seq_len * vocab_size) as usize;
        let zeros_dim = vec![0.0 as Real; dim_size];

        let grad_logits = Arc::new(Tensor::init_from_cpu(
            ctx.clone(),
            &vec![0.0 as Real; vocab_out_size],
        ));
        let g_lmhead_in = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros_dim));

        let edges: Vec<Arc<Tensor<B>>> = (0..=num_layers)
            .map(|_| Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros_dim)))
            .collect();

        let embedding = Embedding::new(
            ctx.clone(),
            vocab_size,
            dim,
            seq_len,
            &weights.embedding,
            input_tokens,
            &edges[0],
        );

        let mut current_input = embedding.out_buffer.clone();
        let mut layers = Vec::with_capacity(num_layers);

        for i in 0..num_layers {
            let block = TransformerBlock::new(
                ctx.clone(),
                &cfg,
                &weights.blocks[i],
                &current_input,
                &edges[i + 1],
                &edges[i],
            );
            current_input = block.add_2.in_out_buffer.clone();
            layers.push(block);
        }

        let last_block = layers.last().expect("At least should be one layer!");

        let final_norm = RMSNorm::new(
            ctx.clone(),
            dim,
            seq_len,
            &weights.final_norm,
            &last_block.add_2.in_out_buffer,
            &g_lmhead_in,
            &edges[num_layers],
        );

        let lm_head = Linear::new(
            ctx.clone(),
            dim,
            vocab_size,
            seq_len,
            &weights.lm_head,
            &final_norm.out_buffer,
            &grad_logits,
            &g_lmhead_in,
        );

        let cross_entropy = CrossEntropy::new(
            ctx.clone(),
            vocab_size,
            seq_len,
            &lm_head.out_buffer,
            &grad_logits,
        );

        let trainable_params = collect_trainable_params(&embedding, &layers, &final_norm, &lm_head);
        let optimizer = AdamW::new(ctx.clone(), &trainable_params);

        // ---- fused forward ----
        let mut forward_graphs: Vec<&ComputeGraph<B>> = vec![&embedding.forward_graph];
        for layer in &layers {
            forward_graphs.push(&layer.norm_1.forward_graph);
            forward_graphs.push(&layer.qkv_proj.forward_graph);
            forward_graphs.push(&layer.qkv_split_forward);
            forward_graphs.push(&layer.rope_forward);
            forward_graphs.push(&layer.attention.forward_graph);
            forward_graphs.push(&layer.out_proj.forward_graph);
            forward_graphs.push(&layer.add_1.forward_graph);
            forward_graphs.push(&layer.norm_2.forward_graph);
            forward_graphs.push(&layer.ffn_up.forward_graph);
            forward_graphs.push(&layer.silu.forward_graph);
            forward_graphs.push(&layer.ffn_down.forward_graph);
            forward_graphs.push(&layer.add_2.forward_graph);
        }
        forward_graphs.push(&final_norm.forward_graph);
        forward_graphs.push(&lm_head.forward_graph);
        forward_graphs.push(&cross_entropy.forward_graph);
        let fused_forward_graph = fuse_compute_graphs(ctx.clone(), &forward_graphs);

        // ---- fused backward ----
        let mut backward_graphs: Vec<&ComputeGraph<B>> = vec![
            &cross_entropy.backward_graph,
            &lm_head.backward_graph,
            &final_norm.backward_graph,
        ];
        for layer in layers.iter().rev() {
            backward_graphs.push(&layer.backward_graph);
        }
        backward_graphs.push(&embedding.backward_graph);
        let fused_backward_graph = fuse_compute_graphs(ctx.clone(), &backward_graphs);

        Self {
            ctx,
            cfg,
            weights,
            input_tokens: input_tokens.clone(),
            embedding,
            layers,
            final_norm,
            lm_head,
            grad_logits,
            cross_entropy,
            optimizer,
            fused_forward_graph,
            fused_backward_graph,
        }
    }

    pub fn train_step(
        &self,
        input_tokens: &[u32],
        target_tokens: &[u32],
        batch_size: usize,
        lr: f32,
        step: usize,
        accumulation_steps: usize,
    ) -> Option<f32> {
        let seq_len = self.cross_entropy.seq_len as usize;
        assert_eq!(
            input_tokens.len(),
            batch_size * seq_len,
            "train_step: input_tokens must be batch_size * seq_len long"
        );
        assert_eq!(
            target_tokens.len(),
            batch_size * seq_len,
            "train_step: target_tokens must be batch_size * seq_len long"
        );
        assert!(accumulation_steps >= 1, "accumulation_steps must be >= 1");

        self.cross_entropy
            .set_grad_scale(1.0 / (seq_len * batch_size * accumulation_steps) as Real);

        let is_first_in_cycle = step % accumulation_steps == 0;
        let is_last_in_cycle = (step + 1) % accumulation_steps == 0;

        if is_first_in_cycle {
            self.zero_grad();
        }

        let read_loss = step % READ_LOSS == 0;
        let mut total_loss = 0.0 as Real;
        for i in 0..batch_size {
            let window = i * seq_len..(i + 1) * seq_len;
            self.input_tokens
                .copy_from_cpu(&input_tokens[window.clone()]);
            self.cross_entropy
                .target_tokens
                .copy_from_cpu(&target_tokens[window]);

            self.zero_transient_grads();

            self.fused_forward_graph.execute();
            if read_loss {
                total_loss += self.cross_entropy.loss();
            }

            self.backward_fused();
        }

        if is_last_in_cycle {
            self.clip_grad_norm(GRAD_CLIP_NORM);
            self.optimizer
                .step(lr, ADAM_BETA1, ADAM_BETA2, ADAM_WEIGHT_DECAY);
        }

        if read_loss {
            Some(total_loss / batch_size as Real)
        } else {
            None
        }
    }

    pub fn clip_grad_norm(&self, max_norm: f32) {
        let params = self.trainable_params();
        let grads: Vec<Vec<Real>> = params.iter().map(|(_, grad)| grad.to_cpu()).collect();

        let total_sq: f64 = grads
            .iter()
            .map(|g| g.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>())
            .sum();
        let total_norm = total_sq.sqrt() as f32;

        if total_norm > max_norm {
            let scale = max_norm / (total_norm + 1e-6);
            for ((_, grad_tensor), grad_data) in params.iter().zip(grads.iter()) {
                let scaled: Vec<Real> = grad_data.iter().map(|&x| x * scale).collect();
                grad_tensor.copy_from_cpu(&scaled);
            }
        }
    }

    pub fn trainable_params(&self) -> Vec<(Arc<Tensor<B>>, Arc<Tensor<B>>)> {
        collect_trainable_params(
            &self.embedding,
            &self.layers,
            &self.final_norm,
            &self.lm_head,
        )
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

    pub fn backward_fused(&self) {
        self.fused_backward_graph.execute();
    }

    pub fn forward_fused(&self) {
        self.fused_forward_graph.execute();
    }

    pub fn zero_grad(&self) {
        zero_tensor(&self.embedding.grad_table);
        zero_tensor(&self.final_norm.grad_weight);
        zero_tensor(&self.lm_head.grad_weight);
        for layer in self.layers.iter() {
            layer.zero_grad();
        }
    }

    pub fn zero_transient_grads(&self) {
        for layer in self.layers.iter() {
            layer.zero_transient_grads();
        }
    }

    pub fn save_weights(&self, path: &str) -> Result<(), Box<dyn Error>> {
        checkpoint::save_v2(&self.weights, path)
    }

    pub fn load_weights(&self, path: &str) -> Result<(), Box<dyn Error>> {
        let loaded = checkpoint::load(&self.weights, path)?;
        if let Some(grads) = loaded.v1_grads {
            for ((_, grad_tensor), grad_data) in self.trainable_params().iter().zip(grads.iter()) {
                grad_tensor.copy_from_cpu(grad_data);
            }
        }
        Ok(())
    }
}

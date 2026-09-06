use crate::Real;
use std::error::Error;
use std::sync::Arc;
use wilupgu::{Backend, ComputeGraph, Tensor, fuse_compute_graphs};

use super::checkpoint;
use super::layers::{CrossEntropy, Embedding, Layer, Linear, RMSNorm, TransformerBlock};
use super::kernels::{self, GraphBuilder};
use super::weights::ModelWeights;
use crate::config::{ModelConfig, TrainConfig};
use crate::optim::{AdamW, AdamWSchedule};

const ADAM_BETA1: Real = 0.9;
const ADAM_BETA2: Real = 0.95;

// The boolean flag enables weight decay.
// Applied only to matmul weights; norm gains and embeddings are excluded to prevent parameter collapse.
fn collect_trainable_params<B: Backend>(
    embedding: &Embedding<B>,
    layers: &[TransformerBlock<B>],
    final_norm: &RMSNorm<B>,
    lm_head: &Linear<B>,
) -> Vec<(Arc<Tensor<B>>, Arc<Tensor<B>>, bool)> {
    let mut params: Vec<(Arc<Tensor<B>>, Arc<Tensor<B>>, bool)> =
        vec![(embedding.table.clone(), embedding.grad_table.clone(), false)];
    for layer in layers.iter() {
        params.push((
            layer.norm_1.weight.clone(),
            layer.norm_1.grad_weight.clone(),
            false,
        ));
        params.push((
            layer.qkv_proj.weight.clone(),
            layer.qkv_proj.grad_weight.clone(),
            true,
        ));
        params.push((
            layer.out_proj.weight.clone(),
            layer.out_proj.grad_weight.clone(),
            true,
        ));
        params.push((
            layer.norm_2.weight.clone(),
            layer.norm_2.grad_weight.clone(),
            false,
        ));
        params.push((
            layer.ffn_up.weight.clone(),
            layer.ffn_up.grad_weight.clone(),
            true,
        ));
        params.push((
            layer.ffn_down.weight.clone(),
            layer.ffn_down.grad_weight.clone(),
            true,
        ));
    }
    params.push((
        final_norm.weight.clone(),
        final_norm.grad_weight.clone(),
        false,
    ));
    params.push((lm_head.weight.clone(), lm_head.grad_weight.clone(), true));
    params
}

pub struct Trainer<B: Backend> {
    pub ctx: Arc<B>,
    pub cfg: ModelConfig,
    pub train_cfg: TrainConfig,
    pub weights: Arc<ModelWeights<B>>,
    pub input_tokens: Arc<Tensor<B>>,
    pub embedding: Embedding<B>,
    pub layers: Vec<TransformerBlock<B>>,
    pub final_norm: RMSNorm<B>,
    pub lm_head: Linear<B>,
    pub logits: Arc<Tensor<B>>,
    pub cross_entropy: CrossEntropy<B>,
    pub optimizer: AdamW<B>,
    pub fused_forward_graph: ComputeGraph<B>,
    pub fused_backward_graph: ComputeGraph<B>,
    zero_grads_graph: ComputeGraph<B>,
    zero_transient_graph: ComputeGraph<B>,
    clip_grads_graph: ComputeGraph<B>,
}

fn elems<B: Backend>(t: &Tensor<B>) -> u32 {
    (t.size / std::mem::size_of::<Real>() as u64) as u32
}

impl<B: Backend> Trainer<B> {
    pub fn new(
        ctx: Arc<B>,
        weights: Arc<ModelWeights<B>>,
        input_tokens: &Arc<Tensor<B>>,
        train_cfg: TrainConfig,
    ) -> Self {
        let cfg = weights.cfg;
        let ModelConfig {
            vocab_size,
            dim,
            seq_len,
            num_layers,
            batch_size,
            ..
        } = cfg;
        let rows = batch_size * seq_len;
        assert_eq!(
            elems(input_tokens),
            rows,
            "Trainer::new: input_tokens tensor must hold batch_size * seq_len tokens"
        );

        let dim_size = (rows * dim) as usize;
        let vocab_out_size = (rows * vocab_size) as usize;
        let zeros_dim = vec![0.0 as Real; dim_size];

        let logits = Arc::new(Tensor::init_from_cpu(
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
            rows,
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
            current_input = block.add_2.out_buffer.clone();
            layers.push(block);
        }

        let last_block = layers.last().expect("At least should be one layer!");

        let final_norm = RMSNorm::new(
            ctx.clone(),
            dim,
            rows,
            cfg.norm_eps,
            &weights.final_norm,
            &last_block.add_2.out_buffer,
            &g_lmhead_in,
            &edges[num_layers],
        );

        let lm_head = Linear::new(
            ctx.clone(),
            dim,
            vocab_size,
            rows,
            &weights.lm_head,
            &final_norm.out_buffer,
            &logits,
            &logits,
            &g_lmhead_in,
        );

        let cross_entropy = CrossEntropy::new(ctx.clone(), vocab_size, rows, &logits);

        let trainable_params = collect_trainable_params(&embedding, &layers, &final_norm, &lm_head);
        let optimizer = AdamW::new(
            ctx.clone(),
            &trainable_params,
            AdamWSchedule {
                lr_max: train_cfg.lr_max,
                lr_min: train_cfg.lr_min,
                warmup_steps: (train_cfg.warmup_steps / train_cfg.accumulation_steps) as u32,
                max_steps: (train_cfg.max_steps / train_cfg.accumulation_steps) as u32,
            },
            ADAM_BETA1,
            ADAM_BETA2,
            train_cfg.adam_weight_decay,
        );

        // ---- zero-grad graphs ----
        // the weight grads only per accumulation cycle.
        let mut zero_transient_graph = ComputeGraph::new(ctx.clone());
        {
            let mut gb = GraphBuilder::train(&mut zero_transient_graph);
            for layer in &layers {
                for grad in layer.transient_grads() {
                    kernels::zero(&mut gb, grad, elems(grad));
                }
            }
        }

        let mut zero_grads_graph = ComputeGraph::new(ctx.clone());
        {
            let mut gb = GraphBuilder::train(&mut zero_grads_graph);
            for (_, grad, _) in &trainable_params {
                kernels::zero(&mut gb, grad, elems(grad));
            }
        }

        // ---- grad clip graph ----
        // (a factor of 1.0 when the norm is under train_cfg.grad_clip_norm).
        let total_partials: u32 = trainable_params
            .iter()
            .map(|(_, grad, _)| kernels::grad_sumsq_wgs(elems(grad)))
            .sum();
        let norm_partials = Arc::new(Tensor::init_from_cpu(
            ctx.clone(),
            &vec![0.0 as Real; total_partials as usize],
        ));
        let clip_scale = Arc::new(Tensor::init_from_cpu(ctx.clone(), &[1.0 as Real]));

        let mut clip_grads_graph = ComputeGraph::new(ctx.clone());
        {
            let mut gb = GraphBuilder::train(&mut clip_grads_graph);
            let mut out_offset = 0;
            for (_, grad, _) in &trainable_params {
                let len = elems(grad);
                kernels::grad_sumsq(
                    &mut gb,
                    grad,
                    &norm_partials,
                    kernels::meta::GradSumSqMeta { len, out_offset },
                );
                out_offset += kernels::grad_sumsq_wgs(len);
            }
            kernels::grad_norm_scale(
                &mut gb,
                &norm_partials,
                &clip_scale,
                kernels::meta::GradNormMeta {
                    num_partials: total_partials,
                    max_norm: train_cfg.grad_clip_norm,
                },
            );
            for (_, grad, _) in &trainable_params {
                kernels::grad_scale(&mut gb, grad, &clip_scale, elems(grad));
            }
        }

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
            train_cfg,
            weights,
            input_tokens: input_tokens.clone(),
            embedding,
            layers,
            final_norm,
            lm_head,
            logits,
            cross_entropy,
            optimizer,
            fused_forward_graph,
            fused_backward_graph,
            zero_grads_graph,
            zero_transient_graph,
            clip_grads_graph,
        }
    }

    pub fn train_step(
        &self,
        input_tokens: &[u32],
        target_tokens: &[u32],
        batch_size: usize,
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
        assert!(
            batch_size == 1 || self.cfg.batch_size == 1,
            "train_step: the batch_size argument (host-loop count) and cfg.batch_size \
             (real batching) cannot both be > 1 — see ARCHITECTURE.md invariants"
        );

        self.cross_entropy
            .set_grad_scale(1.0 / (seq_len * batch_size * accumulation_steps) as Real);

        let is_first_in_cycle = step % accumulation_steps == 0;
        let is_last_in_cycle = (step + 1) % accumulation_steps == 0;

        if is_first_in_cycle {
            self.zero_grad();
        }

        let read_loss = step % self.train_cfg.log_every == 0;
        let mut total_loss = 0.0 as Real;
        for i in 0..batch_size {
            let window = i * seq_len..(i + 1) * seq_len;
            self.input_tokens
                .copy_from_cpu(&input_tokens[window.clone()]);
            self.cross_entropy
                .target_tokens
                .copy_from_cpu(&target_tokens[window]);

            self.zero_transient_grads();

            self.fused_forward_graph.execute_captured();
            if read_loss {
                total_loss += self.cross_entropy.loss();
            }

            self.backward_fused();
        }

        if is_last_in_cycle {
            self.clip_grad_norm();
            self.optimizer.step();
        }

        if read_loss {
            Some(total_loss / batch_size as Real)
        } else {
            None
        }
    }

    pub fn clip_grad_norm(&self) {
        self.clip_grads_graph.execute_captured();
    }

    pub fn trainable_params(&self) -> Vec<(Arc<Tensor<B>>, Arc<Tensor<B>>)> {
        collect_trainable_params(
            &self.embedding,
            &self.layers,
            &self.final_norm,
            &self.lm_head,
        )
        .into_iter()
        .map(|(w, g, _)| (w, g))
        .collect()
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
        self.fused_backward_graph.execute_captured();
    }

    pub fn forward_fused(&self) {
        self.fused_forward_graph.execute_captured();
    }

    pub fn zero_grad(&self) {
        self.zero_grads_graph.execute_captured();
    }

    pub fn zero_transient_grads(&self) {
        self.zero_transient_graph.execute_captured();
    }

    /// Saves weights + full optimizer state (V3)
    pub fn save_checkpoint(&self, path: &str, train_step: u64) -> Result<(), Box<dyn Error>> {
        let (schedule_step, _) = self.optimizer.current_schedule();
        checkpoint::save(
            &self.weights,
            Some((&self.optimizer.moments, schedule_step)),
            train_step,
            path,
        )
    }

    /// weights-only/migrated files start the optimizer cold
    pub fn load_checkpoint(&self, path: &str) -> Result<u64, Box<dyn Error>> {
        let loaded = checkpoint::load(&self.weights, path)?;
        if let Some(state) = loaded.optimizer {
            self.optimizer
                .load_state(&state.moments, state.schedule_step);
        }
        Ok(loaded.train_step)
    }

    pub fn to_flat_weights(&self) -> Vec<Real> {
        self.weights
            .params()
            .iter()
            .flat_map(|t| t.to_cpu::<Real>())
            .collect()
    }

    pub fn set_flat_weights(&self, flat: &[Real]) {
        let params = self.weights.params();
        let total: usize = params
            .iter()
            .map(|t| (t.size / std::mem::size_of::<Real>() as u64) as usize)
            .sum();
        assert_eq!(
            flat.len(),
            total,
            "set_flat_weights: flat length {} doesn't match model's {total} parameters",
            flat.len()
        );
        let mut offset = 0;
        for t in &params {
            let len = (t.size / std::mem::size_of::<Real>() as u64) as usize;
            t.copy_from_cpu(&flat[offset..offset + len]);
            offset += len;
        }
    }
}


#[cfg(test)]
#[path = "../tests/train_tests.rs"]
mod tests;

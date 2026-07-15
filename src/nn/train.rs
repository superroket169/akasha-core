use crate::READ_LOSS;
use crate::Real;
use std::error::Error;
use std::sync::Arc;
use wilupgu::{Backend, ComputeGraph, Tensor, fuse_compute_graphs};

use super::checkpoint;
use super::layers::{CrossEntropy, Embedding, Layer, Linear, RMSNorm, TransformerBlock};
use super::ops::{self, GraphBuilder};
use super::weights::ModelWeights;
use crate::config::{
    ADAM_WEIGHT_DECAY, GRAD_CLIP_NORM, LR_MAX, LR_MIN, MAX_STEPS, ModelConfig, WARMUP_STEPS,
};
use crate::optim::{AdamW, AdamWSchedule};

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
    pub fn new(ctx: Arc<B>, weights: Arc<ModelWeights<B>>, input_tokens: &Arc<Tensor<B>>) -> Self {
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
                lr_max: LR_MAX,
                lr_min: LR_MIN,
                warmup_steps: WARMUP_STEPS as u32,
                max_steps: MAX_STEPS as u32,
            },
            ADAM_BETA1,
            ADAM_BETA2,
            ADAM_WEIGHT_DECAY,
        );

        // ---- zero-grad graphs ----
        // the weight grads only per accumulation cycle.
        let mut zero_transient_graph = ComputeGraph::new(ctx.clone());
        {
            let mut gb = GraphBuilder::train(&mut zero_transient_graph);
            for layer in &layers {
                for grad in layer.transient_grads() {
                    ops::zero(&mut gb, grad, elems(grad));
                }
            }
        }

        let mut zero_grads_graph = ComputeGraph::new(ctx.clone());
        {
            let mut gb = GraphBuilder::train(&mut zero_grads_graph);
            for (_, grad) in &trainable_params {
                ops::zero(&mut gb, grad, elems(grad));
            }
            for layer in &layers {
                for grad in layer.transient_grads() {
                    ops::zero(&mut gb, grad, elems(grad));
                }
            }
        }

        // ---- grad clip graph ----
        // (a factor of 1.0 when the norm is under GRAD_CLIP_NORM).
        let total_partials: u32 = trainable_params
            .iter()
            .map(|(_, grad)| ops::grad_sumsq_wgs(elems(grad)))
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
            for (_, grad) in &trainable_params {
                let len = elems(grad);
                ops::grad_sumsq(
                    &mut gb,
                    grad,
                    &norm_partials,
                    ops::meta::GradSumSqMeta { len, out_offset },
                );
                out_offset += ops::grad_sumsq_wgs(len);
            }
            ops::grad_norm_scale(
                &mut gb,
                &norm_partials,
                &clip_scale,
                ops::meta::GradNormMeta {
                    num_partials: total_partials,
                    max_norm: GRAD_CLIP_NORM,
                },
            );
            for (_, grad) in &trainable_params {
                ops::grad_scale(&mut gb, grad, &clip_scale, elems(grad));
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
             (real batching) cannot both be > 1 — see BATCHING_PLAN.md (B6)"
        );

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

#[cfg(test)]
mod fused_ops_integration {
    use super::*;
    use wilupgu::WgpuBackend;

    #[test]
    fn trainer_forward_backward_stays_finite_with_fused_ops() {
        let ctx = Arc::new(pollster::block_on(WgpuBackend::new()));

        let cfg = ModelConfig::new(37, 16, 4, 2, 11);

        let tokens: Vec<u32> = (0..cfg.seq_len).map(|i| i % cfg.vocab_size).collect();
        let targets: Vec<u32> = (0..cfg.seq_len).map(|i| (i + 1) % cfg.vocab_size).collect();
        let input_tokens = Arc::new(Tensor::init_from_cpu(ctx.clone(), &tokens));

        let weights = Arc::new(ModelWeights::random(ctx.clone(), &cfg));
        let trainer = Trainer::new(ctx.clone(), weights, &input_tokens);
        trainer.cross_entropy.target_tokens.copy_from_cpu(&targets);
        trainer.zero_grad();

        trainer.fused_forward_graph.execute_captured();
        let loss = trainer.cross_entropy.loss();
        trainer.backward_fused();
        ctx.synchronize();

        assert!(loss.is_finite(), "loss is not finite: {loss}");
        assert!(
            loss > 0.0,
            "loss should be positive cross-entropy, got {loss}"
        );

        let embed_grad = trainer.embedding.grad_table.to_cpu::<Real>();
        assert!(
            embed_grad.iter().all(|g| g.is_finite()),
            "embedding grad contains non-finite values"
        );
        assert!(
            embed_grad.iter().any(|&g| g != 0.0),
            "embedding grad is all zero -- gradient did not flow back through the fused attention/rope/qkv ops"
        );
    }
}

/// Numerical gradcheck through the REAL fused forward/backward chain test
#[cfg(test)]
mod full_chain_gradcheck {
    use super::*;
    use wilupgu::WgpuBackend;

    fn loss_at(trainer: &Trainer<WgpuBackend>) -> Real {
        trainer.fused_forward_graph.execute_captured();
        trainer.cross_entropy.loss()
    }

    fn check_param(
        trainer: &Trainer<WgpuBackend>,
        name: &str,
        weight: &Arc<Tensor<WgpuBackend>>,
        analytic: &[Real],
        indices: &[usize],
    ) {
        let eps = 1e-2 as Real;
        for &i in indices {
            let mut w: Vec<Real> = weight.to_cpu();
            let orig = w[i];

            w[i] = orig + eps;
            weight.copy_from_cpu(&w);
            let loss_plus = loss_at(trainer);

            w[i] = orig - eps;
            weight.copy_from_cpu(&w);
            let loss_minus = loss_at(trainer);

            w[i] = orig;
            weight.copy_from_cpu(&w);

            let numeric = (loss_plus - loss_minus) / (2.0 * eps);
            let denom = numeric.abs().max(analytic[i].abs()).max(1e-3);
            let rel = (numeric - analytic[i]).abs() / denom;
            assert!(
                rel < 0.08,
                "{name}[{i}]: analytic={} numeric={numeric} rel_err={rel}",
                analytic[i]
            );
        }
    }

    #[test]
    fn backward_matches_numerical_gradients_through_full_chain() {
        let ctx = Arc::new(pollster::block_on(WgpuBackend::new()));
        let cfg = ModelConfig::new(37, 16, 4, 2, 11);
        let seq_len = cfg.seq_len as usize;

        let tokens: Vec<u32> = (0..cfg.seq_len)
            .map(|i| (i * 7 + 3) % cfg.vocab_size)
            .collect();
        let targets: Vec<u32> = (0..cfg.seq_len)
            .map(|i| (i * 5 + 1) % cfg.vocab_size)
            .collect();
        let input_tokens = Arc::new(Tensor::init_from_cpu(ctx.clone(), &tokens));

        let weights = Arc::new(ModelWeights::random(ctx.clone(), &cfg));
        let trainer = Trainer::new(ctx.clone(), weights, &input_tokens);
        trainer.cross_entropy.target_tokens.copy_from_cpu(&targets);
        trainer.cross_entropy.set_grad_scale(1.0 / seq_len as Real);

        trainer.zero_grad();
        trainer.zero_transient_grads();
        trainer.fused_forward_graph.execute_captured();
        trainer.backward_fused();
        ctx.synchronize();

        let l0 = &trainer.layers[0];
        let l1 = &trainer.layers[1];
        let idx_norm = [0usize, 5, 11, 15];
        check_param(
            &trainer,
            "layer0.norm_1.weight",
            &l0.norm_1.weight,
            &l0.norm_1.grad_weight.to_cpu::<Real>(),
            &idx_norm,
        );
        check_param(
            &trainer,
            "layer1.norm_2.weight",
            &l1.norm_2.weight,
            &l1.norm_2.grad_weight.to_cpu::<Real>(),
            &idx_norm,
        );

        let idx_mat = [0usize, 100, 500, 1023];
        check_param(
            &trainer,
            "layer0.ffn_up.weight",
            &l0.ffn_up.weight,
            &l0.ffn_up.grad_weight.to_cpu::<Real>(),
            &idx_mat,
        );

        let idx_qkv = [0usize, 200, 767];
        check_param(
            &trainer,
            "layer1.qkv_proj.weight",
            &l1.qkv_proj.weight,
            &l1.qkv_proj.grad_weight.to_cpu::<Real>(),
            &idx_qkv,
        );
    }
}

/// Proves the row_offset-based real-batching design
#[cfg(test)]
mod batching_validation {
    use super::*;
    use crate::nn::weights::BlockWeights;
    use wilupgu::WgpuBackend;

    fn clone_weights<B: Backend>(w: &ModelWeights<B>, cfg: ModelConfig) -> ModelWeights<B> {
        ModelWeights {
            cfg,
            embedding: w.embedding.clone(),
            blocks: w
                .blocks
                .iter()
                .map(|b| BlockWeights {
                    norm_1: b.norm_1.clone(),
                    qkv_proj: b.qkv_proj.clone(),
                    out_proj: b.out_proj.clone(),
                    norm_2: b.norm_2.clone(),
                    ffn_up: b.ffn_up.clone(),
                    ffn_down: b.ffn_down.clone(),
                })
                .collect(),
            final_norm: w.final_norm.clone(),
            lm_head: w.lm_head.clone(),
        }
    }

    fn rand_tokens(n: usize, vocab: u32, seed: u64) -> Vec<u32> {
        let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 33) as u32) % vocab
            })
            .collect()
    }

    fn max_abs_diff(a: &[Real], b: &[Real]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn real_batching_matches_sequential_accumulation() {
        let ctx = Arc::new(pollster::block_on(WgpuBackend::new()));
        let batch: u32 = 3;
        let base_cfg = ModelConfig::new(37, 16, 4, 2, 11);
        let seq_len = base_cfg.seq_len as usize;
        let vocab = base_cfg.vocab_size;
        let dim = base_cfg.dim as usize;
        let scale = 1.0 / (batch as usize * seq_len) as Real;

        let cfg_batched = base_cfg.with_batch_size(batch);
        let weights_batched = Arc::new(ModelWeights::random(ctx.clone(), &cfg_batched));
        let weights_ref = Arc::new(clone_weights(&weights_batched, base_cfg));

        let all_inputs: Vec<u32> = (0..batch as usize)
            .flat_map(|b| rand_tokens(seq_len, vocab, 100 + b as u64))
            .collect();
        let all_targets: Vec<u32> = (0..batch as usize)
            .flat_map(|b| rand_tokens(seq_len, vocab, 200 + b as u64))
            .collect();

        // ---- reference: today's production path -- one batch_size=1
        // Trainer, called `batch` times sequentially, gradients accumulating
        // across the loop (no zero_grad in between) exactly like an
        // accumulation cycle in Trainer::train_step ----
        let ref_input_tokens = Arc::new(Tensor::init_from_cpu(ctx.clone(), &vec![0u32; seq_len]));
        let ref_trainer = Trainer::new(ctx.clone(), weights_ref, &ref_input_tokens);
        ref_trainer.cross_entropy.set_grad_scale(scale);
        ref_trainer.zero_grad();
        let mut ref_total_loss = 0.0 as Real;
        for b in 0..batch as usize {
            let window = b * seq_len..(b + 1) * seq_len;
            ref_trainer
                .input_tokens
                .copy_from_cpu(&all_inputs[window.clone()]);
            ref_trainer
                .cross_entropy
                .target_tokens
                .copy_from_cpu(&all_targets[window]);
            ref_trainer.zero_transient_grads();
            ref_trainer.fused_forward_graph.execute_captured();
            ref_total_loss += ref_trainer.cross_entropy.loss();
            ref_trainer.backward_fused();
        }
        ctx.synchronize();
        let ref_avg_loss = ref_total_loss / batch as Real;

        // last iteration's (b = batch-1) forward output is what's left in
        // ref_trainer's buffers -- compare it against the batched trainer's
        // matching row_offset slice below.
        let last_block = ref_trainer.layers.last().unwrap();
        let ref_last_out = last_block.add_2.out_buffer.to_cpu::<Real>();

        // ---- new: one batch_size=3 Trainer, ONE execute over all 33 rows ----
        let rows = batch as usize * seq_len;
        let batched_input_tokens = Arc::new(Tensor::init_from_cpu(ctx.clone(), &vec![0u32; rows]));
        let batched_trainer = Trainer::new(ctx.clone(), weights_batched, &batched_input_tokens);
        batched_trainer.cross_entropy.set_grad_scale(scale);
        batched_trainer.zero_grad();
        batched_trainer.input_tokens.copy_from_cpu(&all_inputs);
        batched_trainer
            .cross_entropy
            .target_tokens
            .copy_from_cpu(&all_targets);
        batched_trainer.zero_transient_grads();
        batched_trainer.fused_forward_graph.execute_captured();
        let batched_loss = batched_trainer.cross_entropy.loss();
        batched_trainer.backward_fused();
        ctx.synchronize();

        assert!(
            (ref_avg_loss - batched_loss).abs() < 1e-3,
            "loss mismatch: sequential avg={ref_avg_loss} batched={batched_loss}"
        );

        let batched_last_block = batched_trainer.layers.last().unwrap();
        let batched_out_full = batched_last_block.add_2.out_buffer.to_cpu::<Real>();
        let last_window = (batch as usize - 1) * seq_len * dim..batch as usize * seq_len * dim;
        let batched_last_out = &batched_out_full[last_window];
        let out_diff = max_abs_diff(&ref_last_out, batched_last_out);
        assert!(
            out_diff < 1e-3,
            "forward residual-stream mismatch at last batch item: max_abs_diff={out_diff}"
        );

        let ref_params = ref_trainer.trainable_params();
        let batched_params = batched_trainer.trainable_params();
        assert_eq!(ref_params.len(), batched_params.len());
        for (i, ((_, ref_grad), (_, batched_grad))) in
            ref_params.iter().zip(batched_params.iter()).enumerate()
        {
            let a = ref_grad.to_cpu::<Real>();
            let b = batched_grad.to_cpu::<Real>();
            let diff = max_abs_diff(&a, &b);
            assert!(
                diff < 1e-3,
                "grad mismatch at trainable_params()[{i}]: max_abs_diff={diff}"
            );
        }
    }
}

/// GPU grad clip vs. the old host-side reference formula
#[cfg(test)]
mod grad_clip_validation {
    use super::*;
    use wilupgu::WgpuBackend;

    fn check_clip(amplitude: Real) {
        let ctx = Arc::new(pollster::block_on(WgpuBackend::new()));
        let cfg = ModelConfig::new(37, 16, 4, 2, 11);

        let tokens: Vec<u32> = (0..cfg.seq_len).map(|i| i % cfg.vocab_size).collect();
        let input_tokens = Arc::new(Tensor::init_from_cpu(ctx.clone(), &tokens));
        let weights = Arc::new(ModelWeights::random(ctx.clone(), &cfg));
        let trainer = Trainer::new(ctx.clone(), weights, &input_tokens);

        let params = trainer.trainable_params();
        let mut host_grads: Vec<Vec<Real>> = Vec::new();
        for (t, (_, grad)) in params.iter().enumerate() {
            let len = (grad.size / std::mem::size_of::<Real>() as u64) as usize;
            let data: Vec<Real> = (0..len)
                .map(|i| ((t * 31 + i) as Real * 0.7).sin() * amplitude)
                .collect();
            grad.copy_from_cpu(&data);
            host_grads.push(data);
        }

        trainer.clip_grad_norm();
        ctx.synchronize();

        let total_sq: f64 = host_grads
            .iter()
            .flatten()
            .map(|&g| (g as f64) * (g as f64))
            .sum();
        let norm = total_sq.sqrt() as f32;
        let scale = if norm > GRAD_CLIP_NORM {
            GRAD_CLIP_NORM / (norm + 1e-6)
        } else {
            1.0
        };

        for (host, (_, grad)) in host_grads.iter().zip(params.iter()) {
            let gpu: Vec<Real> = grad.to_cpu();
            for (i, (&h, &g)) in host.iter().zip(gpu.iter()).enumerate() {
                let expected = h * scale;
                assert!(
                    (expected - g).abs() < 1e-6 + expected.abs() * 1e-4,
                    "amplitude {amplitude}: grad[{i}] expected {expected}, gpu {g} (norm {norm}, scale {scale})"
                );
            }
        }
    }

    #[test]
    fn clips_when_norm_exceeds_max() {
        check_clip(0.1);
    }

    #[test]
    fn leaves_grads_alone_under_max() {
        check_clip(1e-4);
    }
}

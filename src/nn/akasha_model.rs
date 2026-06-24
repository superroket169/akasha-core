use crate::READ_LOSS;
use crate::Real;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::sync::Arc;
use wilupgu::context::WgpuContext;
use wilupgu::graph::{ComputeGraph, fuse_compute_graphs};
use wilupgu::tensor::Tensor;

use super::cross_entropy::CrossEntropy;
use super::embedding::Embedding;
use super::init::{random_normal_vec, xavier_std};
use super::linear::Linear;
use super::pipeline::TransformerBlock;
use super::rmsnorm::RMSNorm;
use super::traits::Layer;
use crate::optim::AdamW;

const ADAM_BETA1: Real = 0.9;
const ADAM_BETA2: Real = 0.95;
const ADAM_WEIGHT_DECAY: Real = 0.0;

fn zero_tensor(t: &Tensor) {
    let len = (t.size / std::mem::size_of::<Real>() as u64) as usize;
    t.copy_from_cpu(&vec![0.0 as Real; len]);
}

fn collect_trainable_params(
    embedding: &Embedding,
    layers: &[TransformerBlock],
    final_norm: &RMSNorm,
    lm_head: &Linear,
) -> Vec<(Arc<Tensor>, Arc<Tensor>)> {
    let mut params: Vec<(Arc<Tensor>, Arc<Tensor>)> =
        vec![(embedding.table.clone(), embedding.grad_table.clone())];
    for layer in layers.iter() {
        params.push((
            layer.norm_1.weight.clone(),
            layer.norm_1.grad_weight.clone(),
        ));
        params.push((
            layer.q_proj.weight.clone(),
            layer.q_proj.grad_weight.clone(),
        ));
        params.push((
            layer.k_proj.weight.clone(),
            layer.k_proj.grad_weight.clone(),
        ));
        params.push((
            layer.v_proj.weight.clone(),
            layer.v_proj.grad_weight.clone(),
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

pub struct AkashaModel {
    pub ctx: Arc<WgpuContext>,
    pub input_tokens: Arc<Tensor>,
    pub embedding: Embedding,
    pub layers: Vec<TransformerBlock>,
    pub final_norm: RMSNorm,
    pub lm_head: Linear,
    pub grad_logits: Arc<Tensor>,
    pub cross_entropy: CrossEntropy,
    pub optimizer: AdamW,
    pub fused_forward_graph: ComputeGraph,
    pub fused_backward_graph: ComputeGraph,
}

impl AkashaModel {
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
            self.optimizer
                .step(lr, ADAM_BETA1, ADAM_BETA2, ADAM_WEIGHT_DECAY);
        }

        if read_loss {
            Some(total_loss / batch_size as Real)
        } else {
            None
        }
    }

    pub fn trainable_params(&self) -> Vec<(Arc<Tensor>, Arc<Tensor>)> {
        collect_trainable_params(
            &self.embedding,
            &self.layers,
            &self.final_norm,
            &self.lm_head,
        )
    }

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

        let emb_w = random_normal_vec((vocab_size * dim) as usize, 0.0, xavier_std(dim));
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

        let head_w = random_normal_vec((dim * vocab_size) as usize, 0.0, xavier_std(dim));
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
        let mut forward_graphs: Vec<&ComputeGraph> = vec![&embedding.forward_graph];
        for layer in &layers {
            forward_graphs.push(&layer.norm_1.forward_graph);
            forward_graphs.push(&layer.q_proj.forward_graph);
            forward_graphs.push(&layer.k_proj.forward_graph);
            forward_graphs.push(&layer.v_proj.forward_graph);
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
        let mut backward_graphs: Vec<&ComputeGraph> = vec![
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

    pub fn save_weights(&self, path: &str) -> bincode::Result<()> {
        let params: Vec<(Vec<Real>, Vec<Real>)> = self
            .trainable_params()
            .iter()
            .map(|(weight, grad)| (weight.to_cpu(), grad.to_cpu()))
            .collect();

        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        bincode::serialize_into(&mut writer, &params)?;
        Ok(())
    }

    pub fn load_weights(&self, path: &str) -> bincode::Result<()> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let params: Vec<(Vec<Real>, Vec<Real>)> = bincode::deserialize_from(&mut reader)?;

        let targets = self.trainable_params();
        assert_eq!(
            params.len(),
            targets.len(),
            "load_weights: saved parameter count doesn't match this model's architecture"
        );

        for ((weight, grad), (w_data, g_data)) in targets.iter().zip(params.iter()) {
            weight.copy_from_cpu(w_data);
            grad.copy_from_cpu(g_data);
        }

        Ok(())
    }
}

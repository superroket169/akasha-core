use super::arch::{build_block, build_decode_forward, build_prefill_forward};
use super::checkpoint;
use super::grad_clip::{AnyGradClip, GlobalNormClip};
use super::kernels::GraphBuilder;
use super::kernels::meta::{MatMulMeta, NormMeta};
use super::loss::{AnyLoss, CrossEntropyOp};
use super::ops::cached::DecodeOp;
use super::ops::full_seq::{EmbeddingOp, LinearOp, RmsNormOp, TrainOp};
use super::sampling;
use super::tape::{NodeId, Tape, zeros};
use super::weights::ModelWeights;
use crate::ModelError;
use crate::Real;
use crate::config::{GradClipKind, ModelConfig, OptimizerKind, TrainConfig};
use crate::optim::{AdamW, AdamWSchedule, AnyOptimizer};
use std::sync::Arc;
use wilupgu::{Backend, ComputeGraph, Tensor};

pub(crate) struct BuiltBlock<B: Backend> {
    pub(crate) tape: Tape<B, TrainOp<B>>,
    pub(crate) block_input_id: NodeId,
    pub(crate) output: NodeId,
}

pub(crate) struct DecodeGraph<B: Backend> {
    pub(crate) graph: ComputeGraph<B>,
    pub(crate) tape: Tape<B, DecodeOp<B>>,
    pub(crate) tokens: Arc<Tensor<B>>,
    pub(crate) logits_id: NodeId,
}

struct ChatState<B: Backend> {
    cache_k: Vec<Arc<Tensor<B>>>,
    cache_v: Vec<Arc<Tensor<B>>>,
    max_context_len: u32,
    decode: DecodeGraph<B>,
}

struct TrainState<B: Backend> {
    ctx: Arc<B>,
    // streaming: rebuild + redispatch a fresh graph every train_step instead
    // of capturing one forever. Required for grad_checkpoint -- a captured
    // graph's nodes each keep their own buffer clone (see wilupgu's
    // WgpuNode::buffers/CudaNode::bindings), so freed activations never
    // actually return to the pool as long as the capture is alive.
    streaming: bool,
    grad_checkpoint: bool,
    fwd_graph: Option<ComputeGraph<B>>, // None when streaming
    bwd_graph: Option<ComputeGraph<B>>,
    tokens: Arc<Tensor<B>>, // embedding's input handle -- see EmbeddingOp::tokens_handle
    head: Tape<B, TrainOp<B>>,
    embedding_id: NodeId,
    blocks: Vec<BuiltBlock<B>>,
    tail: Tape<B, TrainOp<B>>,
    tail_input_id: NodeId,
    logits_id: NodeId,
    loss: AnyLoss<B>,
    optimizer: AnyOptimizer<B>,
    grad_clip: AnyGradClip<B>,
}

pub struct Model<B: Backend> {
    weights: ModelWeights<B>,
    train: Option<TrainState<B>>,
    chat: Option<ChatState<B>>,
}

impl<B: Backend> Model<B> {
    pub fn for_training(
        ctx: Arc<B>,
        weights: ModelWeights<B>,
        cfg: ModelConfig,
        train_cfg: TrainConfig,
    ) -> Self {
        let streaming = train_cfg.run.streaming;
        let grad_checkpoint = train_cfg.run.grad_checkpoint;
        assert!(
            !grad_checkpoint || streaming,
            "grad_checkpoint requires streaming"
        );

        let rows = cfg.batch_size * cfg.seq_len;

        let mut fwd_graph = ComputeGraph::new(ctx.clone());
        let mut gb = GraphBuilder::train(&mut fwd_graph);

        let embedding_op = EmbeddingOp::new(&weights.embedding, rows, cfg.vocab_size, cfg.dim);
        let tokens = embedding_op.tokens_handle();
        let mut head = Tape::new();
        let embedding_id = head.push(&mut gb, TrainOp::Embedding(embedding_op), &[]);

        let mut x = head.output(embedding_id);
        let mut blocks = Vec::with_capacity(weights.blocks.len());
        for (bw, kind) in weights.blocks.iter().zip(cfg.layers()) {
            let built = build_block(&mut gb, bw, &cfg, kind, x);
            x = built.tape.output(built.output);
            blocks.push(built);
        }

        let mut tail = Tape::new();
        let tail_input_id = tail.input(&mut gb, x);
        let norm_shape = NormMeta {
            seq_len: rows,
            size: cfg.dim,
            eps: cfg.norm_eps,
        };
        let final_norm = tail.push(
            &mut gb,
            TrainOp::RmsNorm(RmsNormOp::new(&weights.final_norm, norm_shape)),
            &[(tail_input_id, 0)],
        );
        let lm_shape = MatMulMeta {
            m: rows,
            n: cfg.vocab_size,
            k: cfg.dim,
        };
        let logits_id = tail.push(
            &mut gb,
            TrainOp::Linear(LinearOp::new(&weights.lm_head, lm_shape, true)),
            &[(final_norm, 0)],
        );

        let loss = AnyLoss::CrossEntropy(CrossEntropyOp::new(&ctx, cfg.vocab_size, rows));
        // node only -- real targets are set per step via `train_step`.
        loss.set_targets(&vec![0u32; rows as usize]);
        loss.forward(&mut gb, &tail.output(logits_id));
        loss.set_grad_scale(1.0 / (rows * train_cfg.run.accumulation_steps as u32) as Real);

        // ---- backward: tail -> blocks (reverse) -> head, one shared graph ----
        let mut bwd_graph = ComputeGraph::new(ctx.clone());
        let mut gb_bwd = GraphBuilder::train(&mut bwd_graph);

        let logits_buf = tail.output(logits_id);
        loss.backward(&mut gb_bwd, &logits_buf);
        tail.backward(&mut gb_bwd, (logits_id, 0), &logits_buf);
        let mut grad = tail
            .grad_of((tail_input_id, 0))
            .expect("tail backward didn't reach its input");

        for block in blocks.iter_mut().rev() {
            block.tape.backward(&mut gb_bwd, (block.output, 0), &grad);
            grad = block
                .tape
                .grad_of((block.block_input_id, 0))
                .expect("block backward didn't reach its input");
        }
        head.backward(&mut gb_bwd, (embedding_id, 0), &grad);

        // ---- params, in push order (checkpoint/AdamW-moment contract) ----
        let mut params: Vec<(Arc<Tensor<B>>, Arc<Tensor<B>>, bool)> = Vec::new();
        params.extend(
            head.params()
                .into_iter()
                .map(|(w, g, d)| (w.clone(), g.clone(), d)),
        );
        for block in &blocks {
            params.extend(
                block
                    .tape
                    .params()
                    .into_iter()
                    .map(|(w, g, d)| (w.clone(), g.clone(), d)),
            );
        }
        params.extend(
            tail.params()
                .into_iter()
                .map(|(w, g, d)| (w.clone(), g.clone(), d)),
        );

        let optimizer = match train_cfg.optimizer.kind {
            OptimizerKind::AdamW => AnyOptimizer::AdamW(AdamW::new(
                ctx.clone(),
                &params,
                AdamWSchedule {
                    lr_max: train_cfg.optimizer.lr_max,
                    lr_min: train_cfg.optimizer.lr_min,
                    warmup_steps: train_cfg.optimizer.warmup_steps as u32,
                    max_steps: train_cfg.optimizer.max_steps as u32,
                },
                train_cfg.optimizer.beta1,
                train_cfg.optimizer.beta2,
                train_cfg.optimizer.weight_decay,
            )),
        };

        let grads: Vec<Arc<Tensor<B>>> = params.iter().map(|(_, g, _)| g.clone()).collect();
        let grad_clip = match train_cfg.grad_clip.kind {
            GradClipKind::GlobalNorm => AnyGradClip::GlobalNorm(GlobalNormClip::new(
                ctx.clone(),
                &grads,
                train_cfg.grad_clip.max_norm,
            )),
        };

        Self {
            weights,
            chat: None,
            train: Some(TrainState {
                streaming,
                grad_checkpoint,
                fwd_graph: if streaming { None } else { Some(fwd_graph) },
                bwd_graph: if streaming { None } else { Some(bwd_graph) },
                tokens,
                head,
                embedding_id,
                blocks,
                tail,
                tail_input_id,
                logits_id,
                loss,
                optimizer,
                grad_clip,
                ctx,
            }),
        }
    }

    pub fn for_chat(
        ctx: Arc<B>,
        weights: ModelWeights<B>,
        cfg: ModelConfig,
        max_context_len: u32,
    ) -> Self {
        let cache_len = max_context_len * cfg.dim;
        let cache_k: Vec<_> = weights
            .blocks
            .iter()
            .map(|_| zeros(&ctx, cache_len))
            .collect();

        let cache_v: Vec<_> = weights
            .blocks
            .iter()
            .map(|_| zeros(&ctx, cache_len))
            .collect();

        let decode =
            build_decode_forward(&ctx, &weights, &cfg, &cache_k, &cache_v, max_context_len);

        Self {
            weights,
            train: None,
            chat: Some(ChatState {
                cache_k,
                cache_v,
                max_context_len,
                decode,
            }),
        }
    }

    pub fn prefill_logits(&mut self, prompt: &[u32]) -> Result<Vec<Real>, ModelError> {
        if prompt.is_empty() {
            return Err(ModelError::EmptyPrompt);
        }

        let cfg = self.weights.cfg;
        let ctx = self.weights.embedding.ctx.clone();
        let chat = self
            .chat
            .as_mut()
            .expect("prefill_logits called on a training-only Model");

        let prompt_len = prompt.len() as u32;
        if prompt_len > chat.max_context_len {
            return Err(ModelError::PromptTooLong {
                len: prompt_len,
                max: chat.max_context_len,
            });
        }

        let mut prefill_graph = ComputeGraph::new(ctx.clone());
        let mut gb = GraphBuilder::prefill(&mut prefill_graph);
        let (prefill_tape, prefill_tokens, prefill_logits) = build_prefill_forward(
            &mut gb,
            &self.weights,
            &cfg,
            prompt_len,
            &chat.cache_k,
            &chat.cache_v,
        );
        drop(gb);
        prefill_tokens.copy_from_cpu(prompt);
        prefill_graph.execute();

        let vocab = cfg.vocab_size as usize;
        let all_logits: Vec<Real> = prefill_tape.output(prefill_logits).to_cpu();
        Ok(all_logits[(prompt_len as usize - 1) * vocab..prompt_len as usize * vocab].to_vec())
    }

    pub fn decode_step_logits(&mut self, token: u32, pos: u32) -> Result<Vec<Real>, ModelError> {
        let chat = self
            .chat
            .as_mut()
            .expect("decode_step_logits called on a training-only Model");
        if pos >= chat.max_context_len {
            return Err(ModelError::ContextFull {
                max: chat.max_context_len,
            });
        }
        chat.decode.tape.advance(pos);
        chat.decode.tokens.copy_from_cpu(&[token]);
        chat.decode.graph.execute();
        Ok(chat.decode.tape.output(chat.decode.logits_id).to_cpu())
    }

    pub fn reset_cache(&mut self) {
        let chat = self
            .chat
            .as_mut()
            .expect("reset_cache called on a training-only Model");
        for k in chat.cache_k.iter().chain(chat.cache_v.iter()) {
            let n = (k.size / std::mem::size_of::<Real>() as u64) as usize;
            k.copy_from_cpu(&vec![0.0 as Real; n]);
        }
    }

    pub fn generate(
        &mut self,
        prompt: &[u32],
        max_new_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        repetition_penalty: f32,
    ) -> Result<Vec<u32>, ModelError> {
        let last = self.prefill_logits(prompt)?;
        let cfg = self.weights.cfg;
        let max_context_len = self.chat.as_ref().unwrap().max_context_len;

        let eos = cfg.eos_token;
        let mut seen: Vec<u32> = prompt.to_vec();
        let mut generated = Vec::with_capacity(max_new_tokens);
        let mut next =
            sampling::sample_token(&last, temperature, top_k, top_p, &seen, repetition_penalty);
        generated.push(next);
        seen.push(next);

        let mut pos = prompt.len() as u32;
        for _ in 1..max_new_tokens {
            if next == eos || pos >= max_context_len {
                break;
            }
            let logits = self.decode_step_logits(next, pos)?;
            next = sampling::sample_token(
                &logits,
                temperature,
                top_k,
                top_p,
                &seen,
                repetition_penalty,
            );
            generated.push(next);
            seen.push(next);
            pos += 1;
        }

        Ok(generated)
    }

    fn streaming_forward(t: &mut TrainState<B>) -> ComputeGraph<B> {
        let mut fwd_graph = ComputeGraph::new(t.ctx.clone());
        let mut gb = GraphBuilder::train(&mut fwd_graph);
        t.head.redispatch(&mut gb);

        for block in &mut t.blocks {
            block.tape.redispatch(&mut gb);
            if t.grad_checkpoint {
                block.tape.free_activations();
            }
        }

        t.tail.redispatch(&mut gb);
        let logits = t.tail.output(t.logits_id);
        t.loss.forward(&mut gb, &logits);
        drop(gb);
        fwd_graph
    }

    pub fn train_step(&mut self, tokens: &[u32], targets: &[u32]) -> Real {
        let t = self
            .train
            .as_mut()
            .expect("train_step called on a chat-only Model");
        t.tokens.copy_from_cpu(tokens);
        t.loss.set_targets(targets);

        if !t.streaming {
            t.fwd_graph
                .as_ref()
                .expect("captured mode always keeps fwd_graph")
                .execute_captured();
            let loss = t.loss.loss();
            t.bwd_graph
                .as_ref()
                .expect("captured mode always keeps bwd_graph")
                .execute_captured();
            return loss;
        }

        let fwd_graph = Self::streaming_forward(t);
        fwd_graph.execute();
        let loss = t.loss.loss();

        let mut bwd_graph = ComputeGraph::new(t.ctx.clone());
        {
            let mut gb_bwd = GraphBuilder::train(&mut bwd_graph);
            let logits_buf = t.tail.output(t.logits_id);
            t.loss.backward(&mut gb_bwd, &logits_buf);
            t.tail.backward(&mut gb_bwd, (t.logits_id, 0), &logits_buf);

            let mut grad = t
                .tail
                .grad_of((t.tail_input_id, 0))
                .expect("tail backward didn't reach its input");

            for block in t.blocks.iter_mut().rev() {
                if t.grad_checkpoint {
                    block.tape.redispatch(&mut gb_bwd); // recompute saved_input/saved
                }

                block.tape.backward(&mut gb_bwd, (block.output, 0), &grad);
                grad = block
                    .tape
                    .grad_of((block.block_input_id, 0))
                    .expect("block backward didn't reach its input");

                if t.grad_checkpoint {
                    block.tape.free_activations();
                }
            }
            t.head.backward(&mut gb_bwd, (t.embedding_id, 0), &grad);
        }
        bwd_graph.execute();

        loss
    }

    pub fn eval_loss(&mut self, tokens: &[u32], targets: &[u32]) -> Real {
        let t = self
            .train
            .as_mut()
            .expect("eval_loss called on a chat-only Model");
        t.tokens.copy_from_cpu(tokens);
        t.loss.set_targets(targets);

        if !t.streaming {
            t.fwd_graph
                .as_ref()
                .expect("captured mode always keeps fwd_graph")
                .execute_captured();
            return t.loss.loss();
        }

        let fwd_graph = Self::streaming_forward(t);
        fwd_graph.execute();
        t.loss.loss()
    }

    pub fn current_lr(&self) -> (u32, Real) {
        let t = self
            .train
            .as_ref()
            .expect("current_lr called on a chat-only Model");
        t.optimizer.current_schedule()
    }

    pub fn params(&self) -> Vec<(Arc<Tensor<B>>, Arc<Tensor<B>>, bool)> {
        let t = self
            .train
            .as_ref()
            .expect("params called on a chat-only Model");
        t.head
            .params()
            .into_iter()
            .map(|(w, g, d)| (w.clone(), g.clone(), d))
            .chain(
                t.blocks
                    .iter()
                    .flat_map(|b| b.tape.params())
                    .map(|(w, g, d)| (w.clone(), g.clone(), d)),
            )
            .chain(
                t.tail
                    .params()
                    .into_iter()
                    .map(|(w, g, d)| (w.clone(), g.clone(), d)),
            )
            .collect()
    }

    pub fn zero_grad(&self) {
        for (_, grad, _) in self.params() {
            grad.copy_from_cpu(&vec![
                0.0 as Real;
                (grad.size / std::mem::size_of::<Real>() as u64)
                    as usize
            ]);
        }
    }

    pub fn optimizer_step(&self) {
        let t = self
            .train
            .as_ref()
            .expect("optimizer_step called on a chat-only Model");
        t.grad_clip.clip();
        t.optimizer.step();
    }

    pub fn weights(&self) -> &ModelWeights<B> {
        &self.weights
    }

    pub fn max_context_len(&self) -> u32 {
        self.chat
            .as_ref()
            .expect("max_context_len called on a training-only Model")
            .max_context_len
    }

    /// Saves weights + full optimizer state (V3)
    pub fn save_checkpoint(
        &self,
        path: &str,
        train_step: u64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let t = self
            .train
            .as_ref()
            .expect("save_checkpoint called on a chat-only Model");
        let (schedule_step, _) = t.optimizer.current_schedule();

        checkpoint::save(
            &self.weights,
            Some((t.optimizer.moments(), schedule_step)),
            train_step,
            path,
        )
    }

    /// weights-only/migrated files start the optimizer cold
    pub fn load_checkpoint(&self, path: &str) -> Result<u64, Box<dyn std::error::Error>> {
        let t = self
            .train
            .as_ref()
            .expect("load_checkpoint called on a chat-only Model");
        let loaded = checkpoint::load(&self.weights, path)?;

        if let Some(state) = loaded.optimizer {
            t.optimizer.load_state(&state.moments, state.schedule_step);
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
#[path = "../tests/model_tests.rs"]
mod tests;

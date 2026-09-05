use super::chain::{
    CacheWriteOp, CachedAttentionOp, DecodeOp, HeadGatherOp, PrefillOp, RopeOffsetOp,
};
use super::grad_clip::{AnyGradClip, GlobalNormClip};
use super::loss::{AnyLoss, CrossEntropyOp};
use super::ops::meta::{MatMulMeta, NormMeta};
use super::ops::{GraphBuilder, Prefill, Train};
use super::sampling;
use super::tape::{NodeId, Tape, zeros};
use super::transformer::{
    AddOp, AttentionOp, EmbeddingOp, LinearOp, QkvSplitOp, RmsNormOp, RopeQkOp, SiluOp,
    TrainOp,
};
use super::weights::{BlockWeights, ModelWeights};
use crate::Real;
use crate::config::{BlockKind, GradClipKind, ModelConfig, OptimizerKind, TrainConfig};
use crate::optim::{AdamW, AdamWSchedule, AnyOptimizer};
use std::sync::Arc;
use wilupgu::{Backend, ComputeGraph, Tensor};

struct BuiltBlock<B: Backend> {
    tape: Tape<B, TrainOp<B>>,
    block_input_id: NodeId,
    output: NodeId,
}

#[allow(clippy::too_many_arguments)]
fn build_transformer_block<B: Backend>(
    gb: &mut GraphBuilder<'_, B, Train>,
    bw: &BlockWeights<B>,
    cfg: &ModelConfig,
    block_input: Arc<Tensor<B>>,
) -> BuiltBlock<B> {
    let rows = cfg.batch_size * cfg.seq_len;
    let dim = cfg.dim;
    let head_dim = cfg.head_dim();
    let hidden = cfg.ffn_hidden;
    let ctx = &bw.qkv_proj.ctx;

    let mut tape = Tape::new();
    let block_input_id = tape.input(gb, block_input);

    let norm_shape = NormMeta {
        seq_len: rows,
        size: dim,
        eps: cfg.norm_eps,
    };
    let n1 = tape.push(
        gb,
        TrainOp::RmsNorm(RmsNormOp::new(&bw.norm_1, norm_shape)),
        &[(block_input_id, 0)],
    );

    let qkv_shape = MatMulMeta {
        m: rows,
        n: dim * 3,
        k: dim,
    };
    let qkv = tape.push(
        gb,
        TrainOp::Linear(LinearOp::new(&bw.qkv_proj, qkv_shape, true)),
        &[(n1, 0)],
    );

    // 3 outputs: q=slot0, k=slot1, v=slot2
    let split = tape.push(
        gb,
        TrainOp::QkvSplit(QkvSplitOp::new(ctx, rows, dim)),
        &[(qkv, 0)],
    );

    // 2 outputs: rotated q=slot0, k=slot1
    let rope = tape.push(
        gb,
        TrainOp::RopeQk(RopeQkOp::new(cfg.seq_len, dim, head_dim, cfg.batch_size)),
        &[(split, 0), (split, 1)],
    );

    let attn = tape.push(
        gb,
        TrainOp::Attention(AttentionOp::new(
            ctx,
            cfg.seq_len,
            dim,
            head_dim,
            cfg.batch_size,
        )),
        &[(rope, 0), (rope, 1), (split, 2)],
    );

    let out_proj_shape = MatMulMeta {
        m: rows,
        n: dim,
        k: dim,
    };
    let proj = tape.push(
        gb,
        TrainOp::Linear(LinearOp::new(&bw.out_proj, out_proj_shape, true)),
        &[(attn, 0)],
    );

    let add1 = tape.push(
        gb,
        TrainOp::Add(AddOp::new(ctx, rows * dim)),
        &[(block_input_id, 0), (proj, 0)],
    );

    let n2 = tape.push(
        gb,
        TrainOp::RmsNorm(RmsNormOp::new(&bw.norm_2, norm_shape)),
        &[(add1, 0)],
    );

    let ffn_up_shape = MatMulMeta {
        m: rows,
        n: hidden,
        k: dim,
    };
    let up = tape.push(
        gb,
        TrainOp::Linear(LinearOp::new(&bw.ffn_up, ffn_up_shape, true)),
        &[(n2, 0)],
    );

    let silu = tape.push(
        gb,
        TrainOp::Silu(SiluOp::new(ctx, rows * hidden)),
        &[(up, 0)],
    );

    let ffn_down_shape = MatMulMeta {
        m: rows,
        n: dim,
        k: hidden,
    };
    let down = tape.push(
        gb,
        TrainOp::Linear(LinearOp::new(&bw.ffn_down, ffn_down_shape, true)),
        &[(silu, 0)],
    );

    let add2 = tape.push(
        gb,
        TrainOp::Add(AddOp::new(ctx, rows * dim)),
        &[(add1, 0), (down, 0)],
    );

    BuiltBlock {
        tape,
        block_input_id,
        output: add2,
    }
}

#[allow(clippy::too_many_arguments)]
fn build_prefill_forward<B: Backend>(
    gb: &mut GraphBuilder<'_, B, Prefill>,
    weights: &ModelWeights<B>,
    cfg: &ModelConfig,
    prompt_len: u32,
    cache_k: &[Arc<Tensor<B>>],
    cache_v: &[Arc<Tensor<B>>],
) -> (Tape<B, PrefillOp<B>>, Arc<Tensor<B>>, NodeId) {
    let dim = cfg.dim;
    let head_dim = cfg.head_dim();
    let hidden = cfg.ffn_hidden;
    let ctx = &weights.embedding.ctx;

    let mut tape = Tape::new();

    let embedding_op = EmbeddingOp::new(&weights.embedding, prompt_len, cfg.vocab_size, dim);
    let tokens = embedding_op.tokens_handle();
    let mut x = tape.push(gb, PrefillOp::Embedding(embedding_op), &[]);

    let norm_shape = NormMeta {
        seq_len: prompt_len,
        size: dim,
        eps: cfg.norm_eps,
    };
    for ((bw, ck), cv) in weights.blocks.iter().zip(cache_k).zip(cache_v) {
        let block_input = x;
        let n1 = tape.push(
            gb,
            PrefillOp::RmsNorm(RmsNormOp::new(&bw.norm_1, norm_shape)),
            &[(block_input, 0)],
        );
        let qkv_shape = MatMulMeta {
            m: prompt_len,
            n: dim * 3,
            k: dim,
        };
        let qkv = tape.push(
            gb,
            PrefillOp::Linear(LinearOp::new(&bw.qkv_proj, qkv_shape, true)),
            &[(n1, 0)],
        );

        let split = tape.push(
            gb,
            PrefillOp::QkvSplit(QkvSplitOp::new(ctx, prompt_len, dim)),
            &[(qkv, 0)],
        );
        let rope = tape.push(
            gb,
            PrefillOp::RopeQk(RopeQkOp::new(prompt_len, dim, head_dim, 1)),
            &[(split, 0), (split, 1)],
        );
        let k_written = tape.push(
            gb,
            PrefillOp::CacheWrite(CacheWriteOp::new(ck.clone(), prompt_len, dim)),
            &[(rope, 1)],
        );
        let v_written = tape.push(
            gb,
            PrefillOp::CacheWrite(CacheWriteOp::new(cv.clone(), prompt_len, dim)),
            &[(split, 2)],
        );
        let attn = tape.push(
            gb,
            PrefillOp::Attention(AttentionOp::new(ctx, prompt_len, dim, head_dim, 1)),
            &[(rope, 0), (k_written, 0), (v_written, 0)],
        );

        let out_proj_shape = MatMulMeta {
            m: prompt_len,
            n: dim,
            k: dim,
        };
        let proj = tape.push(
            gb,
            PrefillOp::Linear(LinearOp::new(&bw.out_proj, out_proj_shape, true)),
            &[(attn, 0)],
        );
        let add1 = tape.push(
            gb,
            PrefillOp::Add(AddOp::new(ctx, prompt_len * dim)),
            &[(block_input, 0), (proj, 0)],
        );
        let n2 = tape.push(
            gb,
            PrefillOp::RmsNorm(RmsNormOp::new(&bw.norm_2, norm_shape)),
            &[(add1, 0)],
        );
        let ffn_up_shape = MatMulMeta {
            m: prompt_len,
            n: hidden,
            k: dim,
        };
        let up = tape.push(
            gb,
            PrefillOp::Linear(LinearOp::new(&bw.ffn_up, ffn_up_shape, true)),
            &[(n2, 0)],
        );
        let silu = tape.push(
            gb,
            PrefillOp::Silu(SiluOp::new(ctx, prompt_len * hidden)),
            &[(up, 0)],
        );
        let ffn_down_shape = MatMulMeta {
            m: prompt_len,
            n: dim,
            k: hidden,
        };
        let down = tape.push(
            gb,
            PrefillOp::Linear(LinearOp::new(&bw.ffn_down, ffn_down_shape, true)),
            &[(silu, 0)],
        );
        x = tape.push(
            gb,
            PrefillOp::Add(AddOp::new(ctx, prompt_len * dim)),
            &[(add1, 0), (down, 0)],
        );
    }

    let final_norm = tape.push(
        gb,
        PrefillOp::RmsNorm(RmsNormOp::new(&weights.final_norm, norm_shape)),
        &[(x, 0)],
    );
    let lm_shape = MatMulMeta {
        m: prompt_len,
        n: cfg.vocab_size,
        k: dim,
    };
    let logits = tape.push(
        gb,
        PrefillOp::Linear(LinearOp::new(&weights.lm_head, lm_shape, true)),
        &[(final_norm, 0)],
    );

    (tape, tokens, logits)
}

struct DecodeGraph<B: Backend> {
    graph: ComputeGraph<B>,
    tape: Tape<B, DecodeOp<B>>,
    tokens: Arc<Tensor<B>>,
    logits_id: NodeId,
}

fn build_decode_forward<B: Backend>(
    ctx: &Arc<B>,
    weights: &ModelWeights<B>,
    cfg: &ModelConfig,
    cache_k: &[Arc<Tensor<B>>],
    cache_v: &[Arc<Tensor<B>>],
    max_context_len: u32,
) -> DecodeGraph<B> {
    let dim = cfg.dim;
    let head_dim = cfg.head_dim();
    let hidden = cfg.ffn_hidden;

    let mut graph = ComputeGraph::new(ctx.clone());
    let mut gb = GraphBuilder::decode(&mut graph);
    let mut tape = Tape::new();

    let embedding_op = EmbeddingOp::new(&weights.embedding, 1, cfg.vocab_size, dim);
    let tokens = embedding_op.tokens_handle();
    let mut x = tape.push(&mut gb, DecodeOp::Embedding(embedding_op), &[]);

    let norm_shape = NormMeta {
        seq_len: 1,
        size: dim,
        eps: cfg.norm_eps,
    };
    for ((bw, ck), cv) in weights.blocks.iter().zip(cache_k).zip(cache_v) {
        let block_input = x;
        let n1 = tape.push(
            &mut gb,
            DecodeOp::RmsNorm(RmsNormOp::new(&bw.norm_1, norm_shape)),
            &[(block_input, 0)],
        );
        let qkv_shape = MatMulMeta {
            m: 1,
            n: dim * 3,
            k: dim,
        };
        let qkv = tape.push(
            &mut gb,
            DecodeOp::Linear(LinearOp::new(&bw.qkv_proj, qkv_shape, true)),
            &[(n1, 0)],
        );

        let q = tape.push(
            &mut gb,
            DecodeOp::HeadGather(HeadGatherOp::new(ctx, dim, 0)),
            &[(qkv, 0)],
        );
        let k = tape.push(
            &mut gb,
            DecodeOp::HeadGather(HeadGatherOp::new(ctx, dim, dim)),
            &[(qkv, 0)],
        );
        let v = tape.push(
            &mut gb,
            DecodeOp::HeadGather(HeadGatherOp::new(ctx, dim, 2 * dim)),
            &[(qkv, 0)],
        );

        let rope_q = tape.push(
            &mut gb,
            DecodeOp::RopeOffset(RopeOffsetOp::new(ctx, dim, head_dim)),
            &[(q, 0)],
        );
        let rope_k = tape.push(
            &mut gb,
            DecodeOp::RopeOffset(RopeOffsetOp::new(ctx, dim, head_dim)),
            &[(k, 0)],
        );
        tape.push(
            &mut gb,
            DecodeOp::CacheWrite(CacheWriteOp::new(ck.clone(), 1, dim)),
            &[(rope_k, 0)],
        );
        tape.push(
            &mut gb,
            DecodeOp::CacheWrite(CacheWriteOp::new(cv.clone(), 1, dim)),
            &[(v, 0)],
        );
        let attn = tape.push(
            &mut gb,
            DecodeOp::CachedAttention(CachedAttentionOp::new(
                ck.clone(),
                cv.clone(),
                cfg.num_heads,
                dim,
                head_dim,
                max_context_len,
            )),
            &[(rope_q, 0)],
        );

        let out_proj_shape = MatMulMeta {
            m: 1,
            n: dim,
            k: dim,
        };
        let proj = tape.push(
            &mut gb,
            DecodeOp::Linear(LinearOp::new(&bw.out_proj, out_proj_shape, true)),
            &[(attn, 0)],
        );
        let add1 = tape.push(
            &mut gb,
            DecodeOp::Add(AddOp::new(ctx, dim)),
            &[(block_input, 0), (proj, 0)],
        );
        let n2 = tape.push(
            &mut gb,
            DecodeOp::RmsNorm(RmsNormOp::new(&bw.norm_2, norm_shape)),
            &[(add1, 0)],
        );
        let ffn_up_shape = MatMulMeta {
            m: 1,
            n: hidden,
            k: dim,
        };
        let up = tape.push(
            &mut gb,
            DecodeOp::Linear(LinearOp::new(&bw.ffn_up, ffn_up_shape, true)),
            &[(n2, 0)],
        );
        let silu = tape.push(
            &mut gb,
            DecodeOp::Silu(SiluOp::new(ctx, hidden)),
            &[(up, 0)],
        );
        let ffn_down_shape = MatMulMeta {
            m: 1,
            n: dim,
            k: hidden,
        };
        let down = tape.push(
            &mut gb,
            DecodeOp::Linear(LinearOp::new(&bw.ffn_down, ffn_down_shape, true)),
            &[(silu, 0)],
        );
        x = tape.push(
            &mut gb,
            DecodeOp::Add(AddOp::new(ctx, dim)),
            &[(add1, 0), (down, 0)],
        );
    }

    let final_norm = tape.push(
        &mut gb,
        DecodeOp::RmsNorm(RmsNormOp::new(&weights.final_norm, norm_shape)),
        &[(x, 0)],
    );
    let lm_shape = MatMulMeta {
        m: 1,
        n: cfg.vocab_size,
        k: dim,
    };
    let logits_id = tape.push(
        &mut gb,
        DecodeOp::Linear(LinearOp::new(&weights.lm_head, lm_shape, true)),
        &[(final_norm, 0)],
    );

    DecodeGraph {
        graph,
        tape,
        tokens,
        logits_id,
    }
}

struct ChatState<B: Backend> {
    cache_k: Vec<Arc<Tensor<B>>>,
    cache_v: Vec<Arc<Tensor<B>>>,
    max_context_len: u32,
    decode: DecodeGraph<B>,
}

struct TrainState<B: Backend> {
    fwd_graph: ComputeGraph<B>,
    bwd_graph: ComputeGraph<B>,
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
            match kind {
                BlockKind::Transformer => {
                    let built = build_transformer_block(&mut gb, bw, &cfg, x);
                    x = built.tape.output(built.output);
                    blocks.push(built);
                }
            }
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
                fwd_graph,
                bwd_graph,
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

    /// Prefills `prompt`, then decodes up to `max_new_tokens` more, one at a
    /// time, via the persistent Decode tape (`tape.advance(pos)` per step --
    /// see `tape.rs`'s `Advance` trait).
    pub fn generate(
        &mut self,
        prompt: &[u32],
        max_new_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        repetition_penalty: f32,
    ) -> Vec<u32> {
        let cfg = self.weights.cfg;
        let ctx = self.weights.embedding.ctx.clone();
        let chat = self
            .chat
            .as_mut()
            .expect("generate called on a training-only Model");

        let prompt_len = prompt.len() as u32;
        assert!(
            prompt_len + max_new_tokens as u32 <= chat.max_context_len,
            "prompt + max_new_tokens ({}) exceeds max_context_len ({})",
            prompt_len + max_new_tokens as u32,
            chat.max_context_len,
        );

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
        let last = &all_logits[(prompt_len as usize - 1) * vocab..prompt_len as usize * vocab];

        let mut seen: Vec<u32> = prompt.to_vec();
        let mut generated = Vec::with_capacity(max_new_tokens);
        let mut next =
            sampling::sample_token(last, temperature, top_k, top_p, &seen, repetition_penalty);
        generated.push(next);
        seen.push(next);

        let mut pos = prompt_len;
        for _ in 1..max_new_tokens {
            chat.decode.tape.advance(pos);
            chat.decode.tokens.copy_from_cpu(&[next]);
            chat.decode.graph.execute();
            let logits: Vec<Real> = chat.decode.tape.output(chat.decode.logits_id).to_cpu();
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

        generated
    }

    pub fn train_step(&mut self, tokens: &[u32], targets: &[u32]) -> Real {
        let t = self
            .train
            .as_mut()
            .expect("train_step called on a chat-only Model");
        t.tokens.copy_from_cpu(tokens);
        t.loss.set_targets(targets);
        t.fwd_graph.execute_captured();
        let loss = t.loss.loss();
        t.bwd_graph.execute_captured();
        loss
    }

    pub fn zero_grad(&self) {
        let t = self
            .train
            .as_ref()
            .expect("zero_grad called on a chat-only Model");
        for (_, grad, _) in t
            .head
            .params()
            .into_iter()
            .chain(t.blocks.iter().flat_map(|b| b.tape.params()))
            .chain(t.tail.params())
        {
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
}

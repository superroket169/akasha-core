use super::grad_clip::{AnyGradClip, GlobalNormClip};
use super::loss::{AnyLoss, CrossEntropyOp};
use super::ops::meta::{MatMulMeta, NormMeta};
use super::ops::{GraphBuilder, Train};
use super::tape::{NodeId, Tape};
use super::transformer_ops::{
    AddOp, AttentionOp, EmbeddingOp, LinearOp, QkvSplitOp, RmsNormOp, RopeQkOp, SiluOp,
    TransformerOp,
};
use super::weights::{BlockWeights, ModelWeights};
use crate::Real;
use crate::config::{BlockKind, GradClipKind, ModelConfig, OptimizerKind, TrainConfig};
use crate::optim::{AdamW, AdamWSchedule, AnyOptimizer};
use std::sync::Arc;
use wilupgu::{Backend, ComputeGraph, Tensor};

struct BuiltBlock<B: Backend> {
    tape: Tape<B, TransformerOp<B>>,
    x0: NodeId,
    output: NodeId,
}

#[allow(clippy::too_many_arguments)]
fn build_transformer_block<B: Backend>(
    gb: &mut GraphBuilder<'_, B, Train>,
    bw: &BlockWeights<B>,
    cfg: &ModelConfig,
    x0_value: Arc<Tensor<B>>,
) -> BuiltBlock<B> {
    let rows = cfg.batch_size * cfg.seq_len;
    let dim = cfg.dim;
    let head_dim = cfg.head_dim();
    let hidden = cfg.ffn_hidden;
    let ctx = &bw.qkv_proj.ctx;

    let mut tape = Tape::new();
    let x0 = tape.input(gb, x0_value);

    let norm_shape = NormMeta {
        seq_len: rows,
        size: dim,
        eps: cfg.norm_eps,
    };
    let n1 = tape.push(
        gb,
        TransformerOp::RmsNorm(RmsNormOp::new(&bw.norm_1, norm_shape)),
        &[(x0, 0)],
    );

    let qkv_shape = MatMulMeta {
        m: rows,
        n: dim * 3,
        k: dim,
    };
    let qkv = tape.push(
        gb,
        TransformerOp::Linear(LinearOp::new(&bw.qkv_proj, qkv_shape, true)),
        &[(n1, 0)],
    );

    // 3 outputs: q=slot0, k=slot1, v=slot2
    let split = tape.push(
        gb,
        TransformerOp::QkvSplit(QkvSplitOp::new(ctx, rows, dim)),
        &[(qkv, 0)],
    );

    // 2 outputs: rotated q=slot0, k=slot1
    let rope = tape.push(
        gb,
        TransformerOp::RopeQk(RopeQkOp::new(cfg.seq_len, dim, head_dim, cfg.batch_size)),
        &[(split, 0), (split, 1)],
    );

    let attn = tape.push(
        gb,
        TransformerOp::Attention(AttentionOp::new(
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
        TransformerOp::Linear(LinearOp::new(&bw.out_proj, out_proj_shape, true)),
        &[(attn, 0)],
    );

    let add1 = tape.push(
        gb,
        TransformerOp::Add(AddOp::new(ctx, rows * dim)),
        &[(x0, 0), (proj, 0)],
    );

    let n2 = tape.push(
        gb,
        TransformerOp::RmsNorm(RmsNormOp::new(&bw.norm_2, norm_shape)),
        &[(add1, 0)],
    );

    let ffn_up_shape = MatMulMeta {
        m: rows,
        n: hidden,
        k: dim,
    };
    let up = tape.push(
        gb,
        TransformerOp::Linear(LinearOp::new(&bw.ffn_up, ffn_up_shape, true)),
        &[(n2, 0)],
    );

    let silu = tape.push(
        gb,
        TransformerOp::Silu(SiluOp::new(ctx, rows * hidden)),
        &[(up, 0)],
    );

    let ffn_down_shape = MatMulMeta {
        m: rows,
        n: dim,
        k: hidden,
    };
    let down = tape.push(
        gb,
        TransformerOp::Linear(LinearOp::new(&bw.ffn_down, ffn_down_shape, true)),
        &[(silu, 0)],
    );

    let add2 = tape.push(
        gb,
        TransformerOp::Add(AddOp::new(ctx, rows * dim)),
        &[(add1, 0), (down, 0)],
    );

    BuiltBlock {
        tape,
        x0,
        output: add2,
    }
}

struct TrainState<B: Backend> {
    fwd_graph: ComputeGraph<B>,
    bwd_graph: ComputeGraph<B>,
    tokens: Arc<Tensor<B>>, // embedding's input handle -- see EmbeddingOp::tokens_handle
    head: Tape<B, TransformerOp<B>>,
    head_out: NodeId,
    blocks: Vec<BuiltBlock<B>>,
    tail: Tape<B, TransformerOp<B>>,
    tail_x0: NodeId,
    logits: NodeId,
    loss: AnyLoss<B>,
    optimizer: AnyOptimizer<B>,
    grad_clip: AnyGradClip<B>,
}

pub struct Model<B: Backend> {
    weights: ModelWeights<B>,
    train: Option<TrainState<B>>,
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
        let head_out = head.push(&mut gb, TransformerOp::Embedding(embedding_op), &[]);

        let mut x = head.output(head_out);
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
        let tail_x0 = tail.input(&mut gb, x);
        let norm_shape = NormMeta {
            seq_len: rows,
            size: cfg.dim,
            eps: cfg.norm_eps,
        };
        let final_norm = tail.push(
            &mut gb,
            TransformerOp::RmsNorm(RmsNormOp::new(&weights.final_norm, norm_shape)),
            &[(tail_x0, 0)],
        );
        let lm_shape = MatMulMeta {
            m: rows,
            n: cfg.vocab_size,
            k: cfg.dim,
        };
        let logits = tail.push(
            &mut gb,
            TransformerOp::Linear(LinearOp::new(&weights.lm_head, lm_shape, true)),
            &[(final_norm, 0)],
        );

        let loss = AnyLoss::CrossEntropy(CrossEntropyOp::new(&ctx, cfg.vocab_size, rows));
        // node only -- real targets are set per step via `train_step`.
        loss.set_targets(&vec![0u32; rows as usize]);
        loss.forward(&mut gb, &tail.output(logits));
        loss.set_grad_scale(1.0 / (rows * train_cfg.run.accumulation_steps as u32) as Real);

        // ---- backward: tail -> blocks (reverse) -> head, one shared graph ----
        let mut bwd_graph = ComputeGraph::new(ctx.clone());
        let mut gb_bwd = GraphBuilder::train(&mut bwd_graph);

        let logits_buf = tail.output(logits);
        loss.backward(&mut gb_bwd, &logits_buf);
        tail.backward(&mut gb_bwd, (logits, 0), &logits_buf);
        let mut grad = tail
            .grad_of((tail_x0, 0))
            .expect("tail backward didn't reach its input");

        for block in blocks.iter_mut().rev() {
            block.tape.backward(&mut gb_bwd, (block.output, 0), &grad);
            grad = block
                .tape
                .grad_of((block.x0, 0))
                .expect("block backward didn't reach its input");
        }
        head.backward(&mut gb_bwd, (head_out, 0), &grad);

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
            train: Some(TrainState {
                fwd_graph,
                bwd_graph,
                tokens,
                head,
                head_out,
                blocks,
                tail,
                tail_x0,
                logits,
                loss,
                optimizer,
                grad_clip,
            }),
        }
    }

    /// No training state at all -- see the module doc.
    pub fn for_chat(_ctx: Arc<B>, weights: ModelWeights<B>, _cfg: ModelConfig) -> Self {
        Self {
            weights,
            train: None,
        }
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

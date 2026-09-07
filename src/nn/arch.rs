//! `Architecture<B>`: what a block-kind must provide to plug into `Model`'s
//! three phases. One implementor today (`blocks::transformer::Transformer`);
//! Mamba/GDN mean a new `BlockKind` variant, new `TrainOp`/`PrefillOp`/
//! `DecodeOp` variants for their ops, a new `nn::blocks::*` module
//! implementing this trait, and one new match arm in each `build_*` below --
//! no generic `Model<B, A>` (dispatch stays a plain `match kind`, same shape
//! as `AnyOptimizer`/`AnyGradClip`; `TrainOp` etc. are themselves the
//! extension point, not this trait).

use super::blocks::transformer::Transformer;
use super::kernels::meta::{MatMulMeta, NormMeta};
use super::kernels::{GraphBuilder, Prefill, Train};
use super::model::{BuiltBlock, DecodeGraph};
use super::ops::cached::{DecodeOp, PrefillOp};
use super::ops::full_seq::{EmbeddingOp, LinearOp, RmsNormOp, TrainOp};
use super::tape::{NodeId, NodeSpec, Tape};
use super::weights::{BlockWeights, ModelWeights};
use crate::config::{BlockKind, ModelConfig};
use std::collections::HashMap;
use std::sync::Arc;
use wilupgu::{Backend, ComputeGraph, Tensor};

pub(crate) trait Architecture<B: Backend> {
    fn train_specs(
        &self,
        bw: &BlockWeights<B>,
        cfg: &ModelConfig,
        rows: u32,
    ) -> Vec<NodeSpec<TrainOp<B>>>;

    #[allow(clippy::too_many_arguments)]
    fn prefill_specs(
        &self,
        bw: &BlockWeights<B>,
        ctx: &Arc<B>,
        cache_k: &Arc<Tensor<B>>,
        cache_v: &Arc<Tensor<B>>,
        cfg: &ModelConfig,
        prompt_len: u32,
    ) -> Vec<NodeSpec<PrefillOp<B>>>;

    #[allow(clippy::too_many_arguments)]
    fn decode_specs(
        &self,
        bw: &BlockWeights<B>,
        ctx: &Arc<B>,
        cache_k: &Arc<Tensor<B>>,
        cache_v: &Arc<Tensor<B>>,
        cfg: &ModelConfig,
        max_context_len: u32,
    ) -> Vec<NodeSpec<DecodeOp<B>>>;
}

pub(crate) fn build_block<B: Backend>(
    gb: &mut GraphBuilder<'_, B, Train>,
    bw: &BlockWeights<B>,
    cfg: &ModelConfig,
    kind: BlockKind,
    block_input: Arc<Tensor<B>>,
) -> BuiltBlock<B> {
    let rows = cfg.batch_size * cfg.seq_len;
    let mut tape = Tape::new();
    let block_input_id = tape.input(gb, block_input);
    let mut names = HashMap::from([("input", block_input_id)]);
    let specs = match kind {
        BlockKind::Transformer => Transformer.train_specs(bw, cfg, rows),
    };
    let output = tape.extend(gb, &mut names, specs);

    BuiltBlock {
        tape,
        block_input_id,
        output,
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_prefill_forward<B: Backend>(
    gb: &mut GraphBuilder<'_, B, Prefill>,
    weights: &ModelWeights<B>,
    cfg: &ModelConfig,
    prompt_len: u32,
    cache_k: &[Arc<Tensor<B>>],
    cache_v: &[Arc<Tensor<B>>],
) -> (Tape<B, PrefillOp<B>>, Arc<Tensor<B>>, NodeId) {
    let dim = cfg.dim;
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

    let mut names: HashMap<&'static str, NodeId> = HashMap::new();
    for (((bw, ck), cv), kind) in weights
        .blocks
        .iter()
        .zip(cache_k)
        .zip(cache_v)
        .zip(cfg.layers())
    {
        names.insert("input", x);
        let specs = match kind {
            BlockKind::Transformer => Transformer.prefill_specs(bw, ctx, ck, cv, cfg, prompt_len),
        };
        x = tape.extend(gb, &mut names, specs);
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

pub(crate) fn build_decode_forward<B: Backend>(
    ctx: &Arc<B>,
    weights: &ModelWeights<B>,
    cfg: &ModelConfig,
    cache_k: &[Arc<Tensor<B>>],
    cache_v: &[Arc<Tensor<B>>],
    max_context_len: u32,
) -> DecodeGraph<B> {
    let dim = cfg.dim;

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
    let mut names: HashMap<&'static str, NodeId> = HashMap::new();
    for (((bw, ck), cv), kind) in weights
        .blocks
        .iter()
        .zip(cache_k)
        .zip(cache_v)
        .zip(cfg.layers())
    {
        names.insert("input", x);
        let specs = match kind {
            BlockKind::Transformer => {
                Transformer.decode_specs(bw, ctx, ck, cv, cfg, max_context_len)
            }
        };
        x = tape.extend(&mut gb, &mut names, specs);
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

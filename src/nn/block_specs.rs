use super::cached_ops::{
    CacheWriteOp, CachedAttentionOp, DecodeOp, HeadGatherOp, PrefillOp, RopeOffsetOp,
};
use super::core_ops::{
    AddOp, AttentionOp, EmbeddingOp, LinearOp, QkvSplitOp, RmsNormOp, RopeQkOp, SiluOp, TrainOp,
};
use super::model::{BuiltBlock, DecodeGraph};
use super::ops::meta::{MatMulMeta, NormMeta};
use super::ops::{GraphBuilder, Prefill, Train};
use super::tape::{NodeId, NodeSpec, Tape};
use super::weights::{BlockWeights, ModelWeights};
use crate::config::ModelConfig;
use std::collections::HashMap;
use std::sync::Arc;
use wilupgu::{Backend, ComputeGraph, Tensor};

macro_rules! node {
    ($name:literal <- $inputs:expr, $op:expr) => {
        NodeSpec {
            name: $name,
            inputs: $inputs,
            op: $op,
        }
    };
}

fn transformer_block_specs<B: Backend>(
    bw: &BlockWeights<B>,
    cfg: &ModelConfig,
    rows: u32,
) -> Vec<NodeSpec<TrainOp<B>>> {
    let dim = cfg.dim;
    let hidden = cfg.ffn_hidden;
    let head_dim = cfg.head_dim();
    let ctx = &bw.qkv_proj.ctx;
    let norm_shape = NormMeta { seq_len: rows, size: dim, eps: cfg.norm_eps };

    vec![
        node!("n1" <- &[("input", 0)], TrainOp::RmsNorm(RmsNormOp::new(&bw.norm_1, norm_shape))),
        node!("qkv" <- &[("n1", 0)], TrainOp::Linear(LinearOp::new(
            &bw.qkv_proj, MatMulMeta { m: rows, n: dim * 3, k: dim }, true))),
        node!("split" <- &[("qkv", 0)], TrainOp::QkvSplit(QkvSplitOp::new(ctx, rows, dim))),
        node!("rope" <- &[("split", 0), ("split", 1)],
            TrainOp::RopeQk(RopeQkOp::new(cfg.seq_len, dim, head_dim, cfg.batch_size))),
        node!("attn" <- &[("rope", 0), ("rope", 1), ("split", 2)],
            TrainOp::Attention(AttentionOp::new(ctx, cfg.seq_len, dim, head_dim, cfg.batch_size))),
        node!("proj" <- &[("attn", 0)], TrainOp::Linear(LinearOp::new(
            &bw.out_proj, MatMulMeta { m: rows, n: dim, k: dim }, true))),
        node!("add1" <- &[("input", 0), ("proj", 0)], TrainOp::Add(AddOp::new(ctx, rows * dim))),
        node!("n2" <- &[("add1", 0)], TrainOp::RmsNorm(RmsNormOp::new(&bw.norm_2, norm_shape))),
        node!("up" <- &[("n2", 0)], TrainOp::Linear(LinearOp::new(
            &bw.ffn_up, MatMulMeta { m: rows, n: hidden, k: dim }, true))),
        node!("silu" <- &[("up", 0)], TrainOp::Silu(SiluOp::new(ctx, rows * hidden))),
        node!("down" <- &[("silu", 0)], TrainOp::Linear(LinearOp::new(
            &bw.ffn_down, MatMulMeta { m: rows, n: dim, k: hidden }, true))),
        node!("add2" <- &[("add1", 0), ("down", 0)], TrainOp::Add(AddOp::new(ctx, rows * dim))),
    ]
}

pub(crate) fn build_transformer_block<B: Backend>(
    gb: &mut GraphBuilder<'_, B, Train>,
    bw: &BlockWeights<B>,
    cfg: &ModelConfig,
    block_input: Arc<Tensor<B>>,
) -> BuiltBlock<B> {
    let rows = cfg.batch_size * cfg.seq_len;
    let mut tape = Tape::new();
    let block_input_id = tape.input(gb, block_input);
    let mut names = HashMap::from([("input", block_input_id)]);
    let output = tape.extend(gb, &mut names, transformer_block_specs(bw, cfg, rows));

    BuiltBlock {
        tape,
        block_input_id,
        output,
    }
}

fn prefill_block_specs<B: Backend>(
    bw: &BlockWeights<B>,
    ctx: &Arc<B>,
    cache_k: &Arc<Tensor<B>>,
    cache_v: &Arc<Tensor<B>>,
    cfg: &ModelConfig,
    prompt_len: u32,
) -> Vec<NodeSpec<PrefillOp<B>>> {
    let dim = cfg.dim;
    let hidden = cfg.ffn_hidden;
    let head_dim = cfg.head_dim();
    let norm_shape = NormMeta { seq_len: prompt_len, size: dim, eps: cfg.norm_eps };

    vec![
        node!("n1" <- &[("input", 0)], PrefillOp::RmsNorm(RmsNormOp::new(&bw.norm_1, norm_shape))),
        node!("qkv" <- &[("n1", 0)], PrefillOp::Linear(LinearOp::new(
            &bw.qkv_proj, MatMulMeta { m: prompt_len, n: dim * 3, k: dim }, true))),
        node!("split" <- &[("qkv", 0)], PrefillOp::QkvSplit(QkvSplitOp::new(ctx, prompt_len, dim))),
        node!("rope" <- &[("split", 0), ("split", 1)],
            PrefillOp::RopeQk(RopeQkOp::new(prompt_len, dim, head_dim, 1))),
        node!("k_written" <- &[("rope", 1)],
            PrefillOp::CacheWrite(CacheWriteOp::new(cache_k.clone(), prompt_len, dim))),
        node!("v_written" <- &[("split", 2)],
            PrefillOp::CacheWrite(CacheWriteOp::new(cache_v.clone(), prompt_len, dim))),
        node!("attn" <- &[("rope", 0), ("k_written", 0), ("v_written", 0)],
            PrefillOp::Attention(AttentionOp::new(ctx, prompt_len, dim, head_dim, 1))),
        node!("proj" <- &[("attn", 0)], PrefillOp::Linear(LinearOp::new(
            &bw.out_proj, MatMulMeta { m: prompt_len, n: dim, k: dim }, true))),
        node!("add1" <- &[("input", 0), ("proj", 0)], PrefillOp::Add(AddOp::new(ctx, prompt_len * dim))),
        node!("n2" <- &[("add1", 0)], PrefillOp::RmsNorm(RmsNormOp::new(&bw.norm_2, norm_shape))),
        node!("up" <- &[("n2", 0)], PrefillOp::Linear(LinearOp::new(
            &bw.ffn_up, MatMulMeta { m: prompt_len, n: hidden, k: dim }, true))),
        node!("silu" <- &[("up", 0)], PrefillOp::Silu(SiluOp::new(ctx, prompt_len * hidden))),
        node!("down" <- &[("silu", 0)], PrefillOp::Linear(LinearOp::new(
            &bw.ffn_down, MatMulMeta { m: prompt_len, n: dim, k: hidden }, true))),
        node!("add2" <- &[("add1", 0), ("down", 0)], PrefillOp::Add(AddOp::new(ctx, prompt_len * dim))),
    ]
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
    for ((bw, ck), cv) in weights.blocks.iter().zip(cache_k).zip(cache_v) {
        names.insert("input", x);
        x = tape.extend(
            gb,
            &mut names,
            prefill_block_specs(bw, ctx, ck, cv, cfg, prompt_len),
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

fn decode_block_specs<B: Backend>(
    bw: &BlockWeights<B>,
    ctx: &Arc<B>,
    cache_k: &Arc<Tensor<B>>,
    cache_v: &Arc<Tensor<B>>,
    cfg: &ModelConfig,
    max_context_len: u32,
) -> Vec<NodeSpec<DecodeOp<B>>> {
    let dim = cfg.dim;
    let hidden = cfg.ffn_hidden;
    let head_dim = cfg.head_dim();
    let norm_shape = NormMeta { seq_len: 1, size: dim, eps: cfg.norm_eps };

    vec![
        node!("n1" <- &[("input", 0)], DecodeOp::RmsNorm(RmsNormOp::new(&bw.norm_1, norm_shape))),
        node!("qkv" <- &[("n1", 0)], DecodeOp::Linear(LinearOp::new(
            &bw.qkv_proj, MatMulMeta { m: 1, n: dim * 3, k: dim }, true))),
        node!("q" <- &[("qkv", 0)], DecodeOp::HeadGather(HeadGatherOp::new(ctx, dim, 0))),
        node!("k" <- &[("qkv", 0)], DecodeOp::HeadGather(HeadGatherOp::new(ctx, dim, dim))),
        node!("v" <- &[("qkv", 0)], DecodeOp::HeadGather(HeadGatherOp::new(ctx, dim, 2 * dim))),
        node!("rope_q" <- &[("q", 0)], DecodeOp::RopeOffset(RopeOffsetOp::new(ctx, dim, head_dim))),
        node!("rope_k" <- &[("k", 0)], DecodeOp::RopeOffset(RopeOffsetOp::new(ctx, dim, head_dim))),
        node!("k_written" <- &[("rope_k", 0)],
            DecodeOp::CacheWrite(CacheWriteOp::new(cache_k.clone(), 1, dim))),
        node!("v_written" <- &[("v", 0)],
            DecodeOp::CacheWrite(CacheWriteOp::new(cache_v.clone(), 1, dim))),
        node!("attn" <- &[("rope_q", 0)], DecodeOp::CachedAttention(CachedAttentionOp::new(
            cache_k.clone(),
            cache_v.clone(),
            cfg.num_heads,
            dim,
            head_dim,
            max_context_len,
        ))),
        node!("proj" <- &[("attn", 0)], DecodeOp::Linear(LinearOp::new(
            &bw.out_proj, MatMulMeta { m: 1, n: dim, k: dim }, true))),
        node!("add1" <- &[("input", 0), ("proj", 0)], DecodeOp::Add(AddOp::new(ctx, dim))),
        node!("n2" <- &[("add1", 0)], DecodeOp::RmsNorm(RmsNormOp::new(&bw.norm_2, norm_shape))),
        node!("up" <- &[("n2", 0)], DecodeOp::Linear(LinearOp::new(
            &bw.ffn_up, MatMulMeta { m: 1, n: hidden, k: dim }, true))),
        node!("silu" <- &[("up", 0)], DecodeOp::Silu(SiluOp::new(ctx, hidden))),
        node!("down" <- &[("silu", 0)], DecodeOp::Linear(LinearOp::new(
            &bw.ffn_down, MatMulMeta { m: 1, n: dim, k: hidden }, true))),
        node!("add2" <- &[("add1", 0), ("down", 0)], DecodeOp::Add(AddOp::new(ctx, dim))),
    ]
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
    for ((bw, ck), cv) in weights.blocks.iter().zip(cache_k).zip(cache_v) {
        names.insert("input", x);
        x = tape.extend(
            &mut gb,
            &mut names,
            decode_block_specs(bw, ctx, ck, cv, cfg, max_context_len),
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

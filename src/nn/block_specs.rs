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

/// `n1` -> `qkv` projection -- identical in every phase (same `bw.qkv_proj`
/// weight, same `dim*3` shape), shared by all 3 `*_block_specs` below.
fn block_prologue_specs<B: Backend, Node>(
    bw: &BlockWeights<B>,
    rows: u32,
    dim: u32,
    eps: f32,
) -> Vec<NodeSpec<Node>>
where
    Node: From<RmsNormOp<B>> + From<LinearOp<B>>,
{
    let norm_shape = NormMeta { seq_len: rows, size: dim, eps };
    vec![
        NodeSpec {
            name: "n1",
            inputs: &[("input", 0)],
            op: Node::from(RmsNormOp::new(&bw.norm_1, norm_shape)),
        },
        NodeSpec {
            name: "qkv",
            inputs: &[("n1", 0)],
            op: Node::from(LinearOp::new(
                &bw.qkv_proj,
                MatMulMeta { m: rows, n: dim * 3, k: dim },
                true,
            )),
        },
    ]
}

/// `out_proj` -> add -> `n2` -> ffn -> add -- identical in every phase,
/// consumes an `"attn"`-named node the caller pushed first. Pairs with
/// `block_prologue_specs`.
fn block_epilogue_specs<B: Backend, Node>(
    bw: &BlockWeights<B>,
    ctx: &Arc<B>,
    rows: u32,
    dim: u32,
    hidden: u32,
    eps: f32,
) -> Vec<NodeSpec<Node>>
where
    Node: From<LinearOp<B>> + From<AddOp<B>> + From<RmsNormOp<B>> + From<SiluOp<B>>,
{
    let norm_shape = NormMeta { seq_len: rows, size: dim, eps };
    vec![
        NodeSpec {
            name: "proj",
            inputs: &[("attn", 0)],
            op: Node::from(LinearOp::new(
                &bw.out_proj,
                MatMulMeta { m: rows, n: dim, k: dim },
                true,
            )),
        },
        NodeSpec {
            name: "add1",
            inputs: &[("input", 0), ("proj", 0)],
            op: Node::from(AddOp::new(ctx, rows * dim)),
        },
        NodeSpec {
            name: "n2",
            inputs: &[("add1", 0)],
            op: Node::from(RmsNormOp::new(&bw.norm_2, norm_shape)),
        },
        NodeSpec {
            name: "up",
            inputs: &[("n2", 0)],
            op: Node::from(LinearOp::new(
                &bw.ffn_up,
                MatMulMeta { m: rows, n: hidden, k: dim },
                true,
            )),
        },
        NodeSpec {
            name: "silu",
            inputs: &[("up", 0)],
            op: Node::from(SiluOp::new(ctx, rows * hidden)),
        },
        NodeSpec {
            name: "down",
            inputs: &[("silu", 0)],
            op: Node::from(LinearOp::new(
                &bw.ffn_down,
                MatMulMeta { m: rows, n: dim, k: hidden },
                true,
            )),
        },
        NodeSpec {
            name: "add2",
            inputs: &[("add1", 0), ("down", 0)],
            op: Node::from(AddOp::new(ctx, rows * dim)),
        },
    ]
}

fn transformer_block_specs<B: Backend>(
    bw: &BlockWeights<B>,
    cfg: &ModelConfig,
    rows: u32,
) -> Vec<NodeSpec<TrainOp<B>>> {
    let dim = cfg.dim;
    let head_dim = cfg.head_dim();
    let ctx = &bw.qkv_proj.ctx;

    let mut specs = block_prologue_specs(bw, rows, dim, cfg.norm_eps);
    specs.extend([
        NodeSpec {
            name: "split",
            inputs: &[("qkv", 0)],
            op: TrainOp::QkvSplit(QkvSplitOp::new(ctx, rows, dim)),
        },
        NodeSpec {
            name: "rope",
            inputs: &[("split", 0), ("split", 1)],
            op: TrainOp::RopeQk(RopeQkOp::new(cfg.seq_len, dim, head_dim, cfg.batch_size)),
        },
        NodeSpec {
            name: "attn",
            inputs: &[("rope", 0), ("rope", 1), ("split", 2)],
            op: TrainOp::Attention(AttentionOp::new(
                ctx,
                cfg.seq_len,
                dim,
                head_dim,
                cfg.batch_size,
            )),
        },
    ]);
    specs.extend(block_epilogue_specs(bw, ctx, rows, dim, cfg.ffn_hidden, cfg.norm_eps));
    specs
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
    let head_dim = cfg.head_dim();

    let mut specs = block_prologue_specs(bw, prompt_len, dim, cfg.norm_eps);
    specs.extend([
        NodeSpec {
            name: "split",
            inputs: &[("qkv", 0)],
            op: PrefillOp::QkvSplit(QkvSplitOp::new(ctx, prompt_len, dim)),
        },
        NodeSpec {
            name: "rope",
            inputs: &[("split", 0), ("split", 1)],
            op: PrefillOp::RopeQk(RopeQkOp::new(prompt_len, dim, head_dim, 1)),
        },
        NodeSpec {
            name: "k_written",
            inputs: &[("rope", 1)],
            op: PrefillOp::CacheWrite(CacheWriteOp::new(cache_k.clone(), prompt_len, dim)),
        },
        NodeSpec {
            name: "v_written",
            inputs: &[("split", 2)],
            op: PrefillOp::CacheWrite(CacheWriteOp::new(cache_v.clone(), prompt_len, dim)),
        },
        NodeSpec {
            name: "attn",
            inputs: &[("rope", 0), ("k_written", 0), ("v_written", 0)],
            op: PrefillOp::Attention(AttentionOp::new(ctx, prompt_len, dim, head_dim, 1)),
        },
    ]);
    specs.extend(block_epilogue_specs(
        bw,
        ctx,
        prompt_len,
        dim,
        cfg.ffn_hidden,
        cfg.norm_eps,
    ));
    specs
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
    let head_dim = cfg.head_dim();

    let mut specs = block_prologue_specs(bw, 1, dim, cfg.norm_eps);
    specs.extend([
        NodeSpec {
            name: "q",
            inputs: &[("qkv", 0)],
            op: DecodeOp::HeadGather(HeadGatherOp::new(ctx, dim, 0)),
        },
        NodeSpec {
            name: "k",
            inputs: &[("qkv", 0)],
            op: DecodeOp::HeadGather(HeadGatherOp::new(ctx, dim, dim)),
        },
        NodeSpec {
            name: "v",
            inputs: &[("qkv", 0)],
            op: DecodeOp::HeadGather(HeadGatherOp::new(ctx, dim, 2 * dim)),
        },
        NodeSpec {
            name: "rope_q",
            inputs: &[("q", 0)],
            op: DecodeOp::RopeOffset(RopeOffsetOp::new(ctx, dim, head_dim)),
        },
        NodeSpec {
            name: "rope_k",
            inputs: &[("k", 0)],
            op: DecodeOp::RopeOffset(RopeOffsetOp::new(ctx, dim, head_dim)),
        },
        NodeSpec {
            name: "k_written",
            inputs: &[("rope_k", 0)],
            op: DecodeOp::CacheWrite(CacheWriteOp::new(cache_k.clone(), 1, dim)),
        },
        NodeSpec {
            name: "v_written",
            inputs: &[("v", 0)],
            op: DecodeOp::CacheWrite(CacheWriteOp::new(cache_v.clone(), 1, dim)),
        },
        NodeSpec {
            name: "attn",
            inputs: &[("rope_q", 0)],
            op: DecodeOp::CachedAttention(CachedAttentionOp::new(
                cache_k.clone(),
                cache_v.clone(),
                cfg.num_heads,
                dim,
                head_dim,
                max_context_len,
            )),
        },
    ]);
    specs.extend(block_epilogue_specs(bw, ctx, 1, dim, cfg.ffn_hidden, cfg.norm_eps));
    specs
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

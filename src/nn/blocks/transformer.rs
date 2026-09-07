use super::super::arch::Architecture;
use super::super::kernels::meta::{MatMulMeta, NormMeta};
use super::super::ops::cached::{
    CacheWriteOp, CachedAttentionOp, DecodeOp, HeadGatherOp, PrefillOp, RopeOffsetOp,
};
use super::super::ops::full_seq::{
    AddOp, AttentionOp, LinearOp, QkvSplitOp, RmsNormOp, RopeQkOp, SiluOp, TrainOp,
};
use super::super::tape::{NodeSpec, node};
use super::super::weights::BlockWeights;
use crate::config::ModelConfig;
use std::sync::Arc;
use wilupgu::{Backend, Tensor};

pub(crate) struct Transformer;

impl<B: Backend> Architecture<B> for Transformer {
    fn train_specs(
        &self,
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

    fn prefill_specs(
        &self,
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

    fn decode_specs(
        &self,
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
}

use super::ops;
use super::ops::{CachedPhase, GraphBuilder};
use super::ops::meta::{
    AttnCachedMeta, CacheWriteMeta, EmbeddingMeta, FlashAttnMeta, HeadMoveMeta, KernelMeta,
    MatMulMeta, NormMeta, RopeMeta, RopeOffsetMeta, SoftmaxRectMeta,
};
use super::weights::BlockWeights;
use crate::Real;
use crate::config::ModelConfig;
use std::sync::Arc;
use wilupgu::{Backend, Tensor};

pub(crate) struct DecodeScratch<B: Backend> {
    pub(crate) hidden: Arc<Tensor<B>>,
    pub(crate) norm_out: Arc<Tensor<B>>,
    pub(crate) qkv_out: Arc<Tensor<B>>,
    pub(crate) q_buf: Arc<Tensor<B>>,
    pub(crate) k_buf: Arc<Tensor<B>>,
    pub(crate) v_buf: Arc<Tensor<B>>,
    pub(crate) scores: Arc<Tensor<B>>, // packed [num_heads, attn_len]
    pub(crate) attn_out: Arc<Tensor<B>>,
    pub(crate) ffn_up_out: Arc<Tensor<B>>,
    pub(crate) final_norm_out: Arc<Tensor<B>>,
    pub(crate) logits: Arc<Tensor<B>>,

    // ---- constant Meta buffers: written once here, read forever after ----
    pub(crate) norm_meta: Arc<Tensor<B>>, // NormMeta -- norm_1, norm_2, final_norm
    pub(crate) qkv_meta: Arc<Tensor<B>>,  // MatMulMeta{1,dim,dim} -- out_proj
    pub(crate) qkv_proj_meta: Arc<Tensor<B>>, // MatMulMeta{1,3*dim,dim} -- fused qkv proj
    pub(crate) qkv_split_meta: Vec<Arc<Tensor<B>>>, // 3x HeadMoveMeta (q/k/v slice of fused qkv)
    pub(crate) ffnup_meta: Arc<Tensor<B>>, // MatMulMeta{1,ffn_hidden,dim}
    pub(crate) ffndown_meta: Arc<Tensor<B>>, // MatMulMeta{1,dim,ffn_hidden}
    pub(crate) emb_meta: Arc<Tensor<B>>,  // EmbeddingMeta (seq_len=1)
    pub(crate) lm_meta: Arc<Tensor<B>>,   // MatMulMeta{1,vocab_size,dim}

    // ---- dynamic Meta buffers: updated once per decode step ----
    pub(crate) rope_meta: Arc<Tensor<B>>, // RopeOffsetMeta (pos advances)
    pub(crate) cache_write_meta: Arc<Tensor<B>>, // CacheWriteMeta (dst_row_offset advances)
    pub(crate) attn_meta: Arc<Tensor<B>>, // AttnCachedMeta (attn_len advances)
    pub(crate) softmax_meta: Arc<Tensor<B>>, // SoftmaxRectMeta (width=attn_len)
}

impl<B: Backend> DecodeScratch<B> {
    pub(crate) fn new(ctx: Arc<B>, cfg: &ModelConfig, max_context_len: u32) -> Self {
        let ModelConfig {
            dim,
            num_heads,
            ffn_hidden: ffn_hidden_dim,
            vocab_size,
            ..
        } = *cfg;
        let head_dim = cfg.head_dim();
        let zeros = |n: usize| vec![0.0 as Real; n];
        let dim1 = zeros(dim as usize);

        let norm_meta = NormMeta {
            seq_len: 1,
            size: dim,
            eps: cfg.norm_eps,
        }
        .upload(&ctx);
        let qkv_meta = MatMulMeta {
            m: 1,
            n: dim,
            k: dim,
        }
        .upload(&ctx);
        let qkv_proj_meta = MatMulMeta {
            m: 1,
            n: dim * 3,
            k: dim,
        }
        .upload(&ctx);
        let qkv_split_meta = (0..3u32)
            .map(|i| HeadMoveMeta::qkv_slice(1, dim, i * dim).upload(&ctx))
            .collect();
        let ffnup_meta = MatMulMeta {
            m: 1,
            n: ffn_hidden_dim,
            k: dim,
        }
        .upload(&ctx);
        let ffndown_meta = MatMulMeta {
            m: 1,
            n: dim,
            k: ffn_hidden_dim,
        }
        .upload(&ctx);
        let emb_meta = EmbeddingMeta {
            vocab_size,
            dim,
            seq_len: 1,
        }
        .upload(&ctx);
        let lm_meta = MatMulMeta {
            m: 1,
            n: vocab_size,
            k: dim,
        }
        .upload(&ctx);

        let rope_meta = RopeOffsetMeta {
            seq_len: 1,
            dim,
            head_dim,
            pos: 0,
        }
        .upload(&ctx);
        let cache_write_meta = CacheWriteMeta {
            row_count: 1,
            width: dim,
            dst_row_offset: 0,
        }
        .upload(&ctx);
        let attn_meta = AttnCachedMeta {
            attn_len: 1,
            dim,
            head_dim,
        }
        .upload(&ctx);
        let softmax_meta = SoftmaxRectMeta {
            num_rows: num_heads,
            width: 1,
            scale: 1.0,
        }
        .upload(&ctx);

        Self {
            hidden: Arc::new(Tensor::init_from_cpu(ctx.clone(), &dim1)),
            norm_out: Arc::new(Tensor::init_from_cpu(ctx.clone(), &dim1)),
            qkv_out: Arc::new(Tensor::init_from_cpu(
                ctx.clone(),
                &zeros((dim * 3) as usize),
            )),
            q_buf: Arc::new(Tensor::init_from_cpu(ctx.clone(), &dim1)),
            k_buf: Arc::new(Tensor::init_from_cpu(ctx.clone(), &dim1)),
            v_buf: Arc::new(Tensor::init_from_cpu(ctx.clone(), &dim1)),
            scores: Arc::new(Tensor::init_from_cpu(
                ctx.clone(),
                &zeros((num_heads * max_context_len) as usize),
            )),
            attn_out: Arc::new(Tensor::init_from_cpu(ctx.clone(), &dim1)),
            ffn_up_out: Arc::new(Tensor::init_from_cpu(
                ctx.clone(),
                &zeros(ffn_hidden_dim as usize),
            )),
            final_norm_out: Arc::new(Tensor::init_from_cpu(ctx.clone(), &dim1)),
            logits: Arc::new(Tensor::init_from_cpu(ctx, &zeros(vocab_size as usize))),
            norm_meta,
            qkv_meta,
            qkv_proj_meta,
            qkv_split_meta,
            ffnup_meta,
            ffndown_meta,
            emb_meta,
            lm_meta,
            rope_meta,
            cache_write_meta,
            attn_meta,
            softmax_meta,
        }
    }

    pub(crate) fn update_for_step(&self, pos: u32, cfg: &ModelConfig) {
        let ModelConfig { dim, num_heads, .. } = *cfg;
        let head_dim = cfg.head_dim();
        let scale = 1.0 / (head_dim as f32).sqrt();
        let attn_len = pos + 1;

        RopeOffsetMeta {
            seq_len: 1,
            dim,
            head_dim,
            pos,
        }
        .write_to(&self.rope_meta);
        CacheWriteMeta {
            row_count: 1,
            width: dim,
            dst_row_offset: pos,
        }
        .write_to(&self.cache_write_meta);
        AttnCachedMeta {
            attn_len,
            dim,
            head_dim,
        }
        .write_to(&self.attn_meta);
        SoftmaxRectMeta {
            num_rows: num_heads,
            width: attn_len,
            scale,
        }
        .write_to(&self.softmax_meta);
    }
}

/// Shared prefix of a transformer block, identical for Prefill and Decode:
/// norm1 -> fused qkv projection -> split into q/k/v. Caller does
/// rope+attention+cache_write next (that part genuinely differs per phase),
/// then calls `block_post_attn`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn block_pre_attn<B: Backend, P: CachedPhase>(
    gb: &mut GraphBuilder<'_, B, P>,
    bw: &BlockWeights<B>,
    hidden: &Arc<Tensor<B>>,
    norm_out: &Arc<Tensor<B>>,
    qkv_out: &Arc<Tensor<B>>,
    q_buf: &Arc<Tensor<B>>,
    k_buf: &Arc<Tensor<B>>,
    v_buf: &Arc<Tensor<B>>,
    rows: u32,
    dim: u32,
    norm_meta: (NormMeta, &Arc<Tensor<B>>),
    qkv_proj_meta: (MatMulMeta, &Arc<Tensor<B>>),
    qkv_split_meta: &[Arc<Tensor<B>>],
) {
    let (norm_shape, norm_meta) = norm_meta;
    ops::rmsnorm_with(gb, hidden, &bw.norm_1, norm_out, norm_shape, norm_meta);

    let (qkv_shape, qkv_meta) = qkv_proj_meta;
    ops::matmul_with(gb, norm_out, &bw.qkv_proj, qkv_out, qkv_shape, qkv_meta);

    for (i, (dst, off)) in [q_buf, k_buf, v_buf].into_iter().zip([0, dim, 2 * dim]).enumerate() {
        ops::head_gather_with(
            gb,
            qkv_out,
            dst,
            HeadMoveMeta::qkv_slice(rows, dim, off),
            &qkv_split_meta[i],
        );
    }
}

/// Shared suffix of a transformer block, identical for Prefill and Decode:
/// out_proj+residual -> norm2 -> ffn_up -> silu -> ffn_down+residual.
/// `hidden` is read AND written in place (both matmul_add calls fuse the
/// residual add into the projection's output write).
#[allow(clippy::too_many_arguments)]
pub(crate) fn block_post_attn<B: Backend, P: CachedPhase>(
    gb: &mut GraphBuilder<'_, B, P>,
    bw: &BlockWeights<B>,
    hidden: &Arc<Tensor<B>>,
    attn_out: &Arc<Tensor<B>>,
    norm_out: &Arc<Tensor<B>>,
    ffn_up_out: &Arc<Tensor<B>>,
    rows: u32,
    ffn_hidden_dim: u32,
    out_proj_meta: (MatMulMeta, &Arc<Tensor<B>>),
    norm_meta: (NormMeta, &Arc<Tensor<B>>),
    ffnup_meta: (MatMulMeta, &Arc<Tensor<B>>),
    ffndown_meta: (MatMulMeta, &Arc<Tensor<B>>),
) {
    let (out_proj_shape, out_proj_meta) = out_proj_meta;
    ops::matmul_add_with(gb, attn_out, &bw.out_proj, hidden, out_proj_shape, out_proj_meta);

    let (norm_shape, norm_meta) = norm_meta;
    ops::rmsnorm_with(gb, hidden, &bw.norm_2, norm_out, norm_shape, norm_meta);

    let (ffnup_shape, ffnup_meta) = ffnup_meta;
    ops::matmul_with(gb, norm_out, &bw.ffn_up, ffn_up_out, ffnup_shape, ffnup_meta);

    ops::silu(gb, ffn_up_out, rows * ffn_hidden_dim);

    let (ffndown_shape, ffndown_meta) = ffndown_meta;
    ops::matmul_add_with(gb, ffn_up_out, &bw.ffn_down, hidden, ffndown_shape, ffndown_meta);
}

pub(crate) fn build_prefill_layer<B: Backend>(
    gb: &mut GraphBuilder<'_, B, ops::Prefill>,
    ctx: &Arc<B>,
    bw: &BlockWeights<B>,
    hidden_in: &Arc<Tensor<B>>,
    cache_k: &Arc<Tensor<B>>,
    cache_v: &Arc<Tensor<B>>,
    prompt_len: u32,
    cfg: &ModelConfig,
) -> Arc<Tensor<B>> {
    let ModelConfig {
        dim,
        ffn_hidden: ffn_hidden_dim,
        ..
    } = *cfg;
    let head_dim = cfg.head_dim();
    let zeros_dim = vec![0.0 as Real; (prompt_len * dim) as usize];
    let norm_shape = NormMeta {
        seq_len: prompt_len,
        size: dim,
        eps: cfg.norm_eps,
    };

    // ---- shared block skeleton (identical code path to decode) ----
    let norm1_out = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros_dim));
    let qkv_out = Arc::new(Tensor::init_from_cpu(
        ctx.clone(),
        &vec![0.0 as Real; (prompt_len * dim * 3) as usize],
    ));
    let q_buf = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros_dim));
    let k_buf = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros_dim));
    let v_buf = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros_dim));

    let qkv_proj_shape = MatMulMeta {
        m: prompt_len,
        n: dim * 3,
        k: dim,
    };
    let qkv_split_meta = [
        HeadMoveMeta::qkv_slice(prompt_len, dim, 0).upload(ctx),
        HeadMoveMeta::qkv_slice(prompt_len, dim, dim).upload(ctx),
        HeadMoveMeta::qkv_slice(prompt_len, dim, 2 * dim).upload(ctx),
    ];
    block_pre_attn(
        gb,
        bw,
        hidden_in,
        &norm1_out,
        &qkv_out,
        &q_buf,
        &k_buf,
        &v_buf,
        prompt_len,
        dim,
        (norm_shape, &norm_shape.upload(ctx)),
        (qkv_proj_shape, &qkv_proj_shape.upload(ctx)),
        &qkv_split_meta,
    );

    // ---- RoPE + cache write ----
    let rope_shape = RopeMeta {
        seq_len: prompt_len,
        dim,
        head_dim,
        row_offset: 0,
    };
    ops::rope(gb, &q_buf, rope_shape);
    ops::rope(gb, &k_buf, rope_shape);

    let cache_shape = CacheWriteMeta {
        row_count: prompt_len,
        width: dim,
        dst_row_offset: 0,
    };
    ops::cache_write(gb, &k_buf, cache_k, cache_shape);
    ops::cache_write(gb, &v_buf, cache_v, cache_shape);

    // ---- attention (flash: no per-head scores buffers) ----
    let attn_out = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros_dim));
    let _saved = ops::flash_attention(
        gb,
        &q_buf,
        &k_buf,
        &v_buf,
        &attn_out,
        FlashAttnMeta {
            seq_len: prompt_len,
            dim,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
            row_offset: 0,
        },
    );

    // ---- shared block skeleton, second half ----
    let norm2_out = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros_dim));
    let ffn_up_out = Arc::new(Tensor::init_from_cpu(
        ctx.clone(),
        &vec![0.0 as Real; (prompt_len * ffn_hidden_dim) as usize],
    ));
    let out_proj_shape = MatMulMeta {
        m: prompt_len,
        n: dim,
        k: dim,
    };
    let ffnup_shape = MatMulMeta {
        m: prompt_len,
        n: ffn_hidden_dim,
        k: dim,
    };
    let ffndown_shape = MatMulMeta {
        m: prompt_len,
        n: dim,
        k: ffn_hidden_dim,
    };
    block_post_attn(
        gb,
        bw,
        hidden_in,
        &attn_out,
        &norm2_out,
        &ffn_up_out,
        prompt_len,
        ffn_hidden_dim,
        (out_proj_shape, &out_proj_shape.upload(ctx)),
        (norm_shape, &norm_shape.upload(ctx)),
        (ffnup_shape, &ffnup_shape.upload(ctx)),
        (ffndown_shape, &ffndown_shape.upload(ctx)),
    );

    hidden_in.clone()
}

pub(crate) fn build_decode_layer<B: Backend>(
    gb: &mut GraphBuilder<'_, B, ops::Decode>,
    bw: &BlockWeights<B>,
    scratch: &DecodeScratch<B>,
    cache_k: &Arc<Tensor<B>>,
    cache_v: &Arc<Tensor<B>>,
    max_attn_len: u32,
    cfg: &ModelConfig,
) {
    let ModelConfig {
        dim,
        num_heads,
        ffn_hidden: ffn_hidden_dim,
        ..
    } = *cfg;
    let head_dim = cfg.head_dim();
    let attn_len = max_attn_len;

    let norm_shape = NormMeta {
        seq_len: 1,
        size: dim,
        eps: cfg.norm_eps,
    };

    // ---- shared block skeleton (identical code path to prefill) ----
    let qkv_proj_shape = MatMulMeta {
        m: 1,
        n: dim * 3,
        k: dim,
    };
    block_pre_attn(
        gb,
        bw,
        &scratch.hidden,
        &scratch.norm_out,
        &scratch.qkv_out,
        &scratch.q_buf,
        &scratch.k_buf,
        &scratch.v_buf,
        1,
        dim,
        (norm_shape, &scratch.norm_meta),
        (qkv_proj_shape, &scratch.qkv_proj_meta),
        &scratch.qkv_split_meta,
    );

    // ---- RoPE + cache write ----
    let rope_shape = RopeOffsetMeta {
        seq_len: 1,
        dim,
        head_dim,
        pos: 0,
    };
    ops::rope_offset_with(gb, &scratch.q_buf, rope_shape, &scratch.rope_meta);
    ops::rope_offset_with(gb, &scratch.k_buf, rope_shape, &scratch.rope_meta);

    let cache_shape = CacheWriteMeta {
        row_count: 1,
        width: dim,
        dst_row_offset: 0,
    };
    ops::cache_write_with(
        gb,
        &scratch.k_buf,
        cache_k,
        cache_shape,
        &scratch.cache_write_meta,
    );
    ops::cache_write_with(
        gb,
        &scratch.v_buf,
        cache_v,
        cache_shape,
        &scratch.cache_write_meta,
    );

    // ---- cached attention: strided cache reads, no per-head copies ----
    let scale = 1.0 / (head_dim as f32).sqrt();
    ops::attn_qk_cached_with(
        gb,
        &scratch.q_buf,
        cache_k,
        &scratch.scores,
        num_heads,
        attn_len,
        &scratch.attn_meta,
    );
    ops::softmax_rect_with(
        gb,
        &scratch.scores,
        SoftmaxRectMeta {
            num_rows: num_heads,
            width: attn_len,
            scale,
        },
        &scratch.softmax_meta,
    );
    ops::attn_av_cached_with(
        gb,
        &scratch.scores,
        cache_v,
        &scratch.attn_out,
        dim,
        &scratch.attn_meta,
    );

    // ---- shared block skeleton, second half ----
    let out_proj_shape = MatMulMeta {
        m: 1,
        n: dim,
        k: dim,
    };
    let ffnup_shape = MatMulMeta {
        m: 1,
        n: ffn_hidden_dim,
        k: dim,
    };
    let ffndown_shape = MatMulMeta {
        m: 1,
        n: dim,
        k: ffn_hidden_dim,
    };
    block_post_attn(
        gb,
        bw,
        &scratch.hidden,
        &scratch.attn_out,
        &scratch.norm_out,
        &scratch.ffn_up_out,
        1,
        ffn_hidden_dim,
        (out_proj_shape, &scratch.qkv_meta),
        (norm_shape, &scratch.norm_meta),
        (ffnup_shape, &scratch.ffnup_meta),
        (ffndown_shape, &scratch.ffndown_meta),
    );
}

pub struct Cache<B: Backend> {
    pub num_layers: usize,
    pub dim: u32,
    pub max_context_len: u32,
    pub cur_len: u32,
    pub k: Vec<Arc<Tensor<B>>>,
    pub v: Vec<Arc<Tensor<B>>>,
}

impl<B: Backend> Cache<B> {
    pub fn new(ctx: Arc<B>, num_layers: usize, dim: u32, max_context_len: u32) -> Self {
        let zeros = vec![0.0 as Real; (max_context_len * dim) as usize];
        let k = (0..num_layers)
            .map(|_| Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros)))
            .collect();
        let v = (0..num_layers)
            .map(|_| Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros)))
            .collect();

        Self {
            num_layers,
            dim,
            max_context_len,
            cur_len: 0,
            k,
            v,
        }
    }

    pub fn reset(&mut self) {
        self.cur_len = 0;
    }
}

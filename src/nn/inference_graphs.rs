use super::ops;
use super::ops::GraphBuilder;
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

    // ---- norm_1 + fused q/k/v projection ----
    let norm1_out = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros_dim));
    ops::rmsnorm(gb, hidden_in, &bw.norm_1, &norm1_out, norm_shape);

    let qkv_out = Arc::new(Tensor::init_from_cpu(
        ctx.clone(),
        &vec![0.0 as Real; (prompt_len * dim * 3) as usize],
    ));
    ops::matmul(
        gb,
        &norm1_out,
        &bw.qkv_proj,
        &qkv_out,
        MatMulMeta {
            m: prompt_len,
            n: dim * 3,
            k: dim,
        },
    );

    let q_buf = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros_dim));
    let k_buf = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros_dim));
    let v_buf = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros_dim));
    for (buf, off) in [(&q_buf, 0), (&k_buf, dim), (&v_buf, 2 * dim)] {
        ops::head_gather(
            gb,
            &qkv_out,
            buf,
            HeadMoveMeta::qkv_slice(prompt_len, dim, off),
        );
    }

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

    // out_proj fused with the residual add: writes into hidden_in
    ops::matmul_add(
        gb,
        &attn_out,
        &bw.out_proj,
        hidden_in,
        MatMulMeta {
            m: prompt_len,
            n: dim,
            k: dim,
        },
    );

    // ---- FFN + residual ----
    let norm2_out = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros_dim));
    ops::rmsnorm(gb, hidden_in, &bw.norm_2, &norm2_out, norm_shape);

    let ffn_up_out = Arc::new(Tensor::init_from_cpu(
        ctx.clone(),
        &vec![0.0 as Real; (prompt_len * ffn_hidden_dim) as usize],
    ));
    ops::matmul(
        gb,
        &norm2_out,
        &bw.ffn_up,
        &ffn_up_out,
        MatMulMeta {
            m: prompt_len,
            n: ffn_hidden_dim,
            k: dim,
        },
    );

    ops::silu(gb, &ffn_up_out, prompt_len * ffn_hidden_dim);

    // ffn_down fused with the residual add
    ops::matmul_add(
        gb,
        &ffn_up_out,
        &bw.ffn_down,
        hidden_in,
        MatMulMeta {
            m: prompt_len,
            n: dim,
            k: ffn_hidden_dim,
        },
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

    // ---- norm_1 + fused q/k/v projection ----
    ops::rmsnorm_with(
        gb,
        &scratch.hidden,
        &bw.norm_1,
        &scratch.norm_out,
        norm_shape,
        &scratch.norm_meta,
    );

    ops::matmul_with(
        gb,
        &scratch.norm_out,
        &bw.qkv_proj,
        &scratch.qkv_out,
        MatMulMeta {
            m: 1,
            n: dim * 3,
            k: dim,
        },
        &scratch.qkv_proj_meta,
    );
    for (i, dst) in [&scratch.q_buf, &scratch.k_buf, &scratch.v_buf]
        .into_iter()
        .enumerate()
    {
        ops::head_gather_with(
            gb,
            &scratch.qkv_out,
            dst,
            HeadMoveMeta::qkv_slice(1, dim, i as u32 * dim),
            &scratch.qkv_split_meta[i],
        );
    }

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

    // ---- out_proj + residual ----
    ops::matmul_add_with(
        gb,
        &scratch.attn_out,
        &bw.out_proj,
        &scratch.hidden,
        MatMulMeta {
            m: 1,
            n: dim,
            k: dim,
        },
        &scratch.qkv_meta,
    );

    // ---- FFN + residual ----
    ops::rmsnorm_with(
        gb,
        &scratch.hidden,
        &bw.norm_2,
        &scratch.norm_out,
        norm_shape,
        &scratch.norm_meta,
    );

    ops::matmul_with(
        gb,
        &scratch.norm_out,
        &bw.ffn_up,
        &scratch.ffn_up_out,
        MatMulMeta {
            m: 1,
            n: ffn_hidden_dim,
            k: dim,
        },
        &scratch.ffnup_meta,
    );

    ops::silu(gb, &scratch.ffn_up_out, ffn_hidden_dim);

    ops::matmul_add_with(
        gb,
        &scratch.ffn_up_out,
        &bw.ffn_down,
        &scratch.hidden,
        MatMulMeta {
            m: 1,
            n: dim,
            k: ffn_hidden_dim,
        },
        &scratch.ffndown_meta,
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

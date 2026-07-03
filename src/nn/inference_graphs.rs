use super::ops;
use super::ops::meta::{
    CacheWriteMeta, EmbeddingMeta, HeadMoveMeta, KernelMeta, MatMulMeta, NormMeta, RopeMeta,
    RopeOffsetMeta, SoftmaxRectMeta,
};
use super::pipeline::qkv_slice;
use super::weights::BlockWeights;
use crate::Real;
use crate::config::ModelConfig;
use std::sync::Arc;
use wilupgu::{Backend, ComputeGraph, Tensor};

pub(crate) struct DecodeScratch<B: Backend> {
    pub(crate) hidden: Arc<Tensor<B>>,
    pub(crate) norm_out: Arc<Tensor<B>>,
    pub(crate) qkv_out: Arc<Tensor<B>>,
    pub(crate) q_buf: Arc<Tensor<B>>,
    pub(crate) k_buf: Arc<Tensor<B>>,
    pub(crate) v_buf: Arc<Tensor<B>>,
    pub(crate) q_head: Arc<Tensor<B>>,
    pub(crate) k_head: Arc<Tensor<B>>,
    pub(crate) v_head: Arc<Tensor<B>>,
    pub(crate) scores: Arc<Tensor<B>>,
    pub(crate) out_head: Arc<Tensor<B>>,
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
    pub(crate) q_move_meta: Vec<Arc<Tensor<B>>>, // per head: HeadMoveMeta (seq_len=1)

    // ---- dynamic Meta buffers: updated once per decode step ----
    pub(crate) rope_meta: Arc<Tensor<B>>, // RopeOffsetMeta (pos advances)
    pub(crate) cache_write_meta: Arc<Tensor<B>>, // CacheWriteMeta (dst_row_offset advances)
    pub(crate) qkt_meta: Arc<Tensor<B>>,  // MatMulMeta{1,attn_len,head_dim}
    pub(crate) softmax_meta: Arc<Tensor<B>>, // SoftmaxRectMeta (width=attn_len)
    pub(crate) av_meta: Arc<Tensor<B>>,   // MatMulMeta{1,head_dim,attn_len}
    pub(crate) cache_move_meta: Vec<Arc<Tensor<B>>>, // per head: HeadMoveMeta (seq_len=attn_len)
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
            .map(|i| qkv_slice(1, dim, i * dim).upload(&ctx))
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
        let q_move_meta = (0..num_heads)
            .map(|h| {
                HeadMoveMeta {
                    seq_len: 1,
                    full_dim: dim,
                    head_dim,
                    head_offset: h * head_dim,
                }
                .upload(&ctx)
            })
            .collect();

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
        let qkt_meta = MatMulMeta {
            m: 1,
            n: 1,
            k: head_dim,
        }
        .upload(&ctx);
        let softmax_meta = SoftmaxRectMeta {
            num_rows: 1,
            width: 1,
            scale: 1.0,
        }
        .upload(&ctx);
        let av_meta = MatMulMeta {
            m: 1,
            n: head_dim,
            k: 1,
        }
        .upload(&ctx);
        let cache_move_meta = (0..num_heads)
            .map(|h| {
                HeadMoveMeta {
                    seq_len: 1,
                    full_dim: dim,
                    head_dim,
                    head_offset: h * head_dim,
                }
                .upload(&ctx)
            })
            .collect();

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
            q_head: Arc::new(Tensor::init_from_cpu(
                ctx.clone(),
                &zeros(head_dim as usize),
            )),
            k_head: Arc::new(Tensor::init_from_cpu(
                ctx.clone(),
                &zeros((max_context_len * head_dim) as usize),
            )),
            v_head: Arc::new(Tensor::init_from_cpu(
                ctx.clone(),
                &zeros((max_context_len * head_dim) as usize),
            )),
            scores: Arc::new(Tensor::init_from_cpu(
                ctx.clone(),
                &zeros(max_context_len as usize),
            )),
            out_head: Arc::new(Tensor::init_from_cpu(
                ctx.clone(),
                &zeros(head_dim as usize),
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
            q_move_meta,
            rope_meta,
            cache_write_meta,
            qkt_meta,
            softmax_meta,
            av_meta,
            cache_move_meta,
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
        MatMulMeta {
            m: 1,
            n: attn_len,
            k: head_dim,
        }
        .write_to(&self.qkt_meta);
        SoftmaxRectMeta {
            num_rows: 1,
            width: attn_len,
            scale,
        }
        .write_to(&self.softmax_meta);
        MatMulMeta {
            m: 1,
            n: head_dim,
            k: attn_len,
        }
        .write_to(&self.av_meta);
        for h in 0..num_heads as usize {
            HeadMoveMeta {
                seq_len: attn_len,
                full_dim: dim,
                head_dim,
                head_offset: h as u32 * head_dim,
            }
            .write_to(&self.cache_move_meta[h]);
        }
    }
}

pub(crate) fn build_prefill_layer<B: Backend>(
    graph: &mut ComputeGraph<B>,
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
        num_heads,
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
    ops::rmsnorm(graph, hidden_in, &bw.norm_1, &norm1_out, norm_shape);

    let qkv_out = Arc::new(Tensor::init_from_cpu(
        ctx.clone(),
        &vec![0.0 as Real; (prompt_len * dim * 3) as usize],
    ));
    ops::matmul(
        graph,
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
        ops::head_gather(graph, &qkv_out, buf, qkv_slice(prompt_len, dim, off));
    }

    // ---- RoPE + cache write ----
    let rope_shape = RopeMeta {
        seq_len: prompt_len,
        dim,
        head_dim,
    };
    ops::rope(graph, &q_buf, rope_shape);
    ops::rope(graph, &k_buf, rope_shape);

    let cache_shape = CacheWriteMeta {
        row_count: prompt_len,
        width: dim,
        dst_row_offset: 0,
    };
    ops::cache_write(graph, &k_buf, cache_k, cache_shape);
    ops::cache_write(graph, &v_buf, cache_v, cache_shape);

    // ---- attention ----
    let attn_out = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros_dim));
    let _saved = ops::causal_attention(
        graph, &q_buf, &k_buf, &v_buf, &attn_out, prompt_len, dim, num_heads,
    );

    // out_proj fused with the residual add: writes into hidden_in
    ops::matmul_add(
        graph,
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
    ops::rmsnorm(graph, hidden_in, &bw.norm_2, &norm2_out, norm_shape);

    let ffn_up_out = Arc::new(Tensor::init_from_cpu(
        ctx.clone(),
        &vec![0.0 as Real; (prompt_len * ffn_hidden_dim) as usize],
    ));
    ops::matmul(
        graph,
        &norm2_out,
        &bw.ffn_up,
        &ffn_up_out,
        MatMulMeta {
            m: prompt_len,
            n: ffn_hidden_dim,
            k: dim,
        },
    );

    ops::silu(graph, &ffn_up_out, prompt_len * ffn_hidden_dim);

    // ffn_down fused with the residual add
    ops::matmul_add(
        graph,
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
    graph: &mut ComputeGraph<B>,
    bw: &BlockWeights<B>,
    scratch: &DecodeScratch<B>,
    cache_k: &Arc<Tensor<B>>,
    cache_v: &Arc<Tensor<B>>,
    pos: u32,
    cfg: &ModelConfig,
) {
    let ModelConfig {
        dim,
        num_heads,
        ffn_hidden: ffn_hidden_dim,
        ..
    } = *cfg;
    let head_dim = cfg.head_dim();
    let attn_len = pos + 1;

    let norm_shape = NormMeta {
        seq_len: 1,
        size: dim,
        eps: cfg.norm_eps,
    };

    // ---- norm_1 + fused q/k/v projection ----
    ops::rmsnorm_with(
        graph,
        &scratch.hidden,
        &bw.norm_1,
        &scratch.norm_out,
        norm_shape,
        &scratch.norm_meta,
    );

    ops::matmul_with(
        graph,
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
            graph,
            &scratch.qkv_out,
            dst,
            qkv_slice(1, dim, i as u32 * dim),
            &scratch.qkv_split_meta[i],
        );
    }

    // ---- RoPE + cache write ----
    let rope_shape = RopeOffsetMeta {
        seq_len: 1,
        dim,
        head_dim,
        pos,
    };
    ops::rope_offset_with(graph, &scratch.q_buf, rope_shape, &scratch.rope_meta);
    ops::rope_offset_with(graph, &scratch.k_buf, rope_shape, &scratch.rope_meta);

    let cache_shape = CacheWriteMeta {
        row_count: 1,
        width: dim,
        dst_row_offset: pos,
    };
    ops::cache_write_with(
        graph,
        &scratch.k_buf,
        cache_k,
        cache_shape,
        &scratch.cache_write_meta,
    );
    ops::cache_write_with(
        graph,
        &scratch.v_buf,
        cache_v,
        cache_shape,
        &scratch.cache_write_meta,
    );

    // ---- cached attention ----
    let scale = 1.0 / (head_dim as f32).sqrt();
    for h in 0..num_heads as usize {
        let q_move = HeadMoveMeta {
            seq_len: 1,
            full_dim: dim,
            head_dim,
            head_offset: h as u32 * head_dim,
        };
        let cache_move = HeadMoveMeta {
            seq_len: attn_len,
            ..q_move
        };

        ops::head_gather_with(
            graph,
            &scratch.q_buf,
            &scratch.q_head,
            q_move,
            &scratch.q_move_meta[h],
        );
        ops::head_gather_with(
            graph,
            cache_k,
            &scratch.k_head,
            cache_move,
            &scratch.cache_move_meta[h],
        );
        ops::head_gather_with(
            graph,
            cache_v,
            &scratch.v_head,
            cache_move,
            &scratch.cache_move_meta[h],
        );

        ops::matmul_trp_with(
            graph,
            &scratch.q_head,
            &scratch.k_head,
            &scratch.scores,
            MatMulMeta {
                m: 1,
                n: attn_len,
                k: head_dim,
            },
            &scratch.qkt_meta,
        );

        ops::softmax_rect_with(
            graph,
            &scratch.scores,
            SoftmaxRectMeta {
                num_rows: 1,
                width: attn_len,
                scale,
            },
            &scratch.softmax_meta,
        );

        ops::matmul_with(
            graph,
            &scratch.scores,
            &scratch.v_head,
            &scratch.out_head,
            MatMulMeta {
                m: 1,
                n: head_dim,
                k: attn_len,
            },
            &scratch.av_meta,
        );

        ops::head_scatter_with(
            graph,
            &scratch.out_head,
            &scratch.attn_out,
            q_move,
            &scratch.q_move_meta[h],
        );
    }

    // ---- out_proj + residual ----
    ops::matmul_add_with(
        graph,
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
        graph,
        &scratch.hidden,
        &bw.norm_2,
        &scratch.norm_out,
        norm_shape,
        &scratch.norm_meta,
    );

    ops::matmul_with(
        graph,
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

    ops::silu(graph, &scratch.ffn_up_out, ffn_hidden_dim);

    ops::matmul_add_with(
        graph,
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

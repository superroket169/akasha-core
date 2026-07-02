use super::cache::Cache;
use super::ops;
use super::ops::meta::{
    CacheWriteMeta, EmbeddingMeta, HeadMoveMeta, KernelMeta, MatMulMeta, NormMeta, RopeMeta,
    RopeOffsetMeta, SoftmaxRectMeta,
};
use super::pipeline::qkv_slice;
use super::sampling::sample_token;
use super::weights::{BlockWeights, ModelWeights};
use crate::Real;
use crate::config::ModelConfig;
use crate::tokenizer::AkashaTokenizer;
use std::sync::Arc;
use wilupgu::{Backend, ComputeGraph, Tensor};

struct DecodeScratch<B: Backend> {
    hidden: Arc<Tensor<B>>,
    norm_out: Arc<Tensor<B>>,
    qkv_out: Arc<Tensor<B>>,
    q_buf: Arc<Tensor<B>>,
    k_buf: Arc<Tensor<B>>,
    v_buf: Arc<Tensor<B>>,
    q_head: Arc<Tensor<B>>,
    k_head: Arc<Tensor<B>>,
    v_head: Arc<Tensor<B>>,
    scores: Arc<Tensor<B>>,
    out_head: Arc<Tensor<B>>,
    attn_out: Arc<Tensor<B>>,
    ffn_up_out: Arc<Tensor<B>>,
    final_norm_out: Arc<Tensor<B>>,
    logits: Arc<Tensor<B>>,

    // ---- constant Meta buffers: written once here, read forever after ----
    norm_meta: Arc<Tensor<B>>,     // NormMeta -- norm_1, norm_2, final_norm
    qkv_meta: Arc<Tensor<B>>,      // MatMulMeta{1,dim,dim} -- out_proj
    qkv_proj_meta: Arc<Tensor<B>>, // MatMulMeta{1,3*dim,dim} -- fused qkv proj
    qkv_split_meta: Vec<Arc<Tensor<B>>>, // 3x HeadMoveMeta (q/k/v slice of fused qkv)
    ffnup_meta: Arc<Tensor<B>>,    // MatMulMeta{1,ffn_hidden,dim}
    ffndown_meta: Arc<Tensor<B>>,  // MatMulMeta{1,dim,ffn_hidden}
    emb_meta: Arc<Tensor<B>>,      // EmbeddingMeta (seq_len=1)
    lm_meta: Arc<Tensor<B>>,       // MatMulMeta{1,vocab_size,dim}
    q_move_meta: Vec<Arc<Tensor<B>>>, // per head: HeadMoveMeta (seq_len=1)

    // ---- dynamic Meta buffers: updated once per decode step ----
    rope_meta: Arc<Tensor<B>>,            // RopeOffsetMeta (pos advances)
    cache_write_meta: Arc<Tensor<B>>,     // CacheWriteMeta (dst_row_offset advances)
    qkt_meta: Arc<Tensor<B>>,             // MatMulMeta{1,attn_len,head_dim}
    softmax_meta: Arc<Tensor<B>>,         // SoftmaxRectMeta (width=attn_len)
    av_meta: Arc<Tensor<B>>,              // MatMulMeta{1,head_dim,attn_len}
    cache_move_meta: Vec<Arc<Tensor<B>>>, // per head: HeadMoveMeta (seq_len=attn_len)
}

impl<B: Backend> DecodeScratch<B> {
    fn new(ctx: Arc<B>, cfg: &ModelConfig, max_context_len: u32) -> Self {
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

    fn update_for_step(&self, pos: u32, cfg: &ModelConfig) {
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

fn build_prefill_layer<B: Backend>(
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
    ops::norm::rmsnorm(graph, hidden_in, &bw.norm_1, &norm1_out, norm_shape);

    let qkv_out = Arc::new(Tensor::init_from_cpu(
        ctx.clone(),
        &vec![0.0 as Real; (prompt_len * dim * 3) as usize],
    ));
    ops::matmul::matmul(
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
        ops::head_move::head_gather(graph, &qkv_out, buf, qkv_slice(prompt_len, dim, off));
    }

    // ---- RoPE + cache write ----
    let rope_shape = RopeMeta {
        seq_len: prompt_len,
        dim,
        head_dim,
    };
    ops::rope::rope(graph, &q_buf, rope_shape);
    ops::rope::rope(graph, &k_buf, rope_shape);

    let cache_shape = CacheWriteMeta {
        row_count: prompt_len,
        width: dim,
        dst_row_offset: 0,
    };
    ops::cache::cache_write(graph, &k_buf, cache_k, cache_shape);
    ops::cache::cache_write(graph, &v_buf, cache_v, cache_shape);

    // ---- attention ----
    let attn_out = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros_dim));
    let _saved = ops::attention::causal_attention(
        graph, &q_buf, &k_buf, &v_buf, &attn_out, prompt_len, dim, num_heads,
    );

    // out_proj fused with the residual add: writes into hidden_in
    ops::matmul::matmul_add(
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
    ops::norm::rmsnorm(graph, hidden_in, &bw.norm_2, &norm2_out, norm_shape);

    let ffn_up_out = Arc::new(Tensor::init_from_cpu(
        ctx.clone(),
        &vec![0.0 as Real; (prompt_len * ffn_hidden_dim) as usize],
    ));
    ops::matmul::matmul(
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

    ops::elementwise::silu(graph, &ffn_up_out, prompt_len * ffn_hidden_dim);

    // ffn_down fused with the residual add
    ops::matmul::matmul_add(
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

fn build_decode_layer<B: Backend>(
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
    ops::norm::rmsnorm_with(
        graph,
        &scratch.hidden,
        &bw.norm_1,
        &scratch.norm_out,
        norm_shape,
        &scratch.norm_meta,
    );

    ops::matmul::matmul_with(
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
        ops::head_move::head_gather_with(
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
    ops::rope::rope_offset_with(graph, &scratch.q_buf, rope_shape, &scratch.rope_meta);
    ops::rope::rope_offset_with(graph, &scratch.k_buf, rope_shape, &scratch.rope_meta);

    let cache_shape = CacheWriteMeta {
        row_count: 1,
        width: dim,
        dst_row_offset: pos,
    };
    ops::cache::cache_write_with(
        graph,
        &scratch.k_buf,
        cache_k,
        cache_shape,
        &scratch.cache_write_meta,
    );
    ops::cache::cache_write_with(
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

        ops::head_move::head_gather_with(
            graph,
            &scratch.q_buf,
            &scratch.q_head,
            q_move,
            &scratch.q_move_meta[h],
        );
        ops::head_move::head_gather_with(
            graph,
            cache_k,
            &scratch.k_head,
            cache_move,
            &scratch.cache_move_meta[h],
        );
        ops::head_move::head_gather_with(
            graph,
            cache_v,
            &scratch.v_head,
            cache_move,
            &scratch.cache_move_meta[h],
        );

        ops::matmul::matmul_trp_with(
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

        ops::attention::softmax_rect_with(
            graph,
            &scratch.scores,
            SoftmaxRectMeta {
                num_rows: 1,
                width: attn_len,
                scale,
            },
            &scratch.softmax_meta,
        );

        ops::matmul::matmul_with(
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

        ops::head_move::head_scatter_with(
            graph,
            &scratch.out_head,
            &scratch.attn_out,
            q_move,
            &scratch.q_move_meta[h],
        );
    }

    // ---- out_proj + residual ----
    ops::matmul::matmul_add_with(
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
    ops::norm::rmsnorm_with(
        graph,
        &scratch.hidden,
        &bw.norm_2,
        &scratch.norm_out,
        norm_shape,
        &scratch.norm_meta,
    );

    ops::matmul::matmul_with(
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

    ops::elementwise::silu(graph, &scratch.ffn_up_out, ffn_hidden_dim);

    ops::matmul::matmul_add_with(
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

pub struct InferenceSession<B: Backend> {
    ctx: Arc<B>,
    weights: Arc<ModelWeights<B>>,
    cfg: ModelConfig,
    max_context_len: u32,
    cache: Option<Cache<B>>,
    scratch: DecodeScratch<B>,
    input_token_buf: Arc<Tensor<B>>,
}

impl<B: Backend> InferenceSession<B> {
    pub fn new(ctx: Arc<B>, weights: Arc<ModelWeights<B>>, max_context_len: u32) -> Self {
        let cfg = weights.cfg;
        let scratch = DecodeScratch::new(ctx.clone(), &cfg, max_context_len);
        let input_token_buf = Arc::new(Tensor::init_from_cpu(ctx.clone(), &[0u32]));

        Self {
            ctx,
            weights,
            cfg,
            max_context_len,
            cache: None,
            scratch,
            input_token_buf,
        }
    }

    pub fn replace_cache(&mut self, new: Cache<B>) -> Option<Cache<B>> {
        assert_eq!(
            new.num_layers, self.cfg.num_layers,
            "Cache/model layer count mismatch"
        );
        assert_eq!(new.dim, self.cfg.dim, "Cache/model dim mismatch");
        assert_eq!(
            new.max_context_len, self.max_context_len,
            "Cache/session max_context_len mismatch"
        );
        std::mem::replace(&mut self.cache, Some(new))
    }

    pub fn take_cache(&mut self) -> Option<Cache<B>> {
        self.cache.take()
    }

    pub fn prefill(&mut self, prompt_tokens: &[u32]) -> Vec<Real> {
        let prompt_len = prompt_tokens.len() as u32;
        assert!(prompt_len >= 1, "prefill: prompt must be non-empty");
        assert!(
            prompt_len <= self.max_context_len,
            "prefill: prompt longer than max_context_len"
        );
        assert_eq!(
            self.cache
                .as_ref()
                .expect("prefill: no cache attached (call replace_cache first)")
                .cur_len,
            0,
            "prefill: v1 only supports an empty cache; loop decode_step for a resumed cache"
        );

        let ctx = self.ctx.clone();
        let weights = self.weights.clone();
        let cfg = self.cfg;
        let dim = cfg.dim;
        let vocab_size = cfg.vocab_size;

        let tokens_buf = Arc::new(Tensor::init_from_cpu(ctx.clone(), prompt_tokens));

        let mut graph = ComputeGraph::new(ctx.clone());

        let mut hidden = Arc::new(Tensor::init_from_cpu(
            ctx.clone(),
            &vec![0.0 as Real; (prompt_len * dim) as usize],
        ));
        ops::embedding::embedding(
            &mut graph,
            &tokens_buf,
            &weights.embedding,
            &hidden,
            EmbeddingMeta {
                vocab_size,
                dim,
                seq_len: prompt_len,
            },
        );

        {
            let cache = self.cache.as_ref().unwrap();
            for (i, bw) in weights.blocks.iter().enumerate() {
                hidden = build_prefill_layer(
                    &mut graph,
                    &ctx,
                    bw,
                    &hidden,
                    &cache.k[i],
                    &cache.v[i],
                    prompt_len,
                    &cfg,
                );
            }
        }

        let final_out = Arc::new(Tensor::init_from_cpu(
            ctx.clone(),
            &vec![0.0 as Real; (prompt_len * dim) as usize],
        ));
        ops::norm::rmsnorm(
            &mut graph,
            &hidden,
            &weights.final_norm,
            &final_out,
            NormMeta {
                seq_len: prompt_len,
                size: dim,
                eps: cfg.norm_eps,
            },
        );

        let logits = Arc::new(Tensor::init_from_cpu(
            ctx.clone(),
            &vec![0.0 as Real; (prompt_len * vocab_size) as usize],
        ));
        ops::matmul::matmul(
            &mut graph,
            &final_out,
            &weights.lm_head,
            &logits,
            MatMulMeta {
                m: prompt_len,
                n: vocab_size,
                k: dim,
            },
        );

        graph.execute();

        let all_logits: Vec<Real> = logits.to_cpu();
        let last_row = (prompt_len - 1) as usize * vocab_size as usize;
        let result = all_logits[last_row..last_row + vocab_size as usize].to_vec();

        self.cache.as_mut().unwrap().cur_len = prompt_len;
        result
    }

    pub fn decode_step(&mut self, token: u32) -> Vec<Real> {
        let pos = {
            let cache = self
                .cache
                .as_ref()
                .expect("decode_step: no cache attached (call replace_cache first)");
            assert!(
                cache.cur_len < self.max_context_len,
                "decode_step: context full"
            );
            cache.cur_len
        };

        self.input_token_buf.copy_from_cpu(&[token]);

        let ctx = self.ctx.clone();
        let weights = self.weights.clone();
        let cfg = self.cfg;
        let dim = cfg.dim;
        let vocab_size = cfg.vocab_size;

        self.scratch.update_for_step(pos, &cfg);

        let mut graph = ComputeGraph::new(ctx.clone());

        ops::embedding::embedding_with(
            &mut graph,
            &self.input_token_buf,
            &weights.embedding,
            &self.scratch.hidden,
            EmbeddingMeta {
                vocab_size,
                dim,
                seq_len: 1,
            },
            &self.scratch.emb_meta,
        );

        {
            let cache = self.cache.as_ref().unwrap();
            for (i, bw) in weights.blocks.iter().enumerate() {
                build_decode_layer(
                    &mut graph,
                    bw,
                    &self.scratch,
                    &cache.k[i],
                    &cache.v[i],
                    pos,
                    &cfg,
                );
            }
        }

        ops::norm::rmsnorm_with(
            &mut graph,
            &self.scratch.hidden,
            &weights.final_norm,
            &self.scratch.final_norm_out,
            NormMeta {
                seq_len: 1,
                size: dim,
                eps: cfg.norm_eps,
            },
            &self.scratch.norm_meta,
        );

        ops::matmul::matmul_with(
            &mut graph,
            &self.scratch.final_norm_out,
            &weights.lm_head,
            &self.scratch.logits,
            MatMulMeta {
                m: 1,
                n: vocab_size,
                k: dim,
            },
            &self.scratch.lm_meta,
        );

        graph.execute();

        self.cache.as_mut().unwrap().cur_len = pos + 1;

        self.scratch.logits.to_cpu()
    }

    pub fn generate(
        &mut self,
        tokenizer: &AkashaTokenizer,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        temperature: f32,
    ) -> String {
        if self.cache.is_none() {
            self.cache = Some(Cache::new(
                self.ctx.clone(),
                self.cfg.num_layers,
                self.cfg.dim,
                self.max_context_len,
            ));
        }

        let mut logits = if self.cache.as_ref().unwrap().cur_len == 0 {
            self.prefill(prompt_tokens)
        } else {
            let mut last = Vec::new();
            for &t in prompt_tokens {
                last = self.decode_step(t);
            }
            last
        };

        const EOS: u32 = 50256;
        let mut generated: Vec<u32> = Vec::with_capacity(max_new_tokens);
        for _ in 0..max_new_tokens {
            let next = sample_token(&logits, temperature);
            generated.push(next);
            if next == EOS {
                break;
            }
            logits = self.decode_step(next);
        }

        tokenizer.decode(&generated)
    }
}

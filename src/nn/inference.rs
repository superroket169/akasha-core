use super::akasha_model::{AkashaModel, sample_token};
use super::attention::SelfAttention;
use super::cache::Cache;
use super::embedding::Embedding;
use super::linear::Linear;
use super::pipeline::{TransformerBlock, add_qkv_slice_node, add_rope_node};
use super::ops::meta::{
    CacheWriteMeta, EmbeddingMeta, HeadMoveMeta, KernelMeta, MatMulMeta, NormMeta, RopeMeta,
    RopeOffsetMeta, SoftmaxRectMeta,
};
use super::rmsnorm::RMSNorm;
use crate::Real;
use crate::config::ModelConfig;
use crate::tokenizer::AkashaTokenizer;
use std::sync::Arc;
use wilupgu::{Backend, Binding, ComputeGraph, Tensor, TensorMode};

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
    rope_meta: Arc<Tensor<B>>,        // RopeOffsetMeta (pos advances)
    cache_write_meta: Arc<Tensor<B>>, // CacheWriteMeta (dst_row_offset advances)
    qkt_meta: Arc<Tensor<B>>,         // MatMulMeta{1,attn_len,head_dim}
    softmax_meta: Arc<Tensor<B>>,     // SoftmaxRectMeta (width=attn_len)
    av_meta: Arc<Tensor<B>>,          // MatMulMeta{1,head_dim,attn_len}
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
            .map(|i| {
                HeadMoveMeta {
                    seq_len: 1,
                    full_dim: dim * 3,
                    head_dim: dim,
                    head_offset: i * dim,
                }
                .upload(&ctx)
            })
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
    layer: &TransformerBlock<B>,
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

    let norm1_out = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros_dim));
    RMSNorm::forward_nodes(
        graph,
        &layer.norm_1.weight,
        hidden_in,
        &norm1_out,
        prompt_len,
        dim,
        cfg.norm_eps,
    );

    let q_buf = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros_dim));
    let k_buf = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros_dim));
    let v_buf = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros_dim));
    let qkv_out = Arc::new(Tensor::init_from_cpu(
        ctx.clone(),
        &vec![0.0 as Real; (prompt_len * dim * 3) as usize],
    ));
    Linear::forward_nodes(
        graph,
        &layer.qkv_proj.weight,
        &norm1_out,
        &qkv_out,
        prompt_len,
        dim,
        dim * 3,
    );
    add_qkv_slice_node(graph, "HeadGather", &qkv_out, &q_buf, dim, prompt_len, 0);
    add_qkv_slice_node(graph, "HeadGather", &qkv_out, &k_buf, dim, prompt_len, dim);
    add_qkv_slice_node(
        graph,
        "HeadGather",
        &qkv_out,
        &v_buf,
        dim,
        prompt_len,
        2 * dim,
    );

    let rope_meta = RopeMeta {
        seq_len: prompt_len,
        dim,
        head_dim,
    }
    .upload(ctx);
    add_rope_node(graph, "RoPE", &q_buf, &rope_meta, dim, prompt_len);
    add_rope_node(graph, "RoPE", &k_buf, &rope_meta, dim, prompt_len);

    let cache_write_meta = CacheWriteMeta {
        row_count: prompt_len,
        width: dim,
        dst_row_offset: 0,
    }
    .upload(ctx);
    graph.add_node(
        "CacheWrite",
        &[
            Binding::new(0, &k_buf.buffer, TensorMode::Input),
            Binding::new(1, &cache_k.buffer, TensorMode::InOut),
            Binding::new(2, &cache_write_meta.buffer, TensorMode::Meta),
        ],
        [(dim + 15) / 16, (prompt_len + 15) / 16, 1],
    );
    graph.add_node(
        "CacheWrite",
        &[
            Binding::new(0, &v_buf.buffer, TensorMode::Input),
            Binding::new(1, &cache_v.buffer, TensorMode::InOut),
            Binding::new(2, &cache_write_meta.buffer, TensorMode::Meta),
        ],
        [(dim + 15) / 16, (prompt_len + 15) / 16, 1],
    );

    let attn_out = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros_dim));
    let mut q_heads = Vec::new();
    let mut k_heads = Vec::new();
    let mut v_heads = Vec::new();
    let mut t_scores_heads = Vec::new();
    SelfAttention::forward_nodes(
        graph,
        prompt_len,
        dim,
        num_heads,
        &q_buf,
        &k_buf,
        &v_buf,
        &attn_out,
        &mut q_heads,
        &mut k_heads,
        &mut v_heads,
        &mut t_scores_heads,
    );

    let outproj_meta = MatMulMeta {
        m: prompt_len,
        n: dim,
        k: dim,
    }
    .upload(ctx);
    graph.add_node(
        "MatMulAdd",
        &[
            Binding::new(0, &attn_out.buffer, TensorMode::Input),
            Binding::new(1, &layer.out_proj.weight.buffer, TensorMode::Input),
            Binding::new(2, &hidden_in.buffer, TensorMode::InOut),
            Binding::new(3, &outproj_meta.buffer, TensorMode::Meta),
        ],
        [(dim + 15) / 16, (prompt_len + 15) / 16, 1],
    );

    let norm2_out = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros_dim));
    RMSNorm::forward_nodes(
        graph,
        &layer.norm_2.weight,
        hidden_in,
        &norm2_out,
        prompt_len,
        dim,
        cfg.norm_eps,
    );

    let ffn_up_out = Arc::new(Tensor::init_from_cpu(
        ctx.clone(),
        &vec![0.0 as Real; (prompt_len * ffn_hidden_dim) as usize],
    ));
    Linear::forward_nodes(
        graph,
        &layer.ffn_up.weight,
        &norm2_out,
        &ffn_up_out,
        prompt_len,
        dim,
        ffn_hidden_dim,
    );

    graph.add_node(
        "SiLU",
        &[Binding::new(0, &ffn_up_out.buffer, TensorMode::InOut)],
        [(prompt_len * ffn_hidden_dim + 255) / 256, 1, 1],
    );

    // Fused ffn_down-MatMul + residual-add, same reasoning as out_proj above.
    let ffndown_meta = MatMulMeta {
        m: prompt_len,
        n: dim,
        k: ffn_hidden_dim,
    }
    .upload(ctx);
    graph.add_node(
        "MatMulAdd",
        &[
            Binding::new(0, &ffn_up_out.buffer, TensorMode::Input),
            Binding::new(1, &layer.ffn_down.weight.buffer, TensorMode::Input),
            Binding::new(2, &hidden_in.buffer, TensorMode::InOut),
            Binding::new(3, &ffndown_meta.buffer, TensorMode::Meta),
        ],
        [(dim + 15) / 16, (prompt_len + 15) / 16, 1],
    );

    hidden_in.clone()
}

fn build_decode_layer<B: Backend>(
    graph: &mut ComputeGraph<B>,
    layer: &TransformerBlock<B>,
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

    // ---- norm_1 ----
    graph.add_node(
        "RMSNorm",
        &[
            Binding::new(0, &scratch.hidden.buffer, TensorMode::Input),
            Binding::new(1, &layer.norm_1.weight.buffer, TensorMode::Input),
            Binding::new(2, &scratch.norm_out.buffer, TensorMode::Output),
            Binding::new(3, &scratch.norm_meta.buffer, TensorMode::Meta),
        ],
        [1, 1, 1],
    );

    // ---- fused q/k/v projection ----
    graph.add_node(
        "MatMul",
        &[
            Binding::new(0, &scratch.norm_out.buffer, TensorMode::Input),
            Binding::new(1, &layer.qkv_proj.weight.buffer, TensorMode::Input),
            Binding::new(2, &scratch.qkv_out.buffer, TensorMode::Output),
            Binding::new(3, &scratch.qkv_proj_meta.buffer, TensorMode::Meta),
        ],
        [(dim * 3 + 15) / 16, 1, 1],
    );
    for (dst, meta) in [
        (&scratch.q_buf, &scratch.qkv_split_meta[0]),
        (&scratch.k_buf, &scratch.qkv_split_meta[1]),
        (&scratch.v_buf, &scratch.qkv_split_meta[2]),
    ] {
        graph.add_node(
            "HeadGather",
            &[
                Binding::new(0, &scratch.qkv_out.buffer, TensorMode::Input),
                Binding::new(1, &dst.buffer, TensorMode::Output),
                Binding::new(2, &meta.buffer, TensorMode::Meta),
            ],
            [(dim + 15) / 16, 1, 1],
        );
    }

    // ---- RoPE ----
    let rope_grid = [(head_dim / 2 + 15) / 16, 1, 1];
    graph.add_node(
        "RoPEOffset",
        &[
            Binding::new(0, &scratch.q_buf.buffer, TensorMode::InOut),
            Binding::new(1, &scratch.rope_meta.buffer, TensorMode::Meta),
        ],
        rope_grid,
    );
    graph.add_node(
        "RoPEOffset",
        &[
            Binding::new(0, &scratch.k_buf.buffer, TensorMode::InOut),
            Binding::new(1, &scratch.rope_meta.buffer, TensorMode::Meta),
        ],
        rope_grid,
    );

    graph.add_node(
        "CacheWrite",
        &[
            Binding::new(0, &scratch.k_buf.buffer, TensorMode::Input),
            Binding::new(1, &cache_k.buffer, TensorMode::InOut),
            Binding::new(2, &scratch.cache_write_meta.buffer, TensorMode::Meta),
        ],
        [(dim + 15) / 16, 1, 1],
    );
    graph.add_node(
        "CacheWrite",
        &[
            Binding::new(0, &scratch.v_buf.buffer, TensorMode::Input),
            Binding::new(1, &cache_v.buffer, TensorMode::InOut),
            Binding::new(2, &scratch.cache_write_meta.buffer, TensorMode::Meta),
        ],
        [(dim + 15) / 16, 1, 1],
    );

    for h in 0..num_heads as usize {
        let q_move_meta = &scratch.q_move_meta[h];
        let cache_move_meta = &scratch.cache_move_meta[h];

        graph.add_node(
            "HeadGather",
            &[
                Binding::new(0, &scratch.q_buf.buffer, TensorMode::Input),
                Binding::new(1, &scratch.q_head.buffer, TensorMode::Output),
                Binding::new(2, &q_move_meta.buffer, TensorMode::Meta),
            ],
            [(head_dim + 15) / 16, 1, 1],
        );
        graph.add_node(
            "HeadGather",
            &[
                Binding::new(0, &cache_k.buffer, TensorMode::Input),
                Binding::new(1, &scratch.k_head.buffer, TensorMode::Output),
                Binding::new(2, &cache_move_meta.buffer, TensorMode::Meta),
            ],
            [(head_dim + 15) / 16, (attn_len + 15) / 16, 1],
        );
        graph.add_node(
            "HeadGather",
            &[
                Binding::new(0, &cache_v.buffer, TensorMode::Input),
                Binding::new(1, &scratch.v_head.buffer, TensorMode::Output),
                Binding::new(2, &cache_move_meta.buffer, TensorMode::Meta),
            ],
            [(head_dim + 15) / 16, (attn_len + 15) / 16, 1],
        );

        graph.add_node(
            "MatMulTrp",
            &[
                Binding::new(0, &scratch.q_head.buffer, TensorMode::Input),
                Binding::new(1, &scratch.k_head.buffer, TensorMode::Input),
                Binding::new(2, &scratch.scores.buffer, TensorMode::Output),
                Binding::new(3, &scratch.qkt_meta.buffer, TensorMode::Meta),
            ],
            [(attn_len + 15) / 16, 1, 1],
        );

        graph.add_node(
            "SoftmaxRect",
            &[
                Binding::new(0, &scratch.scores.buffer, TensorMode::InOut),
                Binding::new(1, &scratch.softmax_meta.buffer, TensorMode::Meta),
            ],
            [1, 1, 1],
        );

        graph.add_node(
            "MatMul",
            &[
                Binding::new(0, &scratch.scores.buffer, TensorMode::Input),
                Binding::new(1, &scratch.v_head.buffer, TensorMode::Input),
                Binding::new(2, &scratch.out_head.buffer, TensorMode::Output),
                Binding::new(3, &scratch.av_meta.buffer, TensorMode::Meta),
            ],
            [(head_dim + 15) / 16, 1, 1],
        );

        graph.add_node(
            "HeadScatter",
            &[
                Binding::new(0, &scratch.out_head.buffer, TensorMode::Input),
                Binding::new(1, &scratch.attn_out.buffer, TensorMode::Output),
                Binding::new(2, &q_move_meta.buffer, TensorMode::Meta),
            ],
            [(head_dim + 15) / 16, 1, 1],
        );
    }

    // ---- out_proj + residual ----
    graph.add_node(
        "MatMulAdd",
        &[
            Binding::new(0, &scratch.attn_out.buffer, TensorMode::Input),
            Binding::new(1, &layer.out_proj.weight.buffer, TensorMode::Input),
            Binding::new(2, &scratch.hidden.buffer, TensorMode::InOut),
            Binding::new(3, &scratch.qkv_meta.buffer, TensorMode::Meta),
        ],
        [(dim + 15) / 16, 1, 1],
    );

    // ---- FFN + residual ----
    graph.add_node(
        "RMSNorm",
        &[
            Binding::new(0, &scratch.hidden.buffer, TensorMode::Input),
            Binding::new(1, &layer.norm_2.weight.buffer, TensorMode::Input),
            Binding::new(2, &scratch.norm_out.buffer, TensorMode::Output),
            Binding::new(3, &scratch.norm_meta.buffer, TensorMode::Meta),
        ],
        [1, 1, 1],
    );

    graph.add_node(
        "MatMul",
        &[
            Binding::new(0, &scratch.norm_out.buffer, TensorMode::Input),
            Binding::new(1, &layer.ffn_up.weight.buffer, TensorMode::Input),
            Binding::new(2, &scratch.ffn_up_out.buffer, TensorMode::Output),
            Binding::new(3, &scratch.ffnup_meta.buffer, TensorMode::Meta),
        ],
        [(ffn_hidden_dim + 15) / 16, 1, 1],
    );

    graph.add_node(
        "SiLU",
        &[Binding::new(
            0,
            &scratch.ffn_up_out.buffer,
            TensorMode::InOut,
        )],
        [(ffn_hidden_dim + 255) / 256, 1, 1],
    );

    graph.add_node(
        "MatMulAdd",
        &[
            Binding::new(0, &scratch.ffn_up_out.buffer, TensorMode::Input),
            Binding::new(1, &layer.ffn_down.weight.buffer, TensorMode::Input),
            Binding::new(2, &scratch.hidden.buffer, TensorMode::InOut),
            Binding::new(3, &scratch.ffndown_meta.buffer, TensorMode::Meta),
        ],
        [(dim + 15) / 16, 1, 1],
    );
}

pub struct InferenceSession<B: Backend> {
    ctx: Arc<B>,
    model: Arc<AkashaModel<B>>,
    cfg: ModelConfig,
    max_context_len: u32,
    cache: Option<Cache<B>>,
    scratch: DecodeScratch<B>,
    input_token_buf: Arc<Tensor<B>>,
}

impl<B: Backend> InferenceSession<B> {
    pub fn new(ctx: Arc<B>, model: Arc<AkashaModel<B>>, max_context_len: u32) -> Self {
        let cfg = model.cfg;
        let scratch = DecodeScratch::new(ctx.clone(), &cfg, max_context_len);
        let input_token_buf = Arc::new(Tensor::init_from_cpu(ctx.clone(), &[0u32]));

        Self {
            ctx,
            model,
            cfg,
            max_context_len,
            cache: None,
            scratch,
            input_token_buf,
        }
    }

    pub fn replace_cache(&mut self, new: Cache<B>) -> Option<Cache<B>> {
        assert_eq!(
            new.num_layers,
            self.model.layers.len(),
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
        let model = self.model.clone();
        let cfg = self.cfg;
        let dim = cfg.dim;
        let vocab_size = cfg.vocab_size;

        let tokens_buf = Arc::new(Tensor::init_from_cpu(ctx.clone(), prompt_tokens));

        let mut graph = ComputeGraph::new(ctx.clone());

        let mut hidden = Arc::new(Tensor::init_from_cpu(
            ctx.clone(),
            &vec![0.0 as Real; (prompt_len * dim) as usize],
        ));
        Embedding::forward_nodes(
            &mut graph,
            &model.embedding.table,
            &tokens_buf,
            &hidden,
            vocab_size,
            dim,
            prompt_len,
        );

        {
            let cache = self.cache.as_ref().unwrap();
            for (i, layer) in model.layers.iter().enumerate() {
                hidden = build_prefill_layer(
                    &mut graph,
                    &ctx,
                    layer,
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
        RMSNorm::forward_nodes(
            &mut graph,
            &model.final_norm.weight,
            &hidden,
            &final_out,
            prompt_len,
            dim,
            cfg.norm_eps,
        );

        let logits = Arc::new(Tensor::init_from_cpu(
            ctx.clone(),
            &vec![0.0 as Real; (prompt_len * vocab_size) as usize],
        ));
        Linear::forward_nodes(
            &mut graph,
            &model.lm_head.weight,
            &final_out,
            &logits,
            prompt_len,
            dim,
            vocab_size,
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
        let model = self.model.clone();
        let cfg = self.cfg;
        let dim = cfg.dim;
        let vocab_size = cfg.vocab_size;

        self.scratch.update_for_step(pos, &cfg);

        let mut graph = ComputeGraph::new(ctx.clone());

        graph.add_node(
            "Embedding",
            &[
                Binding::new(0, &self.input_token_buf.buffer, TensorMode::Input),
                Binding::new(1, &model.embedding.table.buffer, TensorMode::Input),
                Binding::new(2, &self.scratch.hidden.buffer, TensorMode::Output),
                Binding::new(3, &self.scratch.emb_meta.buffer, TensorMode::Meta),
            ],
            [(dim + 255) / 256, 1, 1],
        );

        {
            let cache = self.cache.as_ref().unwrap();
            for (i, layer) in model.layers.iter().enumerate() {
                build_decode_layer(
                    &mut graph,
                    layer,
                    &self.scratch,
                    &cache.k[i],
                    &cache.v[i],
                    pos,
                    &cfg,
                );
            }
        }

        graph.add_node(
            "RMSNorm",
            &[
                Binding::new(0, &self.scratch.hidden.buffer, TensorMode::Input),
                Binding::new(1, &model.final_norm.weight.buffer, TensorMode::Input),
                Binding::new(2, &self.scratch.final_norm_out.buffer, TensorMode::Output),
                Binding::new(3, &self.scratch.norm_meta.buffer, TensorMode::Meta),
            ],
            [1, 1, 1],
        );

        graph.add_node(
            "MatMul",
            &[
                Binding::new(0, &self.scratch.final_norm_out.buffer, TensorMode::Input),
                Binding::new(1, &model.lm_head.weight.buffer, TensorMode::Input),
                Binding::new(2, &self.scratch.logits.buffer, TensorMode::Output),
                Binding::new(3, &self.scratch.lm_meta.buffer, TensorMode::Meta),
            ],
            [(vocab_size + 15) / 16, 1, 1],
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
                self.model.layers.len(),
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

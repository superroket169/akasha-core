use super::akasha_model::{AkashaModel, sample_token};
use super::attention::SelfAttention;
use super::cache::Cache;
use super::embedding::Embedding;
use super::linear::Linear;
use super::pipeline::{TransformerBlock, add_qkv_slice_node, add_rope_node};
use super::rmsnorm::RMSNorm;
use crate::Real;
use crate::tokenizer::AkashaTokenizer;
use std::sync::Arc;
use wilupgu::{Backend, Binding, ComputeGraph, Tensor, TensorMode};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct NormMeta {
    seq_len: u32,
    size: u32,
    eps: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct HeadMoveMeta {
    seq_len: u32,
    full_dim: u32,
    head_dim: u32,
    head_offset: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SoftmaxRectMeta {
    num_rows: u32,
    width: u32,
    scale: f32,
}

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
    norm_meta: Arc<Tensor<B>>, // {seq_len:1,size:dim,eps} -- norm_1, norm_2, final_norm
    qkv_meta: Arc<Tensor<B>>,  // {1,dim,dim} -- out_proj
    qkv_proj_meta: Arc<Tensor<B>>, // {1,dim,3*dim} -- fused qkv proj matmul
    qkv_split_meta: Vec<Arc<Tensor<B>>>, // 3x: {seq_len:1,full_dim:3*dim,head_dim:dim,head_offset:0/dim/2*dim}
    ffnup_meta: Arc<Tensor<B>>,          // {1,ffn_hidden_dim,dim}
    ffndown_meta: Arc<Tensor<B>>,        // {1,dim,ffn_hidden_dim}
    emb_meta: Arc<Tensor<B>>,            // {vocab_size,dim,1}
    lm_meta: Arc<Tensor<B>>,             // {1,vocab_size,dim}
    q_move_meta: Vec<Arc<Tensor<B>>>,    // per head: {seq_len:1,full_dim:dim,head_dim,head_offset}

    // ---- dynamic Meta buffers: updated once per decode step ----
    rope_meta: Arc<Tensor<B>>,            // {1,dim,head_dim,pos}
    cache_write_meta: Arc<Tensor<B>>,     // {1,dim,pos}
    qkt_meta: Arc<Tensor<B>>,             // {1,attn_len,head_dim}
    softmax_meta: Arc<Tensor<B>>,         // {num_rows:1,width:attn_len,scale}
    av_meta: Arc<Tensor<B>>,              // {1,head_dim,attn_len}
    cache_move_meta: Vec<Arc<Tensor<B>>>, // per head: {seq_len:attn_len,full_dim:dim,head_dim,head_offset}
}

impl<B: Backend> DecodeScratch<B> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        ctx: Arc<B>,
        dim: u32,
        num_heads: u32,
        head_dim: u32,
        max_context_len: u32,
        ffn_hidden_dim: u32,
        vocab_size: u32,
    ) -> Self {
        let zeros = |n: usize| vec![0.0 as Real; n];
        let dim1 = zeros(dim as usize);

        let norm_meta = Arc::new(Tensor::init_from_cpu(
            ctx.clone(),
            &[NormMeta {
                seq_len: 1,
                size: dim,
                eps: 1e-5,
            }],
        ));
        let qkv_meta = Arc::new(Tensor::init_from_cpu(ctx.clone(), &[1u32, dim, dim]));
        let qkv_proj_meta = Arc::new(Tensor::init_from_cpu(ctx.clone(), &[1u32, dim * 3, dim]));
        let qkv_split_meta = (0..3u32)
            .map(|i| {
                Arc::new(Tensor::init_from_cpu(
                    ctx.clone(),
                    &[HeadMoveMeta {
                        seq_len: 1,
                        full_dim: dim * 3,
                        head_dim: dim,
                        head_offset: i * dim,
                    }],
                ))
            })
            .collect();
        let ffnup_meta = Arc::new(Tensor::init_from_cpu(
            ctx.clone(),
            &[1u32, ffn_hidden_dim, dim],
        ));
        let ffndown_meta = Arc::new(Tensor::init_from_cpu(
            ctx.clone(),
            &[1u32, dim, ffn_hidden_dim],
        ));
        let emb_meta = Arc::new(Tensor::init_from_cpu(ctx.clone(), &[vocab_size, dim, 1u32]));
        let lm_meta = Arc::new(Tensor::init_from_cpu(ctx.clone(), &[1u32, vocab_size, dim]));
        let q_move_meta = (0..num_heads)
            .map(|h| {
                Arc::new(Tensor::init_from_cpu(
                    ctx.clone(),
                    &[HeadMoveMeta {
                        seq_len: 1,
                        full_dim: dim,
                        head_dim,
                        head_offset: h * head_dim,
                    }],
                ))
            })
            .collect();

        let rope_meta = Arc::new(Tensor::init_from_cpu(
            ctx.clone(),
            &[1u32, dim, head_dim, 0],
        ));
        let cache_write_meta = Arc::new(Tensor::init_from_cpu(ctx.clone(), &[1u32, dim, 0]));
        let qkt_meta = Arc::new(Tensor::init_from_cpu(ctx.clone(), &[1u32, 1u32, head_dim]));
        let softmax_meta = Arc::new(Tensor::init_from_cpu(
            ctx.clone(),
            &[SoftmaxRectMeta {
                num_rows: 1,
                width: 1,
                scale: 1.0,
            }],
        ));
        let av_meta = Arc::new(Tensor::init_from_cpu(ctx.clone(), &[1u32, head_dim, 1u32]));
        let cache_move_meta = (0..num_heads)
            .map(|h| {
                Arc::new(Tensor::init_from_cpu(
                    ctx.clone(),
                    &[HeadMoveMeta {
                        seq_len: 1,
                        full_dim: dim,
                        head_dim,
                        head_offset: h * head_dim,
                    }],
                ))
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

    fn update_for_step(&self, pos: u32, dim: u32, head_dim: u32, num_heads: u32, scale: f32) {
        let attn_len = pos + 1;
        self.rope_meta.copy_from_cpu(&[1u32, dim, head_dim, pos]);
        self.cache_write_meta.copy_from_cpu(&[1u32, dim, pos]);
        self.qkt_meta.copy_from_cpu(&[1u32, attn_len, head_dim]);
        self.softmax_meta.copy_from_cpu(&[SoftmaxRectMeta {
            num_rows: 1,
            width: attn_len,
            scale,
        }]);
        self.av_meta.copy_from_cpu(&[1u32, head_dim, attn_len]);
        for h in 0..num_heads as usize {
            self.cache_move_meta[h].copy_from_cpu(&[HeadMoveMeta {
                seq_len: attn_len,
                full_dim: dim,
                head_dim,
                head_offset: h as u32 * head_dim,
            }]);
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
    dim: u32,
    num_heads: u32,
    head_dim: u32,
) -> Arc<Tensor<B>> {
    let zeros_dim = vec![0.0 as Real; (prompt_len * dim) as usize];
    let ffn_hidden_dim = dim * 4;

    let norm1_out = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros_dim));
    RMSNorm::forward_nodes(
        graph,
        &layer.norm_1.weight,
        hidden_in,
        &norm1_out,
        prompt_len,
        dim,
        1e-5,
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

    let rope_meta = Arc::new(Tensor::init_from_cpu(
        ctx.clone(),
        &[prompt_len, dim, head_dim],
    ));
    add_rope_node(graph, "RoPE", &q_buf, &rope_meta, dim, prompt_len);
    add_rope_node(graph, "RoPE", &k_buf, &rope_meta, dim, prompt_len);

    let cache_write_meta = Arc::new(Tensor::init_from_cpu(ctx.clone(), &[prompt_len, dim, 0]));
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

    let outproj_meta = Arc::new(Tensor::init_from_cpu(ctx.clone(), &[prompt_len, dim, dim]));
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
        1e-5,
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
    let ffndown_meta = Arc::new(Tensor::init_from_cpu(
        ctx.clone(),
        &[prompt_len, dim, ffn_hidden_dim],
    ));
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

#[allow(clippy::too_many_arguments)]
fn build_decode_layer<B: Backend>(
    graph: &mut ComputeGraph<B>,
    layer: &TransformerBlock<B>,
    scratch: &DecodeScratch<B>,
    cache_k: &Arc<Tensor<B>>,
    cache_v: &Arc<Tensor<B>>,
    pos: u32,
    dim: u32,
    num_heads: u32,
    head_dim: u32,
    ffn_hidden_dim: u32,
) {
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
    dim: u32,
    num_heads: u32,
    head_dim: u32,
    ffn_hidden_dim: u32,
    max_context_len: u32,
    cache: Option<Cache<B>>,
    scratch: DecodeScratch<B>,
    input_token_buf: Arc<Tensor<B>>,
}

impl<B: Backend> InferenceSession<B> {
    pub fn new(
        ctx: Arc<B>,
        model: Arc<AkashaModel<B>>,
        dim: u32,
        num_heads: u32,
        max_context_len: u32,
    ) -> Self {
        assert_eq!(dim % num_heads, 0, "dim must be divisible by num_heads");
        let head_dim = dim / num_heads;
        let ffn_hidden_dim = dim * 4;
        let vocab_size = model.vocab_size;

        let scratch = DecodeScratch::new(
            ctx.clone(),
            dim,
            num_heads,
            head_dim,
            max_context_len,
            ffn_hidden_dim,
            vocab_size,
        );
        let input_token_buf = Arc::new(Tensor::init_from_cpu(ctx.clone(), &[0u32]));

        Self {
            ctx,
            model,
            dim,
            num_heads,
            head_dim,
            ffn_hidden_dim,
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
        assert_eq!(new.dim, self.dim, "Cache/model dim mismatch");
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
        let dim = self.dim;
        let num_heads = self.num_heads;
        let head_dim = self.head_dim;
        let vocab_size = model.vocab_size;

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
                    dim,
                    num_heads,
                    head_dim,
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
            1e-5,
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
        let dim = self.dim;
        let num_heads = self.num_heads;
        let head_dim = self.head_dim;
        let ffn_hidden_dim = self.ffn_hidden_dim;
        let vocab_size = model.vocab_size;
        let scale = 1.0 / (head_dim as f32).sqrt();

        self.scratch
            .update_for_step(pos, dim, head_dim, num_heads, scale);

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
                    dim,
                    num_heads,
                    head_dim,
                    ffn_hidden_dim,
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
                self.dim,
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

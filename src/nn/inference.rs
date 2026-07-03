use super::cache::Cache;
use super::inference_graphs::{DecodeScratch, build_decode_layer, build_prefill_layer};
use super::ops;
use super::ops::meta::{EmbeddingMeta, MatMulMeta, NormMeta};
use super::sampling::sample_token;
use super::weights::ModelWeights;
use crate::Real;
use crate::config::ModelConfig;
use crate::tokenizer::AkashaTokenizer;
use std::sync::Arc;
use wilupgu::{Backend, ComputeGraph, Tensor};

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
        ops::embedding(
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
        ops::rmsnorm(
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
        ops::matmul(
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

        ops::embedding_with(
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

        ops::rmsnorm_with(
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

        ops::matmul_with(
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

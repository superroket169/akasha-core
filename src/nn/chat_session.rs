use super::model::Model;
use super::sampling;
use crate::AkashaError;
use wilupgu::Backend;

pub struct ChatSession<B: Backend> {
    model: Model<B>,
    pos: u32,
    seen: Vec<u32>,
    last_token: Option<u32>,
    eos: u32,
    finished: bool,
}

impl<B: Backend> ChatSession<B> {
    pub fn new(model: Model<B>) -> Self {
        let eos = model.weights().cfg.eos_token;
        Self {
            model,
            pos: 0,
            seen: Vec::new(),
            last_token: None,
            eos,
            finished: false,
        }
    }

    pub fn feed_prompt(
        &mut self,
        prompt: &[u32],
        temperature: f32,
        top_k: usize,
        top_p: f32,
        repetition_penalty: f32,
    ) -> Result<u32, AkashaError> {
        if self.pos != 0 {
            return Err(AkashaError::CacheNotEmpty { cur_len: self.pos });
        }
        let logits = self.model.prefill_logits(prompt)?;
        self.seen = prompt.to_vec();
        self.pos = prompt.len() as u32;

        let next = sampling::sample_token(
            &logits,
            temperature,
            top_k,
            top_p,
            &self.seen,
            repetition_penalty,
        );
        self.seen.push(next);
        self.last_token = Some(next);
        self.finished = next == self.eos;
        Ok(next)
    }

    pub fn next(
        &mut self,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        repetition_penalty: f32,
    ) -> Result<Option<u32>, AkashaError> {
        if self.finished || self.pos >= self.model.max_context_len() {
            return Ok(None);
        }
        let prev = self
            .last_token
            .expect("ChatSession::next called before feed_prompt");

        let logits = self.model.decode_step_logits(prev, self.pos)?;
        self.pos += 1;

        let next = sampling::sample_token(
            &logits,
            temperature,
            top_k,
            top_p,
            &self.seen,
            repetition_penalty,
        );
        self.seen.push(next);
        self.last_token = Some(next);
        self.finished = next == self.eos;
        Ok(Some(next))
    }

    /// Clears the KV cache and this session's bookkeeping for a fresh prompt.
    pub fn reset(&mut self) {
        self.model.reset_cache();
        self.pos = 0;
        self.seen.clear();
        self.last_token = None;
        self.finished = false;
    }

    pub fn is_finished(&self) -> bool {
        self.finished
    }
}

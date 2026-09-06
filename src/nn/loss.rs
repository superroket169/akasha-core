//! Loss + `AnyLoss`
//! Stays outside the Tape/Op system on purpose -- V1 aliasing
//! (logits -> probs -> grad_logits, same buffer) isn't a plain N-in/M-out op.

use super::kernels;
use super::kernels::GraphBuilder;
use super::kernels::Train;
use super::kernels::meta::CrossEntropyMeta;
use crate::Real;
use std::sync::Arc;
use wilupgu::{Backend, Tensor};

pub(crate) struct CrossEntropyOp<B: Backend> {
    target_tokens: Arc<Tensor<B>>,
    losses: Arc<Tensor<B>>,
    d_losses: Arc<Tensor<B>>,
    shape: CrossEntropyMeta,
}

impl<B: Backend> CrossEntropyOp<B> {
    pub(crate) fn new(ctx: &Arc<B>, vocab_size: u32, seq_len: u32) -> Self {
        let n = seq_len as usize;
        Self {
            target_tokens: Arc::new(Tensor::init_from_cpu(ctx.clone(), &vec![0u32; n])),
            losses: Arc::new(Tensor::init_from_cpu(ctx.clone(), &vec![0.0 as Real; n])),
            d_losses: Arc::new(Tensor::init_from_cpu(
                ctx.clone(),
                &vec![1.0 as Real / seq_len as Real; n],
            )),
            shape: CrossEntropyMeta {
                vocab_size,
                num_rows: seq_len,
            },
        }
    }

    pub(crate) fn set_grad_scale(&self, scale: Real) {
        self.d_losses
            .copy_from_cpu(&vec![scale; self.shape.num_rows as usize]);
    }

    pub(crate) fn loss(&self) -> Real {
        let losses: Vec<Real> = self.losses.to_cpu();
        losses.iter().sum::<Real>() / losses.len() as Real
    }

    /// Call before executing the graph each step -- separate from
    /// `forward()`, which only emits the node once at construction.
    pub(crate) fn set_targets(&self, targets: &[u32]) {
        self.target_tokens.copy_from_cpu(targets);
    }

    /// `logits` becomes probs in place.
    pub(crate) fn forward(&self, gb: &mut GraphBuilder<'_, B, Train>, logits: &Arc<Tensor<B>>) {
        kernels::cross_entropy(gb, logits, &self.target_tokens, &self.losses, self.shape);
    }

    /// `logits` (currently holding probs) becomes grad_logits in place.
    pub(crate) fn backward(&self, gb: &mut GraphBuilder<'_, B, Train>, logits: &Arc<Tensor<B>>) {
        kernels::cross_entropy_bwd(gb, logits, &self.target_tokens, &self.d_losses, self.shape);
    }
}

pub(crate) enum AnyLoss<B: Backend> {
    CrossEntropy(CrossEntropyOp<B>),
}

impl<B: Backend> AnyLoss<B> {
    pub(crate) fn set_targets(&self, targets: &[u32]) {
        match self {
            AnyLoss::CrossEntropy(op) => op.set_targets(targets),
        }
    }

    pub(crate) fn forward(&self, gb: &mut GraphBuilder<'_, B, Train>, logits: &Arc<Tensor<B>>) {
        match self {
            AnyLoss::CrossEntropy(op) => op.forward(gb, logits),
        }
    }

    pub(crate) fn backward(&self, gb: &mut GraphBuilder<'_, B, Train>, logits: &Arc<Tensor<B>>) {
        match self {
            AnyLoss::CrossEntropy(op) => op.backward(gb, logits),
        }
    }

    pub(crate) fn loss(&self) -> Real {
        match self {
            AnyLoss::CrossEntropy(op) => op.loss(),
        }
    }

    pub(crate) fn set_grad_scale(&self, scale: Real) {
        match self {
            AnyLoss::CrossEntropy(op) => op.set_grad_scale(scale),
        }
    }
}

use super::ops;
use super::ops::meta::CrossEntropyMeta;
use super::traits::Layer;
use crate::Real;
use std::sync::Arc;
use wilupgu::{Backend, ComputeGraph, Tensor};

pub struct CrossEntropy<B: Backend> {
    pub seq_len: u32,
    pub target_tokens: Arc<Tensor<B>>,
    pub probs: Arc<Tensor<B>>,
    pub losses: Arc<Tensor<B>>,
    pub d_losses: Arc<Tensor<B>>,
    pub grad_logits: Arc<Tensor<B>>,
    pub forward_graph: ComputeGraph<B>,
    pub backward_graph: ComputeGraph<B>,
}

impl<B: Backend> CrossEntropy<B> {
    pub fn new(
        ctx: Arc<B>,
        vocab_size: u32,
        seq_len: u32,
        logits: &Arc<Tensor<B>>,
        grad_logits: &Arc<Tensor<B>>,
    ) -> Self {
        let target_tokens = Arc::new(Tensor::init_from_cpu(
            ctx.clone(),
            &vec![0u32; seq_len as usize],
        ));
        let probs = Arc::new(Tensor::init_from_cpu(
            ctx.clone(),
            &vec![0.0 as Real; (seq_len * vocab_size) as usize],
        ));
        let losses = Arc::new(Tensor::init_from_cpu(
            ctx.clone(),
            &vec![0.0 as Real; seq_len as usize],
        ));
        let d_losses = Arc::new(Tensor::init_from_cpu(
            ctx.clone(),
            &vec![1.0 as Real / seq_len as Real; seq_len as usize],
        ));
        let grad_logits = grad_logits.clone();

        let shape = CrossEntropyMeta {
            vocab_size,
            num_rows: seq_len,
        };

        let mut forward_graph = ComputeGraph::new(ctx.clone());
        ops::cross_entropy(
            &mut forward_graph,
            logits,
            &target_tokens,
            &probs,
            &losses,
            shape,
        );

        let mut backward_graph = ComputeGraph::new(ctx.clone());
        ops::cross_entropy_bwd(
            &mut backward_graph,
            &probs,
            &target_tokens,
            &d_losses,
            &grad_logits,
            shape,
        );

        Self {
            seq_len,
            target_tokens,
            probs,
            losses,
            d_losses,
            grad_logits,
            forward_graph,
            backward_graph,
        }
    }

    pub fn set_grad_scale(&self, scale: Real) {
        self.d_losses
            .copy_from_cpu(&vec![scale; self.seq_len as usize]);
    }

    pub fn loss(&self) -> Real {
        let losses: Vec<Real> = self.losses.to_cpu();
        losses.iter().sum::<Real>() / losses.len() as Real
    }
}

impl<B: Backend> Layer for CrossEntropy<B> {
    fn forward(&self) {
        self.forward_graph.execute();
    }
    fn backward(&self) {
        self.backward_graph.execute();
    }
}

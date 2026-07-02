use super::ops::meta::{CrossEntropyMeta, KernelMeta};
use super::traits::Layer;
use crate::Real;
use std::sync::Arc;
use wilupgu::{Backend, Binding, ComputeGraph, Tensor, TensorMode};

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

        let t_meta = CrossEntropyMeta {
            vocab_size,
            num_rows: seq_len,
        }
        .upload(&ctx);

        let dispatch_x = (seq_len + 255) / 256;

        let mut forward_graph = ComputeGraph::new(ctx.clone());
        forward_graph.add_node(
            "CrossEntropy",
            &[
                Binding::new(0, &logits.buffer, TensorMode::Input),
                Binding::new(1, &target_tokens.buffer, TensorMode::Input),
                Binding::new(2, &probs.buffer, TensorMode::Output),
                Binding::new(3, &losses.buffer, TensorMode::Output),
                Binding::new(4, &t_meta.buffer, TensorMode::Meta),
            ],
            [dispatch_x, 1, 1],
        );

        let mut backward_graph = ComputeGraph::new(ctx.clone());
        backward_graph.add_node(
            "CrossEntropyBwd",
            &[
                Binding::new(0, &probs.buffer, TensorMode::Input),
                Binding::new(1, &target_tokens.buffer, TensorMode::Input),
                Binding::new(2, &d_losses.buffer, TensorMode::Input),
                Binding::new(3, &grad_logits.buffer, TensorMode::Output),
                Binding::new(4, &t_meta.buffer, TensorMode::Meta),
            ],
            [dispatch_x, 1, 1],
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

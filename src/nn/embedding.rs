use super::ops;
use super::ops::meta::EmbeddingMeta;
use super::traits::Layer;
use crate::Real;
use std::sync::Arc;
use wilupgu::{Backend, ComputeGraph, Tensor};

pub struct Embedding<B: Backend> {
    pub table: Arc<Tensor<B>>,
    pub grad_table: Arc<Tensor<B>>,
    pub out_buffer: Arc<Tensor<B>>,
    pub forward_graph: ComputeGraph<B>,
    pub backward_graph: ComputeGraph<B>,
}

impl<B: Backend> Embedding<B> {
    pub fn new(
        ctx: Arc<B>,
        vocab_size: u32,
        dim: u32,
        seq_len: u32,
        table: &Arc<Tensor<B>>,
        tokens_buffer: &Arc<Tensor<B>>,
        grad_output: &Arc<Tensor<B>>,
    ) -> Self {
        assert_eq!(
            table.size,
            (vocab_size * dim) as u64 * std::mem::size_of::<Real>() as u64,
            "Dict size mismatch!"
        );

        let shape = EmbeddingMeta {
            vocab_size,
            dim,
            seq_len,
        };

        let table = table.clone();
        let zero_table = vec![0.0 as Real; (vocab_size * dim) as usize];
        let grad_table = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zero_table));

        let out_size = (seq_len * dim) as usize;
        let zero_out = vec![0.0 as Real; out_size];
        let out_buffer = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zero_out));

        let mut forward_graph = ComputeGraph::new(ctx.clone());
        ops::embedding(
            &mut forward_graph,
            tokens_buffer,
            &table,
            &out_buffer,
            shape,
        );

        let mut backward_graph = ComputeGraph::new(ctx.clone());
        ops::embedding_bwd(
            &mut backward_graph,
            tokens_buffer,
            grad_output,
            &grad_table,
            shape,
        );

        Self {
            table,
            grad_table,
            out_buffer,
            forward_graph,
            backward_graph,
        }
    }
}

impl<B: Backend> Layer for Embedding<B> {
    fn forward(&self) {
        self.forward_graph.execute();
    }
    fn backward(&self) {
        self.backward_graph.execute();
    }
}

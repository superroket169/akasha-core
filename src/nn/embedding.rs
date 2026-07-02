use super::traits::Layer;
use crate::Real;
use std::sync::Arc;
use wilupgu::{Backend, Binding, ComputeGraph, Tensor, TensorMode};

pub struct Embedding<B: Backend> {
    pub table: Arc<Tensor<B>>,
    pub grad_table: Arc<Tensor<B>>,
    pub out_buffer: Arc<Tensor<B>>,
    pub forward_graph: ComputeGraph<B>,
    pub backward_graph: ComputeGraph<B>,
}

impl<B: Backend> Embedding<B> {
    pub(crate) fn forward_nodes(
        graph: &mut ComputeGraph<B>,
        table: &Arc<Tensor<B>>,
        tokens: &Arc<Tensor<B>>,
        output: &Arc<Tensor<B>>,
        vocab_size: u32,
        dim: u32,
        seq_len: u32,
    ) {
        let ctx = tokens.ctx.clone();
        let meta_data = vec![vocab_size, dim, seq_len];
        let t_meta = Arc::new(Tensor::init_from_cpu(ctx, &meta_data));

        graph.add_node(
            "Embedding",
            &[
                Binding::new(0, &tokens.buffer, TensorMode::Input),
                Binding::new(1, &table.buffer, TensorMode::Input),
                Binding::new(2, &output.buffer, TensorMode::Output),
                Binding::new(3, &t_meta.buffer, TensorMode::Meta),
            ],
            [(dim + 255) / 256, seq_len, 1],
        );
    }

    pub fn new(
        ctx: Arc<B>,
        vocab_size: u32,
        dim: u32,
        seq_len: u32,
        table_data: &[Real],
        tokens_buffer: &Arc<Tensor<B>>,
        grad_output: &Arc<Tensor<B>>,
    ) -> Self {
        assert_eq!(
            table_data.len(),
            (vocab_size * dim) as usize,
            "Dict size mismatch!"
        );

        let table = Arc::new(Tensor::init_from_cpu(ctx.clone(), table_data));
        let zero_table = vec![0.0 as Real; (vocab_size * dim) as usize];
        let grad_table = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zero_table));

        let meta_data = vec![vocab_size, dim, seq_len];
        let t_meta = Arc::new(Tensor::init_from_cpu(ctx.clone(), &meta_data));

        let out_size = (seq_len * dim) as usize;
        let zero_out = vec![0.0 as Real; out_size];
        let out_buffer = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zero_out));

        let mut forward_graph = ComputeGraph::new(ctx.clone());
        Self::forward_nodes(
            &mut forward_graph,
            &table,
            tokens_buffer,
            &out_buffer,
            vocab_size,
            dim,
            seq_len,
        );

        let mut backward_graph = ComputeGraph::new(ctx.clone());
        backward_graph.add_node(
            "EmbeddingBwd",
            &[
                Binding::new(0, &tokens_buffer.buffer, TensorMode::Input),
                Binding::new(1, &grad_output.buffer, TensorMode::Input),
                Binding::new(2, &grad_table.buffer, TensorMode::Output),
                Binding::new(3, &t_meta.buffer, TensorMode::Meta),
            ],
            [(dim + 255) / 256, seq_len, 1],
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

use super::traits::Layer;
use crate::Real;
use std::sync::Arc;
use wilupgu::context::WgpuContext;
use wilupgu::graph::{ComputeGraph, TensorBind, TensorMode};
use wilupgu::nn::shaders::BuiltInShader;
use wilupgu::tensor::Tensor;

pub struct Embedding {
    pub table: Arc<Tensor>,
    pub grad_table: Arc<Tensor>,
    pub out_buffer: Arc<Tensor>,
    pub forward_graph: ComputeGraph,
    pub backward_graph: ComputeGraph,
}

impl Embedding {
    pub fn new(
        ctx: Arc<WgpuContext>,
        vocab_size: u32,
        dim: u32,
        seq_len: u32,
        table_data: &[Real],
        tokens_buffer: &Arc<Tensor>,
        grad_output: &Arc<Tensor>,
    ) -> Self {
        assert_eq!(
            table_data.len(),
            (vocab_size * dim) as usize,
            "Dict size mismatch!"
        );

        let table = Arc::new(Tensor::init_from_cpu(ctx.clone(), table_data));
        let zero_table = vec![0.0 as Real; (vocab_size * dim) as usize];
        let grad_table = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zero_table));

        let meta_data = vec![vocab_size, dim];
        let t_meta = Arc::new(Tensor::init_from_cpu(ctx.clone(), &meta_data));

        let out_size = (seq_len * dim) as usize;
        let zero_out = vec![0.0 as Real; out_size];
        let out_buffer = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zero_out));

        let shader = BuiltInShader::Embedding.get_def();
        let total_threads = seq_len * dim;

        // --- FORWARD ---
        let mut forward_graph = ComputeGraph::new(ctx.clone());
        forward_graph.add_node(
            &shader,
            &[
                TensorBind {
                    binding: 0,
                    tensor: tokens_buffer,
                    mode: TensorMode::Input,
                },
                TensorBind {
                    binding: 1,
                    tensor: &table,
                    mode: TensorMode::Input,
                },
                TensorBind {
                    binding: 2,
                    tensor: &out_buffer,
                    mode: TensorMode::Output,
                },
                TensorBind {
                    binding: 3,
                    tensor: &t_meta,
                    mode: TensorMode::Meta,
                },
            ],
            [(total_threads + 255) / 256, 1, 1],
        );

        // --- BACKWARD ---
        let mut backward_graph = ComputeGraph::new(ctx.clone());
        backward_graph.add_node(
            &shader,
            &[
                TensorBind {
                    binding: 0,
                    tensor: tokens_buffer,
                    mode: TensorMode::Input,
                },
                TensorBind {
                    binding: 1,
                    tensor: grad_output,
                    mode: TensorMode::Input,
                },
                TensorBind {
                    binding: 2,
                    tensor: &grad_table,
                    mode: TensorMode::Output,
                },
                TensorBind {
                    binding: 3,
                    tensor: &t_meta,
                    mode: TensorMode::Meta,
                },
            ],
            [(seq_len * dim + 255) / 256, 1, 1],
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

impl Layer for Embedding {
    fn forward(&self) {
        self.forward_graph.execute();
    }
    fn backward(&self) {
        self.backward_graph.execute();
    }
}

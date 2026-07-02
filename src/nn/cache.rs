use crate::Real;
use std::sync::Arc;
use wilupgu::{Backend, Tensor};

pub struct Cache<B: Backend> {
    pub num_layers: usize,
    pub dim: u32,
    pub max_context_len: u32,
    pub cur_len: u32,
    pub k: Vec<Arc<Tensor<B>>>,
    pub v: Vec<Arc<Tensor<B>>>,
}

impl<B: Backend> Cache<B> {
    pub fn new(ctx: Arc<B>, num_layers: usize, dim: u32, max_context_len: u32) -> Self {
        let zeros = vec![0.0 as Real; (max_context_len * dim) as usize];
        let k = (0..num_layers)
            .map(|_| Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros)))
            .collect();
        let v = (0..num_layers)
            .map(|_| Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros)))
            .collect();

        Self {
            num_layers,
            dim,
            max_context_len,
            cur_len: 0,
            k,
            v,
        }
    }

    pub fn reset(&mut self) {
        self.cur_len = 0;
    }
}

use crate::Real;
use std::cell::Cell;
use std::sync::Arc;
use wilupgu::context::WgpuContext;
use wilupgu::graph::{ComputeGraph, TensorBind, TensorMode};
use wilupgu::nn::shaders::BuiltInShader;
use wilupgu::tensor::Tensor;

const DEFAULT_EPS: Real = 1e-8;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ParamMeta {
    size: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct StepConfig {
    step: u32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
}

fn elem_count(t: &Tensor) -> usize {
    (t.size / std::mem::size_of::<Real>() as u64) as usize
}

pub struct AdamW {
    cfg: Arc<Tensor>,
    eps: Real,
    step_count: Cell<u32>,
    graph: ComputeGraph,
    pub moments: Vec<(Arc<Tensor>, Arc<Tensor>)>, // (weight, grad)
}

impl AdamW {
    pub fn new(ctx: Arc<WgpuContext>, params: &[(Arc<Tensor>, Arc<Tensor>)]) -> Self {
        let cfg = Arc::new(Tensor::init_from_cpu(
            ctx.clone(),
            &[StepConfig {
                step: 0,
                lr: 0.0,
                beta1: 0.0,
                beta2: 0.0,
                eps: DEFAULT_EPS,
                weight_decay: 0.0,
            }],
        ));

        let def = BuiltInShader::AdamW.get_def();
        let mut graph = ComputeGraph::new(ctx.clone());
        let mut moments = Vec::with_capacity(params.len());

        for (weight, grad) in params {
            let len = elem_count(weight);
            assert_eq!(
                len,
                elem_count(grad),
                "AdamW: weight/grad tensor size mismatch"
            );

            let zeros = vec![0.0 as Real; len];
            let m = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros));
            let v = Arc::new(Tensor::init_from_cpu(ctx.clone(), &zeros));
            let param_meta = Arc::new(Tensor::init_from_cpu(
                ctx.clone(),
                &[ParamMeta { size: len as u32 }],
            ));

            graph.add_node(
                &def,
                &[
                    TensorBind {
                        binding: 0,
                        tensor: weight,
                        mode: TensorMode::InOut,
                    },
                    TensorBind {
                        binding: 1,
                        tensor: grad,
                        mode: TensorMode::Input,
                    },
                    TensorBind {
                        binding: 2,
                        tensor: &m,
                        mode: TensorMode::InOut,
                    },
                    TensorBind {
                        binding: 3,
                        tensor: &v,
                        mode: TensorMode::InOut,
                    },
                    TensorBind {
                        binding: 4,
                        tensor: &param_meta,
                        mode: TensorMode::Meta,
                    },
                    TensorBind {
                        binding: 5,
                        tensor: &cfg,
                        mode: TensorMode::Meta,
                    },
                ],
                [((len as u32) + 255) / 256, 1, 1],
            );

            moments.push((m, v));
        }

        Self {
            cfg,
            eps: DEFAULT_EPS,
            step_count: Cell::new(0),
            graph,
            moments,
        }
    }

    pub fn step(&self, lr: Real, beta1: Real, beta2: Real, weight_decay: Real) {
        let t = self.step_count.get() + 1;
        self.step_count.set(t);

        self.cfg.copy_from_cpu(&[StepConfig {
            step: t,
            lr,
            beta1,
            beta2,
            eps: self.eps,
            weight_decay,
        }]);

        self.graph.execute();
    }
}

use crate::Real;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use wilupgu::builtin;
use wilupgu::{Backend, Binding, ComputeGraph, Tensor, TensorMode};

const DEFAULT_EPS: Real = 1e-8;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ParamMeta {
    size: u32,
    groups_x: u32,
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

fn elem_count<B: Backend>(t: &Tensor<B>) -> usize {
    (t.size / std::mem::size_of::<Real>() as u64) as usize
}

pub struct AdamW<B: Backend> {
    cfg: Arc<Tensor<B>>,
    eps: Real,
    step_count: AtomicU32,
    graph: ComputeGraph<B>,
    pub moments: Vec<(Arc<Tensor<B>>, Arc<Tensor<B>>)>, // (weight, grad)
}

impl<B: Backend> AdamW<B> {
    pub fn new(ctx: Arc<B>, params: &[(Arc<Tensor<B>>, Arc<Tensor<B>>)]) -> Self {
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

            let total_groups = (((len as u32) + 255) / 256).max(1);
            let groups_x = total_groups.min(8192);
            let groups_y = (total_groups + groups_x - 1) / groups_x;

            let param_meta = Arc::new(Tensor::init_from_cpu(
                ctx.clone(),
                &[ParamMeta {
                    size: len as u32,
                    groups_x,
                }],
            ));

            graph.add_node(
                &builtin::ADAMW,
                &[
                    Binding::new(0, &weight.buffer, TensorMode::InOut),
                    Binding::new(1, &grad.buffer, TensorMode::Input),
                    Binding::new(2, &m.buffer, TensorMode::InOut),
                    Binding::new(3, &v.buffer, TensorMode::InOut),
                    Binding::new(4, &param_meta.buffer, TensorMode::Meta),
                    Binding::new(5, &cfg.buffer, TensorMode::Meta),
                ],
                [groups_x, groups_y, 1],
            );

            moments.push((m, v));
        }

        Self {
            cfg,
            eps: DEFAULT_EPS,
            step_count: AtomicU32::new(0),
            graph,
            moments,
        }
    }

    pub fn step(&self, lr: Real, beta1: Real, beta2: Real, weight_decay: Real) {
        let t = self.step_count.fetch_add(1, Ordering::Relaxed) + 1;

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

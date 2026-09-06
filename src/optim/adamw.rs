use crate::Real;
use std::sync::Arc;
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
struct ScheduleState {
    step: u32,
    lr: f32,
}

#[derive(Clone, Copy)]
pub struct AdamWSchedule {
    pub lr_max: Real,
    pub lr_min: Real,
    pub warmup_steps: u32,
    pub max_steps: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ScheduleConfig {
    lr_max: f32,
    lr_min: f32,
    warmup_steps: u32,
    max_steps: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ConstCfg {
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
}

fn elem_count<B: Backend>(t: &Tensor<B>) -> usize {
    (t.size / std::mem::size_of::<Real>() as u64) as usize
}

pub struct AdamW<B: Backend> {
    graph: ComputeGraph<B>,
    pub moments: Vec<(Arc<Tensor<B>>, Arc<Tensor<B>>)>,
    schedule_state: Arc<Tensor<B>>,
}

impl<B: Backend> AdamW<B> {
    /// `params` entries are (weight, grad, decay): weight decay is applied
    /// only where the flag is true (matmul weights yes, norm gains and the embedding table no).
    pub fn new(
        ctx: Arc<B>,
        params: &[(Arc<Tensor<B>>, Arc<Tensor<B>>, bool)],
        schedule: AdamWSchedule,
        beta1: Real,
        beta2: Real,
        weight_decay: Real,
    ) -> Self {
        let schedule_state = Arc::new(Tensor::init_from_cpu(
            ctx.clone(),
            &[ScheduleState { step: 0, lr: 0.0 }],
        ));
        let schedule_cfg = Arc::new(Tensor::init_from_cpu(
            ctx.clone(),
            &[ScheduleConfig {
                lr_max: schedule.lr_max,
                lr_min: schedule.lr_min,
                warmup_steps: schedule.warmup_steps,
                max_steps: schedule.max_steps,
            }],
        ));
        let const_cfg = Arc::new(Tensor::init_from_cpu(
            ctx.clone(),
            &[ConstCfg {
                beta1,
                beta2,
                eps: DEFAULT_EPS,
                weight_decay,
            }],
        ));
        let const_cfg_no_decay = Arc::new(Tensor::init_from_cpu(
            ctx.clone(),
            &[ConstCfg {
                beta1,
                beta2,
                eps: DEFAULT_EPS,
                weight_decay: 0.0,
            }],
        ));

        let mut graph = ComputeGraph::new(ctx.clone());

        graph.add_node(
            &builtin::ADAMW_SCHEDULE,
            &[
                Binding::new(0, &schedule_state.buffer, TensorMode::InOut),
                Binding::new(1, &schedule_cfg.buffer, TensorMode::Meta),
            ],
            [1, 1, 1],
        );

        let mut moments = Vec::with_capacity(params.len());

        for (weight, grad, decay) in params {
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
                    Binding::new(5, &schedule_state.buffer, TensorMode::Input),
                    Binding::new(
                        6,
                        if *decay {
                            &const_cfg.buffer
                        } else {
                            &const_cfg_no_decay.buffer
                        },
                        TensorMode::Meta,
                    ),
                ],
                [groups_x, groups_y, 1],
            );

            moments.push((m, v));
        }

        Self {
            graph,
            moments,
            schedule_state,
        }
    }

    pub fn step(&self) {
        self.graph.execute_captured();
    }

    pub fn current_schedule(&self) -> (u32, Real) {
        let raw: Vec<u32> = self.schedule_state.to_cpu();
        (raw[0], f32::from_bits(raw[1]))
    }

    /// Restores a V3 checkpoint's optimizer state: m/v moments (in param order — the format contract)
    /// and the schedule step counter
    /// The lr field is left at 0; the schedule kernel recomputes it from the step
    /// counter before the next AdamW node runs.
    pub fn load_state(&self, moments: &[(Vec<Real>, Vec<Real>)], schedule_step: u32) {
        assert_eq!(
            moments.len(),
            self.moments.len(),
            "AdamW::load_state: checkpoint moment count doesn't match model"
        );
        for ((m_t, v_t), (m_d, v_d)) in self.moments.iter().zip(moments) {
            m_t.copy_from_cpu(m_d);
            v_t.copy_from_cpu(v_d);
        }
        self.schedule_state.copy_from_cpu(&[ScheduleState {
            step: schedule_step,
            lr: 0.0,
        }]);
    }
}

// test for loss diffs
#[cfg(test)]
#[path = "../tests/adamw_tests.rs"]
mod tests;

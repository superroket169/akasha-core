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
    pub fn new(
        ctx: Arc<B>,
        params: &[(Arc<Tensor<B>>, Arc<Tensor<B>>)],
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
                    Binding::new(5, &schedule_state.buffer, TensorMode::Input),
                    Binding::new(6, &const_cfg.buffer, TensorMode::Meta),
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
}

// test for loss diffs
#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::cosine_lr;
    use wilupgu::WgpuBackend;

    #[test]
    fn schedule_matches_host_formula_and_updates_weights() {
        let ctx = Arc::new(pollster::block_on(WgpuBackend::new()));

        let weight_data = vec![1.0 as Real, 2.0, 3.0, 4.0];
        let grad_data = vec![0.1 as Real, -0.2, 0.3, -0.4];
        let weight = Arc::new(Tensor::init_from_cpu(ctx.clone(), &weight_data));
        let grad = Arc::new(Tensor::init_from_cpu(ctx.clone(), &grad_data));

        let (lr_max, lr_min, warmup_steps, max_steps) = (6e-4 as Real, 6e-5 as Real, 5u32, 50u32);
        let opt = AdamW::new(
            ctx.clone(),
            &[(weight.clone(), grad.clone())],
            AdamWSchedule {
                lr_max,
                lr_min,
                warmup_steps,
                max_steps,
            },
            0.9,
            0.95,
            0.01,
        );

        for expected_step in 1..=15u32 {
            opt.step();
            ctx.synchronize();

            let (step, lr) = opt.current_schedule();
            assert_eq!(step, expected_step, "on-device step counter drifted");

            let expected_lr = cosine_lr(
                expected_step as usize,
                warmup_steps as usize,
                max_steps as usize,
                lr_max,
                lr_min,
            );
            let diff = (lr - expected_lr).abs();
            assert!(
                diff < 1e-6,
                "step {expected_step}: on-device lr {lr} vs. host cosine_lr {expected_lr} (diff {diff})"
            );
        }

        // weight[0]  -> should decrease
        // weight[1]  -> should increase
        let final_weights: Vec<Real> = weight.to_cpu();
        assert!(
            final_weights[0] < weight_data[0],
            "weight[0] should have decreased (positive grad): {} -> {}",
            weight_data[0],
            final_weights[0]
        );
        assert!(
            final_weights[1] > weight_data[1],
            "weight[1] should have increased (negative grad): {} -> {}",
            weight_data[1],
            final_weights[1]
        );
        assert!(
            final_weights.iter().all(|w| w.is_finite()),
            "weights contain non-finite values after {} steps",
            final_weights.len()
        );
    }
}

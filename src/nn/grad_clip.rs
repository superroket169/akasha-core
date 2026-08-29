//! Grad clipping + `AnyGradClip` (Tape/Op tasarımı: ARCHITECTURE.md → Big
//! Refactor). Enum even at one variant -- global-norm clipping has real
//! alternatives (AGC, per-parameter clipping), unlike e.g. `AddOp`.

use super::ops;
use super::tape::elem_count;
use crate::Real;
use std::sync::Arc;
use wilupgu::{Backend, ComputeGraph, Tensor};

pub(crate) struct GlobalNormClip<B: Backend> {
    graph: ComputeGraph<B>,
}

impl<B: Backend> GlobalNormClip<B> {
    pub(crate) fn new(ctx: Arc<B>, grads: &[Arc<Tensor<B>>], max_norm: Real) -> Self {
        let total_partials: u32 = grads
            .iter()
            .map(|g| ops::grad_sumsq_wgs(elem_count(g)))
            .sum();
        let norm_partials = Arc::new(Tensor::init_from_cpu(
            ctx.clone(),
            &vec![0.0 as Real; total_partials as usize],
        ));
        let clip_scale = Arc::new(Tensor::init_from_cpu(ctx.clone(), &[1.0 as Real]));

        let mut graph = ComputeGraph::new(ctx.clone());
        let mut gb = super::ops::GraphBuilder::train(&mut graph);
        let mut out_offset = 0;

        for g in grads {
            let len = elem_count(g);
            ops::grad_sumsq(
                &mut gb,
                g,
                &norm_partials,
                ops::meta::GradSumSqMeta { len, out_offset },
            );
            out_offset += ops::grad_sumsq_wgs(len);
        }

        ops::grad_norm_scale(
            &mut gb,
            &norm_partials,
            &clip_scale,
            ops::meta::GradNormMeta {
                num_partials: total_partials,
                max_norm,
            },
        );
        for g in grads {
            ops::grad_scale(&mut gb, g, &clip_scale, elem_count(g));
        }

        Self { graph }
    }

    pub(crate) fn clip(&self) {
        self.graph.execute_captured();
    }
}

pub(crate) enum AnyGradClip<B: Backend> {
    GlobalNorm(GlobalNormClip<B>),
}

impl<B: Backend> AnyGradClip<B> {
    pub(crate) fn clip(&self) {
        match self {
            AnyGradClip::GlobalNorm(c) => c.clip(),
        }
    }
}

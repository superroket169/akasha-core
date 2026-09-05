pub mod adamw;

pub use adamw::{AdamW, AdamWSchedule};

use crate::Real;
use std::sync::Arc;
use wilupgu::{Backend, Tensor};

/// Same dispatch pattern as `TrainOp` (transformer.rs): a closed enum,
/// matched, no `dyn Trait`. One variant today (AdamW) -- a second
/// optimizer means a second arm here, same as adding a new `Op` kind.
pub enum AnyOptimizer<B: Backend> {
    AdamW(AdamW<B>),
}

impl<B: Backend> AnyOptimizer<B> {
    pub fn step(&self) {
        match self {
            AnyOptimizer::AdamW(o) => o.step(),
        }
    }

    pub fn current_schedule(&self) -> (u32, Real) {
        match self {
            AnyOptimizer::AdamW(o) => o.current_schedule(),
        }
    }

    pub fn moments(&self) -> &[(Arc<Tensor<B>>, Arc<Tensor<B>>)] {
        match self {
            AnyOptimizer::AdamW(o) => &o.moments,
        }
    }

    pub fn load_state(&self, moments: &[(Vec<Real>, Vec<Real>)], schedule_step: u32) {
        match self {
            AnyOptimizer::AdamW(o) => o.load_state(moments, schedule_step),
        }
    }
}

/// The exact shape `Tape::params()` already produces -- construction takes
/// this owned form because it's stored past the borrow that built it.
pub type ParamList<B> = Vec<(Arc<Tensor<B>>, Arc<Tensor<B>>, bool)>;

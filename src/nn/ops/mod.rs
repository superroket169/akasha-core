//! One emitter per kernel; each owns its binding layout and grid formula.
//! `foo(shape)` uploads a constant meta; `foo_with(shape, meta)` takes a
//! caller-owned meta updated between executions. `shape` drives the grid.

pub mod attention;
pub mod cache;
pub mod elementwise;
pub mod embedding;
pub mod head_move;
pub mod loss;
pub mod matmul;
pub mod meta;
pub mod norm;
pub mod rope;

use crate::Real;
use wilupgu::{Backend, Tensor};

/// Host-side zeroing (not a graph node).
pub(crate) fn zero_tensor<B: Backend>(t: &Tensor<B>) {
    let len = (t.size / std::mem::size_of::<Real>() as u64) as usize;
    t.copy_from_cpu(&vec![0.0 as Real; len]);
}

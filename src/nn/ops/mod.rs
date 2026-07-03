//! Two things live here: typed kernel metas (`meta`) and one graph-node
//! emitter per kernel (`emit`, re-exported flat as `ops::*`).
//! `foo(shape)` uploads a constant meta; `foo_with(shape, meta)` takes a
//! caller-owned meta updated between executions. `shape` drives the grid.

mod emit;
pub mod meta;

pub(crate) use emit::*;

use crate::Real;
use wilupgu::{Backend, Tensor};

/// Host-side zeroing (not a graph node).
pub(crate) fn zero_tensor<B: Backend>(t: &Tensor<B>) {
    let len = (t.size / std::mem::size_of::<Real>() as u64) as usize;
    t.copy_from_cpu(&vec![0.0 as Real; len]);
}

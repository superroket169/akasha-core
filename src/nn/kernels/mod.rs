//! Two things live here: typed kernel metas (`meta`) and one graph-node
//! emitter per kernel (`emit`, re-exported flat as `ops::*`).
//! `foo(shape)` uploads a constant meta; `foo_with(shape, meta)` takes a
//! caller-owned meta updated between executions. `shape` drives the grid.
//!
//! Live metas hold on every backend: kernels read the meta buffer at
//! execution time (CUDA generic kernels get it as a device pointer, cuBLAS
//! matmuls dtoh it per dispatch). The one exception is a CAPTURED graph
//! containing matmuls -- cuBLAS dims get frozen at capture, so only
//! constant-meta graphs (training) may use `execute_captured`.

mod emit;
pub mod meta;

pub(crate) use emit::*;

use std::marker::PhantomData;
use wilupgu::ComputeGraph;

pub struct Train;
pub struct Prefill;
pub struct Decode;

pub trait Phase {}
pub trait FwdPhase: Phase {}
pub trait FullSeqPhase: FwdPhase {} // square causal attention / full-seq RoPE
pub trait CachedPhase: FwdPhase {} // reads/writes the KV cache

impl Phase for Train {}
impl FwdPhase for Train {}
impl FullSeqPhase for Train {}

impl Phase for Prefill {}
impl FwdPhase for Prefill {}
impl FullSeqPhase for Prefill {}
impl CachedPhase for Prefill {}

impl Phase for Decode {}
impl FwdPhase for Decode {}
impl CachedPhase for Decode {}

pub(crate) struct GraphBuilder<'g, B: Backend, P: Phase> {
    pub(crate) graph: &'g mut ComputeGraph<B>,
    _phase: PhantomData<P>,
}

impl<'g, B: Backend> GraphBuilder<'g, B, Train> {
    pub(crate) fn train(graph: &'g mut ComputeGraph<B>) -> Self {
        Self {
            graph,
            _phase: PhantomData,
        }
    }
}

impl<'g, B: Backend> GraphBuilder<'g, B, Prefill> {
    pub(crate) fn prefill(graph: &'g mut ComputeGraph<B>) -> Self {
        Self {
            graph,
            _phase: PhantomData,
        }
    }
}

impl<'g, B: Backend> GraphBuilder<'g, B, Decode> {
    pub(crate) fn decode(graph: &'g mut ComputeGraph<B>) -> Self {
        Self {
            graph,
            _phase: PhantomData,
        }
    }
}

use wilupgu::Backend;

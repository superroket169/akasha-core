//! Generic op-graph engine (Tape/Op tasarımı: ARCHITECTURE.md → Big Refactor).

use super::ops::FwdPhase;
use super::ops::GraphBuilder;
use crate::Real;
use std::sync::Arc;
use wilupgu::{Backend, Tensor};

/// `elems` zero-initialized elements. Every concrete op's `new()` allocates
/// its own scratch/gradient buffers through this -- one place to change if
/// that ever needs to route through the pool differently.
pub(crate) fn zeros<B: Backend>(ctx: &Arc<B>, elems: u32) -> Arc<Tensor<B>> {
    Arc::new(Tensor::init_from_cpu(
        ctx.clone(),
        &vec![0.0 as Real; elems as usize],
    ))
}

/// A zeroed buffer the same size as `t` -- for a weight's gradient
/// accumulator, which is always exactly the weight's own shape.
pub(crate) fn zeros_like<B: Backend>(t: &Arc<Tensor<B>>) -> Arc<Tensor<B>> {
    zeros(&t.ctx, t.size as u32 / std::mem::size_of::<Real>() as u32)
}

pub(crate) fn elem_count<B: Backend>(t: &Arc<Tensor<B>>) -> u32 {
    (t.size / std::mem::size_of::<Real>() as u64) as u32
}

/// A node's forward, for one specific phase `P`. Ops shared across phases
/// (Linear, RMSNorm, ...) implement this generically over `P: FwdPhase`;
/// an enum that wraps several such ops for one committed phase (e.g.
/// `TransformerOp` for `Train`) implements it concretely for just that `P`.
pub(crate) trait Forward<B: Backend, P: FwdPhase> {
    fn forward(
        &mut self,
        gb: &mut GraphBuilder<'_, B, P>,
        xs: &[Arc<Tensor<B>>],
    ) -> Vec<Arc<Tensor<B>>>;
}

/// Backward always runs in `Train` -- there is no such thing as a Prefill
/// or Decode backward, inference never computes gradients. Only op-kinds
/// used by training implement this.
pub(crate) trait Backward<B: Backend> {
    /// Emits this op's backward node(s) and, if it owns a weight, also
    /// accumulates into that weight's gradient as a side effect. Returns
    /// the gradient w.r.t. each of this op's inputs, same order as `xs`
    /// was in `forward`.
    fn backward(
        &mut self,
        gb: &mut GraphBuilder<'_, B, super::ops::Train>,
        grad_outputs: &[Arc<Tensor<B>>],
    ) -> Vec<Arc<Tensor<B>>>;

    /// (weight, grad, weight_decay_eligible). None for weightless ops.
    /// Tape::params() preserves push order -- that order IS the
    /// checkpoint/AdamW-moment format contract (weights.params() today),
    /// don't reorder ops after the fact without knowing that.
    fn param(&self) -> Option<(&Arc<Tensor<B>>, &Arc<Tensor<B>>, bool)> {
        None
    }
}

/// Handle to a node already on the tape.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct NodeId(usize);

/// A specific output slot of a node -- almost always slot `0`.
pub(crate) type Out = (NodeId, usize);

struct TapeNode<B: Backend, Node> {
    op: Node,
    inputs: Vec<Out>,
    outputs: Vec<Arc<Tensor<B>>>,
}

/// A DAG of `Node`s. `Node` is `TransformerOp<B>` for training,
/// `PrefillOp<B>`/`DecodeOp<B>` for inference.
pub(crate) struct Tape<B: Backend, Node> {
    nodes: Vec<TapeNode<B, Node>>,
    grads: Vec<Vec<Option<Arc<Tensor<B>>>>>,
}

impl<B: Backend, Node> Tape<B, Node> {
    pub(crate) fn new() -> Self {
        Self {
            nodes: Vec::new(),
            grads: Vec::new(),
        }
    }

    /// Registers a value that already exists (a block's own input, or
    /// another Tape's output) as a leaf node other nodes can reference.
    /// Works for any `Node` -- `Identity` below is the shared leaf marker.
    pub(crate) fn input<P: FwdPhase>(
        &mut self,
        gb: &mut GraphBuilder<'_, B, P>,
        x: Arc<Tensor<B>>,
    ) -> NodeId
    where
        Node: From<Identity<B>> + Forward<B, P>,
    {
        self.push(gb, Node::from(Identity(x)), &[])
    }

    /// Runs `op.forward(inputs' values)`, records it, returns its handle.
    pub(crate) fn push<P: FwdPhase>(
        &mut self,
        gb: &mut GraphBuilder<'_, B, P>,
        mut op: Node,
        inputs: &[Out],
    ) -> NodeId
    where
        Node: Forward<B, P>,
    {
        let input_vals: Vec<Arc<Tensor<B>>> = inputs
            .iter()
            .map(|(id, slot)| self.nodes[id.0].outputs[*slot].clone())
            .collect();
        let outputs = op.forward(gb, &input_vals);
        self.nodes.push(TapeNode {
            op,
            inputs: inputs.to_vec(),
            outputs,
        });
        NodeId(self.nodes.len() - 1)
    }

    /// Slot 0 of `id`'s output(s) -- the common case.
    pub(crate) fn output(&self, id: NodeId) -> Arc<Tensor<B>> {
        self.nodes[id.0].outputs[0].clone()
    }

    pub(crate) fn output_slot(&self, out: Out) -> Arc<Tensor<B>> {
        self.nodes[out.0.0].outputs[out.1].clone()
    }
}

/// Only compiles for a `Node` that actually has backward math -- calling
/// `.backward()` on a `Tape<B, PrefillOp<B>>` is a compile error, not a
/// runtime one, because `PrefillOp` never implements `Backward`.
impl<B: Backend, Node: Backward<B>> Tape<B, Node> {
    /// Seeds `output`'s gradient, walks every node in reverse, and
    /// accumulates (`+=`) fan-in. Read a leaf's total via `grad_of` after.
    pub(crate) fn backward(
        &mut self,
        gb: &mut GraphBuilder<'_, B, super::ops::Train>,
        output: Out,
        grad_output: &Arc<Tensor<B>>,
    ) {
        let mut grads: Vec<Vec<Option<Arc<Tensor<B>>>>> = self
            .nodes
            .iter()
            .map(|n| vec![None; n.outputs.len()])
            .collect();
        grads[output.0.0][output.1] = Some(grad_output.clone());

        for i in (0..self.nodes.len()).rev() {
            let slot_grads = std::mem::take(&mut grads[i]);
            if slot_grads.iter().all(Option::is_none) {
                continue; // dead end: nothing routed here, this node's output(s) were never consumed
            }

            let filled: Vec<Arc<Tensor<B>>> = slot_grads
                .into_iter()
                .enumerate()
                .map(|(slot, g)| g.unwrap_or_else(|| zeros_like(&self.nodes[i].outputs[slot])))
                .collect();

            let inputs = self.nodes[i].inputs.clone();
            let input_grads = self.nodes[i].op.backward(gb, &filled);

            for ((id, slot), ig) in inputs.into_iter().zip(input_grads) {
                match grads[id.0][slot].take() {
                    None => grads[id.0][slot] = Some(ig),
                    Some(existing) => {
                        super::ops::add_inplace_bwd(gb, &existing, &ig, elem_count(&existing));
                        grads[id.0][slot] = Some(existing);
                    }
                }
            }
        }
        self.grads = grads;
    }

    /// A node's total accumulated gradient after `backward()`. `None`
    /// before `backward()` runs, or if nothing routed a gradient to it.
    pub(crate) fn grad_of(&self, out: Out) -> Option<Arc<Tensor<B>>> {
        self.grads
            .get(out.0.0)
            .and_then(|slots| slots.get(out.1))
            .and_then(|g| g.clone())
    }

    /// (weight, grad, decay) triples in push order -- see `Backward::param`.
    pub(crate) fn params(&self) -> Vec<(&Arc<Tensor<B>>, &Arc<Tensor<B>>, bool)> {
        self.nodes
            .iter()
            .filter_map(|node| node.op.param())
            .collect()
    }
}

/// The shared leaf op: wraps a value that already exists (a block's own
/// input, or another Tape's output). No inputs; forward ignores `xs` and
/// just returns the stored value. Every `Node` enum needs a
/// `From<Identity<B>>` impl (one match-free line) so `Tape::input` works
/// the same way regardless of which op family the tape holds.
pub(crate) struct Identity<B: Backend>(pub Arc<Tensor<B>>);

impl<B: Backend, P: FwdPhase> Forward<B, P> for Identity<B> {
    fn forward(
        &mut self,
        _gb: &mut GraphBuilder<'_, B, P>,
        _xs: &[Arc<Tensor<B>>],
    ) -> Vec<Arc<Tensor<B>>> {
        vec![self.0.clone()]
    }
}

impl<B: Backend> Backward<B> for Identity<B> {
    fn backward(
        &mut self,
        _gb: &mut GraphBuilder<'_, B, super::ops::Train>,
        _grad_outputs: &[Arc<Tensor<B>>],
    ) -> Vec<Arc<Tensor<B>>> {
        vec![]
    }
}

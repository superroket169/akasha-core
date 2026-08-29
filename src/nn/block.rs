//! Autograd DAG for `Trainer`. `Tape` is an append-only node list; a node's
//! inputs are `(NodeId, output_slot)` pairs into earlier entries -- most
//! nodes have exactly one output, so slot is almost always `0`. Push order
//! is already a valid topological order for forward (a node can only
//! reference a `NodeId` that already exists, so no separate sort is
//! needed); iterating in reverse is a valid reverse-topological order for
//! backward, for the same reason.
//!
//! The one thing that DOES need real bookkeeping: an output slot with more
//! than one consumer (residual fan-out) receives more than one gradient
//! contribution during backward, and those must be summed before that
//! node's own backward runs. `Tape::backward` accumulates them with
//! `ops::add_inplace_bwd` (wilupgu's `BWD_ADD_INPLACE`, Accumulate-mode
//! `+=` kernel -- already exists, already used by hand for exactly this in
//! layers.rs today, no new shader needed).
//!
//! `Tape` itself never inspects which concrete op a node holds -- `push`/
//! `backward`/`params` only ever call through the `Op` trait. `TransformerOp`
//! is a closed enum purely for static dispatch (no `dyn Trait`, no heap
//! alloc/vtable, chosen because the op-kind set here is fixed -- every
//! transformer block reuses the same handful of kernels). If a second,
//! different op set is ever needed (Mamba/hybrid stacks), that's a real
//! trait-ification of `Tape` itself -- deliberately not done now.
//!
//! AdamW stays OUTSIDE Tape on purpose: it's Trainer's job to call
//! `tape.backward(...)` then hand `tape.params()` to the existing,
//! untouched `optim::AdamW` -- embedding the optimizer inside Tape would
//! mean Tape knowing a specific algorithm, exactly what it isn't supposed
//! to know.
//!
//! Not yet wired into layers.rs/train.rs -- integration is a separate step.

use super::ops;
use super::ops::FlashAttnBuffers;
use super::ops::GraphBuilder;
use super::ops::Train;
use super::ops::meta::{FlashAttnMeta, HeadMoveMeta, MatMulMeta, NormMeta, RopeMeta};
use crate::Real;
use std::sync::Arc;
use wilupgu::{Backend, Tensor};

/// `elems` zero-initialized elements. Every `Op::new` allocates its own
/// scratch/gradient buffers through this -- one place to change if that
/// ever needs to route through the pool differently.
fn zeros<B: Backend>(ctx: &Arc<B>, elems: u32) -> Arc<Tensor<B>> {
    Arc::new(Tensor::init_from_cpu(
        ctx.clone(),
        &vec![0.0 as Real; elems as usize],
    ))
}

/// A zeroed buffer the same size as `t` -- for a weight's gradient
/// accumulator, which is always exactly the weight's own shape.
fn zeros_like<B: Backend>(t: &Arc<Tensor<B>>) -> Arc<Tensor<B>> {
    zeros(&t.ctx, t.size as u32 / std::mem::size_of::<Real>() as u32)
}

fn elem_count<B: Backend>(t: &Arc<Tensor<B>>) -> u32 {
    (t.size / std::mem::size_of::<Real>() as u64) as u32
}

/// One forward/backward pair for a kernel. `&mut self` because a concrete
/// Op saves whatever its own backward needs (an input, an rsqrt cache, ...)
/// during forward. `xs`/`grad_outputs` are ordered the same as the node's
/// `inputs`/its own output count -- most ops have exactly one of each.
pub(crate) trait Op<B: Backend> {
    fn forward(
        &mut self,
        gb: &mut GraphBuilder<'_, B, Train>,
        xs: &[Arc<Tensor<B>>],
    ) -> Vec<Arc<Tensor<B>>>;

    /// Emits this op's backward node(s) and, if it owns a weight, also
    /// accumulates into that weight's gradient as a side effect. Returns
    /// the gradient w.r.t. each of this op's inputs, same order as `xs`.
    fn backward(
        &mut self,
        gb: &mut GraphBuilder<'_, B, Train>,
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

pub(crate) enum TransformerOp<B: Backend> {
    Linear(LinearOp<B>),
    RmsNorm(RmsNormOp<B>),
    Silu(SiluOp<B>),
    Add(AddOp<B>),
    RopeQk(RopeQkOp),
    QkvSplit(QkvSplitOp<B>),
    Attention(AttentionOp<B>),
    /// A leaf: wraps a value that already exists (a block's own input, or
    /// another Tape segment's output). No inputs, forward ignores `xs` and
    /// just returns the stored value; backward returns nothing to
    /// propagate to (it IS where propagation stops) -- `Tape::grad_of`
    /// reads its accumulated gradient off afterward.
    Input(Arc<Tensor<B>>),
}

impl<B: Backend> Op<B> for TransformerOp<B> {
    fn forward(
        &mut self,
        gb: &mut GraphBuilder<'_, B, Train>,
        xs: &[Arc<Tensor<B>>],
    ) -> Vec<Arc<Tensor<B>>> {
        match self {
            TransformerOp::Linear(op) => op.forward(gb, xs),
            TransformerOp::RmsNorm(op) => op.forward(gb, xs),
            TransformerOp::Silu(op) => op.forward(gb, xs),
            TransformerOp::Add(op) => op.forward(gb, xs),
            TransformerOp::RopeQk(op) => op.forward(gb, xs),
            TransformerOp::QkvSplit(op) => op.forward(gb, xs),
            TransformerOp::Attention(op) => op.forward(gb, xs),
            TransformerOp::Input(x) => vec![x.clone()],
        }
    }

    fn backward(
        &mut self,
        gb: &mut GraphBuilder<'_, B, Train>,
        grad_outputs: &[Arc<Tensor<B>>],
    ) -> Vec<Arc<Tensor<B>>> {
        match self {
            TransformerOp::Linear(op) => op.backward(gb, grad_outputs),
            TransformerOp::RmsNorm(op) => op.backward(gb, grad_outputs),
            TransformerOp::Silu(op) => op.backward(gb, grad_outputs),
            TransformerOp::Add(op) => op.backward(gb, grad_outputs),
            TransformerOp::RopeQk(op) => op.backward(gb, grad_outputs),
            TransformerOp::QkvSplit(op) => op.backward(gb, grad_outputs),
            TransformerOp::Attention(op) => op.backward(gb, grad_outputs),
            TransformerOp::Input(_) => vec![],
        }
    }

    fn param(&self) -> Option<(&Arc<Tensor<B>>, &Arc<Tensor<B>>, bool)> {
        match self {
            TransformerOp::Linear(op) => op.param(),
            TransformerOp::RmsNorm(op) => op.param(),
            TransformerOp::Silu(op) => op.param(),
            TransformerOp::Add(op) => op.param(),
            TransformerOp::RopeQk(op) => op.param(),
            TransformerOp::QkvSplit(op) => op.param(),
            TransformerOp::Attention(op) => op.param(),
            TransformerOp::Input(_) => None,
        }
    }
}

/// Handle to a node already on the tape.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct NodeId(usize);

/// A specific output slot of a node -- almost always slot `0`; only
/// `QkvSplit` (3 outputs) and `Attention`'s backward (3 outputs) need
/// anything else.
pub(crate) type Out = (NodeId, usize);

struct TapeNode<B: Backend> {
    op: TransformerOp<B>,
    inputs: Vec<Out>,
    outputs: Vec<Arc<Tensor<B>>>,
}

/// A DAG covering one block's worth of ops (or a whole model's, nothing
/// stops that -- scope is a modeling choice, not a limit of this struct).
/// See the module doc for how backward's accumulation works.
pub(crate) struct Tape<B: Backend> {
    nodes: Vec<TapeNode<B>>,
    grads: Vec<Vec<Option<Arc<Tensor<B>>>>>,
}

impl<B: Backend> Tape<B> {
    pub(crate) fn new() -> Self {
        Self {
            nodes: Vec::new(),
            grads: Vec::new(),
        }
    }

    /// Registers a value that already exists (a block's own input, or
    /// another Tape's output) as a node other nodes can reference.
    pub(crate) fn input(
        &mut self,
        gb: &mut GraphBuilder<'_, B, Train>,
        x: Arc<Tensor<B>>,
    ) -> NodeId {
        self.push(gb, TransformerOp::Input(x), &[])
    }

    /// Runs `op.forward(inputs' values)`, records it, returns its handle.
    /// `out(id)` (slot 0) is a convenience for the overwhelmingly common
    /// single-output-predecessor case.
    pub(crate) fn push(
        &mut self,
        gb: &mut GraphBuilder<'_, B, Train>,
        mut op: TransformerOp<B>,
        inputs: &[Out],
    ) -> NodeId {
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

    /// Seeds `output`'s gradient with `grad_output`, walks every node in
    /// reverse (already a valid reverse-topological order -- see module
    /// doc), and accumulates (`+=`) whenever more than one node routes a
    /// contribution to the same earlier output slot. Read a leaf's total
    /// via `grad_of` afterward.
    pub(crate) fn backward(
        &mut self,
        gb: &mut GraphBuilder<'_, B, Train>,
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
            // an op's backward wants one gradient per output slot; a slot
            // nobody wrote to (an output some OTHER slot's consumer used,
            // this one didn't) gets a zeroed contribution.
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
                        ops::add_inplace_bwd(gb, &existing, &ig, elem_count(&existing));
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

    /// (weight, grad, decay) triples in push order -- see `Op::param`.
    pub(crate) fn params(&self) -> Vec<(&Arc<Tensor<B>>, &Arc<Tensor<B>>, bool)> {
        self.nodes
            .iter()
            .filter_map(|node| node.op.param())
            .collect()
    }
}

/// `y = x @ weight`. `weight` comes straight from `BlockWeights` (e.g.
/// `&bw.qkv_proj`) -- everything else (`grad_weight`, `out`, `grad_in`) is
/// allocated here, nobody upstream invents a buffer name for it.
pub(crate) struct LinearOp<B: Backend> {
    weight: Arc<Tensor<B>>,
    grad_weight: Arc<Tensor<B>>,
    out: Arc<Tensor<B>>,
    grad_in: Arc<Tensor<B>>,
    shape: MatMulMeta,
    saved_input: Option<Arc<Tensor<B>>>,
    decay: bool,
}

impl<B: Backend> LinearOp<B> {
    pub(crate) fn new(weight: &Arc<Tensor<B>>, shape: MatMulMeta, decay: bool) -> Self {
        let ctx = &weight.ctx;
        Self {
            weight: weight.clone(),
            grad_weight: zeros_like(weight),
            out: zeros(ctx, shape.m * shape.n),
            grad_in: zeros(ctx, shape.m * shape.k),
            shape,
            saved_input: None,
            decay,
        }
    }
}

impl<B: Backend> Op<B> for LinearOp<B> {
    fn forward(
        &mut self,
        gb: &mut GraphBuilder<'_, B, Train>,
        xs: &[Arc<Tensor<B>>],
    ) -> Vec<Arc<Tensor<B>>> {
        let x = &xs[0];
        ops::matmul(gb, x, &self.weight, &self.out, self.shape);
        self.saved_input = Some(x.clone());
        vec![self.out.clone()]
    }

    fn backward(
        &mut self,
        gb: &mut GraphBuilder<'_, B, Train>,
        grad_outputs: &[Arc<Tensor<B>>],
    ) -> Vec<Arc<Tensor<B>>> {
        let grad_output = &grad_outputs[0];
        let x = self
            .saved_input
            .take()
            .expect("LinearOp::backward called before forward");
        // dW[k,n] += x[m,k]^T @ dY[m,n]
        ops::matmul_weight_bwd(gb, &x, grad_output, &self.grad_weight, self.shape);
        // dX[m,k] = dY[m,n] @ W[k,n]^T. matmul_trp wants its own (m,n,k)
        // describing the OUTPUT as [m,n] and B stored as [n,k] -- here the
        // output is dX[m,k] and B (weight) is already stored as [k,n], so
        // n/k swap relative to the forward shape. Verified against the
        // real, shipped Linear::new in layers.rs (identical swap there).
        let trp_shape = MatMulMeta {
            m: self.shape.m,
            n: self.shape.k,
            k: self.shape.n,
        };
        ops::matmul_trp(gb, grad_output, &self.weight, &self.grad_in, trp_shape);
        vec![self.grad_in.clone()]
    }

    fn param(&self) -> Option<(&Arc<Tensor<B>>, &Arc<Tensor<B>>, bool)> {
        Some((&self.weight, &self.grad_weight, self.decay))
    }
}

/// `weight` comes straight from `BlockWeights` (e.g. `&bw.norm_1`).
pub(crate) struct RmsNormOp<B: Backend> {
    weight: Arc<Tensor<B>>,
    grad_weight: Arc<Tensor<B>>,
    out: Arc<Tensor<B>>,
    grad_in: Arc<Tensor<B>>,
    rsqrt_cache: Arc<Tensor<B>>,
    shape: NormMeta,
    saved_input: Option<Arc<Tensor<B>>>,
}

impl<B: Backend> RmsNormOp<B> {
    pub(crate) fn new(weight: &Arc<Tensor<B>>, shape: NormMeta) -> Self {
        let ctx = &weight.ctx;
        Self {
            weight: weight.clone(),
            grad_weight: zeros_like(weight),
            out: zeros(ctx, shape.seq_len * shape.size),
            grad_in: zeros(ctx, shape.seq_len * shape.size),
            rsqrt_cache: zeros(ctx, shape.seq_len),
            shape,
            saved_input: None,
        }
    }
}

impl<B: Backend> Op<B> for RmsNormOp<B> {
    fn forward(
        &mut self,
        gb: &mut GraphBuilder<'_, B, Train>,
        xs: &[Arc<Tensor<B>>],
    ) -> Vec<Arc<Tensor<B>>> {
        let x = &xs[0];
        ops::rmsnorm(gb, x, &self.weight, &self.out, self.shape);
        self.saved_input = Some(x.clone());
        vec![self.out.clone()]
    }

    fn backward(
        &mut self,
        gb: &mut GraphBuilder<'_, B, Train>,
        grad_outputs: &[Arc<Tensor<B>>],
    ) -> Vec<Arc<Tensor<B>>> {
        let x = self
            .saved_input
            .take()
            .expect("RmsNormOp::backward called before forward");
        ops::rmsnorm_bwd(
            gb,
            &grad_outputs[0],
            &x,
            &self.weight,
            &self.grad_in,
            &self.rsqrt_cache,
            &self.grad_weight,
            self.shape,
        );
        vec![self.grad_in.clone()]
    }

    fn param(&self) -> Option<(&Arc<Tensor<B>>, &Arc<Tensor<B>>, bool)> {
        // norm gains are decay-exempt (E1, ARCHITECTURE.md Invariantlar).
        Some((&self.weight, &self.grad_weight, false))
    }
}

/// Weightless -- silu_out/silu_bwd (not the in-place `silu`), because
/// backward needs the pre-activation and in-place would have overwritten
/// it. Matches what layers.rs::SiLU already does today, for the same reason.
pub(crate) struct SiluOp<B: Backend> {
    out: Arc<Tensor<B>>,
    grad_in: Arc<Tensor<B>>,
    len: u32,
    saved_input: Option<Arc<Tensor<B>>>,
}

impl<B: Backend> SiluOp<B> {
    pub(crate) fn new(ctx: &Arc<B>, len: u32) -> Self {
        Self {
            out: zeros(ctx, len),
            grad_in: zeros(ctx, len),
            len,
            saved_input: None,
        }
    }
}

impl<B: Backend> Op<B> for SiluOp<B> {
    fn forward(
        &mut self,
        gb: &mut GraphBuilder<'_, B, Train>,
        xs: &[Arc<Tensor<B>>],
    ) -> Vec<Arc<Tensor<B>>> {
        let x = &xs[0];
        ops::silu_out(gb, x, &self.out, self.len);
        self.saved_input = Some(x.clone());
        vec![self.out.clone()]
    }

    fn backward(
        &mut self,
        gb: &mut GraphBuilder<'_, B, Train>,
        grad_outputs: &[Arc<Tensor<B>>],
    ) -> Vec<Arc<Tensor<B>>> {
        let x = self
            .saved_input
            .take()
            .expect("SiluOp::backward called before forward");
        ops::silu_bwd(gb, &x, &grad_outputs[0], &self.grad_in, self.len);
        vec![self.grad_in.clone()]
    }
}

/// `y = a + b` (residual). Backward: `d(a+b)/da = d(a+b)/db = identity`, so
/// the SAME incoming gradient is handed back for both inputs unchanged --
/// no kernel, no state to save. `Tape::backward`'s accumulation is what
/// used to be `zero_transient_grads`/`add_inplace_bwd` by hand.
pub(crate) struct AddOp<B: Backend> {
    out: Arc<Tensor<B>>,
    len: u32,
}

impl<B: Backend> AddOp<B> {
    pub(crate) fn new(ctx: &Arc<B>, len: u32) -> Self {
        Self {
            out: zeros(ctx, len),
            len,
        }
    }
}

impl<B: Backend> Op<B> for AddOp<B> {
    fn forward(
        &mut self,
        gb: &mut GraphBuilder<'_, B, Train>,
        xs: &[Arc<Tensor<B>>],
    ) -> Vec<Arc<Tensor<B>>> {
        ops::add_out(gb, &xs[0], &xs[1], &self.out, self.len);
        vec![self.out.clone()]
    }

    fn backward(
        &mut self,
        _gb: &mut GraphBuilder<'_, B, Train>,
        grad_outputs: &[Arc<Tensor<B>>],
    ) -> Vec<Arc<Tensor<B>>> {
        vec![grad_outputs[0].clone(), grad_outputs[0].clone()]
    }
}

/// Fused RoPE on Q and K in one dispatch. In-place (`InOut`): forward
/// rotates `xs = [q, k]` and hands the SAME two buffers back as its
/// "output"; backward does the same in reverse on the incoming gradients.
/// No weight, no saved state -- the kernel derives the rotation from
/// position alone.
pub(crate) struct RopeQkOp {
    shape: RopeMeta,
}

impl RopeQkOp {
    pub(crate) fn new(shape: RopeMeta) -> Self {
        Self { shape }
    }
}

impl<B: Backend> Op<B> for RopeQkOp {
    fn forward(
        &mut self,
        gb: &mut GraphBuilder<'_, B, Train>,
        xs: &[Arc<Tensor<B>>],
    ) -> Vec<Arc<Tensor<B>>> {
        ops::rope_qk(gb, &xs[0], &xs[1], self.shape);
        vec![xs[0].clone(), xs[1].clone()]
    }

    fn backward(
        &mut self,
        gb: &mut GraphBuilder<'_, B, Train>,
        grad_outputs: &[Arc<Tensor<B>>],
    ) -> Vec<Arc<Tensor<B>>> {
        ops::rope_bwd_qk(gb, &grad_outputs[0], &grad_outputs[1], self.shape);
        vec![grad_outputs[0].clone(), grad_outputs[1].clone()]
    }
}

/// Fused qkv split: one `[rows, 3*dim]` input -> three `[rows, dim]`
/// outputs (q, k, v), one dispatch. Backward (`qkv_scatter`) is the mirror:
/// three gradient slices -> one fused `[rows, 3*dim]` gradient, also one
/// dispatch. This is the op qkv_split/qkv_scatter needed a real Tape node
/// for -- 1-in/3-out doesn't fit a single-output trait, which is why it
/// used to be hand-wired outside any tape.
pub(crate) struct QkvSplitOp<B: Backend> {
    q: Arc<Tensor<B>>,
    k: Arc<Tensor<B>>,
    v: Arc<Tensor<B>>,
    grad_qkv: Arc<Tensor<B>>,
    shape: HeadMoveMeta,
}

impl<B: Backend> QkvSplitOp<B> {
    pub(crate) fn new(ctx: &Arc<B>, rows: u32, dim: u32) -> Self {
        Self {
            q: zeros(ctx, rows * dim),
            k: zeros(ctx, rows * dim),
            v: zeros(ctx, rows * dim),
            grad_qkv: zeros(ctx, rows * dim * 3),
            shape: HeadMoveMeta::qkv_slice(rows, dim, 0),
        }
    }
}

impl<B: Backend> Op<B> for QkvSplitOp<B> {
    fn forward(
        &mut self,
        gb: &mut GraphBuilder<'_, B, Train>,
        xs: &[Arc<Tensor<B>>],
    ) -> Vec<Arc<Tensor<B>>> {
        ops::qkv_split(gb, &xs[0], &self.q, &self.k, &self.v, self.shape);
        vec![self.q.clone(), self.k.clone(), self.v.clone()]
    }

    fn backward(
        &mut self,
        gb: &mut GraphBuilder<'_, B, Train>,
        grad_outputs: &[Arc<Tensor<B>>],
    ) -> Vec<Arc<Tensor<B>>> {
        // grad_outputs = [dq, dk, dv]
        ops::qkv_scatter(
            gb,
            &grad_outputs[0],
            &grad_outputs[1],
            &grad_outputs[2],
            &self.grad_qkv,
            self.shape,
        );
        vec![self.grad_qkv.clone()]
    }
}

/// Flash attention: `[q, k, v] -> attn_out` (3 in, 1 out) forward; `d_out
/// -> [dq, dk, dv]` (1 in, 3 out) backward. No weight. Needs q/k/v (not
/// just its own output) saved for backward, since the kernel recomputes
/// attention weights rather than storing the full score matrix.
pub(crate) struct AttentionOp<B: Backend> {
    out: Arc<Tensor<B>>,
    grad_q: Arc<Tensor<B>>,
    grad_k: Arc<Tensor<B>>,
    grad_v: Arc<Tensor<B>>,
    shape: FlashAttnMeta,
    saved: Option<(
        Arc<Tensor<B>>,
        Arc<Tensor<B>>,
        Arc<Tensor<B>>,
        FlashAttnBuffers<B>,
    )>,
}

impl<B: Backend> AttentionOp<B> {
    pub(crate) fn new(ctx: &Arc<B>, shape: FlashAttnMeta) -> Self {
        let dim_size = shape.seq_len * shape.dim;
        Self {
            out: zeros(ctx, dim_size),
            grad_q: zeros(ctx, dim_size),
            grad_k: zeros(ctx, dim_size),
            grad_v: zeros(ctx, dim_size),
            shape,
            saved: None,
        }
    }
}

impl<B: Backend> Op<B> for AttentionOp<B> {
    fn forward(
        &mut self,
        gb: &mut GraphBuilder<'_, B, Train>,
        xs: &[Arc<Tensor<B>>],
    ) -> Vec<Arc<Tensor<B>>> {
        let (q, k, v) = (xs[0].clone(), xs[1].clone(), xs[2].clone());
        let saved_bufs = ops::flash_attention(gb, &q, &k, &v, &self.out, self.shape);
        self.saved = Some((q, k, v, saved_bufs));
        vec![self.out.clone()]
    }

    fn backward(
        &mut self,
        gb: &mut GraphBuilder<'_, B, Train>,
        grad_outputs: &[Arc<Tensor<B>>],
    ) -> Vec<Arc<Tensor<B>>> {
        let (q, k, v, saved_bufs) = self
            .saved
            .take()
            .expect("AttentionOp::backward called before forward");
        ops::flash_attention_bwd(
            gb,
            &q,
            &k,
            &v,
            &saved_bufs,
            &grad_outputs[0],
            &self.grad_q,
            &self.grad_k,
            &self.grad_v,
            self.shape,
        );
        vec![
            self.grad_q.clone(),
            self.grad_k.clone(),
            self.grad_v.clone(),
        ]
    }
}

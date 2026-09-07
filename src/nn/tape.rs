use super::kernels::FwdPhase;
use super::kernels::GraphBuilder;
use crate::Real;
use std::collections::HashMap;
use std::sync::Arc;
use wilupgu::{Backend, Tensor};

pub(crate) fn zeros<B: Backend>(ctx: &Arc<B>, elems: u32) -> Arc<Tensor<B>> {
    Arc::new(Tensor::init_from_cpu(
        ctx.clone(),
        &vec![0.0 as Real; elems as usize],
    ))
}

pub(crate) fn zeros_like<B: Backend>(t: &Arc<Tensor<B>>) -> Arc<Tensor<B>> {
    zeros(&t.ctx, t.size as u32 / std::mem::size_of::<Real>() as u32)
}

pub(crate) fn elem_count<B: Backend>(t: &Arc<Tensor<B>>) -> u32 {
    (t.size / std::mem::size_of::<Real>() as u64) as u32
}

pub(crate) trait Forward<B: Backend, P: FwdPhase> {
    fn forward(
        &mut self,
        gb: &mut GraphBuilder<'_, B, P>,
        xs: &[Arc<Tensor<B>>],
    ) -> Vec<Arc<Tensor<B>>>;
}

pub(crate) trait Backward<B: Backend> {
    fn backward(
        &mut self,
        gb: &mut GraphBuilder<'_, B, super::kernels::Train>,
        grad_outputs: &[Arc<Tensor<B>>],
    ) -> Vec<Arc<Tensor<B>>>;

    // push order here IS the checkpoint/AdamW-moment format contract.
    fn param(&self) -> Option<(&Arc<Tensor<B>>, &Arc<Tensor<B>>, bool)> {
        None
    }
}

pub(crate) trait Advance {
    fn advance(&mut self, step: u32);
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct NodeId(usize);

pub(crate) type Out = (NodeId, usize);

struct TapeNode<B: Backend, Node> {
    op: Node,
    inputs: Vec<Out>,
    outputs: Vec<Arc<Tensor<B>>>,
}

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

    pub(crate) fn input<P: FwdPhase>(
        &mut self,
        gb: &mut GraphBuilder<'_, B, P>,
        x: Arc<Tensor<B>>,
    ) -> NodeId
    where
        Node: From<Leaf<B>> + Forward<B, P>,
    {
        self.push(gb, Node::from(Leaf(x)), &[])
    }

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

    pub(crate) fn output(&self, id: NodeId) -> Arc<Tensor<B>> {
        self.nodes[id.0].outputs[0].clone()
    }

    pub(crate) fn extend<P: FwdPhase>(
        &mut self,
        gb: &mut GraphBuilder<'_, B, P>,
        names: &mut HashMap<&'static str, NodeId>,
        specs: Vec<NodeSpec<Node>>,
    ) -> NodeId
    where
        Node: Forward<B, P>,
    {
        let mut last = None;
        for spec in specs {
            let inputs: Vec<Out> = spec
                .inputs
                .iter()
                .map(|(name, slot)| (names[name], *slot))
                .collect();
            let id = self.push(gb, spec.op, &inputs);
            names.insert(spec.name, id);
            last = Some(id);
        }
        last.expect("Tape::extend called with empty specs")
    }
}

pub(crate) struct NodeSpec<Node> {
    pub(crate) name: &'static str,
    pub(crate) inputs: &'static [(&'static str, usize)],
    pub(crate) op: Node,
}

macro_rules! node {
    ($name:literal <- $inputs:expr, $op:expr) => {
        $crate::nn::tape::NodeSpec {
            name: $name,
            inputs: $inputs,
            op: $op,
        }
    };
}
pub(crate) use node;

impl<B: Backend, Node: Backward<B>> Tape<B, Node> {
    pub(crate) fn backward(
        &mut self,
        gb: &mut GraphBuilder<'_, B, super::kernels::Train>,
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
                continue;
            }

            let filled: Vec<Arc<Tensor<B>>> = slot_grads
                .into_iter()
                .enumerate()
                .map(|(slot, g)| g.unwrap_or_else(|| zeros_like(&self.nodes[i].outputs[slot])))
                .collect();

            // grads[i] was just taken (now empty)
            grads[i] = filled.iter().cloned().map(Some).collect();

            let inputs = self.nodes[i].inputs.clone();
            let input_grads = self.nodes[i].op.backward(gb, &filled);

            for ((id, slot), ig) in inputs.into_iter().zip(input_grads) {
                match grads[id.0][slot].take() {
                    None => grads[id.0][slot] = Some(ig),
                    Some(existing) => {
                        super::kernels::add_inplace_bwd(gb, &existing, &ig, elem_count(&existing));
                        grads[id.0][slot] = Some(existing);
                    }
                }
            }
        }
        self.grads = grads;
    }

    pub(crate) fn grad_of(&self, out: Out) -> Option<Arc<Tensor<B>>> {
        self.grads
            .get(out.0.0)
            .and_then(|slots| slots.get(out.1))
            .and_then(|g| g.clone())
    }

    pub(crate) fn params(&self) -> Vec<(&Arc<Tensor<B>>, &Arc<Tensor<B>>, bool)> {
        self.nodes
            .iter()
            .filter_map(|node| node.op.param())
            .collect()
    }
}

impl<B: Backend, Node: Advance> Tape<B, Node> {
    pub(crate) fn advance(&mut self, step: u32) {
        for node in &mut self.nodes {
            node.op.advance(step);
        }
    }
}

pub(crate) struct Leaf<B: Backend>(pub Arc<Tensor<B>>);

impl<B: Backend, P: FwdPhase> Forward<B, P> for Leaf<B> {
    fn forward(
        &mut self,
        _gb: &mut GraphBuilder<'_, B, P>,
        _xs: &[Arc<Tensor<B>>],
    ) -> Vec<Arc<Tensor<B>>> {
        vec![self.0.clone()]
    }
}

impl<B: Backend> Backward<B> for Leaf<B> {
    fn backward(
        &mut self,
        _gb: &mut GraphBuilder<'_, B, super::kernels::Train>,
        _grad_outputs: &[Arc<Tensor<B>>],
    ) -> Vec<Arc<Tensor<B>>> {
        vec![]
    }
}

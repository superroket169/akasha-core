use super::super::ops::full_seq::TrainOp;
use super::super::kernels::{GraphBuilder, Train};
use super::super::tape::Backward;
use std::sync::Arc;
use wilupgu::{Backend, Tensor};

macro_rules! impl_backward_dispatch {
    ($enum:ident { $($variant:ident),+ $(,)? }) => {
        impl<B: Backend> Backward<B> for $enum<B> {
            fn backward(
                &mut self,
                gb: &mut GraphBuilder<'_, B, Train>,
                grad_outputs: &[Arc<Tensor<B>>],
            ) -> Vec<Arc<Tensor<B>>> {
                match self {
                    $( $enum::$variant(op) => op.backward(gb, grad_outputs), )+
                }
            }

            fn param(&self) -> Option<(&Arc<Tensor<B>>, &Arc<Tensor<B>>, bool)> {
                match self {
                    $( $enum::$variant(op) => op.param(), )+
                }
            }
        }
    };
}

impl_backward_dispatch!(TrainOp {
    Embedding, Linear, RmsNorm, Silu, Add, RopeQk, QkvSplit, Attention, Leaf,
});

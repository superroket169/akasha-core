use super::super::ops::cached::{DecodeOp, PrefillOp};
use super::super::ops::full_seq::TrainOp;
use super::super::kernels::{Decode, GraphBuilder, Prefill, Train};
use super::super::tape::Forward;
use std::sync::Arc;
use wilupgu::{Backend, Tensor};

macro_rules! impl_forward_dispatch {
    ($enum:ident, $phase:ty, { $($variant:ident),+ $(,)? }) => {
        impl<B: Backend> Forward<B, $phase> for $enum<B> {
            fn forward(
                &mut self,
                gb: &mut GraphBuilder<'_, B, $phase>,
                xs: &[Arc<Tensor<B>>],
            ) -> Vec<Arc<Tensor<B>>> {
                match self {
                    $( $enum::$variant(op) => op.forward(gb, xs), )+
                }
            }
        }
    };
}

impl_forward_dispatch!(TrainOp, Train, {
    Embedding, Linear, RmsNorm, Silu, Add, RopeQk, QkvSplit, Attention, Leaf,
});

impl_forward_dispatch!(PrefillOp, Prefill, {
    Embedding, Linear, RmsNorm, Silu, Add, RopeQk, QkvSplit, Attention, CacheWrite, Leaf,
});

impl_forward_dispatch!(DecodeOp, Decode, {
    Embedding, Linear, RmsNorm, Silu, Add, RopeOffset, HeadGather, CacheWrite, CachedAttention, Leaf,
});

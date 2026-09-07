use super::super::ops::full_seq::TrainOp;
use super::super::tape::Checkpointable;
use std::sync::Arc;
use wilupgu::Backend;

macro_rules! impl_checkpoint_dispatch {
    ($enum:ident { $($variant:ident),+ $(,)? }) => {
        impl<B: Backend> Checkpointable<B> for $enum<B> {
            fn free_activations(&mut self) {
                match self {
                    $( $enum::$variant(op) => Checkpointable::<B>::free_activations(op), )+
                }
            }

            fn realloc_activations(&mut self, ctx: &Arc<B>) {
                match self {
                    $( $enum::$variant(op) => Checkpointable::<B>::realloc_activations(op, ctx), )+
                }
            }
        }
    };
}

impl_checkpoint_dispatch!(TrainOp {
    Embedding, Linear, RmsNorm, Silu, Add, RopeQk, QkvSplit, Attention, Leaf,
});

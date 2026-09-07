use super::super::ops::full_seq::TrainOp;
use super::super::tape::Checkpointable;
use wilupgu::Backend;

macro_rules! impl_checkpoint_dispatch {
    ($enum:ident { $($variant:ident),+ $(,)? }) => {
        impl<B: Backend> Checkpointable<B> for $enum<B> {
            fn free_activations(&mut self) {
                match self {
                    $( $enum::$variant(op) => Checkpointable::<B>::free_activations(op), )+
                }
            }
        }
    };
}

impl_checkpoint_dispatch!(TrainOp {
    Embedding,
    Linear,
    RmsNorm,
    Silu,
    Add,
    RopeQk,
    QkvSplit,
    Attention,
    Leaf,
});

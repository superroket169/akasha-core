use super::cached_ops::{DecodeOp, PrefillOp};
use super::core_ops::{AddOp, LinearOp, RmsNormOp, SiluOp, TrainOp};
use super::tape::Leaf;
use wilupgu::Backend;

macro_rules! impl_from_op {
    ($($op:ident => $enum:ident :: $variant:ident),+ $(,)?) => {
        $(
            impl<B: Backend> From<$op<B>> for $enum<B> {
                fn from(op: $op<B>) -> Self {
                    $enum::$variant(op)
                }
            }
        )+
    };
}

impl_from_op! {
    Leaf => TrainOp::Leaf,
    RmsNormOp => TrainOp::RmsNorm,
    LinearOp => TrainOp::Linear,
    AddOp => TrainOp::Add,
    SiluOp => TrainOp::Silu,

    Leaf => PrefillOp::Leaf,
    RmsNormOp => PrefillOp::RmsNorm,
    LinearOp => PrefillOp::Linear,
    AddOp => PrefillOp::Add,
    SiluOp => PrefillOp::Silu,

    Leaf => DecodeOp::Leaf,
    RmsNormOp => DecodeOp::RmsNorm,
    LinearOp => DecodeOp::Linear,
    AddOp => DecodeOp::Add,
    SiluOp => DecodeOp::Silu,
}

use super::super::ops::cached::{DecodeOp, PrefillOp};
use super::super::ops::full_seq::TrainOp;
use super::super::tape::Leaf;
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
    Leaf => PrefillOp::Leaf,
    Leaf => DecodeOp::Leaf,
}

use super::super::ops::cached::{CacheWriteOp, CachedAttentionOp, DecodeOp, RopeOffsetOp};
use super::super::kernels::meta::KernelMeta;
use super::super::tape::Advance;
use wilupgu::Backend;

impl<B: Backend> Advance for CacheWriteOp<B> {
    fn advance(&mut self, step: u32) {
        self.shape.dst_row_offset = step;
        self.shape.write_to(&self.meta);
    }
}

impl<B: Backend> Advance for RopeOffsetOp<B> {
    fn advance(&mut self, step: u32) {
        self.shape.pos = step;
        self.shape.write_to(&self.meta);
    }
}

impl<B: Backend> Advance for CachedAttentionOp<B> {
    // step is the position just written this step -- cache is valid for [0, step].
    fn advance(&mut self, step: u32) {
        let attn_len = step + 1;
        self.attn_shape.attn_len = attn_len;
        self.attn_shape.write_to(&self.attn_meta);
        self.softmax_shape.width = attn_len;
        self.softmax_shape.write_to(&self.softmax_meta);
    }
}

impl<B: Backend> Advance for DecodeOp<B> {
    fn advance(&mut self, step: u32) {
        match self {
            DecodeOp::RopeOffset(op) => op.advance(step),
            DecodeOp::CacheWrite(op) => op.advance(step),
            DecodeOp::CachedAttention(op) => op.advance(step),
            _ => {}
        }
    }
}

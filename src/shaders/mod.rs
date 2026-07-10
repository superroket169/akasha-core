mod cpu;
mod cuda;

use wilupgu::Shader;
use wilupgu::TensorMode::{InOut, Input, Meta, Output};
use wilupgu::{CudaShape, CudaSpec, MetaField};

pub static EMBEDDING: Shader = Shader {
    name: "Embedding",
    layout: &[Input, Input, Output, Meta],
    wgpu: Some(include_str!("fwd/embedding.wgsl")),
    cpu: Some(cpu::embedding),
    cuda: Some(CudaSpec {
        src: cuda::EMBEDDING,
        entry: "embedding_kernel",
        shape: CudaShape::Generic {
            meta_fields: &[MetaField::U32, MetaField::U32, MetaField::U32],
            block_dim: (256, 1, 1),
            append_len: false,
        },
    }),
};

pub static EMBEDDING_BWD: Shader = Shader {
    name: "EmbeddingBwd",
    layout: &[Input, Input, Output, Meta],
    wgpu: Some(include_str!("bwd/embedding_bwd.wgsl")),
    cpu: Some(cpu::embedding_bwd),
    cuda: Some(CudaSpec {
        src: cuda::EMBEDDING_BWD,
        entry: "embedding_bwd_kernel",
        shape: CudaShape::Generic {
            meta_fields: &[MetaField::U32, MetaField::U32, MetaField::U32],
            block_dim: (256, 1, 1),
            append_len: false,
        },
    }),
};

pub static SILU: Shader = Shader {
    name: "SiLU",
    layout: &[InOut],
    wgpu: Some(include_str!("fwd/silu.wgsl")),
    cpu: Some(cpu::silu),
    cuda: Some(CudaSpec {
        src: cuda::SILU,
        entry: "silu_kernel",
        shape: CudaShape::Generic {
            meta_fields: &[],
            block_dim: (256, 1, 1),
            append_len: true,
        },
    }),
};

pub static SILU_OUT: Shader = Shader {
    name: "SiLUOut",
    layout: &[Input, Output],
    wgpu: Some(include_str!("fwd/silu_out.wgsl")),
    cpu: Some(cpu::silu_out),
    cuda: Some(CudaSpec {
        src: cuda::SILU_OUT,
        entry: "silu_out_kernel",
        shape: CudaShape::Generic {
            meta_fields: &[],
            block_dim: (256, 1, 1),
            append_len: true,
        },
    }),
};

pub static ADD: Shader = Shader {
    name: "Add",
    layout: &[Input, Input, Output],
    wgpu: Some(include_str!("fwd/add.wgsl")),
    cpu: Some(cpu::add),
    cuda: Some(CudaSpec {
        src: cuda::ADD,
        entry: "add_kernel",
        shape: CudaShape::Generic {
            meta_fields: &[],
            block_dim: (256, 1, 1),
            append_len: true,
        },
    }),
};

/// No CPU implementation -- pre-existing gap inherited from wilupgu's old
/// string-keyed dispatch (the CPU backend never had a `SiLUBwd` match arm).
pub static SILU_BWD: Shader = Shader {
    name: "SiLUBwd",
    layout: &[Input, Input, Output],
    wgpu: Some(include_str!("bwd/silu_bwd.wgsl")),
    cpu: None,
    cuda: Some(CudaSpec {
        src: cuda::SILU_BWD,
        entry: "silu_bwd_kernel",
        shape: CudaShape::Generic {
            meta_fields: &[],
            block_dim: (256, 1, 1),
            append_len: true,
        },
    }),
};

pub static ROPE: Shader = Shader {
    name: "RoPE",
    layout: &[InOut, Meta],
    wgpu: Some(include_str!("fwd/rope.wgsl")),
    cpu: Some(cpu::rope),
    cuda: Some(CudaSpec {
        src: cuda::ROPE,
        entry: "rope_kernel",
        shape: CudaShape::Generic {
            meta_fields: &[MetaField::U32, MetaField::U32, MetaField::U32],
            block_dim: (16, 16, 1),
            append_len: false,
        },
    }),
};

/// No CPU implementation (pre-existing gap, see `SILU_BWD`).
pub static ROPE_BWD: Shader = Shader {
    name: "RoPEBwd",
    layout: &[InOut, Meta],
    wgpu: Some(include_str!("bwd/rope_bwd.wgsl")),
    cpu: None,
    cuda: Some(CudaSpec {
        src: cuda::ROPE_BWD,
        entry: "rope_bwd_kernel",
        shape: CudaShape::Generic {
            meta_fields: &[MetaField::U32, MetaField::U32, MetaField::U32],
            block_dim: (16, 16, 1),
            append_len: false,
        },
    }),
};

pub static ROPE_QK: Shader = Shader {
    name: "RopeQK",
    layout: &[InOut, InOut, Meta],
    wgpu: Some(include_str!("fwd/rope_qk.wgsl")),
    cpu: None,
    cuda: Some(CudaSpec {
        src: cuda::ROPE_QK,
        entry: "rope_qk_kernel",
        shape: CudaShape::Generic {
            meta_fields: &[
                MetaField::U32,
                MetaField::U32,
                MetaField::U32,
                MetaField::U32,
            ],
            block_dim: (16, 16, 1),
            append_len: false,
        },
    }),
};

pub static ROPE_BWD_QK: Shader = Shader {
    name: "RopeBwdQK",
    layout: &[InOut, InOut, Meta],
    wgpu: Some(include_str!("bwd/rope_bwd_qk.wgsl")),
    cpu: None,
    cuda: Some(CudaSpec {
        src: cuda::ROPE_BWD_QK,
        entry: "rope_bwd_qk_kernel",
        shape: CudaShape::Generic {
            meta_fields: &[
                MetaField::U32,
                MetaField::U32,
                MetaField::U32,
                MetaField::U32,
            ],
            block_dim: (16, 16, 1),
            append_len: false,
        },
    }),
};

pub static ROPE_OFFSET: Shader = Shader {
    name: "RoPEOffset",
    layout: &[InOut, Meta],
    wgpu: Some(include_str!("fwd/rope_offset.wgsl")),
    cpu: Some(cpu::rope_offset),
    cuda: Some(CudaSpec {
        src: cuda::ROPE_OFFSET,
        entry: "rope_offset_kernel",
        shape: CudaShape::Generic {
            meta_fields: &[
                MetaField::U32,
                MetaField::U32,
                MetaField::U32,
                MetaField::U32,
            ],
            block_dim: (16, 16, 1),
            append_len: false,
        },
    }),
};

pub static SOFTMAX: Shader = Shader {
    name: "Softmax",
    layout: &[InOut, Meta],
    wgpu: Some(include_str!("fwd/softmax.wgsl")),
    cpu: Some(cpu::softmax),
    cuda: Some(CudaSpec {
        src: cuda::SOFTMAX,
        entry: "softmax_kernel",
        shape: CudaShape::Generic {
            meta_fields: &[MetaField::U32],
            block_dim: (256, 1, 1),
            append_len: false,
        },
    }),
};

/// No CPU implementation (pre-existing gap, see `SILU_BWD`).
pub static SOFTMAX_BWD: Shader = Shader {
    name: "SoftmaxBwd",
    layout: &[Input, Input, Output, Meta],
    wgpu: Some(include_str!("bwd/softmax_bwd.wgsl")),
    cpu: None,
    cuda: Some(CudaSpec {
        src: cuda::SOFTMAX_BWD,
        entry: "softmax_bwd_kernel",
        shape: CudaShape::Generic {
            meta_fields: &[MetaField::U32, MetaField::F32],
            block_dim: (256, 1, 1),
            append_len: false,
        },
    }),
};

pub static SOFTMAX_RECT: Shader = Shader {
    name: "SoftmaxRect",
    layout: &[InOut, Meta],
    wgpu: Some(include_str!("fwd/softmax_rect.wgsl")),
    cpu: Some(cpu::softmax_rect),
    cuda: Some(CudaSpec {
        src: cuda::SOFTMAX_RECT,
        entry: "softmax_rect_kernel",
        shape: CudaShape::Generic {
            meta_fields: &[MetaField::U32, MetaField::U32, MetaField::F32],
            block_dim: (256, 1, 1),
            append_len: false,
        },
    }),
};

pub static CAUSAL_SOFTMAX: Shader = Shader {
    name: "CausalSoftmax",
    layout: &[InOut, Meta],
    wgpu: Some(include_str!("fwd/causal_softmax.wgsl")),
    cpu: Some(cpu::causal_softmax),
    cuda: Some(CudaSpec {
        src: cuda::CAUSAL_SOFTMAX,
        entry: "causal_softmax_kernel",
        shape: CudaShape::Generic {
            meta_fields: &[MetaField::U32, MetaField::F32],
            block_dim: (256, 1, 1),
            append_len: false,
        },
    }),
};

pub static RMSNORM: Shader = Shader {
    name: "RMSNorm",
    layout: &[Input, Input, Output, Meta],
    wgpu: Some(include_str!("fwd/rmsnorm.wgsl")),
    cpu: Some(cpu::rmsnorm),
    cuda: Some(CudaSpec {
        src: cuda::RMSNORM,
        entry: "rmsnorm_kernel",
        shape: CudaShape::Generic {
            meta_fields: &[MetaField::U32, MetaField::U32, MetaField::F32],
            block_dim: (256, 1, 1),
            append_len: false,
        },
    }),
};

/// No CPU implementation (pre-existing gap, see `SILU_BWD`).
pub static RMSNORM_BWD: Shader = Shader {
    name: "RMSNormBwd",
    layout: &[Input, Input, Input, Output, Output, Meta],
    wgpu: Some(include_str!("bwd/rmsnorm_bwd.wgsl")),
    cpu: None,
    cuda: Some(CudaSpec {
        src: cuda::RMSNORM_BWD,
        entry: "rmsnorm_bwd_kernel",
        shape: CudaShape::Generic {
            meta_fields: &[MetaField::U32, MetaField::U32, MetaField::F32],
            block_dim: (256, 1, 1),
            append_len: false,
        },
    }),
};

/// No CPU implementation (pre-existing gap, see `SILU_BWD`).
pub static RMSNORM_WEIGHT_BWD: Shader = Shader {
    name: "RMSNormWeightBwd",
    layout: &[Input, Input, Input, Output, Meta],
    wgpu: Some(include_str!("bwd/rmsnorm_weight_bwd.wgsl")),
    cpu: None,
    cuda: Some(CudaSpec {
        src: cuda::RMSNORM_WEIGHT_BWD,
        entry: "rmsnorm_weight_bwd_kernel",
        shape: CudaShape::Generic {
            meta_fields: &[MetaField::U32, MetaField::U32],
            block_dim: (256, 1, 1),
            append_len: false,
        },
    }),
};

pub static CROSS_ENTROPY: Shader = Shader {
    name: "CrossEntropy",
    layout: &[Input, Input, Output, Output, Meta],
    wgpu: Some(include_str!("fwd/cross_entropy.wgsl")),
    cpu: Some(cpu::cross_entropy),
    cuda: Some(CudaSpec {
        src: cuda::CROSS_ENTROPY,
        entry: "cross_entropy_kernel",
        shape: CudaShape::Generic {
            meta_fields: &[MetaField::U32, MetaField::U32],
            block_dim: (256, 1, 1),
            append_len: false,
        },
    }),
};

/// No CPU implementation (pre-existing gap, see `SILU_BWD`).
pub static CROSS_ENTROPY_BWD: Shader = Shader {
    name: "CrossEntropyBwd",
    layout: &[Input, Input, Input, Output, Meta],
    wgpu: Some(include_str!("bwd/cross_entropy_bwd.wgsl")),
    cpu: None,
    cuda: Some(CudaSpec {
        src: cuda::CROSS_ENTROPY_BWD,
        entry: "cross_entropy_bwd_kernel",
        shape: CudaShape::Generic {
            meta_fields: &[MetaField::U32, MetaField::U32],
            block_dim: (256, 1, 1),
            append_len: false,
        },
    }),
};

pub static HEAD_GATHER: Shader = Shader {
    name: "HeadGather",
    layout: &[Input, Output, Meta],
    wgpu: Some(include_str!("head_gather.wgsl")),
    cpu: Some(cpu::head_gather),
    cuda: Some(CudaSpec {
        src: cuda::HEAD_GATHER,
        entry: "head_gather_kernel",
        shape: CudaShape::Generic {
            meta_fields: &[
                MetaField::U32,
                MetaField::U32,
                MetaField::U32,
                MetaField::U32,
            ],
            block_dim: (16, 16, 1),
            append_len: false,
        },
    }),
};

pub static HEAD_SCATTER: Shader = Shader {
    name: "HeadScatter",
    layout: &[Input, Output, Meta],
    wgpu: Some(include_str!("head_scatter.wgsl")),
    cpu: Some(cpu::head_scatter),
    cuda: Some(CudaSpec {
        src: cuda::HEAD_SCATTER,
        entry: "head_scatter_kernel",
        shape: CudaShape::Generic {
            meta_fields: &[
                MetaField::U32,
                MetaField::U32,
                MetaField::U32,
                MetaField::U32,
            ],
            block_dim: (16, 16, 1),
            append_len: false,
        },
    }),
};

pub static QKV_SPLIT: Shader = Shader {
    name: "QkvSplit",
    layout: &[Input, Output, Output, Output, Meta],
    wgpu: Some(include_str!("fwd/qkv_split.wgsl")),
    cpu: None,
    cuda: Some(CudaSpec {
        src: cuda::QKV_SPLIT,
        entry: "qkv_split_kernel",
        shape: CudaShape::Generic {
            meta_fields: &[
                MetaField::U32,
                MetaField::U32,
                MetaField::U32,
                MetaField::U32,
            ],
            block_dim: (16, 16, 1),
            append_len: false,
        },
    }),
};

pub static QKV_SCATTER: Shader = Shader {
    name: "QkvScatter",
    layout: &[Input, Input, Input, Output, Meta],
    wgpu: Some(include_str!("bwd/qkv_scatter.wgsl")),
    cpu: None,
    cuda: Some(CudaSpec {
        src: cuda::QKV_SCATTER,
        entry: "qkv_scatter_kernel",
        shape: CudaShape::Generic {
            meta_fields: &[
                MetaField::U32,
                MetaField::U32,
                MetaField::U32,
                MetaField::U32,
            ],
            block_dim: (16, 16, 1),
            append_len: false,
        },
    }),
};

pub static FLASH_ATTENTION: Shader = Shader {
    name: "FlashAttention",
    layout: &[Input, Input, Input, Output, Output, Meta],
    wgpu: Some(include_str!("fwd/flash_attention.wgsl")),
    cpu: None,
    cuda: Some(CudaSpec {
        src: cuda::FLASH_ATTENTION,
        entry: "flash_attention_kernel",
        shape: CudaShape::Generic {
            meta_fields: &[
                MetaField::U32,
                MetaField::U32,
                MetaField::U32,
                MetaField::F32,
                MetaField::U32,
            ],
            block_dim: (64, 1, 1),
            append_len: false,
        },
    }),
};

pub static FLASH_ATTENTION_BWD_DQ: Shader = Shader {
    name: "FlashAttentionBwdDQ",
    layout: &[Input, Input, Input, Input, Input, Input, Output, Meta],
    wgpu: Some(include_str!("bwd/flash_attention_bwd_dq.wgsl")),
    cpu: None,
    cuda: Some(CudaSpec {
        src: cuda::FLASH_ATTENTION_BWD_DQ,
        entry: "flash_attention_bwd_dq_kernel",
        shape: CudaShape::Generic {
            meta_fields: &[
                MetaField::U32,
                MetaField::U32,
                MetaField::U32,
                MetaField::F32,
                MetaField::U32,
            ],
            block_dim: (64, 1, 1),
            append_len: false,
        },
    }),
};

pub static FLASH_ATTENTION_BWD_DKDV: Shader = Shader {
    name: "FlashAttentionBwdDKDV",
    layout: &[
        Input, Input, Input, Input, Input, Input, Output, Output, Meta,
    ],
    wgpu: Some(include_str!("bwd/flash_attention_bwd_dkdv.wgsl")),
    cpu: None,
    cuda: Some(CudaSpec {
        src: cuda::FLASH_ATTENTION_BWD_DKDV,
        entry: "flash_attention_bwd_dkdv_kernel",
        shape: CudaShape::Generic {
            meta_fields: &[
                MetaField::U32,
                MetaField::U32,
                MetaField::U32,
                MetaField::F32,
                MetaField::U32,
            ],
            block_dim: (64, 1, 1),
            append_len: false,
        },
    }),
};

pub static CACHE_WRITE: Shader = Shader {
    name: "CacheWrite",
    layout: &[Input, InOut, Meta],
    wgpu: Some(include_str!("cache_write.wgsl")),
    cpu: Some(cpu::cache_write),
    cuda: Some(CudaSpec {
        src: cuda::CACHE_WRITE,
        entry: "cache_write_kernel",
        shape: CudaShape::Generic {
            meta_fields: &[MetaField::U32, MetaField::U32, MetaField::U32],
            block_dim: (16, 16, 1),
            append_len: false,
        },
    }),
};

pub static GRAD_SUMSQ: Shader = Shader {
    name: "GradSumSq",
    layout: &[Input, Output, Meta],
    wgpu: Some(include_str!("bwd/grad_sumsq.wgsl")),
    cpu: None,
    cuda: Some(CudaSpec {
        src: cuda::GRAD_SUMSQ,
        entry: "grad_sumsq_kernel",
        shape: CudaShape::Generic {
            meta_fields: &[MetaField::U32, MetaField::U32],
            block_dim: (256, 1, 1),
            append_len: false,
        },
    }),
};

pub static GRAD_NORM_SCALE: Shader = Shader {
    name: "GradNormScale",
    layout: &[Input, Output, Meta],
    wgpu: Some(include_str!("bwd/grad_norm_scale.wgsl")),
    cpu: None,
    cuda: Some(CudaSpec {
        src: cuda::GRAD_NORM_SCALE,
        entry: "grad_norm_scale_kernel",
        shape: CudaShape::Generic {
            meta_fields: &[MetaField::U32, MetaField::F32],
            block_dim: (256, 1, 1),
            append_len: false,
        },
    }),
};

pub static GRAD_SCALE: Shader = Shader {
    name: "GradScale",
    layout: &[InOut, Input, Meta],
    wgpu: Some(include_str!("bwd/grad_scale.wgsl")),
    cpu: None,
    cuda: Some(CudaSpec {
        src: cuda::GRAD_SCALE,
        entry: "grad_scale_kernel",
        shape: CudaShape::Generic {
            meta_fields: &[MetaField::U32],
            block_dim: (256, 1, 1),
            append_len: false,
        },
    }),
};

//! akasha-core's own `Shader` constants -- kernels that used to live in
//! wilupgu but aren't universal/structural enough to belong there forever
//! (attention machinery, RoPE, RMSNorm, embedding, cross-entropy, SiLU).
//! See wilupgu's `REFACTOR.md` for the full rationale.

mod cpu;
#[cfg(feature = "cuda")]
mod cuda;

use wilupgu::TensorMode::{InOut, Input, Meta, Output};
use wilupgu::Shader;
#[cfg(feature = "cuda")]
use wilupgu::{CudaShape, CudaSpec};

pub static EMBEDDING: Shader = Shader {
    name: "Embedding",
    layout: &[Input, Input, Output, Meta],
    wgpu: Some(include_str!("fwd/embedding.wgsl")),
    cpu: Some(cpu::embedding),
    #[cfg(feature = "cuda")]
    cuda: Some(CudaSpec {
        src: "",
        entry: "",
        shape: CudaShape::Custom(cuda::embedding),
    }),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

pub static EMBEDDING_BWD: Shader = Shader {
    name: "EmbeddingBwd",
    layout: &[Input, Input, Output, Meta],
    wgpu: Some(include_str!("bwd/embedding_bwd.wgsl")),
    cpu: Some(cpu::embedding_bwd),
    #[cfg(feature = "cuda")]
    cuda: Some(CudaSpec {
        src: "",
        entry: "",
        shape: CudaShape::Custom(cuda::embedding_bwd),
    }),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

pub static SILU: Shader = Shader {
    name: "SiLU",
    layout: &[InOut],
    wgpu: Some(include_str!("fwd/silu.wgsl")),
    cpu: Some(cpu::silu),
    #[cfg(feature = "cuda")]
    cuda: Some(CudaSpec {
        src: cuda::SILU,
        entry: "silu_kernel",
        shape: CudaShape::InOut1,
    }),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// No CPU implementation -- pre-existing gap inherited from wilupgu's old
/// string-keyed dispatch (the CPU backend never had a `SiLUBwd` match arm).
pub static SILU_BWD: Shader = Shader {
    name: "SiLUBwd",
    layout: &[Input, Input, Output],
    wgpu: Some(include_str!("bwd/silu_bwd.wgsl")),
    cpu: None,
    #[cfg(feature = "cuda")]
    cuda: Some(CudaSpec {
        src: cuda::SILU_BWD,
        entry: "silu_bwd_kernel",
        shape: CudaShape::In2Out1,
    }),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

pub static ROPE: Shader = Shader {
    name: "RoPE",
    layout: &[InOut, Meta],
    wgpu: Some(include_str!("fwd/rope.wgsl")),
    cpu: Some(cpu::rope),
    #[cfg(feature = "cuda")]
    cuda: Some(CudaSpec {
        src: "",
        entry: "",
        shape: CudaShape::Custom(cuda::rope),
    }),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// No CPU implementation (pre-existing gap, see `SILU_BWD`).
pub static ROPE_BWD: Shader = Shader {
    name: "RoPEBwd",
    layout: &[InOut, Meta],
    wgpu: Some(include_str!("bwd/rope_bwd.wgsl")),
    cpu: None,
    #[cfg(feature = "cuda")]
    cuda: Some(CudaSpec {
        src: "",
        entry: "",
        shape: CudaShape::Custom(cuda::rope_bwd),
    }),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

pub static ROPE_OFFSET: Shader = Shader {
    name: "RoPEOffset",
    layout: &[InOut, Meta],
    wgpu: Some(include_str!("fwd/rope_offset.wgsl")),
    cpu: Some(cpu::rope_offset),
    #[cfg(feature = "cuda")]
    cuda: Some(CudaSpec {
        src: "",
        entry: "",
        shape: CudaShape::Custom(cuda::rope_offset),
    }),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

pub static SOFTMAX: Shader = Shader {
    name: "Softmax",
    layout: &[InOut, Meta],
    wgpu: Some(include_str!("fwd/softmax.wgsl")),
    cpu: Some(cpu::softmax),
    #[cfg(feature = "cuda")]
    cuda: Some(CudaSpec {
        src: "",
        entry: "",
        shape: CudaShape::Custom(cuda::softmax),
    }),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// No CPU implementation (pre-existing gap, see `SILU_BWD`).
pub static SOFTMAX_BWD: Shader = Shader {
    name: "SoftmaxBwd",
    layout: &[Input, Input, Output, Meta],
    wgpu: Some(include_str!("bwd/softmax_bwd.wgsl")),
    cpu: None,
    #[cfg(feature = "cuda")]
    cuda: Some(CudaSpec {
        src: "",
        entry: "",
        shape: CudaShape::Custom(cuda::softmax_bwd),
    }),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

pub static SOFTMAX_RECT: Shader = Shader {
    name: "SoftmaxRect",
    layout: &[InOut, Meta],
    wgpu: Some(include_str!("fwd/softmax_rect.wgsl")),
    cpu: Some(cpu::softmax_rect),
    #[cfg(feature = "cuda")]
    cuda: Some(CudaSpec {
        src: "",
        entry: "",
        shape: CudaShape::Custom(cuda::softmax_rect),
    }),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

pub static CAUSAL_SOFTMAX: Shader = Shader {
    name: "CausalSoftmax",
    layout: &[InOut, Meta],
    wgpu: Some(include_str!("fwd/causal_softmax.wgsl")),
    cpu: Some(cpu::causal_softmax),
    #[cfg(feature = "cuda")]
    cuda: Some(CudaSpec {
        src: "",
        entry: "",
        shape: CudaShape::Custom(cuda::causal_softmax),
    }),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

pub static RMSNORM: Shader = Shader {
    name: "RMSNorm",
    layout: &[Input, Input, Output, Meta],
    wgpu: Some(include_str!("fwd/rmsnorm.wgsl")),
    cpu: Some(cpu::rmsnorm),
    #[cfg(feature = "cuda")]
    cuda: Some(CudaSpec {
        src: "",
        entry: "",
        shape: CudaShape::Custom(cuda::rmsnorm),
    }),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// No CPU implementation (pre-existing gap, see `SILU_BWD`).
pub static RMSNORM_BWD: Shader = Shader {
    name: "RMSNormBwd",
    layout: &[Input, Input, Input, Output, Output, Meta],
    wgpu: Some(include_str!("bwd/rmsnorm_bwd.wgsl")),
    cpu: None,
    #[cfg(feature = "cuda")]
    cuda: Some(CudaSpec {
        src: "",
        entry: "",
        shape: CudaShape::Custom(cuda::rmsnorm_bwd),
    }),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// No CPU implementation (pre-existing gap, see `SILU_BWD`).
pub static RMSNORM_WEIGHT_BWD: Shader = Shader {
    name: "RMSNormWeightBwd",
    layout: &[Input, Input, Input, Output, Meta],
    wgpu: Some(include_str!("bwd/rmsnorm_weight_bwd.wgsl")),
    cpu: None,
    #[cfg(feature = "cuda")]
    cuda: Some(CudaSpec {
        src: "",
        entry: "",
        shape: CudaShape::Custom(cuda::rmsnorm_weight_bwd),
    }),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

pub static CROSS_ENTROPY: Shader = Shader {
    name: "CrossEntropy",
    layout: &[Input, Input, Output, Output, Meta],
    wgpu: Some(include_str!("fwd/cross_entropy.wgsl")),
    cpu: Some(cpu::cross_entropy),
    #[cfg(feature = "cuda")]
    cuda: Some(CudaSpec {
        src: "",
        entry: "",
        shape: CudaShape::Custom(cuda::cross_entropy),
    }),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// No CPU implementation (pre-existing gap, see `SILU_BWD`).
pub static CROSS_ENTROPY_BWD: Shader = Shader {
    name: "CrossEntropyBwd",
    layout: &[Input, Input, Input, Output, Meta],
    wgpu: Some(include_str!("bwd/cross_entropy_bwd.wgsl")),
    cpu: None,
    #[cfg(feature = "cuda")]
    cuda: Some(CudaSpec {
        src: "",
        entry: "",
        shape: CudaShape::Custom(cuda::cross_entropy_bwd),
    }),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

pub static HEAD_GATHER: Shader = Shader {
    name: "HeadGather",
    layout: &[Input, Output, Meta],
    wgpu: Some(include_str!("head_gather.wgsl")),
    cpu: Some(cpu::head_gather),
    #[cfg(feature = "cuda")]
    cuda: Some(CudaSpec {
        src: "",
        entry: "",
        shape: CudaShape::Custom(cuda::head_gather),
    }),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

pub static HEAD_SCATTER: Shader = Shader {
    name: "HeadScatter",
    layout: &[Input, Output, Meta],
    wgpu: Some(include_str!("head_scatter.wgsl")),
    cpu: Some(cpu::head_scatter),
    #[cfg(feature = "cuda")]
    cuda: Some(CudaSpec {
        src: "",
        entry: "",
        shape: CudaShape::Custom(cuda::head_scatter),
    }),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

pub static CACHE_WRITE: Shader = Shader {
    name: "CacheWrite",
    layout: &[Input, InOut, Meta],
    wgpu: Some(include_str!("cache_write.wgsl")),
    cpu: Some(cpu::cache_write),
    #[cfg(feature = "cuda")]
    cuda: Some(CudaSpec {
        src: "",
        entry: "",
        shape: CudaShape::Custom(cuda::cache_write),
    }),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

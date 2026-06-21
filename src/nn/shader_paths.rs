pub const ADD_INPLACE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src/shaders/add_inplace.spv");
pub const CAUSAL_MASK: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src/shaders/causal_mask.spv");
pub const EMBEDDING: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src/shaders/embedding.spv");
pub const MATMUL: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src/shaders/matmul.spv");
pub const MATMUL_BWD_INPUT_TRP_B: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/shaders/matmul_bwd_input_trp_b.spv"
);
pub const MATMUL_BWD_WEIGHT_TRP_A: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/shaders/matmul_bwd_weight_trp_a.spv"
);
pub const MATMUL_TRP: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src/shaders/matmul_trp.spv");
pub const RMSNORM: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src/shaders/rmsnorm.spv");
pub const RMSNORM_BWD: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src/shaders/rmsnorm_bwd.spv");
pub const ROPE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src/shaders/rope.spv");
pub const ROPE_BWD: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src/shaders/rope_bwd.spv");
pub const SILU: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src/shaders/silu.spv");
pub const SILU_BWD: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src/shaders/silu_bwd.spv");
pub const SOFTMAX: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src/shaders/softmax.spv");
pub const SOFTMAX_BWD: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src/shaders/softmax_bwd.spv");

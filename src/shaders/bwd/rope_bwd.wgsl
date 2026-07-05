struct Meta {
    seq_len: u32,
    dim: u32,
    head_dim: u32,
}

@group(0) @binding(0) var<storage, read_write> d_vec: array<f32>;
@group(0) @binding(1) var<storage, read> config: Meta;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let token_idx = global_id.y;
    let dim_idx = global_id.x * 2u;

    if (token_idx >= config.seq_len || dim_idx >= config.head_dim) { return; }

    let num_heads = config.dim / config.head_dim;
    for (var h: u32 = 0u; h < num_heads; h = h + 1u) {
        let offset = token_idx * config.dim + h * config.head_dim + dim_idx;
        
        let dx0 = d_vec[offset];
        let dx1 = d_vec[offset + 1u];
        
        let freq = 1.0 / pow(10000.0, f32(dim_idx) / f32(config.head_dim));
        let v_angle = f32(token_idx) * freq;
        let v_cos = cos(v_angle);
        let v_sin = sin(v_angle);
        
        d_vec[offset]       = dx0 * v_cos + dx1 * v_sin;
        d_vec[offset + 1u]  = -dx0 * v_sin + dx1 * v_cos;
    }
}

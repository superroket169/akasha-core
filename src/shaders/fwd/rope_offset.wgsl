struct Meta {
    seq_len: u32,
    dim: u32,
    head_dim: u32,
    pos_offset: u32,
}

@group(0) @binding(0) var<storage, read_write> vec: array<f32>;
@group(0) @binding(1) var<storage, read> m: Meta;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let token_idx = global_id.y;
    let dim_idx = global_id.x * 2u;

    if (token_idx >= m.seq_len || dim_idx >= m.head_dim) {
        return;
    }

    let num_heads = m.dim / m.head_dim;
    let abs_pos = token_idx + m.pos_offset;

    for (var h: u32 = 0u; h < num_heads; h = h + 1u) {
        let offset = token_idx * m.dim + h * m.head_dim + dim_idx;

        let x0 = vec[offset];
        let x1 = vec[offset + 1u];

        let freq = 1.0 / pow(10000.0, f32(dim_idx) / f32(m.head_dim));
        let v_angle = f32(abs_pos) * freq;

        let v_cos = cos(v_angle);
        let v_sin = sin(v_angle);

        vec[offset]      = x0 * v_cos - x1 * v_sin;
        vec[offset + 1u] = x0 * v_sin + x1 * v_cos;
    }
}

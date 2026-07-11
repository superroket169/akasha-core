struct Meta {
    seq_len: u32,
    dim: u32,
    head_dim: u32,
    row_offset: u32,
}

@group(0) @binding(0) var<storage, read_write> q: array<f32>;
@group(0) @binding(1) var<storage, read_write> k: array<f32>;
@group(0) @binding(2) var<storage, read> m: Meta;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let token_idx = global_id.y;
    let dim_idx = global_id.x * 2u;

    if (token_idx >= m.seq_len || dim_idx >= m.head_dim) {
        return;
    }

    // grid.z spans the heads
    let h = global_id.z;
    let row = m.row_offset + token_idx;
    let offset = row * m.dim + h * m.head_dim + dim_idx;

    let freq = 1.0 / pow(10000.0, f32(dim_idx) / f32(m.head_dim));
    let v_angle = f32(token_idx) * freq;
    let v_cos = cos(v_angle);
    let v_sin = sin(v_angle);

    let q0 = q[offset];
    let q1 = q[offset + 1u];
    q[offset]      = q0 * v_cos - q1 * v_sin;
    q[offset + 1u] = q0 * v_sin + q1 * v_cos;

    let k0 = k[offset];
    let k1 = k[offset + 1u];
    k[offset]      = k0 * v_cos - k1 * v_sin;
    k[offset + 1u] = k0 * v_sin + k1 * v_cos;
}

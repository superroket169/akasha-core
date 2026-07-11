struct Meta {
    seq_len: u32,
    dim: u32,
    head_dim: u32,
    row_offset: u32,
}

@group(0) @binding(0) var<storage, read_write> d_q: array<f32>;
@group(0) @binding(1) var<storage, read_write> d_k: array<f32>;
@group(0) @binding(2) var<storage, read> m: Meta;

// Fused backward of rope_qk.wgsl -- same angle recompute, both gradients in
// one dispatch.
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

    let dq0 = d_q[offset];
    let dq1 = d_q[offset + 1u];
    d_q[offset]      = dq0 * v_cos + dq1 * v_sin;
    d_q[offset + 1u] = -dq0 * v_sin + dq1 * v_cos;

    let dk0 = d_k[offset];
    let dk1 = d_k[offset + 1u];
    d_k[offset]      = dk0 * v_cos + dk1 * v_sin;
    d_k[offset + 1u] = -dk0 * v_sin + dk1 * v_cos;
}

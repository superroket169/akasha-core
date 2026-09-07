struct Meta {
    seq_len: u32,
    full_dim: u32,
    head_dim: u32,
    head_offset: u32,
}

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;
@group(0) @binding(4) var<storage, read> config: Meta;

// Fuses 3x HeadScatter
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let col = global_id.x;
    let row = global_id.y;
    if (row >= config.seq_len || col >= config.head_dim) {
        return;
    }

    let width = config.head_dim;
    let src_idx = row * width + col;
    let dst_row = row * config.full_dim;

    dst[dst_row + col] = q[src_idx];
    dst[dst_row + width + col] = k[src_idx];
    dst[dst_row + 2u * width + col] = v[src_idx];
}

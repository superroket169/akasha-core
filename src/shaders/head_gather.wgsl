struct Meta {
    seq_len: u32,
    full_dim: u32,
    head_dim: u32,
    head_offset: u32,
}

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> config: Meta;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let col = global_id.x;
    let row = global_id.y;
    if (row >= config.seq_len || col >= config.head_dim) {
        return;
    }
    let src_idx = row * config.full_dim + config.head_offset + col;
    let dst_idx = row * config.head_dim + col;
    dst[dst_idx] = src[src_idx];
}

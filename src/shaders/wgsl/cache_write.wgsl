struct Meta {
    row_count: u32,
    width: u32,
    dst_row_offset: u32,
}

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> config: Meta;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let col = global_id.x;
    let row = global_id.y;
    if (row >= config.row_count || col >= config.width) {
        return;
    }
    let src_idx = row * config.width + col;
    let dst_idx = (config.dst_row_offset + row) * config.width + col;
    dst[dst_idx] = src[src_idx];
}

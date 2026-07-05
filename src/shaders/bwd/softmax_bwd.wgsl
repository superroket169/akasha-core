struct Meta { seq_len: u32, scale: f32, }

@group(0) @binding(0) var<storage, read> Y: array<f32>;
@group(0) @binding(1) var<storage, read> dY: array<f32>;
@group(0) @binding(2) var<storage, read_write> dX: array<f32>;
@group(0) @binding(3) var<storage, read> config: Meta;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    if (row >= config.seq_len) {
        return;
    }
    let offset = row * config.seq_len;

    var sum_ydy: f32 = 0.0;
    for (var i: u32 = 0u; i < config.seq_len; i = i + 1u) {
        sum_ydy = sum_ydy + (Y[offset + i] * dY[offset + i]);
    }

    for (var i: u32 = 0u; i < config.seq_len; i = i + 1u) {
        dX[offset + i] = Y[offset + i] * (dY[offset + i] - sum_ydy) * config.scale;
    }
}

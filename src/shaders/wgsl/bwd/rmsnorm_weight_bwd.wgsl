struct Meta { seq_len: u32, size: u32, eps: f32, }

@group(0) @binding(0) var<storage, read> dY: array<f32>;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> rsqrt_cache: array<f32>;
@group(0) @binding(3) var<storage, read_write> dWeight: array<f32>;
@group(0) @binding(4) var<storage, read> config: Meta;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= config.size) {
        return;
    }

    var acc: f32 = 0.0;
    for (var row: u32 = 0u; row < config.seq_len; row = row + 1u) {
        let offset = row * config.size;
        let norm_x = X[offset + i] * rsqrt_cache[row];
        acc = acc + (dY[offset + i] * norm_x);
    }

    dWeight[i] = dWeight[i] + acc;
}

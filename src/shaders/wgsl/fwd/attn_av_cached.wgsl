struct Meta {
    attn_len: u32,
    dim: u32,
    head_dim: u32,
}

@group(0) @binding(0) var<storage, read> scores: array<f32>;
@group(0) @binding(1) var<storage, read> v_cache: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<storage, read> config: Meta;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let d = global_id.x;

    if (d >= config.dim) {
        return;
    }
    
    let s_off = (d / config.head_dim) * config.attn_len;
    var sum: f32 = 0.0;
    
    for (var j: u32 = 0u; j < config.attn_len; j = j + 1u) {
        sum = sum + scores[s_off + j] * v_cache[j * config.dim + d];
    }

    out[d] = sum;
}

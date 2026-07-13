struct Meta {
    attn_len: u32,
    dim: u32,
    head_dim: u32,
}

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k_cache: array<f32>;
@group(0) @binding(2) var<storage, read_write> scores: array<f32>;
@group(0) @binding(3) var<storage, read> config: Meta;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let j = global_id.x;
    let h = global_id.y;

    if (j >= config.attn_len) {
        return;
    }
    
    let q_off = h * config.head_dim;
    let k_off = j * config.dim + q_off;
    var sum: f32 = 0.0;
    
    for (var c: u32 = 0u; c < config.head_dim; c = c + 1u) {
        sum = sum + q[q_off + c] * k_cache[k_off + c];
    }
    scores[h * config.attn_len + j] = sum;
}

struct Meta {
    vocab_size: u32,
    embed_dim: u32,
    seq_len: u32,
}

@group(0) @binding(0) var<storage, read> tokens: array<u32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read> config: Meta;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let token_idx = global_id.y;
    let dim_idx = global_id.x;

    if (token_idx >= config.seq_len || dim_idx >= config.embed_dim) {
        return;
    }

    let token_id = tokens[token_idx];
    if (token_id < config.vocab_size) {
        let weight_idx = token_id * config.embed_dim + dim_idx;
        let out_idx = token_idx * config.embed_dim + dim_idx;
        output[out_idx] = weight[weight_idx];
    }
}

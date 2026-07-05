struct Meta {
    vocab_size: u32,
    embed_dim: u32,
    seq_len: u32,
}

@group(0) @binding(0) var<storage, read> tokens: array<u32>;
@group(0) @binding(1) var<storage, read> grad_output: array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_table: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> config: Meta;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let token_idx = global_id.y;
    let dim_idx = global_id.x;

    if (token_idx >= config.seq_len || dim_idx >= config.embed_dim) {
        return;
    }

    let token_id = tokens[token_idx];
    if (token_id >= config.vocab_size) {
        return;
    }

    let target_idx = token_id * config.embed_dim + dim_idx;
    let grad_val = grad_output[token_idx * config.embed_dim + dim_idx];

    // CAS loop
    var old_bits = atomicLoad(&grad_table[target_idx]);
    loop {
        let new_bits = bitcast<u32>(bitcast<f32>(old_bits) + grad_val);
        let result = atomicCompareExchangeWeak(&grad_table[target_idx], old_bits, new_bits);
        if (result.exchanged) {
            break;
        }
        old_bits = result.old_value;
    }
}

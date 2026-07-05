struct Meta {
    vocab_size: u32,
    num_rows: u32,
}

@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read> targets: array<u32>;
@group(0) @binding(2) var<storage, read_write> probs: array<f32>;
@group(0) @binding(3) var<storage, read_write> losses: array<f32>;
@group(0) @binding(4) var<storage, read> m: Meta;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    if (row >= m.num_rows) {
        return;
    }
    let offset = row * m.vocab_size;
    let target_id = targets[row];

    var max_val: f32 = -3.4028235e38;
    for (var i: u32 = 0u; i < m.vocab_size; i = i + 1u) {
        max_val = max(max_val, logits[offset + i]);
    }

    var sum_exp: f32 = 0.0;
    for (var i: u32 = 0u; i < m.vocab_size; i = i + 1u) {
        let e = exp(logits[offset + i] - max_val);
        probs[offset + i] = e;
        sum_exp = sum_exp + e;
    }

    let inv_sum = 1.0 / sum_exp;
    for (var i: u32 = 0u; i < m.vocab_size; i = i + 1u) {
        probs[offset + i] = probs[offset + i] * inv_sum;
    }

    let log_sum_exp = log(sum_exp);
    losses[row] = -(logits[offset + target_id] - max_val - log_sum_exp);
}

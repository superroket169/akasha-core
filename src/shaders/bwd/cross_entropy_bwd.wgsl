struct Meta {
    vocab_size: u32,
    num_rows: u32,
}

@group(0) @binding(0) var<storage, read> probs: array<f32>;
@group(0) @binding(1) var<storage, read> targets: array<u32>;
@group(0) @binding(2) var<storage, read> d_losses: array<f32>;
@group(0) @binding(3) var<storage, read_write> d_logits: array<f32>;
@group(0) @binding(4) var<storage, read> m: Meta;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    if (row >= m.num_rows) {
        return;
    }
    let offset = row * m.vocab_size;
    let target_id = targets[row];
    let grad_scale = d_losses[row];

    for (var i: u32 = 0u; i < m.vocab_size; i = i + 1u) {
        let indicator = select(0.0, 1.0, i == target_id);
        d_logits[offset + i] = (probs[offset + i] - indicator) * grad_scale;
    }
}

// grad = (p - onehot(target)) * d_loss, one thread per element
struct Meta {
    vocab_size: u32,
    num_rows: u32,
}

@group(0) @binding(0) var<storage, read_write> probs: array<f32>;
@group(0) @binding(1) var<storage, read> targets: array<u32>;
@group(0) @binding(2) var<storage, read> d_losses: array<f32>;
@group(0) @binding(3) var<storage, read> m: Meta;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;
    if (col >= m.vocab_size || row >= m.num_rows) {
        return;
    }

    let idx = row * m.vocab_size + col;
    var g = probs[idx];
    if (col == targets[row]) {
        g = g - 1.0;
    }
    
    probs[idx] = g * d_losses[row];
}

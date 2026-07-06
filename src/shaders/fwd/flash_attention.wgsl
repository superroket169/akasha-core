struct Meta {
    seq_len: u32,
    dim: u32,
    head_dim: u32,
    scale: f32,
}

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<storage, read_write> l_cache: array<f32>;
@group(0) @binding(5) var<storage, read> m: Meta;

// Fixed accumulator size -- caller asserts head_dim <= 128.
const MAX_HEAD_DIM: u32 = 128u;

// One invocation per (query row, head): fused QK^T + causal softmax + AV,
// online-softmax so the `[seq_len, seq_len]` scores matrix is never
// materialized. `row_max`/`row_sum` follow the standard running-softmax
// recurrence; `l_cache` saves `row_max + log(row_sum)` for the backward pass.
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let head = global_id.y;
    let num_heads = m.dim / m.head_dim;

    if (row >= m.seq_len || head >= num_heads) {
        return;
    }

    let head_off = head * m.head_dim;
    let q_off = row * m.dim + head_off;

    var acc: array<f32, MAX_HEAD_DIM>;
    for (var d: u32 = 0u; d < m.head_dim; d = d + 1u) {
        acc[d] = 0.0;
    }

    var row_max: f32 = -1000000000.0;
    var row_sum: f32 = 0.0;

    for (var j: u32 = 0u; j <= row; j = j + 1u) {
        let kv_off = j * m.dim + head_off;

        var score: f32 = 0.0;
        for (var d: u32 = 0u; d < m.head_dim; d = d + 1u) {
            score = score + q[q_off + d] * k[kv_off + d];
        }
        score = score * m.scale;

        let new_max = max(row_max, score);
        let correction = exp(row_max - new_max);
        let p = exp(score - new_max);

        row_sum = row_sum * correction + p;
        for (var d: u32 = 0u; d < m.head_dim; d = d + 1u) {
            acc[d] = acc[d] * correction + p * v[kv_off + d];
        }

        row_max = new_max;
    }

    let out_off = row * m.dim + head_off;
    for (var d: u32 = 0u; d < m.head_dim; d = d + 1u) {
        out[out_off + d] = acc[d] / row_sum;
    }

    l_cache[row * num_heads + head] = row_max + log(row_sum);
}

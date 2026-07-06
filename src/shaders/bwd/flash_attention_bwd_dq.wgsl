struct Meta {
    seq_len: u32,
    dim: u32,
    head_dim: u32,
    scale: f32,
    row_offset: u32,
}

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read> o: array<f32>;
@group(0) @binding(4) var<storage, read> d_o: array<f32>;
@group(0) @binding(5) var<storage, read> l_cache: array<f32>;
@group(0) @binding(6) var<storage, read_write> d_q: array<f32>;
@group(0) @binding(7) var<storage, read> m: Meta;

const MAX_HEAD_DIM: u32 = 128u;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let head = global_id.y;
    let num_heads = m.dim / m.head_dim;

    if (row >= m.seq_len || head >= num_heads) {
        return;
    }

    let head_off = head * m.head_dim;
    let q_off = (m.row_offset + row) * m.dim + head_off;
    let o_off = (m.row_offset + row) * m.dim + head_off;
    let l_i = l_cache[row * num_heads + head];

    var d_i: f32 = 0.0;
    for (var d: u32 = 0u; d < m.head_dim; d = d + 1u) {
        d_i = d_i + d_o[o_off + d] * o[o_off + d];
    }

    var dq_acc: array<f32, MAX_HEAD_DIM>;
    for (var d: u32 = 0u; d < m.head_dim; d = d + 1u) {
        dq_acc[d] = 0.0;
    }

    for (var j: u32 = 0u; j <= row; j = j + 1u) {
        let kv_off = (m.row_offset + j) * m.dim + head_off;

        var score: f32 = 0.0;
        for (var d: u32 = 0u; d < m.head_dim; d = d + 1u) {
            score = score + q[q_off + d] * k[kv_off + d];
        }
        score = score * m.scale;
        let p = exp(score - l_i);

        var dp: f32 = 0.0;
        for (var d: u32 = 0u; d < m.head_dim; d = d + 1u) {
            dp = dp + d_o[o_off + d] * v[kv_off + d];
        }
        let d_s = p * (dp - d_i);

        for (var d: u32 = 0u; d < m.head_dim; d = d + 1u) {
            dq_acc[d] = dq_acc[d] + d_s * k[kv_off + d];
        }
    }

    let dq_off = (m.row_offset + row) * m.dim + head_off;
    for (var d: u32 = 0u; d < m.head_dim; d = d + 1u) {
        d_q[dq_off + d] = dq_acc[d] * m.scale;
    }
}

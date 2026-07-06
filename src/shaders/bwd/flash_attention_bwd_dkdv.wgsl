struct Meta {
    seq_len: u32,
    dim: u32,
    head_dim: u32,
    scale: f32,
}

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read> o: array<f32>;
@group(0) @binding(4) var<storage, read> d_o: array<f32>;
@group(0) @binding(5) var<storage, read> l_cache: array<f32>;
@group(0) @binding(6) var<storage, read_write> d_k: array<f32>;
@group(0) @binding(7) var<storage, read_write> d_v: array<f32>;
@group(0) @binding(8) var<storage, read> m: Meta;

const MAX_HEAD_DIM: u32 = 128u;

//   dV_j = sum_i P_ij * dO_i
//   dK_j = scale * sum_i dS_ij * Q_i
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let col = global_id.x;
    let head = global_id.y;
    let num_heads = m.dim / m.head_dim;

    if (col >= m.seq_len || head >= num_heads) {
        return;
    }

    let head_off = head * m.head_dim;
    let kv_off = col * m.dim + head_off;

    var dk_acc: array<f32, MAX_HEAD_DIM>;
    var dv_acc: array<f32, MAX_HEAD_DIM>;
    for (var d: u32 = 0u; d < m.head_dim; d = d + 1u) {
        dk_acc[d] = 0.0;
        dv_acc[d] = 0.0;
    }

    for (var i: u32 = col; i < m.seq_len; i = i + 1u) {
        let qo_off = i * m.dim + head_off;
        let l_i = l_cache[i * num_heads + head];

        var score: f32 = 0.0;
        for (var d: u32 = 0u; d < m.head_dim; d = d + 1u) {
            score = score + q[qo_off + d] * k[kv_off + d];
        }
        score = score * m.scale;
        let p = exp(score - l_i);

        var d_i: f32 = 0.0;
        var dp: f32 = 0.0;
        for (var d: u32 = 0u; d < m.head_dim; d = d + 1u) {
            d_i = d_i + d_o[qo_off + d] * o[qo_off + d];
            dp = dp + d_o[qo_off + d] * v[kv_off + d];
        }
        let d_s = p * (dp - d_i);

        for (var d: u32 = 0u; d < m.head_dim; d = d + 1u) {
            dv_acc[d] = dv_acc[d] + p * d_o[qo_off + d];
            dk_acc[d] = dk_acc[d] + d_s * q[qo_off + d];
        }
    }

    for (var d: u32 = 0u; d < m.head_dim; d = d + 1u) {
        d_k[kv_off + d] = dk_acc[d] * m.scale;
        d_v[kv_off + d] = dv_acc[d];
    }
}

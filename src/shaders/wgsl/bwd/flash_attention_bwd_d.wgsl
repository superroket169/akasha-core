struct Meta {
    seq_len: u32,
    dim: u32,
    head_dim: u32,
    scale: f32,
    row_offset: u32,
}

@group(0) @binding(0) var<storage, read> d_o: array<f32>;
@group(0) @binding(1) var<storage, read> o: array<f32>;
@group(0) @binding(2) var<storage, read_write> d_sum: array<f32>;
@group(0) @binding(3) var<storage, read> m: Meta;

const HEAD_DIM: u32 = 64u;

// D[i] = sum_d dO_i[d] * O_i[d], precomputed once per row so
// flash_attention_bwd_dq/dkdv can look it up instead of recomputing it
// once per (row, col) pair they visit.
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let head = global_id.y;
    let num_heads = m.dim / m.head_dim;

    if (row >= m.seq_len || head >= num_heads) {
        return;
    }

    let off = (m.row_offset + row) * m.dim + head * m.head_dim;

    var d_i: f32 = 0.0;
    for (var d: u32 = 0u; d < HEAD_DIM; d = d + 1u) {
        d_i = d_i + d_o[off + d] * o[off + d];
    }

    d_sum[row * num_heads + head] = d_i;
}

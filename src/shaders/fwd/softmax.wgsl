struct Meta {
    seq_len: u32,
}

@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> m: Meta;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    if (row >= m.seq_len) {
        return;
    }
    let offset = row * m.seq_len;

    var max_val: f32 = -1000000.0;
    for (var i: u32 = 0u; i < m.seq_len; i = i + 1u) {
        let val = x[offset + i];
        if (val > max_val) {
            max_val = val;
        }
    }

    var sum_exp: f32 = 0.0;
    for (var i: u32 = 0u; i < m.seq_len; i = i + 1u) {
        let e = exp(x[offset + i] - max_val);
        x[offset + i] = e;
        sum_exp = sum_exp + e;
    }

    for (var i: u32 = 0u; i < m.seq_len; i = i + 1u) {
        x[offset + i] = x[offset + i] / sum_exp;
    }
}

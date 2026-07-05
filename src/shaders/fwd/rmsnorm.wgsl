struct Meta {
    seq_len: u32,
    size: u32,
    eps: f32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read> m: Meta;

var<workgroup> partial: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let row = wg_id.x;
    if (row >= m.seq_len) {
        return;
    }
    let offset = row * m.size;
    let tid = local_id.x;

    var local_ss: f32 = 0.0;
    var i: u32 = tid;
    while (i < m.size) {
        let val = x[offset + i];
        local_ss = local_ss + (val * val);
        i = i + 256u;
    }
    partial[tid] = local_ss;
    workgroupBarrier();

    var stride: u32 = 128u;
    while (stride > 0u) {
        if (tid < stride) {
            partial[tid] = partial[tid] + partial[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    let rsqrt = 1.0 / sqrt((partial[0] / f32(m.size)) + m.eps);

    i = tid;
    while (i < m.size) {
        output[offset + i] = x[offset + i] * rsqrt * weight[i];
        i = i + 256u;
    }
}

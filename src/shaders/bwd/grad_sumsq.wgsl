struct Meta {
    len: u32,
    out_offset: u32,
}

@group(0) @binding(0) var<storage, read> grad: array<f32>;
@group(0) @binding(1) var<storage, read_write> partials: array<f32>;
@group(0) @binding(2) var<storage, read> m: Meta;

var<workgroup> partial: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let tid = local_id.x;
    let stride = num_wg.x * 256u;

    var acc: f32 = 0.0;
    var i: u32 = wg_id.x * 256u + tid;
    while (i < m.len) {
        let v = grad[i];
        acc = acc + v * v;
        i = i + stride;
    }
    partial[tid] = acc;
    workgroupBarrier();

    var s: u32 = 128u;
    while (s > 0u) {
        if (tid < s) {
            partial[tid] = partial[tid] + partial[tid + s];
        }
        workgroupBarrier();
        s = s / 2u;
    }

    if (tid == 0u) {
        partials[m.out_offset + wg_id.x] = partial[0];
    }
}

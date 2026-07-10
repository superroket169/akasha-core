struct Meta {
    num_partials: u32,
    max_norm: f32,
}

@group(0) @binding(0) var<storage, read> partials: array<f32>;
@group(0) @binding(1) var<storage, read_write> scale: array<f32>;
@group(0) @binding(2) var<storage, read> m: Meta;

var<workgroup> partial: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(local_invocation_id) local_id: vec3<u32>) {
    let tid = local_id.x;

    var acc: f32 = 0.0;
    var i: u32 = tid;
    while (i < m.num_partials) {
        acc = acc + partials[i];
        i = i + 256u;
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
        let norm = sqrt(partial[0]);
        scale[0] = select(1.0, m.max_norm / (norm + 1e-6), norm > m.max_norm);
    }
}

struct Meta { seq_len: u32, size: u32, eps: f32, }

@group(0) @binding(0) var<storage, read> dY: array<f32>;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> Weight: array<f32>;
@group(0) @binding(3) var<storage, read_write> dX: array<f32>;
@group(0) @binding(4) var<storage, read_write> rsqrt_cache: array<f32>;
@group(0) @binding(5) var<storage, read> config: Meta;

var<workgroup> partial: array<f32, 256>;

fn reduce(tid: u32) -> f32 {
    workgroupBarrier();
    var stride: u32 = 128u;
    while (stride > 0u) {
        if (tid < stride) {
            partial[tid] = partial[tid] + partial[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    return partial[0];
}

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let row = wg_id.x;
    if (row >= config.seq_len) {
        return;
    }
    let offset = row * config.size;
    let tid = local_id.x;

    var local_ss: f32 = 0.0;
    var i: u32 = tid;
    while (i < config.size) {
        local_ss = local_ss + (X[offset + i] * X[offset + i]);
        i = i + 256u;
    }
    partial[tid] = local_ss;
    let ss = reduce(tid);

    let rsqrt = 1.0 / sqrt((ss / f32(config.size)) + config.eps);
    if (tid == 0u) {
        rsqrt_cache[row] = rsqrt;
    }

    var local_sum_grad: f32 = 0.0;
    i = tid;
    while (i < config.size) {
        let norm_x = X[offset + i] * rsqrt;
        let dy_w = dY[offset + i] * Weight[i];
        local_sum_grad = local_sum_grad + (dy_w * norm_x);
        i = i + 256u;
    }
    partial[tid] = local_sum_grad;
    let sum_grad = reduce(tid);

    i = tid;
    while (i < config.size) {
        let norm_x = X[offset + i] * rsqrt;
        let dy_w = dY[offset + i] * Weight[i];
        dX[offset + i] = rsqrt * (dy_w - (norm_x * sum_grad / f32(config.size)));
        i = i + 256u;
    }
}

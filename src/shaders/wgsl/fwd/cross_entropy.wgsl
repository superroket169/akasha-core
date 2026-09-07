struct Meta {
    vocab_size: u32,
    num_rows: u32,
}

@group(0) @binding(0) var<storage, read_write> logits: array<f32>;
@group(0) @binding(1) var<storage, read> targets: array<u32>;
@group(0) @binding(2) var<storage, read_write> losses: array<f32>;
@group(0) @binding(3) var<storage, read> m: Meta;

var<workgroup> partial: array<f32, 256>;
var<workgroup> row_max: f32;
var<workgroup> row_sum: f32;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let row = wg_id.x;
    if (row >= m.num_rows) {
        return;
    }
    let offset = row * m.vocab_size;
    let tid = local_id.x;

    // ---- max over the row ----
    var local_max: f32 = -3.4028235e38;
    var i: u32 = tid;
    while (i < m.vocab_size) {
        local_max = max(local_max, logits[offset + i]);
        i = i + 256u;
    }
    
    partial[tid] = local_max;
    workgroupBarrier();
    var stride: u32 = 128u;
    
    while (stride > 0u) {
        if (tid < stride) {
            partial[tid] = max(partial[tid], partial[tid + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (tid == 0u) {
        row_max = partial[0];
    }
    workgroupBarrier();
    let max_val = row_max;

    // ---- sum(exp) over the row ----
    var local_sum: f32 = 0.0;
    i = tid;
    while (i < m.vocab_size) {
        local_sum = local_sum + exp(logits[offset + i] - max_val);
        i = i + 256u;
    }

    partial[tid] = local_sum;
    workgroupBarrier();
    stride = 128u;
    
    while (stride > 0u) {
        if (tid < stride) {
            partial[tid] = partial[tid] + partial[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    
    if (tid == 0u) {
        row_sum = partial[0];
    }

    workgroupBarrier();
    let sum_exp = row_sum;

    // loss from the original logit, before the in-place overwrite below
    if (tid == 0u) {
        losses[row] = -(logits[offset + targets[row]] - max_val - log(sum_exp));
    }
    workgroupBarrier();

    // ---- in-place: logits -> probs ----
    i = tid;
    while (i < m.vocab_size) {
        logits[offset + i] = exp(logits[offset + i] - max_val) / sum_exp;
        i = i + 256u;
    }
}

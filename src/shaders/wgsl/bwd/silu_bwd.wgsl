@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> dY: array<f32>;
@group(0) @binding(2) var<storage, read_write> dX: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let idx = (wg_id.y * num_wg.x + wg_id.x) * 256u + local_id.x;

    if (idx >= arrayLength(&x) || idx >= arrayLength(&dX)) {
        return;
    }

    let val = x[idx];
    let sig = 1.0 / (1.0 + exp(-val));

    let grad_silu = sig + val * sig * (1.0 - sig);
    dX[idx] = dY[idx] * grad_silu;
}

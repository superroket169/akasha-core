@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> dY: array<f32>;
@group(0) @binding(2) var<storage, read_write> dX: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let val = x[idx];
    let sig = 1.0 / (1.0 + exp(-val));

    let grad_silu = sig + val * sig * (1.0 - sig);
    dX[idx] = dY[idx] * grad_silu;
}

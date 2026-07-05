@group(0) @binding(0) var<storage, read_write> x: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let val = x[idx];
    x[idx] = val / (1.0 + exp(-val));
}

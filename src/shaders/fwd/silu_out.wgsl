@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= arrayLength(&x) || idx >= arrayLength(&y)) {
        return;
    }
    
    let val = x[idx];
    y[idx] = val / (1.0 + exp(-val));
}

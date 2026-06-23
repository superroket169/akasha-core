use crate::Real;
use rand::Rng;

pub fn random_normal_vec(len: usize, mean: Real, std: Real) -> Vec<Real> {
    let mut rng = rand::thread_rng();
    (0..len)
        .map(|_| {
            let u1: f32 = rng.gen_range(1e-9_f32..1.0);
            let u2: f32 = rng.gen_range(0.0_f32..1.0);
            let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
            mean + z0 * std
        })
        .collect()
}

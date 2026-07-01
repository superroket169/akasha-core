use crate::Real;
use rand::Rng;

pub fn xavier_std(fan_in: u32) -> Real {
    1.0 / (fan_in as Real).sqrt()
}

static INIT_RNG: std::sync::OnceLock<std::sync::Mutex<rand::rngs::StdRng>> =
    std::sync::OnceLock::new();
fn init_rng() -> std::sync::MutexGuard<'static, rand::rngs::StdRng> {
    INIT_RNG
        .get_or_init(|| std::sync::Mutex::new(rand::SeedableRng::seed_from_u64(42)))
        .lock()
        .unwrap()
}

pub fn random_normal_vec(len: usize, mean: Real, std: Real) -> Vec<Real> {
    let mut rng = init_rng();
    (0..len)
        .map(|_| {
            let u1: f32 = rng.gen_range(1e-9_f32..1.0);
            let u2: f32 = rng.gen_range(0.0_f32..1.0);
            let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
            mean + z0 * std
        })
        .collect()
}

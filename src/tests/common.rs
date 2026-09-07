use crate::Real;

pub(crate) fn rand_vec(n: usize, seed: u64) -> Vec<Real> {
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let bits = ((state >> 40) as u32) & 0x00FF_FFFF;
            (bits as f32 / 0x00FF_FFFF as f32) * 2.0 - 1.0
        })
        .collect()
}

pub(crate) fn max_abs_diff(a: &[Real], b: &[Real]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

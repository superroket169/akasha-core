use crate::Real;

/// Temperature-scaled softmax + cumulative sampling.
pub fn sample_token(logits: &[Real], temperature: f32) -> u32 {
    let scaled: Vec<f64> = logits
        .iter()
        .map(|&x| (x as f64) / temperature as f64)
        .collect();
    let max = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scaled.iter().map(|&x| (x - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    let probs: Vec<f64> = exps.iter().map(|&x| x / sum).collect();

    let mut r: f64 = rand::random();
    for (i, &p) in probs.iter().enumerate() {
        if r < p {
            return i as u32;
        }
        r -= p;
    }
    (probs.len() - 1) as u32
}

use crate::Real;

pub fn sample_token(logits: &[Real], temperature: f32, top_k: usize, top_p: f32) -> u32 {
    if temperature <= 0.0 {
        return argmax(logits);
    }

    let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
    idx.sort_unstable_by(|&a, &b| {
        logits[b as usize]
            .partial_cmp(&logits[a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    if top_k > 0 && top_k < idx.len() {
        idx.truncate(top_k);
    }

    let max = logits[idx[0] as usize] as f64;
    let mut probs: Vec<f64> = idx
        .iter()
        .map(|&i| ((logits[i as usize] as f64 - max) / temperature as f64).exp())
        .collect();
    let sum: f64 = probs.iter().sum();
    for p in &mut probs {
        *p /= sum;
    }

    if top_p < 1.0 {
        let mut cum = 0.0;
        let mut keep = probs.len();
        for (n, &p) in probs.iter().enumerate() {
            cum += p;
            if cum >= top_p as f64 {
                keep = n + 1;
                break;
            }
        }
        idx.truncate(keep);
        probs.truncate(keep);
        let s: f64 = probs.iter().sum();
        for p in &mut probs {
            *p /= s;
        }
    }

    let mut r: f64 = rand::random();
    for (n, &p) in probs.iter().enumerate() {
        if r < p {
            return idx[n];
        }
        r -= p;
    }
    *idx.last().unwrap()
}

fn argmax(logits: &[Real]) -> u32 {
    let mut best = 0;
    for (i, &x) in logits.iter().enumerate() {
        if x > logits[best] {
            best = i;
        }
    }
    best as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    const LOGITS: [f32; 5] = [1.0, 4.0, 2.0, 8.0, 3.0];

    #[test]
    fn zero_temperature_is_greedy() {
        for _ in 0..20 {
            assert_eq!(sample_token(&LOGITS, 0.0, 0, 1.0), 3);
        }
    }

    #[test]
    fn top_k_one_is_greedy() {
        for _ in 0..20 {
            assert_eq!(sample_token(&LOGITS, 1.5, 1, 1.0), 3);
        }
    }

    #[test]
    fn top_k_restricts_candidates() {
        for _ in 0..50 {
            let t = sample_token(&LOGITS, 2.0, 2, 1.0);
            assert!(t == 3 || t == 1, "sampled outside top-2: {t}");
        }
    }

    #[test]
    fn tiny_top_p_is_greedy() {
        for _ in 0..20 {
            assert_eq!(sample_token(&LOGITS, 1.0, 0, 0.01), 3);
        }
    }

    #[test]
    fn filters_off_can_reach_every_token() {
        let mut seen = [false; 5];
        for _ in 0..2000 {
            seen[sample_token(&LOGITS, 100.0, 0, 1.0) as usize] = true;
        }
        assert_eq!(seen, [true; 5], "some tokens never sampled: {seen:?}");
    }
}

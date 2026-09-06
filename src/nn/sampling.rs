use crate::Real;

/// CTRL-style repetition penalty
fn apply_repetition_penalty(logits: &mut [Real], seen: &[u32], penalty: f32) {
    if penalty == 1.0 {
        return;
    }
    let mut ids: Vec<u32> = seen.to_vec();
    ids.sort_unstable();
    ids.dedup();
    for id in ids {
        if let Some(l) = logits.get_mut(id as usize) {
            *l = if *l > 0.0 { *l / penalty } else { *l * penalty };
        }
    }
}

pub fn sample_token(
    logits: &[Real],
    temperature: f32,
    top_k: usize,
    top_p: f32,
    seen: &[u32],
    repetition_penalty: f32,
) -> u32 {
    let mut logits = logits.to_vec();
    apply_repetition_penalty(&mut logits, seen, repetition_penalty);
    let logits = logits.as_slice();

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
#[path = "../tests/sampling_tests.rs"]
mod tests;

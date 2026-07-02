use akasha_core::Real;
use akasha_core::config::{DIM, FFN_HIDDEN, NUM_LAYERS, VOCAB_SIZE};
use std::fs::File;
use std::io::{BufReader, BufWriter};

type Param = (Vec<Real>, Vec<Real>);

fn interleave_qkv(dim: usize, q: &[Real], k: &[Real], v: &[Real]) -> Vec<Real> {
    let mut combined = Vec::with_capacity(dim * 3 * dim);
    for r in 0..dim {
        combined.extend_from_slice(&q[r * dim..(r + 1) * dim]);
        combined.extend_from_slice(&k[r * dim..(r + 1) * dim]);
        combined.extend_from_slice(&v[r * dim..(r + 1) * dim]);
    }
    combined
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let input_path = args
        .get(1)
        .map(String::as_str)
        .unwrap_or("checkpoints/model_final.bin");
    let output_path = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("checkpoints/model_final_migrated.bin");

    let dim = DIM as usize;
    let hidden_dim = FFN_HIDDEN as usize;

    println!("Reading old-format checkpoint: {input_path}");
    let file = File::open(input_path).expect("failed to open input checkpoint");
    let params: Vec<Param> =
        bincode::deserialize_from(BufReader::new(file)).expect("failed to deserialize checkpoint");

    let expected_old_len = 1 + NUM_LAYERS * 8 + 2;
    assert_eq!(
        params.len(),
        expected_old_len,
        "unexpected param count {} (expected {expected_old_len} for old q/k/v-separate layout) -- \
         is this already a migrated checkpoint?",
        params.len()
    );

    let mut new_params: Vec<Param> = Vec::with_capacity(1 + NUM_LAYERS * 6 + 2);
    let mut it = params.into_iter();

    // embedding: unchanged
    new_params.push(it.next().unwrap());

    for i in 0..NUM_LAYERS {
        let norm_1 = it.next().unwrap();
        let (q_w, q_g) = it.next().unwrap();
        let (k_w, k_g) = it.next().unwrap();
        let (v_w, v_g) = it.next().unwrap();
        let out_proj = it.next().unwrap();
        let norm_2 = it.next().unwrap();
        let ffn_up = it.next().unwrap();
        let ffn_down = it.next().unwrap();

        assert_eq!(
            q_w.len(),
            dim * dim,
            "layer {i}: q_proj weight size mismatch"
        );
        assert_eq!(
            ffn_up.0.len(),
            dim * hidden_dim,
            "layer {i}: ffn_up size mismatch"
        );

        let qkv_w = interleave_qkv(dim, &q_w, &k_w, &v_w);
        let qkv_g = interleave_qkv(dim, &q_g, &k_g, &v_g);

        new_params.push(norm_1);
        new_params.push((qkv_w, qkv_g));
        new_params.push(out_proj);
        new_params.push(norm_2);
        new_params.push(ffn_up);
        new_params.push(ffn_down);
    }

    // final_norm, lm_head: unchanged
    new_params.push(it.next().unwrap());
    new_params.push(it.next().unwrap());
    assert!(
        it.next().is_none(),
        "leftover params after migration -- count mismatch"
    );

    println!("Writing migrated checkpoint: {output_path}");
    let out_file = File::create(output_path).expect("failed to create output checkpoint");
    bincode::serialize_into(BufWriter::new(out_file), &new_params)
        .expect("failed to serialize migrated checkpoint");

    println!(
        "Done: {} params -> {} params (VOCAB_SIZE={VOCAB_SIZE}, DIM={DIM}, NUM_LAYERS={NUM_LAYERS})",
        1 + NUM_LAYERS * 8 + 2,
        new_params.len()
    );
}

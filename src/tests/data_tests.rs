use super::*;

/// Fake tokenizer: one token per char. Makes chunked-vs-whole tokenization
/// exactly comparable (real BPE only approximately so at boundaries).
fn char_encode(texts: &[&str]) -> Vec<Vec<u32>> {
    texts
        .iter()
        .map(|s| s.chars().map(|c| c as u32).collect())
        .collect()
}

fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("sequexa_data_test_{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// The critical property: chunked tokenization drops/duplicates NOTHING,
/// even with multi-byte chars straddling chunk boundaries and shard cuts.
#[test]
fn sharding_is_lossless() {
    let dir = temp_dir("lossless");
    // Multi-byte chars (2- and 3-byte UTF-8) + words + newlines, sized so
    // tiny chunk/shard limits force many boundary cuts.
    let text = "merhaba dünyağış çok İyi\n".repeat(300);
    let input = dir.join("corpus.txt");
    std::fs::write(&input, &text).unwrap();

    let shard_dir = dir.join("shards");
    tokenize_to_shards(input.to_str().unwrap(), char_encode, &shard_dir, 1000, 64);

    let mut paths: Vec<PathBuf> = std::fs::read_dir(&shard_dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    paths.sort();
    assert!(paths.len() > 1, "test should produce multiple shards");

    let roundtrip: Vec<u32> = paths.iter().flat_map(|p| load_shard(p)).collect();
    let expected: Vec<u32> = text.chars().map(|c| c as u32).collect();
    assert_eq!(roundtrip, expected);
    for p in &paths[..paths.len() - 1] {
        assert_eq!(load_shard(p).len(), 1000);
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn random_batch_windows_are_consistent() {
    let dir = temp_dir("batch");
    let text = "abcdefghij".repeat(500);
    let input = dir.join("corpus.txt");

    std::fs::write(&input, &text).unwrap();
    let shard_dir = dir.join("shards");

    tokenize_to_shards(input.to_str().unwrap(), char_encode, &shard_dir, 700, 128);

    let seq_len = 16;
    let mut ds = Dataset::from_shard_dir(&shard_dir, seq_len);
    assert_eq!(ds.token_count(), 5000);

    let mut rng = rand::thread_rng();

    for _ in 0..600 {
        let (inputs, targets) = ds.random_batch(3, &mut rng);
        assert_eq!(inputs.len(), 3 * seq_len);
        assert_eq!(targets.len(), 3 * seq_len);

        for b in 0..3 {
            let i = &inputs[b * seq_len..(b + 1) * seq_len];
            let t = &targets[b * seq_len..(b + 1) * seq_len];
            assert_eq!(&i[1..], &t[..seq_len - 1]);
        }
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
#[should_panic(expected = "no shard with at least seq_len + 1")]
fn tiny_corpus_panics_instead_of_underflowing() {
    let dir = temp_dir("tiny");
    let input = dir.join("corpus.txt");

    std::fs::write(&input, "abc").unwrap();

    let shard_dir = dir.join("shards");

    tokenize_to_shards(input.to_str().unwrap(), char_encode, &shard_dir, 1000, 64);

    let _ = Dataset::from_shard_dir(&shard_dir, 16);
}

//! Streaming dataset. A large raw corpus is tokenized ONCE, chunk by
//! chunk with a bounded working set, into on-disk token shards; training then
//! samples random windows from a small resident pool of shards instead of
//! holding the whole corpus in RAM.
//!
//! Layout: `data/train.txt` -> `data/train_shards/shard_00000.bin`, ... —
//! raw little-endian u32 tokens, no header (token count = file size / 4).
//! If the shard directory already exists it is reused as-is; delete it to
//! force re-tokenization (e.g. after changing the corpus or tokenizer).

use crate::tokenizer::AkashaTokenizer;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

/// 16M tokens = 64MB per shard file.
const SHARD_TOKENS: usize = 16_000_000;
/// Raw-text read granularity during tokenization (bounds the working set).
const CHUNK_BYTES: usize = 8 * 1024 * 1024;
/// How many shards stay loaded during training (bounds RAM at ~4 x 64MB).
const RESIDENT_SHARDS: usize = 4;
/// A resident shard is swapped for a random cold one every this many batches.
const ROTATE_EVERY: usize = 256;

pub struct Dataset {
    shard_paths: Vec<PathBuf>,
    total_tokens: usize,
    seq_len: usize,
    /// (shard index, tokens) — the pool random_batch samples from.
    resident: Vec<(usize, Vec<u32>)>,
    batches_served: usize,
    next_victim: usize,
}

impl Dataset {
    pub fn from_file(path: &str, tokenizer: &AkashaTokenizer, seq_len: usize) -> Self {
        let shard_dir = shard_dir_for(path);
        if !has_shards(&shard_dir) {
            println!(
                "Tokenizing {path} into shards at {}...",
                shard_dir.display()
            );
            tokenize_to_shards(
                path,
                |s| tokenizer.encode(s),
                &shard_dir,
                SHARD_TOKENS,
                CHUNK_BYTES,
            );
        } else {
            println!(
                "Reusing existing shards at {} (delete the directory to re-tokenize)",
                shard_dir.display()
            );
        }
        Self::from_shard_dir(&shard_dir, seq_len)
    }

    fn from_shard_dir(shard_dir: &Path, seq_len: usize) -> Self {
        let mut shard_paths: Vec<PathBuf> = std::fs::read_dir(shard_dir)
            .expect("cannot read shard directory")
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("shard_") && n.ends_with(".bin"))
            })
            .collect();
        shard_paths.sort();

        // A window needs seq_len inputs + 1 target; shards below that (at most
        // the final partial one) can't serve a single window - drop them
        let mut total_tokens = 0usize;
        shard_paths.retain(|p| {
            let tokens = std::fs::metadata(p)
                .map(|m| m.len() as usize / 4)
                .unwrap_or(0);
            if tokens < seq_len + 1 {
                println!(
                    "Skipping {} ({} tokens < seq_len + 1 = {})",
                    p.display(),
                    tokens,
                    seq_len + 1
                );
                return false;
            }
            total_tokens += tokens;
            true
        });
        assert!(
            !shard_paths.is_empty(),
            "dataset has no shard with at least seq_len + 1 = {} tokens",
            seq_len + 1
        );

        // Initial pool = first N shards; rotation mixes the rest in over time.
        let resident = shard_paths
            .iter()
            .take(RESIDENT_SHARDS)
            .enumerate()
            .map(|(i, p)| (i, load_shard(p)))
            .collect();

        Self {
            shard_paths,
            total_tokens,
            seq_len,
            resident,
            batches_served: 0,
            next_victim: 0,
        }
    }

    pub fn random_batch(
        &mut self,
        batch_size: usize,
        rng: &mut impl rand::Rng,
    ) -> (Vec<u32>, Vec<u32>) {
        self.maybe_rotate(rng);

        let mut inputs = Vec::with_capacity(batch_size * self.seq_len);
        let mut targets = Vec::with_capacity(batch_size * self.seq_len);

        // Weighted by window count so every window across the pool is
        // equally likely regardless of which shard it lives in.
        let total_windows: usize = self
            .resident
            .iter()
            .map(|(_, t)| t.len() - self.seq_len)
            .sum();
        for _ in 0..batch_size {
            let mut r = rng.gen_range(0..total_windows);
            for (_, tokens) in &self.resident {
                let windows = tokens.len() - self.seq_len;
                if r < windows {
                    inputs.extend_from_slice(&tokens[r..r + self.seq_len]);
                    targets.extend_from_slice(&tokens[r + 1..r + self.seq_len + 1]);
                    break;
                }
                r -= windows;
            }
        }

        (inputs, targets)
    }

    /// Swaps one resident shard for a random cold one every ROTATE_EVERY
    /// batches, so long runs see the whole corpus, not just the initial pool.
    fn maybe_rotate(&mut self, rng: &mut impl rand::Rng) {
        self.batches_served += 1;
        if self.shard_paths.len() <= self.resident.len() || self.batches_served % ROTATE_EVERY != 0
        {
            return;
        }
        let incoming = loop {
            let candidate = rng.gen_range(0..self.shard_paths.len());
            if !self.resident.iter().any(|(i, _)| *i == candidate) {
                break candidate;
            }
        };
        let victim = self.next_victim % self.resident.len();
        self.next_victim += 1;
        self.resident[victim] = (incoming, load_shard(&self.shard_paths[incoming]));
    }

    pub fn token_count(&self) -> usize {
        self.total_tokens
    }
}

fn shard_dir_for(path: &str) -> PathBuf {
    let p = Path::new(path);
    let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("dataset");
    p.with_file_name(format!("{stem}_shards"))
}

fn has_shards(dir: &Path) -> bool {
    std::fs::read_dir(dir)
        .map(|entries| {
            entries.filter_map(|e| e.ok()).any(|e| {
                e.file_name()
                    .to_str()
                    .is_some_and(|n| n.starts_with("shard_") && n.ends_with(".bin"))
            })
        })
        .unwrap_or(false)
}

fn load_shard(path: &Path) -> Vec<u32> {
    let bytes = std::fs::read(path).expect("cannot read shard");
    assert_eq!(bytes.len() % 4, 0, "shard {} is truncated", path.display());
    bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// Tokenizes `input_path` into shard files under `dir` with a bounded
/// working set: at most ~chunk_bytes of raw text plus one shard of tokens
/// is in memory at any point. `encode` is a parameter (not a tokenizer)
/// so the chunking mechanics are testable without one.
fn tokenize_to_shards(
    input_path: &str,
    encode: impl Fn(&str) -> Vec<u32>,
    dir: &Path,
    shard_tokens: usize,
    chunk_bytes: usize,
) {
    std::fs::create_dir_all(dir).expect("cannot create shard directory");
    let mut file = std::fs::File::open(input_path).expect("Cannot read dataset");

    let mut carry: Vec<u8> = Vec::new();
    let mut pending: Vec<u32> = Vec::new();
    let mut shard_idx = 0usize;
    let mut total_tokens = 0usize;

    loop {
        let mut buf = vec![0u8; chunk_bytes];
        let n = file.read(&mut buf).expect("read failed");
        let eof = n == 0;
        carry.extend_from_slice(&buf[..n]);

        // Longest valid UTF-8 prefix; the (at most 3-byte) split char waits
        // for the next chunk in carry.
        let valid_len = match std::str::from_utf8(&carry) {
            Ok(_) => carry.len(),
            Err(e) => e.valid_up_to(),
        };
        if eof {
            assert_eq!(valid_len, carry.len(), "{input_path} is not valid UTF-8");
        }
        let text = std::str::from_utf8(&carry[..valid_len]).unwrap();

        // Mid-run, also hold back the last partial line/word so BPE never
        // sees a word cut in half. The separator itself goes to the NEXT
        // chunk: GPT-2 BPE attaches whitespace to the following word
        // (Ġword) — cutting after it would tokenize that word space-less.
        let cut = if eof {
            text.len()
        } else {
            text.rfind('\n')
                .or_else(|| text.rfind(' '))
                .unwrap_or(text.len())
        };

        if cut > 0 {
            pending.extend(encode(&text[..cut]));
            carry.drain(..cut);
        }

        while pending.len() >= shard_tokens {
            let rest = pending.split_off(shard_tokens);
            write_shard(dir, shard_idx, &pending);
            total_tokens += pending.len();
            shard_idx += 1;
            pending = rest;
        }

        if eof {
            if !pending.is_empty() {
                write_shard(dir, shard_idx, &pending);
                total_tokens += pending.len();
                shard_idx += 1;
            }
            break;
        }
    }
    println!("Tokenized into {shard_idx} shard(s), {total_tokens} tokens total");
}

fn write_shard(dir: &Path, idx: usize, tokens: &[u32]) {
    let path = dir.join(format!("shard_{idx:05}.bin"));
    let mut f = std::fs::File::create(&path).expect("cannot create shard");
    f.write_all(bytemuck::cast_slice(tokens))
        .expect("shard write failed");
    println!("  {} ({} tokens)", path.display(), tokens.len());
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fake tokenizer: one token per char. Makes chunked-vs-whole tokenization
    /// exactly comparable (real BPE only approximately so at boundaries).
    fn char_encode(s: &str) -> Vec<u32> {
        s.chars().map(|c| c as u32).collect()
    }

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("akasha_data_test_{name}"));
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
        assert_eq!(roundtrip, char_encode(&text));
        // All but the last shard are exactly full.
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
        // Enough calls to trigger several rotations (ROTATE_EVERY = 256).
        for _ in 0..600 {
            let (inputs, targets) = ds.random_batch(3, &mut rng);
            assert_eq!(inputs.len(), 3 * seq_len);
            assert_eq!(targets.len(), 3 * seq_len);
            // targets are inputs shifted by one within each window
            for b in 0..3 {
                let i = &inputs[b * seq_len..(b + 1) * seq_len];
                let t = &targets[b * seq_len..(b + 1) * seq_len];
                assert_eq!(&i[1..], &t[..seq_len - 1]);
            }
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// B8: a corpus smaller than seq_len + 1 must fail loudly, not underflow.
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
}

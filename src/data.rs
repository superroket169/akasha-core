//! Streaming dataset: a large corpus is tokenized ONCE into on-disk shards
//! (`data/train.txt` -> `data/train_shards/shard_00000.bin`, ..., raw LE u32
//! tokens, no header); training then samples random windows from a small
//! resident pool instead of holding the whole corpus in RAM. Existing shard
//! dirs are reused as-is — delete to force re-tokenization.

use crate::tokenizer::Tokenizer;
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
    pub fn from_file(path: &str, tokenizer: &Tokenizer, seq_len: usize) -> Self {
        let shard_dir = shard_dir_for(path);
        if !has_shards(&shard_dir) {
            println!(
                "Tokenizing {path} into shards at {}...",
                shard_dir.display()
            );
            tokenize_to_shards(
                path,
                |texts| tokenizer.encode_batch(texts),
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

        // A window needs seq_len + 1 tokens; drop shards below that.
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

/// Tokenizes `input_path` into shard files under `dir`, bounded to
/// ~chunk_bytes of text in memory at a time. `encode` takes a batch (order
/// preserved) so a real tokenizer can run it across all cores instead of one.
fn tokenize_to_shards(
    input_path: &str,
    encode: impl Fn(&[&str]) -> Vec<Vec<u32>>,
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

        // Hold back the last partial line/word; GPT-2 BPE attaches
        // whitespace to the following word, so the cut must land on \n or ' '.
        let cut = if eof {
            text.len()
        } else {
            text.rfind('\n')
                .or_else(|| text.rfind(' '))
                .unwrap_or(text.len())
        };

        if cut > 0 {
            // split_inclusive keeps \n attached — it's its own BPE token too.
            let pieces: Vec<&str> = text[..cut].split_inclusive('\n').collect();
            for tokens in encode(&pieces) {
                pending.extend(tokens);
            }
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
#[path = "tests/data_tests.rs"]
mod tests;

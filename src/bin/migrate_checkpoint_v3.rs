//! One-shot v1/v2 -> v3 checkpoint migration. The ONLY place that still
//! understands the legacy formats — the library reads/writes v3 exclusively.
//! The input file is NEVER touched; output is `<input minus .bin>.v3.bin`
//! (or the given path). Pure CPU, no GPU backend needed. Verifies by
//! re-reading both files and comparing every weight tensor bitwise.
//!
//! Migrated files carry no optimizer state (v1's grads are transient
//! accumulation state, worthless for resume; moments never existed) and
//! train_step 0: loading one starts a fresh schedule on trained weights,
//! which is exactly what continued pretraining wants.
//!
//! Usage: migrate_checkpoint_v3 [input] [output]

use akasha_core::Real;
use akasha_core::config::ModelConfig;
use akasha_core::nn::checkpoint::{V3_MAGIC, V3Body};
use serde::Deserialize;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, Write};

// ---- legacy formats, frozen here ----

const V2_MAGIC: &[u8; 4] = b"AKV2";

// header fields are only deserialized, never read — the v3 header is rebuilt
// from the shape-derived architecture below.
#[allow(dead_code)]
#[derive(Deserialize)]
struct V2Body {
    vocab_size: u32,
    dim: u32,
    num_heads: u32,
    num_layers: u64,
    seq_len: u32,
    ffn_hidden: u32,
    params: Vec<Vec<Real>>,
}

/// v1: headerless bincode of (weight, grad) pairs. Grads are dropped.
type V1Pairs = Vec<(Vec<Real>, Vec<Real>)>;

fn read_legacy_params(path: &str) -> Vec<Vec<Real>> {
    let mut reader = BufReader::new(File::open(path).expect("failed to open input"));
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic).expect("input too short");
    assert_ne!(
        &magic, V3_MAGIC,
        "{path} is already a v3 checkpoint, nothing to migrate"
    );
    if &magic == V2_MAGIC {
        println!("Input format: v2");
        let body: V2Body = bincode::deserialize_from(reader).expect("failed to parse v2 body");
        body.params
    } else {
        println!("Input format: v1 (grads will be dropped)");
        reader.rewind().unwrap();
        let pairs: V1Pairs = bincode::deserialize_from(reader)
            .expect("failed to parse input as v1 (weight, grad) pairs");
        pairs.into_iter().map(|(w, _)| w).collect()
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let input = args
        .get(1)
        .map(String::as_str)
        .unwrap_or("checkpoints/model_final.bin");
    let default_output = format!("{}.v3.bin", input.strip_suffix(".bin").unwrap_or(input));
    let output = args.get(2).map(String::as_str).unwrap_or(&default_output);

    assert_ne!(input, output, "input and output must differ");
    assert!(
        !std::path::Path::new(output).exists(),
        "{output} already exists -- refusing to overwrite"
    );

    println!("Reading legacy checkpoint (read-only): {input}");
    let params = read_legacy_params(input);

    // Derive the architecture from tensor shapes; num_heads is not stored in
    // v1, so it comes from the compiled-in config and is sanity-checked.
    let cfg = ModelConfig::akasha_hall_1();
    let num_layers = (params.len() - 3) / 6;
    assert_eq!(
        params.len(),
        3 + num_layers * 6,
        "unexpected param count {}",
        params.len()
    );
    let dim = params[1].len(); // norm_1 of block 0
    let vocab_size = params[0].len() / dim;
    let ffn_hidden = params[5].len() / dim; // ffn_up of block 0
    assert_eq!(
        (vocab_size as u32, dim as u32, num_layers, ffn_hidden as u32),
        (cfg.vocab_size, cfg.dim, cfg.num_layers, cfg.ffn_hidden),
        "file architecture does not match ModelConfig::akasha_hall_1()"
    );
    println!(
        "Architecture: dim={dim}, layers={num_layers}, vocab={vocab_size}, ffn={ffn_hidden}, heads={} (from config)",
        cfg.num_heads
    );

    let body = V3Body {
        vocab_size: cfg.vocab_size,
        dim: cfg.dim,
        num_heads: cfg.num_heads,
        num_layers: cfg.num_layers as u64,
        seq_len: cfg.seq_len,
        ffn_hidden: cfg.ffn_hidden,
        train_step: 0,
        schedule_step: 0,
        params,
        moments: Vec::new(),
    };

    println!("Writing v3 checkpoint: {output}");
    let mut writer = BufWriter::new(File::create(output).expect("failed to create output"));
    writer.write_all(V3_MAGIC).unwrap();
    bincode::serialize_into(&mut writer, &body).expect("failed to serialize v3");
    drop(writer);

    // ---- verification: re-read both files, compare every tensor bitwise ----
    println!("Verifying (bitwise, from disk)...");
    let legacy_again = read_legacy_params(input);
    let mut reader = BufReader::new(File::open(output).unwrap());
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic).unwrap();
    assert_eq!(&magic, V3_MAGIC);
    let v3_again: V3Body = bincode::deserialize_from(reader).expect("failed to re-read v3");

    assert_eq!(legacy_again.len(), v3_again.params.len());
    assert!(v3_again.moments.is_empty());
    for (i, (w_old, w_new)) in legacy_again.iter().zip(v3_again.params.iter()).enumerate() {
        assert_eq!(w_old.len(), w_new.len(), "tensor {i} length mismatch");
        let b1: &[u8] = bytemuck::cast_slice(w_old);
        let b2: &[u8] = bytemuck::cast_slice(w_new);
        assert_eq!(b1, b2, "tensor {i} differs bitwise");
    }

    let in_size = std::fs::metadata(input).unwrap().len();
    let out_size = std::fs::metadata(output).unwrap().len();
    println!("OK: all {} tensors bitwise-identical.", legacy_again.len());
    println!(
        "in: {:.1} MB -> v3: {:.1} MB",
        in_size as f64 / 1e6,
        out_size as f64 / 1e6
    );
    println!(
        "The original file was not modified. Keep it until a chat run on the v3 file is verified too."
    );
}

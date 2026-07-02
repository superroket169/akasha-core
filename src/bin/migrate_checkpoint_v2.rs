//! One-shot v1 -> v2 checkpoint migration (see REFACTOR.md, "checkpoint
//! migration" -- the input file is NEVER touched). Pure CPU, no GPU backend
//! needed. Writes `<input>.v2.bin` (or the given output path), then verifies
//! by re-reading both files and comparing every weight tensor bitwise.
//!
//! Usage: migrate_checkpoint_v2 [input] [output]

use akasha_core::Real;
use akasha_core::config::ModelConfig;
use akasha_core::nn::checkpoint::{V2_MAGIC, V2Body};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, Write};

type Pairs = Vec<(Vec<Real>, Vec<Real>)>;

fn read_v1(path: &str) -> Pairs {
    let mut reader = BufReader::new(File::open(path).expect("failed to open input"));
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic).expect("input too short");
    assert_ne!(
        &magic, V2_MAGIC,
        "{path} is already a v2 checkpoint, nothing to migrate"
    );
    reader.rewind().unwrap();
    bincode::deserialize_from(reader).expect("failed to parse input as v1 (weight, grad) pairs")
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let input = args
        .get(1)
        .map(String::as_str)
        .unwrap_or("checkpoints/model_final.bin");
    let default_output = format!("{input}.v2.bin");
    let output = args.get(2).map(String::as_str).unwrap_or(&default_output);

    assert_ne!(input, output, "input and output must differ");
    assert!(
        !std::path::Path::new(output).exists(),
        "{output} already exists -- refusing to overwrite"
    );

    println!("Reading v1 checkpoint (read-only): {input}");
    let pairs = read_v1(input);

    // Derive the architecture from tensor shapes; num_heads is not stored in
    // v1, so it comes from the compiled-in config and is sanity-checked.
    let cfg = ModelConfig::akasha_hall_1();
    let num_layers = (pairs.len() - 3) / 6;
    assert_eq!(
        pairs.len(),
        3 + num_layers * 6,
        "unexpected v1 param count {}",
        pairs.len()
    );
    let dim = pairs[1].0.len(); // norm_1 of block 0
    let vocab_size = pairs[0].0.len() / dim;
    let ffn_hidden = pairs[5].0.len() / dim; // ffn_up of block 0
    assert_eq!(
        (vocab_size as u32, dim as u32, num_layers, ffn_hidden as u32),
        (cfg.vocab_size, cfg.dim, cfg.num_layers, cfg.ffn_hidden),
        "file architecture does not match ModelConfig::akasha_hall_1()"
    );
    println!(
        "Architecture: dim={dim}, layers={num_layers}, vocab={vocab_size}, ffn={ffn_hidden}, heads={} (from config)",
        cfg.num_heads
    );

    let body = V2Body {
        vocab_size: cfg.vocab_size,
        dim: cfg.dim,
        num_heads: cfg.num_heads,
        num_layers: cfg.num_layers as u64,
        seq_len: cfg.seq_len,
        ffn_hidden: cfg.ffn_hidden,
        params: pairs.iter().map(|(w, _)| w.clone()).collect(),
    };

    println!("Writing v2 checkpoint: {output}");
    let mut writer = BufWriter::new(File::create(output).expect("failed to create output"));
    writer.write_all(V2_MAGIC).unwrap();
    bincode::serialize_into(&mut writer, &body).expect("failed to serialize v2");
    drop(writer);

    // ---- verification: re-read both files, compare every tensor bitwise ----
    println!("Verifying (bitwise, from disk)...");
    let v1_again = read_v1(input);
    let mut reader = BufReader::new(File::open(output).unwrap());
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic).unwrap();
    assert_eq!(&magic, V2_MAGIC);
    let v2_again: V2Body = bincode::deserialize_from(reader).expect("failed to re-read v2");

    assert_eq!(v1_again.len(), v2_again.params.len());
    for (i, ((w1, _), w2)) in v1_again.iter().zip(v2_again.params.iter()).enumerate() {
        assert_eq!(w1.len(), w2.len(), "tensor {i} length mismatch");
        let b1: &[u8] = bytemuck::cast_slice(w1);
        let b2: &[u8] = bytemuck::cast_slice(w2);
        assert_eq!(b1, b2, "tensor {i} differs bitwise");
    }

    let v1_size = std::fs::metadata(input).unwrap().len();
    let v2_size = std::fs::metadata(output).unwrap().len();
    println!("OK: all {} tensors bitwise-identical.", v1_again.len());
    println!(
        "v1: {:.1} MB -> v2: {:.1} MB",
        v1_size as f64 / 1e6,
        v2_size as f64 / 1e6
    );
    println!(
        "The original file was not modified. Keep it until a chat run on the v2 file is verified too."
    );
}

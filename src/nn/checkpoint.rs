//! V3 is the ONLY format this module reads or writes. Legacy v1 (headerless
//! bincode (weight, grad) pairs) and v2 ("AKV2" + weights only) readers live
//! solely in `bin/migrate_checkpoint_v3.rs` — run it once per old file.
//!
//! Layout: 4-byte magic "AKV3", then one bincode `V3Body`. Weights AND
//! optimizer moments follow the `weights.params()` order — that order is the
//! format contract (see ARCHITECTURE.md, Invariantlar).

use super::weights::ModelWeights;
use crate::Real;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::sync::Arc;
use wilupgu::{Backend, Tensor};

pub const V3_MAGIC: &[u8; 4] = b"AKV3";

#[derive(Serialize, Deserialize)]
pub struct V3Body {
    pub vocab_size: u32,
    pub dim: u32,
    pub num_heads: u32,
    pub num_layers: u64,
    pub seq_len: u32,
    pub ffn_hidden: u32,
    /// Training-loop step the file was written at (0 for migrated files).
    pub train_step: u64,
    /// AdamW schedule counter (accumulation cycles, not loop steps).
    pub schedule_step: u32,
    pub params: Vec<Vec<Real>>,
    /// (m, v) per param, in `params` order. Empty = no optimizer state
    /// (weights-only / migrated file): resume starts the optimizer cold.
    pub moments: Vec<(Vec<Real>, Vec<Real>)>,
}

pub struct OptimizerState {
    pub moments: Vec<(Vec<Real>, Vec<Real>)>,
    pub schedule_step: u32,
}

pub struct LoadedCheckpoint {
    pub train_step: u64,
    /// `None` for weights-only files (e.g. migrated v1/v2).
    pub optimizer: Option<OptimizerState>,
}

pub fn save<B: Backend>(
    weights: &ModelWeights<B>,
    optim: Option<(&[(Arc<Tensor<B>>, Arc<Tensor<B>>)], u32)>,
    train_step: u64,
    path: &str,
) -> Result<(), Box<dyn Error>> {
    let cfg = weights.cfg;
    let (moments, schedule_step) = match optim {
        Some((moments, schedule_step)) => (moments, schedule_step),
        None => (&[][..], 0),
    };

    // Streams tensors individually to keep peak RAM at ~150MB (bypassing ~2GB full-body collect).
    // Manually matches bincode 1.x LE fixint layout (structs back-to-back, Vec = u64 len + data).
    // Fully compatible with `bincode::deserialize::<V3Body>`.

    let mut writer = BufWriter::new(File::create(path)?);
    writer.write_all(V3_MAGIC)?;
    bincode::serialize_into(&mut writer, &cfg.vocab_size)?;
    bincode::serialize_into(&mut writer, &cfg.dim)?;
    bincode::serialize_into(&mut writer, &cfg.num_heads)?;
    bincode::serialize_into(&mut writer, &(cfg.num_layers as u64))?;
    bincode::serialize_into(&mut writer, &cfg.seq_len)?;
    bincode::serialize_into(&mut writer, &cfg.ffn_hidden)?;
    bincode::serialize_into(&mut writer, &train_step)?;
    bincode::serialize_into(&mut writer, &schedule_step)?;

    let params = weights.params();
    bincode::serialize_into(&mut writer, &(params.len() as u64))?;
    for t in &params {
        write_tensor(&mut writer, &t.to_cpu())?;
    }

    bincode::serialize_into(&mut writer, &(moments.len() as u64))?;
    for (m, v) in moments {
        write_tensor(&mut writer, &m.to_cpu())?;
        write_tensor(&mut writer, &v.to_cpu())?;
    }
    Ok(())
}

fn write_tensor(writer: &mut impl Write, data: &[Real]) -> Result<(), Box<dyn Error>> {
    bincode::serialize_into(&mut *writer, &(data.len() as u64))?;
    // Real = f32. Native-endian cast safely matches bincode's LE format,
    // since our target architectures (x86_64 / aarch64) are strictly little-endian.
    writer.write_all(bytemuck::cast_slice(data))?;
    Ok(())
}

pub fn load<B: Backend>(
    weights: &ModelWeights<B>,
    path: &str,
) -> Result<LoadedCheckpoint, Box<dyn Error>> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    if &magic != V3_MAGIC {
        return Err(format!(
            "{path} is not a v3 checkpoint — migrate it once with: \
             cargo run --release --bin migrate_checkpoint_v3 -- {path}"
        )
        .into());
    }

    let body: V3Body = bincode::deserialize_from(&mut reader)?;
    let cfg = weights.cfg;
    let file_arch = (
        body.vocab_size,
        body.dim,
        body.num_heads,
        body.num_layers,
        body.ffn_hidden,
    );
    let model_arch = (
        cfg.vocab_size,
        cfg.dim,
        cfg.num_heads,
        cfg.num_layers as u64,
        cfg.ffn_hidden,
    );
    if file_arch != model_arch {
        return Err(format!(
            "checkpoint architecture mismatch: file {file_arch:?} vs model {model_arch:?} \
             (vocab_size, dim, num_heads, num_layers, ffn_hidden)"
        )
        .into());
    }

    let targets = weights.params();
    copy_params(&targets, &body.params)?;

    let optimizer = if body.moments.is_empty() {
        None
    } else {
        if body.moments.len() != targets.len() {
            return Err(format!(
                "checkpoint has {} moment pairs, model expects {}",
                body.moments.len(),
                targets.len()
            )
            .into());
        }
        Some(OptimizerState {
            moments: body.moments,
            schedule_step: body.schedule_step,
        })
    };

    Ok(LoadedCheckpoint {
        train_step: body.train_step,
        optimizer,
    })
}

fn copy_params<B: Backend>(
    targets: &[Arc<Tensor<B>>],
    data: &[Vec<Real>],
) -> Result<(), Box<dyn Error>> {
    if targets.len() != data.len() {
        return Err(format!(
            "checkpoint has {} parameter tensors, model expects {}",
            data.len(),
            targets.len()
        )
        .into());
    }
    for (i, (t, d)) in targets.iter().zip(data).enumerate() {
        let expected = (t.size / std::mem::size_of::<Real>() as u64) as usize;
        if d.len() != expected {
            return Err(format!(
                "checkpoint tensor {i} has {} elements, model expects {expected}",
                d.len()
            )
            .into());
        }
        t.copy_from_cpu(d);
    }
    Ok(())
}

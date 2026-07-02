use super::weights::ModelWeights;
use crate::Real;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, Write};
use std::sync::Arc;
use wilupgu::{Backend, Tensor};

pub const V2_MAGIC: &[u8; 4] = b"AKV2";

#[derive(Serialize, Deserialize)]
pub struct V2Body {
    pub vocab_size: u32,
    pub dim: u32,
    pub num_heads: u32,
    pub num_layers: u64,
    pub seq_len: u32,
    pub ffn_hidden: u32,
    pub params: Vec<Vec<Real>>,
}

pub struct LoadedCheckpoint {
    /// v1 files carry grads; `Trainer::load_weights` restores them for exact
    /// resume behavior. `None` for v2.
    pub v1_grads: Option<Vec<Vec<Real>>>,
}

pub fn save_v2<B: Backend>(weights: &ModelWeights<B>, path: &str) -> Result<(), Box<dyn Error>> {
    let cfg = weights.cfg;
    let body = V2Body {
        vocab_size: cfg.vocab_size,
        dim: cfg.dim,
        num_heads: cfg.num_heads,
        num_layers: cfg.num_layers as u64,
        seq_len: cfg.seq_len,
        ffn_hidden: cfg.ffn_hidden,
        params: weights.params().iter().map(|t| t.to_cpu()).collect(),
    };
    let mut writer = BufWriter::new(File::create(path)?);
    writer.write_all(V2_MAGIC)?;
    bincode::serialize_into(&mut writer, &body)?;
    Ok(())
}

pub fn load<B: Backend>(
    weights: &ModelWeights<B>,
    path: &str,
) -> Result<LoadedCheckpoint, Box<dyn Error>> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;

    let targets = weights.params();

    if &magic == V2_MAGIC {
        let body: V2Body = bincode::deserialize_from(&mut reader)?;
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
        copy_params(&targets, &body.params)?;
        Ok(LoadedCheckpoint { v1_grads: None })
    } else {
        reader.rewind()?;
        let pairs: Vec<(Vec<Real>, Vec<Real>)> = bincode::deserialize_from(&mut reader)?;
        let (weight_data, grads): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
        copy_params(&targets, &weight_data)?;
        Ok(LoadedCheckpoint {
            v1_grads: Some(grads),
        })
    }
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

use crate::Real;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct AkashaWeights {
    pub embedding_table: Vec<Real>,
    pub blocks: Vec<TransformerBlockWeights>,
    pub final_norm: Vec<Real>,
    pub lm_head: Vec<Real>,
}

#[derive(Serialize, Deserialize)]
pub struct TransformerBlockWeights {
    pub norm_1: Vec<Real>,
    pub q_proj: Vec<Real>,
    pub k_proj: Vec<Real>,
    pub v_proj: Vec<Real>,
    pub out_proj: Vec<Real>,
    pub norm_2: Vec<Real>,
    pub ffn_up: Vec<Real>,
    pub ffn_down: Vec<Real>,
}

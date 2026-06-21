use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct AkashaWeights {
    pub embedding_table: Vec<f32>,
    pub blocks: Vec<TransformerBlockWeights>,
    pub final_norm: Vec<f32>,
    pub lm_head: Vec<f32>,
}

#[derive(Serialize, Deserialize)]
pub struct TransformerBlockWeights {
    pub norm_1: Vec<f32>,
    pub q_proj: Vec<f32>,
    pub k_proj: Vec<f32>,
    pub v_proj: Vec<f32>,
    pub out_proj: Vec<f32>,
    pub norm_2: Vec<f32>,
    pub ffn_up: Vec<f32>,
    pub ffn_down: Vec<f32>,
}

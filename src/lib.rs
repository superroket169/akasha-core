pub mod config;
pub mod data;
pub mod nn;
pub mod optim;
pub mod tokenizer;

pub const READ_LOSS: usize = config::LOG_EVERY;
pub type Real = f32;

pub mod config;
pub mod data;
pub mod diagnostic;
pub mod nn;
pub mod optim;
pub mod shaders;
pub mod tokenizer;

#[cfg(test)]
#[path = "tests/common.rs"]
pub(crate) mod test_common;

pub type Real = f32;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelError {
    EmptyPrompt,
    PromptTooLong { len: u32, max: u32 },
    ContextFull { max: u32 },
    NoCache,
    CacheNotEmpty { cur_len: u32 },
}

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelError::EmptyPrompt => write!(f, "prompt is empty"),
            ModelError::PromptTooLong { len, max } => {
                write!(f, "prompt is {len} tokens, context window is {max}")
            }
            ModelError::ContextFull { max } => write!(f, "context window is full ({max} tokens)"),
            ModelError::NoCache => {
                write!(f, "no cache attached (call replace_cache or generate first)")
            }
            ModelError::CacheNotEmpty { cur_len } => write!(
                f,
                "prefill needs an empty cache, this one holds {cur_len} tokens \
                 (loop decode_step for a resumed cache)"
            ),
        }
    }
}

impl std::error::Error for ModelError {}

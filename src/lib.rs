pub mod config;
pub mod data;
pub mod nn;
pub mod optim;
pub mod shaders;
pub mod tokenizer;

pub const READ_LOSS: usize = config::LOG_EVERY;
pub type Real = f32;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AkashaError {
    EmptyPrompt,
    PromptTooLong { len: u32, max: u32 },
    ContextFull { max: u32 },
    NoCache,
    CacheNotEmpty { cur_len: u32 },
}

impl std::fmt::Display for AkashaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AkashaError::EmptyPrompt => write!(f, "prompt is empty"),
            AkashaError::PromptTooLong { len, max } => {
                write!(f, "prompt is {len} tokens, context window is {max}")
            }
            AkashaError::ContextFull { max } => write!(f, "context window is full ({max} tokens)"),
            AkashaError::NoCache => {
                write!(f, "no cache attached (call replace_cache or generate first)")
            }
            AkashaError::CacheNotEmpty { cur_len } => write!(
                f,
                "prefill needs an empty cache, this one holds {cur_len} tokens \
                 (loop decode_step for a resumed cache)"
            ),
        }
    }
}

impl std::error::Error for AkashaError {}

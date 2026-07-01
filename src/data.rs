use crate::tokenizer::AkashaTokenizer;

pub struct Dataset {
    tokens: Vec<u32>,
    seq_len: usize,
}

impl Dataset {
    pub fn from_file(path: &str, tokenizer: &AkashaTokenizer, seq_len: usize) -> Self {
        let text = std::fs::read_to_string(path).expect("Cannot read dataset");
        let mut truncated_len = text.len().min(50_000_000);

        while truncated_len > 0 && !text.is_char_boundary(truncated_len) {
            truncated_len -= 1;
        }

        let text = &text[..truncated_len];
        let tokens = tokenizer.encode(text);

        println!(
            "Tokenized {} chars into {} tokens",
            text.len(),
            tokens.len()
        );
        Self { tokens, seq_len }
    }

    pub fn random_batch(
        &self,
        batch_size: usize,
        rng: &mut impl rand::Rng,
    ) -> (Vec<u32>, Vec<u32>) {
        let mut inputs = Vec::with_capacity(batch_size * self.seq_len);
        let mut targets = Vec::with_capacity(batch_size * self.seq_len);

        for _ in 0..batch_size {
            let max_start = self.tokens.len() - self.seq_len - 1;
            let start = rng.gen_range(0..max_start);
            inputs.extend_from_slice(&self.tokens[start..start + self.seq_len]);
            targets.extend_from_slice(&self.tokens[start + 1..start + self.seq_len + 1]);
        }

        (inputs, targets)
    }

    pub fn token_count(&self) -> usize {
        self.tokens.len()
    }
}

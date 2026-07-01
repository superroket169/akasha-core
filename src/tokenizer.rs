use std::collections::HashMap;

/// GPT-2 BPE tokenizer
pub struct AkashaTokenizer {
    inner: tokenizers::Tokenizer,
}

impl AkashaTokenizer {
    pub fn from_pretrained() -> Self {
        let t = tokenizers::Tokenizer::from_pretrained("gpt2", None)
            .expect("failed to download/load gpt2 tokenizer (requires network access)");
        Self { inner: t }
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.inner.encode(text, false).unwrap().get_ids().to_vec()
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        self.inner.decode(ids, true).unwrap()
    }

    pub fn vocab_size(&self) -> u32 {
        self.inner.get_vocab_size(true) as u32
    }
}

// ---- Tokenizer Struct ----
pub struct Tokenizer {
    pub vocab: HashMap<String, u32>,
    pub inverse_vocab: HashMap<u32, String>,
    pub merges: HashMap<(u32, u32), u32>,
}

impl Tokenizer {
    // ---- Initialize a new Tokenizer ----
    pub fn new() -> Self {
        Self {
            vocab: HashMap::new(),
            inverse_vocab: HashMap::new(),
            merges: HashMap::new(),
        }
    }

    // ---- Train the BPE model on given text to reach target vocabulary size ----
    pub fn train(&mut self, text: &str, target_vocab_size: u32) {
        let mut current_id = 0;

        let mut tokens: Vec<u32> = text
            .chars()
            .map(|c| {
                let s = c.to_string();
                if !self.vocab.contains_key(&s) {
                    self.vocab.insert(s.clone(), current_id);
                    self.inverse_vocab.insert(current_id, s);
                    current_id += 1;
                }
                self.vocab[&c.to_string()]
            })
            .collect();

        // ---- Iteratively merging ----
        while current_id < target_vocab_size {
            let mut pair_counts = HashMap::new();

            for i in 0..tokens.len().saturating_sub(1) {
                let pair = (tokens[i], tokens[i + 1]);
                *pair_counts.entry(pair).or_insert(0) += 1;
            }

            let mut counts_vec: Vec<_> = pair_counts.into_iter().collect();
            counts_vec.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

            let best_pair = match counts_vec.first() {
                Some(&(pair, _)) => pair,
                None => break,
            };

            let new_token_string = format!(
                "{}{}",
                self.inverse_vocab[&best_pair.0], self.inverse_vocab[&best_pair.1]
            );
            self.vocab.insert(new_token_string.clone(), current_id);
            self.inverse_vocab.insert(current_id, new_token_string);
            self.merges.insert(best_pair, current_id);

            // ---- Replace pairs in the token stream with the new ID ----
            let mut new_tokens = Vec::with_capacity(tokens.len());
            let mut i = 0;
            while i < tokens.len() {
                if i < tokens.len() - 1 && (tokens[i], tokens[i + 1]) == best_pair {
                    new_tokens.push(current_id);
                    i += 2;
                } else {
                    new_tokens.push(tokens[i]);
                    i += 1;
                }
            }
            tokens = new_tokens;
            current_id += 1;
        }
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut tokens: Vec<u32> = text
            .chars()
            .filter_map(|c| self.vocab.get(&c.to_string()).copied())
            .collect();

        loop {
            if tokens.len() < 2 {
                break;
            }

            let mut best_pair = None;
            let mut best_idx = 0;
            let mut min_rank = u32::MAX;

            for i in 0..tokens.len() - 1 {
                let pair = (tokens[i], tokens[i + 1]);
                if let Some(&new_token_id) = self.merges.get(&pair) {
                    if new_token_id < min_rank {
                        min_rank = new_token_id;
                        best_pair = Some(pair);
                        best_idx = i;
                    }
                }
            }

            match best_pair {
                Some(pair) => {
                    tokens[best_idx] = self.merges[&pair];
                    tokens.remove(best_idx + 1);
                }
                None => break,
            }
        }

        tokens
    }

    // ---- Decode token IDs back into a string ----
    pub fn decode(&self, tokens: &[u32]) -> String {
        tokens
            .iter()
            .filter_map(|&t| self.inverse_vocab.get(&t))
            .cloned()
            .collect::<Vec<String>>()
            .join("")
    }
}

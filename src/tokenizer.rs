/// GPT-2 BPE tokenizer
pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
}

impl Tokenizer {
    pub fn from_pretrained() -> Self {
        let local =
            std::env::var("SEQUEXA_TOKENIZER").unwrap_or_else(|_| "tokenizer.json".to_string());
        if std::path::Path::new(&local).exists() {
            let t = tokenizers::Tokenizer::from_file(&local)
                .unwrap_or_else(|e| panic!("failed to load tokenizer from `{local}`: {e}"));
            return Self { inner: t };
        }

        let t = tokenizers::Tokenizer::from_pretrained("gpt2", None).expect(
            "failed to download gpt2 tokenizer (needs network once; \
             or provide tokenizer.json / set SEQUEXA_TOKENIZER)",
        );
        if let Err(e) = t.save(&local, false) {
            eprintln!("warning: could not save tokenizer copy to `{local}`: {e}");
        }
        Self { inner: t }
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.inner.encode(text, false).unwrap().get_ids().to_vec()
    }

    /// Unlike `encode`, runs across `tokenizers`' internal rayon pool.
    pub fn encode_batch(&self, texts: &[&str]) -> Vec<Vec<u32>> {
        self.inner
            .encode_batch(texts.to_vec(), false)
            .unwrap()
            .iter()
            .map(|e| e.get_ids().to_vec())
            .collect()
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        self.inner.decode(ids, true).unwrap()
    }

    pub fn vocab_size(&self) -> u32 {
        self.inner.get_vocab_size(true) as u32
    }
}

# akasha-core

Akasha-core is an LLM engine built on [wilupgu](../wilupgu), a
backend-independent tensor/dispatch library (wgpu / CUDA / CPU). It
prioritizes code cleanliness and extensibility over research-driven
experimentation: every kernel sits behind a single emitter, bindings and
kernel configuration are type-checked at zero runtime cost, and the
Train / Prefill / Decode phase system (`GraphBuilder<Phase>`) makes using a
kernel in the wrong phase a compile error. Training and inference share the
same `ModelWeights` but build entirely separate graphs. No ML framework is
involved — every op (matmul, RMSNorm, flash attention, RoPE, AdamW,
cross-entropy and all backward passes) is a hand-written WGSL / CUDA C / CPU
kernel.

## Model: akasha-hall 1.0

| | |
|---|---|
| Params | ~162M (lm_head untied from embedding) |
| Dim / Layers / Heads | 768 / 12 / 12 (head_dim 64) |
| FFN hidden | 3072 (SiLU) |
| Context | 512 |
| Vocab | 50257 (GPT-2 BPE) |
| Position / Norm | RoPE / RMSNorm |
| Trained | 440k steps on an RTX 4050, final loss ~3.6 |

## Usage

```bash
cargo run --release                           # train (expects data/train.txt; resumes from checkpoints/)
cargo run --release -- --chat                 # chat with checkpoints/model_final.bin
cargo run --release -- --chat --weights <p>   # chat with a specific checkpoint
cargo run --release --features cuda           # CUDA backend (NVIDIA)
cargo run --release --features cpu -- --chat --cpu   # CPU backend
```

- **Tokenizer**: a local `tokenizer.json` (or the `AKASHA_TOKENIZER` env var)
  is used if present; otherwise the GPT-2 tokenizer is downloaded once and a
  local copy is saved.
- **Training data**: `data/train.txt`, raw text; tokenized at startup
  (truncated to the first 50M chars). Checkpoints go to
  `checkpoints/model_step_<N>.bin` every `SAVE_EVERY` steps and training
  auto-resumes from the newest one. `scripts/train.{bat,sh}` wrap the run in
  an auto-restart loop.
- **Hyperparameters** all live in `src/config.rs`.
- **Diagnostics**: `src/bin/diagnose.rs` is a 10-check correctness suite
  (gradient flow, gradchecks, accumulation, CE closed-form, KV-cache-vs-naive
  equivalence, ...) — run it whenever training looks wrong before blaming
  hyperparameters.

## Testing

```bash
cargo test -- --test-threads=1   # ALWAYS single-threaded: parallel tests
                                 # create concurrent GPU devices and segfault
```

## Docs

- [ARCHITECTURE.md](ARCHITECTURE.md) — layer map, phase type system, grad
  topology, meta protocol, KV cache, invariants, idea queue
- [TODO.md](TODO.md) — roadmap (continued pretraining → chat fine-tuning)
- [BATCHING_PLAN.md](BATCHING_PLAN.md) — real-batching design & status

## License

No license file is currently present in this repository.

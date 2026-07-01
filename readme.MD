# akasha-core

A from-scratch GPT-2-small-scale (162M parameter) transformer language model,
trained with a hand-written forward/backward pass on top of
[`wilupgu`](https://github.com/2j87/wilupgu), a small GPU compute-graph
library with interchangeable **Vulkan** (default) and **CUDA** (optional,
NVIDIA-only) backends.

Nothing here uses an existing ML framework (no PyTorch/tch-rs/candle) — every
op (matmul, RMSNorm, multi-head attention, RoPE, softmax, AdamW, cross-entropy,
and their backward passes) is implemented as a WGSL compute shader and, for
the optional CUDA path, also as CUDA C / a cuBLAS call.

## Architecture

| | |
|---|---|
| Dim | 768 |
| Layers | 12 |
| Heads | 12 (head_dim = 64) |
| FFN hidden | 3072 (4x dim) |
| Sequence length | 512 |
| Vocab | 50257 (GPT-2 BPE) |
| Position encoding | RoPE |
| Normalization | RMSNorm |
| Activation | SiLU |
| Optimizer | AdamW (with gradient clipping + accumulation) |
| Params | ~162M |

Per transformer block: `norm_1 → {q,k,v}_proj → RoPE → multi-head attention →
out_proj → residual add → norm_2 → ffn_up → SiLU → ffn_down → residual add`.
Multi-head attention processes each head sequentially through a shared set of
scratch buffers (gather the head's Q/K/V columns, scaled-dot-product
attention with a causal mask, scatter the output back into its column
range), rather than allocating per-head buffers — this matters for memory: at
12 heads x 12 layers, naively allocating distinct scratch tensors per head
nearly doubled the model's VRAM footprint for no benefit, since each head's
scratch buffers are fully consumed before the next head's iteration begins.

Forward and backward passes for the whole model are each fused into a single
`wilupgu::ComputeGraph` (`fuse_compute_graphs`), so one training micro-step is
two `execute()` calls (plus the AdamW step at the gradient-accumulation
boundary), not dozens of small per-layer dispatches.

## Current training run

| | |
|---|---|
| Status | Training in progress |
| Hardware | RTX 4050 |
| Dataset | Wikipedia (WikiText-103, see [Training](#training)) |
| Running for | ~3 days |
| Loss so far | 3.86 |
| Target | < 3.5 |
| Launched via | `scripts/train.bat`, see [Training](#training) |

## Building

```bash
# Vulkan backend (default — works on any GPU with a Vulkan driver)
cargo build --release

# CUDA backend (NVIDIA only, requires the CUDA Toolkit; ~2.5x faster than
# Vulkan in practice for this model size, since cuBLAS's GEMM kernels beat
# generic WGSL compute-shader matmuls)
cargo build --release --features cuda
```

## Training

```bash
# Vulkan
cargo run --release --bin akasha-core

# CUDA
cargo run --release --features cuda --bin akasha-core
```

On first run this downloads/tokenizes a dataset into `data/train.txt`
(WikiText-103 by default — see `src/data.rs`; the full raw file is truncated
to the first 50M characters before tokenizing, since GPT-2 BPE tokenizing the
full ~540MB file in one call blew past available RAM) and starts training
from scratch. Checkpoints save to `checkpoints/model_step_<N>.bin` every
`SAVE_EVERY` steps (see `src/config.rs`); re-running automatically resumes
from the latest checkpoint found in that directory. `scripts/train.bat`
(Windows) / `scripts/train.sh` (Linux/macOS) wrap this in an auto-restart
loop (useful since a single long run can outlive transient GPU/driver
hiccups) — both `cd` to the repo root first, so they can be invoked from
either the repo root or `scripts/` itself.

Chat with a trained checkpoint:

```bash
cargo run --release --bin akasha-core -- --chat
```

All hyperparameters (dim, layers, heads, seq_len, batch size, gradient
accumulation steps, LR schedule, weight decay, gradient clip norm, logging/
checkpoint cadence) live in `src/config.rs`.

## Diagnostics

`src/bin/diagnose.rs` is a standalone binary with 8 correctness checks,
useful any time training looks wrong (loss plateaus, NaNs, slow convergence)
before assuming it's a hyperparameter problem:

1. Trainable parameter count sanity check
2. Per-layer/per-tensor gradient flow (catches dead or exploding gradients)
3. HeadGather/HeadScatter round-trip + finite-difference gradient check
4. RMSNorm backward finite-difference gradient check
5. Gradient accumulation correctness (weights unchanged mid-cycle, updated at
   the boundary)
6. AdamW weight-decay grouping (advisory — this codebase uses one shared
   `weight_decay` for every parameter, no separate no-decay group for
   embeddings/norms)
7. Cross-entropy loss correctness against a known-closed-form value
8. **Single-layer memorization test** — train a tiny 1-layer model on a fixed
   repeated batch; it should reach near-zero loss within a few hundred steps.
   If it can't, something in the training loop itself (not just
   hyperparameters) is broken.

```bash
cargo run --release --bin diagnose                  # Vulkan
cargo run --release --features cuda --bin diagnose  # CUDA
DIAGNOSE_ONLY_CHECK8=1 cargo run --release --bin diagnose  # just check 8
```

Check 8 doubles as a CUDA-vs-Vulkan correctness A/B: both `init.rs`'s weight
initialization and `diagnose.rs`'s random batch generation use a shared,
once-seeded `StdRng` (seed 42) instead of `thread_rng()`, specifically so a
CUDA run and a Vulkan run from the same code start from byte-identical
weights and data — any divergence in the resulting training trajectory is
then a genuine backend numerics bug, not different random draws.

## Bugs found and fixed this session

These were root-caused via Check 8 and direct CUDA-vs-Vulkan A/B comparison
of intermediate buffers, and are worth knowing about if you're modifying the
backward pass:

- **`Linear::backward`'s `grad_input` used the wrong meta tensor.** It reused
  the forward pass's `[M, N, K]` meta for the backward `MatMulTrp` dispatch,
  but that op needs `N`/`K` swapped relative to forward. This silently fed a
  corrupted gradient back through the whole network on *every* training
  step, on *both* backends (each backend executed the mislabeled dispatch
  "faithfully" but via different mechanics, so they diverged from each
  other too) — this was the actual root cause of a training run that
  plateaued instead of converging. Fixed in `src/nn/linear.rs` with a
  dedicated, correctly-derived meta tensor for the `grad_input` dispatch.
- **Vulkan: AdamW's dispatch exceeded the 65535-workgroups-per-dimension
  limit** for any tensor over ~16.7M elements (the embedding/lm_head table is
  ~38.6M) — see the wilupgu README for the fix (2D dispatch grid).
- **Vulkan: missing command-buffer synchronization** caused intermittent
  "Parent device is lost" crashes under sustained training — see the wilupgu
  README (periodic `device.poll`).
- **Vulkan: redundant shader/pipeline compilation per node** instead of
  caching by shader name — not a correctness bug, but a real contributor to
  VRAM pressure and slowness at full model scale (thousands of nodes in a
  fused graph) — see the wilupgu README.
- **Multi-head attention's per-head scratch buffers** (`out_head`,
  `grad_out_head`, `grad_q_head`, `grad_k_head`, `grad_v_head`, `grad_y`,
  `grad_raw`, `zero_meta`) were allocated fresh inside the per-head loop in
  `src/nn/attention.rs`, i.e. once per head per layer, even though each is
  pure scratch space fully consumed within that head's own iteration. Hoisted
  outside the loop and reused across all heads — meaningfully reduces VRAM
  use at 12 heads x 12 layers.
- **Dataset tokenization OOM.** Tokenizing the full ~540MB WikiText-103 raw
  file in one `tokenizer.encode()` call spiked to a ~14GB allocation (the
  `tokenizers` crate's working-set overhead is much larger than the input
  text size). `src/data.rs` now truncates to the first 50M characters
  (~15M tokens) before tokenizing.

## License

No license file is currently present in this repository.

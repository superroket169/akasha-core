# TODO

## Where we are (2026-07-02)

**akasha-hall 1.0** — pretraining done. 162M param, RoPE + RMSNorm + SiLU,
12L/12H/768d, GPT-2 BPE vocab (50257), 512 ctx. Trained ~3-4 days on a friend's
RTX 4050, 440,000 steps, loss plateaued at 3.6.

Both backends now produce coherent output:
- **CPU** (`--cpu`, single-threaded, ~1s/token) — validated first, used to
  find/fix most of the bugs below since it panics loudly on corruption instead
  of silently producing garbage.
- **Vulkan (wgpu)** — was completely broken (crash, GPU hang, or silent empty
  output) through this whole debugging session; now working.

Quality read from manual testing: grammar/dialogue formatting/register-switching
is strong (Wikipedia-style vs. novel-dialogue-style both come out syntactically
correct). Semantic coherence and emotional-tone tracking are weak — the model
doesn't follow a prompt's tone or stay on topic past a sentence or two. Read as
a data problem (too little, too repetitive), not an architecture problem.

## Bugs fixed this session (for the record)

- wgpu: `Device::create_bind_group` buffer-binding-size validation error —
  `request_device` was using `wgpu::Limits::default()` (WebGL2-conservative,
  128MB) instead of `adapter.limits()`.
- Buffer pool: `Tensor::drop()` recycled a buffer unconditionally, even when a
  live `ComputeGraph::Node` still referenced it (e.g. `prefill()`'s per-layer
  temporaries, dropped mid-loop while the whole graph executes once at the
  end) — silent data corruption, surfaced as CPU panics / SIGSEGVs / wgpu
  producing all-NaN logits (which `sample_token`'s NaN-fallback maps straight
  to the EOS token id, hence "no answer, ever"). Fixed with
  `Backend::is_sole_owner` (`Arc::strong_count(buf) == 1`), checked before
  recycling. Required a second fix specifically for wgpu: `WgpuNode` never
  retained an `Arc<wgpu::Buffer>` clone (only an opaque `wgpu::BindGroup`), so
  `is_sole_owner` was a no-op there until `WgpuNode` was given its own
  `buffers: Vec<WgpuBuffer>` field.
- `inference.rs`'s `qkv_proj_meta` had `N`/`K` swapped (`[1, dim, dim*3]`
  instead of `[1, dim*3, dim]`) — decode-path fused QKV projection was reading
  out of bounds / computing the wrong shape.
- `wilupgu`'s CPU backend (`backends/cpu.rs`, new this session): a few kernels
  (`head_gather`, `matmul`, `matmul_trp`) assumed the destination buffer was
  exactly `m*n`-sized, but several KV-cache decode buffers (`k_head`,
  `v_head`, `scores`) are persistent scratch allocated at `max_context_len`
  and only partially filled each step — fixed to read-modify-write instead of
  blind-overwrite.

## Short-term

- [x] Push `wilupgu` (already committed) and commit/push `akasha-core`'s
  KV-cache branch — done, including the full 5-stage refactor (see
  REFACTOR.md).
- [ ] Proper documentation pass over the refactored codebase: module-level
  docs, the weights/train/inference mental map, how to add a new kernel
  (meta struct + emitter + phase bound), checkpoint format spec (v1/v2),
  and a rewritten readme.
- [x] Streaming dataset module (2026-07-16) — chunked tokenization into
  16M-token disk shards + resident-pool `random_batch` with rotation; no
  more 50M-char truncation. With this + the V3 checkpoint (optimizer state
  survives restarts), continued pretraining is UNBLOCKED — remaining work
  is data collection itself.
- [ ] Update `readme.MD` — still says "Training in progress ... ~3 days ...
  loss 3.86", needs the actual final numbers (440k steps, loss 3.6) and a
  mention of the two working chat backends.
- [ ] Decide what "good enough to ship v1" means — right now there's no
  automated way to score coherence/tone-following beyond manual read-through.
  Even a handful of held-out prompts checked by eye each run would beat
  nothing.
- [ ] Full fine-tuning for chat/instruction-following — after the continued
  pretraining above (decided over LoRA: model
  is small enough that the RTX 4050 already proved it can full-train it, and a
  chat dataset will be much smaller than pretraining data anyway — LoRA's
  efficiency win doesn't matter here, and full-FT reuses the existing
  backward/AdamW path with zero new kernels). Needs a small
  instruction/chat-formatted dataset (tens of thousands of examples is
  plausibly enough) and a chat template/delimiter convention.

## Medium-term

- [ ] Continued pretraining on a larger, more diverse dataset — current run
  only cycled the ~15M-token WikiText-103 slice once or twice, which reads
  directly in the output (strong local grammar, weak long-range coherence).
- [ ] Custom tokenizer trained on the new dataset instead of reusing GPT-2's
  off-the-shelf BPE vocab (a fixed foreign vocab is a worse fit once the
  dataset isn't WebText-like anymore). Non-trivial: needs a BPE-training
  implementation (or a vetted crate), and a decision on whether vocab size
  changes (currently hardcoded at 50257 in a few places) — if it does, this
  is effectively a fresh pretraining run, not something you can hot-swap into
  the current checkpoint.
- [ ] Re-run `diagnose.rs`'s parallel-test suite now that the buffer-pool bug
  is fixed — worth checking whether that's what caused the previously-noted
  intermittent SIGSEGVs in parallel tests, and whether the causal_mask
  threshold test issue is still there.
- [ ] `wilupgu` CPU backend: currently single-threaded (~1s/token, fine for
  poking at the model, not for real use) — rayon-parallelize the row-wise
  loops (`matmul`, `matmul_trp`, `cross_entropy`, `softmax`, `rmsnorm`) once
  it's actually load-bearing for something (e.g. running without the friend's
  GPU).

## Long-term / open questions

- [ ] Multi-turn chat context (right now every prompt starts a fresh KV-cache
  — `session.take_cache()` before each `generate()` call — no conversation
  memory across turns).
- [ ] Sampling quality: currently plain temperature sampling
  (`sample_token`); top-k / top-p (nucleus) would likely help output quality
  independent of any training changes.
- [ ] Scaling up (more layers/dim) once data quantity/quality stops being the
  obvious bottleneck — premature before the above.
- [ ] License file — repo has none yet.

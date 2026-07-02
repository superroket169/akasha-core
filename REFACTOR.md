# REFACTOR — the great untangling

Status: **Stages 1–3 complete and validated** (2026-07-02). Stage 3
validation: diagnose 10/10 PASS (iGPU, `DIAGNOSE_VOCAB_SIZE=4096`), coherent
chat on both the v1 file and the migrated v2 file. Checkpoint migration ran
against the real akasha-hall 1.0 file (75/75 tensors bitwise-identical,
original untouched, `model_final.bin` md5 eb02fd2de56271a1d5181335a6e103f9).
Next up: Stage 4 (not started).

The KV-cache work made inference fast but turned `inference.rs` /
`akasha_model.rs` / `pipeline.rs` / `attention.rs` into spaghetti: the same
ops exist in three copies, meta buffers are untyped positional arrays, and
weights are welded to the training apparatus. This document is the plan for
fixing that, in stages that each compile and validate on their own.

---

## Diagnosis — what actually makes it spaghetti

**1. Every op exists up to three times, because emitters allocate their meta
internally.**
`RMSNorm` logic lives in `RMSNorm::new` (train), `RMSNorm::forward_nodes`
(prefill) and as a raw `graph.add_node("RMSNorm", ...)` block inside
`build_decode_layer` (decode). Decode can't reuse `forward_nodes` for one
reason only: `forward_nodes` allocates its meta tensor inside, while decode
needs persistent metas updated in place each step (`update_for_step`). Fix
the signature (caller supplies the meta buffer) and three copies collapse
into one.

**2. Metas are untyped positional arrays, and the typed ones are duplicated.**
- `HeadMoveMeta` is defined twice (attention.rs and inference.rs);
  `NormMeta` too (rmsnorm.rs `Meta` / inference.rs `NormMeta`).
- MatMul metas are bare `&[u32; 3]` arrays. The `qkv_proj_meta` N/K-swap bug
  (found during the KV-cache debugging session) is exactly the bug class
  this allows. `MatMulMeta { m, n, k }` would not have compiled wrong.
- `ffn_hidden = dim * 4` is hardcoded independently in pipeline.rs and
  inference.rs; `eps = 1e-5` is sprinkled everywhere; `EOS = 50256` twice;
  rope.rs even has a hardcoded `head_dim = 64`.
- `optim/adamw.rs` is the one place that already does this right
  (`ParamMeta` / `StepConfig` Pod structs) — that's the house style now.

**3. Weights are buried inside the training apparatus.**
`InferenceSession` reaches through `model.layers[i].qkv_proj.weight`, so
chat mode constructs the full `AkashaModel`: grad buffers for every param
(~650 MB), AdamW moments (2 × 650 MB), fused train graphs at seq_len=512,
and 14 grad-edge tensors per block. **Inference currently pays ~2 GB of dead
memory plus the build time of graphs it never runs.** This is also why
constructors have "too many arguments": most of `Linear::new`'s /
`TransformerBlock::new`'s parameters are backward-pass wiring
(`grad_output`/`grad_input` chains) that inference never needs.

---

## Target architecture

### Axis 1 — state split by phase

```
ModelWeights<B>      pure params: Arc<Tensor> per weight + init + save/load. No phase.
Trainer<B>           Arc<ModelWeights> + grads + AdamW + fused train graphs + train_step
InferenceSession<B>  Arc<ModelWeights> + Cache + DecodeScratch + prefill/decode/generate
```

Both engines share one `Arc<ModelWeights>`. Chat mode stops allocating
grads/moments entirely; later, fine-tuning can run a `Trainer` and an
`InferenceSession` side by side on the same weights (eval-during-training
for free).

### Axis 2 — behavior: one emitter per op, phase-typed

A `nn/ops/` module holds **one emitter per kernel**. Emitters compute their
own dispatch grids from shape params and take the meta buffer from the
caller (train passes a constant one, decode passes its persistent updatable
one). Phase support is expressed in the type system, at zero runtime cost:

```rust
pub struct Train;  pub struct Prefill;  pub struct Decode;
pub trait Phase {}
pub trait FwdPhase: Phase {}     // Train + Prefill + Decode
pub trait CachedPhase: Phase {}  // Prefill + Decode

pub struct GraphBuilder<'g, B: Backend, P: Phase> { graph: &'g mut ComputeGraph<B>, _p: PhantomData<P> }

pub fn rope<B, P: FwdPhase>(gb: &mut GraphBuilder<B, P>, ...);        // every phase
pub fn cache_write<B, P: CachedPhase>(gb: &mut GraphBuilder<B, P>, ...); // won't compile in Train
pub fn cross_entropy<B>(gb: &mut GraphBuilder<B, Train>, ...);        // train only
```

"Which shader runs in which phase" becomes readable from signatures:

| op                                    | train fwd | train bwd | prefill | decode |
|---------------------------------------|:---:|:---:|:---:|:---:|
| matmul / linear                        | ✓ | ✓ | ✓ | ✓ (`MatMulAdd` fused) |
| rmsnorm                                | ✓ | ✓ | ✓ | ✓ |
| rope                                   | ✓ | ✓ | ✓ | ✓ (`RoPEOffset`) |
| causal attention (square)              | ✓ | ✓ | ✓ | — |
| cached attention (rect, `SoftmaxRect`) | — | — | — | ✓ |
| cache_write                            | — | — | ✓ | ✓ |
| silu / residual add                    | ✓ | ✓ | ✓ | ✓ |
| cross_entropy                          | ✓ | ✓ | — | — |

Deliberately **not**: one mega-trait with all phase methods (forces empty
stubs on single-phase ops), and **no** `dyn`-dispatch op framework (that's a
tensor-IR project, not a refactor).

### Axis 3 — metas: one module, all typed

`ops/meta.rs` is the single source of truth for every kernel's meta layout:
`MatMulMeta{m,n,k}`, `NormMeta`, `HeadMoveMeta`, `RopeMeta`,
`RopeOffsetMeta`, `SoftmaxRectMeta`, `AttnScaleMeta`, `CacheWriteMeta`,
`EmbeddingMeta`, `CrossEntropyMeta`, `ZeroMeta`. Bare `&[u32]` metas are
banned. Each struct documents which WGSL/CUDA/CPU kernel reads it.

### Helpers — categorized, not a grab-bag

No `helpers.rs` dumping ground. By domain:
- emitter/grid helpers → `ops/` (`grid1d`, `grid2d`)
- `sample_token` (+ future top-k/top-p) → `nn/sampling.rs`
- `save_weights`/`load_weights` + format versioning → `nn/checkpoint.rs`
- `zero_tensor` → one definition in `ops/`
- `interleave_qkv` → checkpoint/weights side (init-time only)

### Constructors — `ModelConfig` first, typestate only at the top

`ModelConfig { vocab_size, dim, num_heads, num_layers, seq_len, ffn_hidden,
norm_eps }` travels as `&cfg`; derived values are methods (`cfg.head_dim()`).
This alone kills most positional-arg risk. Typestate builders (missing
field = compile error) only for the two entry points:

```rust
let weights = ModelWeights::load(ctx, &cfg, path)?;       // or ::random(ctx, &cfg)
let trainer = Trainer::builder(&weights).build();
let session = InferenceSession::builder(&weights)
    .max_context_len(1024)                                 // required: no .build() without it
    .build();
```

### Errors — two classes

Programmer invariants stay `assert!` (`dim % num_heads == 0`). Anything
reachable from user input becomes `Result<_, AkashaError>` (thiserror):
checkpoint IO/shape mismatch, cache attach mismatch, `decode_step` on a full
context (currently a panic — should be `Err(ContextFull)`).

---

## ⚠️ Checkpoint migration — `checkpoints/model_final.bin` is sacred

That file is 4–5 days of RTX 4050 training (440k steps, the akasha-hall 1.0
weights). It is currently in the fused-QKV `(weight, grad)`-pairs bincode
format. Stage 3 introduces a weights-only v2 format (half the file size,
no dead grads at inference). Rules, non-negotiable:

1. **The original file is never modified or deleted. Ever.** Migration is a
   separate read-only tool (`src/bin/migrate_checkpoint_v2.rs`, same spirit
   as `migrate_qkv_checkpoint.rs`) that reads the old file and writes a
   **new** file (`model_final.v2.bin`).
2. The old-format **loader stays in the codebase** (`checkpoint.rs` keeps a
   `load_v1` compat path) at least until v2 is fully verified — the old file
   must remain loadable at all times.
3. Verification before trusting v2: load v1 and v2 side by side and compare
   every weight tensor **bitwise**; then one full chat generation with a
   fixed seed on each, outputs must be identical.
4. Recommended: keep a copy of the original on external storage / cloud
   regardless of this refactor. It is the single most expensive artifact in
   the repo.
5. v2 format gets a magic + version header so future migrations don't need
   filename archaeology, plus a small JSON sidecar with `ModelConfig` +
   training provenance (steps, loss, dataset).

---

## Bonus unlocked: build the decode graph once

`decode_step` currently rebuilds its ComputeGraph **every token**
(inference.rs), even though all buffers and metas are persistent and the
node structure is identical every step — only meta *contents* change
(already updated via `update_for_step`). Once emitters fix dispatch grids at
`max_context_len` (relying on the shaders' existing bounds checks), the
decode graph can be built once per session; per-token work becomes
`update_for_step` + `execute()`. Cheaper per token on CPU and iGPU alike.
Not possible pre-refactor; nearly free after Stage 2.

---

## Stages

Each stage compiles, is committable on its own, and is validated the same
way: `cargo check --all-targets`, a short CPU-backend chat run (same seed →
same output as before the stage), and `diagnose.rs`.

- [x] **Stage 1 — `ModelConfig` + typed metas (`ops/meta.rs`)**
  Mechanical, zero behavior change. Kill duplicate meta structs, replace
  every positional meta array, thread `&ModelConfig` through
  `AkashaModel::new` / `TransformerBlock::new` / `InferenceSession::new`
  (session stops taking `dim`/`num_heads` it can read from the model).
  Fixes the `ffn_hidden = dim*4` double-hardcode.
  Done along the way: `examples/overfit_demo.rs` was already broken by the
  earlier fused-QKV change (still touched `q_proj`/`k_proj`/`v_proj`) —
  fixed to use `qkv_proj`.

- [x] **Stage 2 — `ops/` emitter layer**
  Move `forward_nodes` / `add_rope_node` / `add_qkv_slice_node` and the raw
  `add_node` blocks of `build_prefill_layer` / `build_decode_layer` into one
  emitter per op (meta buffer as parameter). Prefill/decode builders become
  ~20-line lists of emitter calls. `sampling.rs` split happens here.
  Landed as: `ops/{matmul,norm,embedding,rope,head_move,attention,cache,
  elementwise,loss}.rs` — one emitter per kernel owning its binding layout
  and grid formula, `foo(shape)` = const meta / `foo_with(shape, meta)` =
  caller-owned persistent meta; `ops::attention::causal_attention` is the
  composite shared verbatim by train and prefill; zero raw `add_node` calls
  left under `nn/` outside `ops/`. Deliberate leftovers: `diagnose.rs`
  (kernel-level tests want raw nodes) and `optim/adamw.rs` (its own kernel,
  moves into `Trainer` in Stage 3).

- [x] **Stage 3 — weights/engine split**
  Extract `ModelWeights`; `Trainer` takes train_step/clip/AdamW/fused
  graphs; `InferenceSession` depends only on weights. `checkpoint.rs` with
  v1 compat loader + v2 weights-only format + migration tool (see sacred-
  file rules above). `akasha_model.rs` / `pipeline.rs` / `attention.rs`
  dissolve into `weights.rs` + `ops/` + `train.rs`. Riskiest stage — gate on
  diagnose's gradient-flow and memorization checks.
  Landed as: `weights.rs` (`ModelWeights`/`BlockWeights`, RNG draw order
  preserved for seed parity), `train.rs` (`Trainer`, ex-`AkashaModel` minus
  the deprecated sliding-window `generate`), `checkpoint.rs` (v1 sniffed +
  grads restored on v1 resume; v2 = `AKV2` magic + arch header + weights
  only), `bin/migrate_checkpoint_v2.rs` (ran against the real file:
  75/75 tensors bitwise-identical, 1297 MB -> 649 MB, v1 kept). Chat mode
  now builds zero training state.

- [ ] **Stage 4 — phase typing**
  `GraphBuilder<B, P>` + `FwdPhase`/`CachedPhase` bounds on the emitters
  (small diff — Stage 2 already shaped the signatures).

- [ ] **Stage 5 — API polish**
  Typestate builders for the two entry points, `AkashaError`, decode-graph-
  built-once optimization, delete legacy dead code (`rope.rs`'s unused
  `RoPE` struct with its hardcoded `head_dim=64`, old `weights.rs` format
  structs, deprecated `AkashaModel::generate`).

Known debt to carry over (not blocking, tracked in TODO.md): `Layer` trait
in `traits.rs` is nearly vestigial — decide in Stage 3 whether the unfused
`forward()`/`backward()` debug paths earn their keep; config.rs's "~117M
parameters" comment is wrong (real count ~162M, lm_head untied).

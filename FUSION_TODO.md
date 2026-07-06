# Fusion TODO

Kernel-fusion candidates raised but not done yet. Old/unfused kernels stay in
place either way (validation reference), same pattern as FlashAttention and
the RoPE-QK/QKV-split fusions already shipped.

- [ ] **AdamW "foreach" fusion (AdamW A)** — one AdamW dispatch per weight
  tensor right now (~75 for the 12-layer config: embedding + 6 per block +
  final_norm + lm_head). Combine into one (or a handful of) dispatch(es) that
  loop over all tensors. Lives in `wilupgu` (`builtin::ADAMW` is wilupgu's,
  not akasha's). Grouped with the other fusion work below, not with AdamW B
  (schedule-on-GPU) — different problem, same "N dispatches -> fewer"
  category.
- [ ] **Linear+SiLU fused epilogue** — `ffn_up` matmul immediately followed by
  `SiLU`, nothing else consumes the pre-activation in between. Blocked on
  `matmul` being a wilupgu builtin (cuBLAS on CUDA) with no epilogue-fusion
  hook; would need a hand-written fused kernel bypassing cuBLAS for this one
  path, same tradeoff as FlashAttention.
- [ ] **Add+RMSNorm fused epilogue** — `add_1`/`add_2` residual add immediately
  followed by `RMSNorm`. Same blocker: `residual_add` is a wilupgu builtin.
- [ ] **RMSNorm_bwd + RMSNorm_weight_bwd fusion** — flagged early, still
  undecided (not rejected, just parked). Technically fusable but the two
  kernels parallelize over different axes (per-row vs. per-feature reduction
  across all rows), so fusing would need atomic accumulation for dWeight.
  Low complexity/value ratio at current model size; revisit if this stops
  being true.

# Akasha LLM
This llm is an llm. Plan is under there =>

### These are just plans
Parameters => ~310 million => Finished (dinamic architecture)
--
Context Window => 2048 token => Finished
--
Vocubulary Size => 786 x 64 = 50,304 => Finished (dinamic architecture)
--
Embedding Dimension => 1024 => Finished
--
Layers => 24 Transform block => Finished
--
Attention Heads => 16 => Ready to start
--
Position Coding => RoPE => Finished
--
Optimizer => AdamW => Ready to Start
--
Cosine Decay with Warmup => Not ready
--
Gradient Clipping => 1.0 => Not ready
--

----------

## Roadmap to First Inference
- [x] **Core Engine:** Vulkan-based Compute Graph (Filuplex)
- [x] **Primitive Layers:** Linear, RMSNorm, RoPE, SiLU, Attention, Add, Embedding
- [x] **Backpropagation:** Full gradient propagation chain implemented
- [ ] **Shaders:** Finalize `softmax_bwd` and `rmsnorm_bwd`
- [ ] **Optimizer:** AdamW implementation & Weight Update logic
- [ ] **Data Pipeline:** Tokenizer integration & Dataset scrubbing (Python)
- [ ] **Training Loop:** Epochs, Gradient Accumulation, and Save/Load
- [ ] **Validation:** "Hello World" (Overfitting on single batch)

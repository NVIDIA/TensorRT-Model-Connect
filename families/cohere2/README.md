# Cohere2 / Tiny Aya Global

This family builds `Cohere2ForCausalLM` as a native TensorRT `text_continuation`
task. The initial checkpoint is `CohereLabs/tiny-aya-global`, pinned in the E2E
manifest. It uses the checkpoint's complete decoder and original weights.

The supported build precision is FP32, widening the BF16 checkpoint exactly.
The graph implements parallel LayerNorm attention and SwiGLU branches, grouped
query attention, local GPT-J rotary positions, and full-attention layers without
rotary positions. The checkpoint's native byte-level tokenizer applies both
serialized regex splits before BPE and adds its BOS token for text input.
Explicit token IDs are already framed and receive no additional special tokens.

Execution is greedy, batch one, on one GPU. `max_sequence_length` defaults to
128 and can be set up to the checkpoint's context limit at build time. Every
generated token recomputes the bounded prefix; there is no external KV cache.
The prefix and `max_new_tokens` must fit the selected build profile. Sampling,
quantization, parallel execution, and dynamic KV caches are unsupported.

The family E2E tests build the real checkpoint, run the public native CLI,
compare tokenizer IDs with Hugging Face, and require exact greedy tokens and
decoded text against the upstream FP32 eager implementation in Transformers
5.2.0. English, Spanish, and Arabic prefixes share a 128-token bundle.

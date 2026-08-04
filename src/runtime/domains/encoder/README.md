# Encoder and Ranking Backends

No behavior-bearing shared encoder backend remains in this directory.
Encoder-only, embedding, and reranking behavior is model-owned under
`src/runtime/models/<family>/`.

Use each model's `MODEL.toml` as the discovery source, then read its
`plugin.cpp` and `pipeline.*`. For example:

- `src/runtime/models/bert/` owns the `bert_encoder_only` strategy.
- `src/runtime/models/eagle_vlm/` owns the `eagle_vlm_embedding` and
  `eagle_vlm_reranking` strategies.

Keep pooling, preprocessing, tensor binding, and output semantics with the
owning model rather than adding a generic backend here.

<!-- Collaborative review anchor. -->

# Encoder and Ranking Backends

No behavior-bearing shared encoder backend remains in this directory.
Encoder-only, embedding, and reranking behavior is model-owned under
`python/tensorrt_model_connect/models/<family>/runtime/`.

Use each model's `MODEL.toml` as the discovery source, then read its
`plugin.cpp` and `pipeline.*`. For example:

- `python/tensorrt_model_connect/models/bert/runtime/` owns the `bert_encoder_only` strategy.
- `python/tensorrt_model_connect/models/eagle_vlm/runtime/` owns the `eagle_vlm_embedding` and
  `eagle_vlm_reranking` strategies.

Keep pooling, preprocessing, tensor binding, and output semantics with the
owning model rather than adding a generic backend here.

<!-- Collaborative review anchor: batch 2. -->

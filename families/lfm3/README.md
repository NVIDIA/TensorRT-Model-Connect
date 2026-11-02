# LFM3 text generation

This family implements the native TensorRT graph and runtime for
`LiquidAI/LFM3-1.7B`, pinned in the E2E manifest to revision
`ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7`. Actual TensorRT checkpoint parity
and GPU qualification are still required before treating this family as validated.

The initial scope is one request on one GPU, FP16 or BF16, with a fixed cache.
Prefill processes each token in order to update convolution and attention state
together. Sliding attention masks the full allocated KV cache; it does not reduce
the cache allocation. Raw prompts and single-user ChatML prompts support the
checkpoint's thinking toggle. Nonempty system prompts, quantized checkpoints,
tensor parallelism, dynamic caches, batching and speculative decoding are rejected.

Use the public Python `BuildRequest` with `family="lfm3"`,
`task="text_generation"`, and `precision="bf16"` or `"fp16"`. Build and runtime
interfaces follow the repository's public Task API. The native family DSO is
`libtrtmc_model_lfm3.so`. Rebuild both the bundle and this DSO when changing
family topology or runtime metadata.

The E2E manifest declares raw, chat and sliding-window boundary cases with exact
greedy token-ID comparison to the pinned official reference. CPU tests exercise
configuration, safetensors mapping and graph helpers; they do not prove device
execution or checkpoint parity. The optional native tokenizer checkpoint test
requires `TRTMC_LFM3_TOKENIZER_JSON` to point to that revision's `tokenizer.json`
and returns an explicit skip when the asset is absent.

Repository source retains its Apache-2.0 notices. Model weights are distributed
by LiquidAI under the LFM Open License v1.0; users must obtain them separately
under that license. No model weights or external implementation files are included.

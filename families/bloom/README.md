# BLOOM

This family builds BLOOM decoders for `text_continuation` (`bigscience/bloom-560m`
in the manifests, fp32, tensor parallel size 1 and 4). BLOOM uses ALiBi
positions, an embedding layer norm, a fused `query_key_value` projection, a
GELU MLP with biases and a `lm_head` tied to the embedding. A bundle holds a
one-token decode engine and a dynamic-length prefill engine.

## Text continuation Task

This family implements `IModel` and `ITextContinuation` with its own binding
and Config declaration. The existing shared C ABI and header-only C++
`TextContinuation` wrapper discover that binding. No sibling family is imported.

The input is UTF-8 text or checkpoint token IDs. Text uses the existing family
tokenizer and optional chat template. Token IDs are passed directly to the
decoder, without decoding/re-tokenizing or applying a chat template. Results own
only the newly generated text and token IDs, plus setup/prefill/decode timings.
The original graph, weights, sampler, cache and distributed implementation remain.

Build a fresh bundle; the old `text_generation` primary mode is not retained by
this family runtime. The usual CLI remains:

```sh
trtmc build bigscience/bloom-560m -o model.bundle
trtmc run model.bundle --prompt "The capital of France is" --max-new-tokens 20
```

### Config and defaults

`runtime/task_config.h` owns the complete declaration and conversion. Defaults
remain 128 new tokens, temperature 1, top-k 1, top-p 1, min-p 0, seed -1, checkpoint
EOS, autoregressive mode, chat template off, thinking on, answer-stop off and
stop-check interval 16. An explicit token limit of zero returns an empty
continuation. Unknown or mistyped options are errors, not silently ignored.
`repetition_penalty` accepts only the neutral value 1: the existing sampler does
not implement a non-neutral penalty. No new sampling algorithm is added.

### Validation

The existing official checkpoint cases and their reference criteria remain.
Each single-device case also calls the direct public C11 and C++17 consumers with
both input representations. Their complete token IDs and text must agree; text
input must also agree with the existing CLI. Consumers read the owned result
after releasing the model handle. The added checks do not replace the original
reference oracle or certify unexecuted distributed profiles.

```sh
cmake --build build --target trtmc trtmc_backend_trt trtmc_model_bloom test_bloom_task_config
ctest --test-dir build --output-on-failure -R '^bloom_task_config$'
build/test_bloom_sdk_c model.bundle build text "The capital of France is" max_new_tokens=20
build/test_bloom_sdk_cpp model.bundle build text "The capital of France is" max_new_tokens=20
```

The Config CTest target also builds both SDK consumers. Existing E2Es use
`TRTMC_NATIVE_BUILD_DIR` to locate them. The server consumer must support semantic
Tasks before serving a migrated family; legacy-interface inheritance is not kept
in this family to work around an unmigrated application.

## Quantization

`quantization=fp8` (with `precision` fp16 or bf16, tensor parallel size 1)
quantizes the original checkpoint at build time for bloom-560m, 1b1, 1b7, 3b
and 7b1. q, k, v, `dense_h_to_4h` and `dense_4h_to_h` of every layer run as FP8
GEMMs (per-tensor weight scale amax / 448, static activation scales from
`fp8_activation_scales.json`, ModelOpt `FP8_DEFAULT_CFG`). The attention output
projection, `lm_head`, embeddings and norms stay unquantized. On TensorRT 11.4,
B300, decode is 1.01x to 1.41x and prefill (255 tokens) 1.14x to 1.42x faster
than fp16; the decode plan is 17% to 39% smaller.

### NVFP4

NVFP4 was measured on bloom-1b7 and is not shipped. The study built real
TensorRT engines (no NVFP4 code is in the family): NVFP4 MLP (fc1 and fc2) with
q/k/v in FP8 as the main variant, plus four other layouts. All rows come from
one job on one B300, TensorRT 11.4, max cache 256.

| bloom-1b7 | Decode plan MiB | Decode ms (vs fp16) | Prefill 255 ms | KL to HF fp32 (x FP8) | PPL | Prompts with non-finite prefill |
|---|---|---|---|---|---|---|
| fp16 | 4267.8 | 1.305 (1.000x) | 3.855 | 0.00002 | 32.172 | 0 of 192 |
| FP8 (shipped) | 3212.3 | 1.210 (1.078x) | 2.895 | 0.00586 (1.0x) | 32.384 | 0 of 192 |
| NVFP4 MLP, q/k/v FP8 | 2876 | 1.206 (1.082x) | 2.813 | 0.04718 (8.1x) | 33.259 | 192 of 192 |
| NVFP4 fc1 only (fc2 FP8) | 3044 | 1.182 (1.104x) | 2.856 | 0.03329 (5.7x) | 32.867 | 0 of 192 |
| NVFP4 fc2 only (fc1 FP8) | build fails | - | - | - | - | - |
| NVFP4 all five projections | 2750 | 1.193 (1.094x) | 2.771 | 0.05725 (9.8x) | 33.562 | 192 of 192 |

TensorRT 11.1 (same job) gives the same picture for the main variant: decode
1.345 ms against 1.362 ms for FP8, KL 7.9x FP8, non-finite prefill on 192 of
192 prompts, and fc2 only fails to build.

Reasons:

- Speed: NVFP4 adds only 0.3% (MLP) to 2.3% (fc1 only) decode speed over FP8.
  Decode of a 1.7B model at one token is limited by launch latency and fixed
  kernels (norms, attention, quantize), not by weight reads; the FP4 GEMMs are
  not faster than the FP8 GEMMs at one token. The decode plan is 10.5% smaller
  than FP8.
- Quality: KL is 5.7x to 9.8x the FP8 KL (PPL +3.4% against +0.7% for FP8).
- TensorRT problems: every layout with fc2 in NVFP4 returns non-finite values
  from the prefill engine, starting at the output of the first NVFP4 MLP; it
  depends on the number of prompt rows, and the same input is finite through
  the decode engine. The block-scaled FP4 GEMM with a run-time row count is the
  likely cause (not isolated). The fc2-only layout fails to build on 11.1 and
  11.4.

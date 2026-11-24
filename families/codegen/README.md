# CodeGen

This family builds CodeGen decoders for `text_continuation`
(`Salesforce/codegen-350M-mono` in the manifests: fp16 at tensor parallel size
1, fp32 at tensor parallel size 4). CodeGen is a GPT-J-style decoder with a
parallel residual, partial interleaved RoPE, a fused `qkv_proj` matrix and a
GELU MLP. A bundle holds a one-token decode engine and a dynamic-length prefill
engine.

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
trtmc build Salesforce/codegen-350M-mono -o model.bundle
trtmc run model.bundle --prompt "def hello_world():" --max-new-tokens 20
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
cmake --build build --target trtmc trtmc_backend_trt trtmc_model_codegen test_codegen_task_config
ctest --test-dir build --output-on-failure -R '^codegen_task_config$'
build/test_codegen_sdk_c model.bundle build text "def hello_world():" max_new_tokens=20
build/test_codegen_sdk_cpp model.bundle build text "def hello_world():" max_new_tokens=20
```

The Config CTest target also builds both SDK consumers. Existing E2Es use
`TRTMC_NATIVE_BUILD_DIR` to locate them. The server consumer must support semantic
Tasks before serving a migrated family; legacy-interface inheritance is not kept
in this family to work around an unmigrated application.

## Quantization

`quantization=fp8` (with `precision=fp16`, tensor parallel size 1) quantizes the
original checkpoint at build time. q, k, v, the attention output projection,
`fc_in` and `fc_out` of every layer run as FP8 GEMMs (per-tensor weight scale
amax / 448, static activation scales from `fp8_activation_scales.json`, ModelOpt
`FP8_DEFAULT_CFG`); `lm_head`, embeddings and norms stay fp16. On TensorRT 11.4,
B300, codegen-350M-mono: decode 1.09x faster than fp16 in a CUDA graph, decode
plan 31% smaller.

### NVFP4

NVFP4 was measured and is not shipped. The study built real TensorRT engines
(no NVFP4 code is in the family): MLP (`fc_in` and `fc_out`) in NVFP4 with FP8
attention projections. B300, TensorRT 11.4, default build, codegen-350M-mono.
The FP8 column is the FP8 measurement job; the NVFP4 job reproduced its KL,
top-1 agreement and CUDA-graph decode speed.

| Item | FP8 (shipped) | NVFP4 MLP, attention FP8 |
|---|---|---|
| Decode plan | 541.8 MiB | 472 MiB (0.87x of FP8) |
| Decode vs fp16, eager / CUDA graph | 1.05x / 1.09x | 0.97x / 1.00x |
| Prefill 128 tokens vs fp16 | 1.01x | 0.89x |
| KL to HF, wiki / code | 0.0069 / 0.0033 | 0.053 / 0.026 (8x FP8) |
| Top-1 agreement with HF, wiki / code | 93.2% / 96.7% | 84.2% / 92.9% |
| Non-finite values in the prompt scan | none | logits and K/V values |

Other layouts: only `fc_in` in NVFP4 gives KL 0.031 / 0.015 and decode 1.03x
of fp16; keeping the first and last two layers in FP8 gives KL 0.040 / 0.020
and decode 1.00x. Both are lossier and slower than FP8.

Reasons: at 350M parameters, decode is not bound by weight reads, so the
smaller weights do not pay; NVFP4 is slower than FP8, has about 8x the FP8 KL
and returns non-finite values.

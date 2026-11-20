# XGLM

This family builds XGLM multilingual decoders for `text_continuation`
(`facebook/xglm-564M` in the manifests: fp16 with tensor parallel size 1, fp32
with size 4, max sequence length 256). XGLM uses sinusoidal positions,
LayerNorm, separate q/k/v/out projections with biases and a GELU `fc1`/`fc2`
MLP. A bundle holds a one-token decode engine and a dynamic-length prefill
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
trtmc build facebook/xglm-564M -o model.bundle
trtmc run model.bundle --prompt "Hello" --max-new-tokens 20
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
cmake --build build --target trtmc trtmc_backend_trt trtmc_model_xglm test_xglm_task_config
ctest --test-dir build --output-on-failure -R '^xglm_task_config$'
build/test_xglm_sdk_c model.bundle build text "Hello" max_new_tokens=20
build/test_xglm_sdk_cpp model.bundle build text "Hello" max_new_tokens=20
```

The Config CTest target also builds both SDK consumers. Existing E2Es use
`TRTMC_NATIVE_BUILD_DIR` to locate them. The server consumer must support semantic
Tasks before serving a migrated family; legacy-interface inheritance is not kept
in this family to work around an unmigrated application.

## Quantization

`quantization=fp8` (fp16 precision, tensor parallel size 1) quantizes the
original checkpoint at build time for xglm-564M and xglm-1.7B. q, k, v, `fc1`
and `fc2` of every layer run as FP8 GEMMs (per-tensor weight scale
amax / 448, static activation scales from `fp8_activation_scales.json`,
ModelOpt `FP8_DEFAULT_CFG`, fp16 Dequantize). The attention output projection,
`lm_head`, embeddings and LayerNorms stay fp16. On TensorRT 11.4, decode is
1.075x faster than fp16 on 564M (B200) and 1.079x on 1.7B (B300), prefill 1.10x
to 1.44x, and the decode plan is 16.7% to 24.5% smaller. Other XGLM sizes raise
and name the calibration tool.

### NVFP4

NVFP4 was measured on both calibrated sizes and is not shipped. The study built
real TensorRT engines (no NVFP4 code is in the family). Weights are packed FP4
with an FP8 scale per block of 16; activations are quantized to FP4 at run time
with the FP8 activation scales. Variants: `all` = q/k/v/`fc1`/`fc2` in NVFP4;
`mlp + FP8 attn` = `fc1`/`fc2` NVFP4, q/k/v FP8; `mlp` = `fc1`/`fc2` NVFP4,
q/k/v fp16; `fc1 only` = `fc1` NVFP4, the rest FP8. All rows: one B200,
TensorRT 11.4, max sequence length 256. KL is against the Hugging Face model on
wikitext-2, through the decode engine.

| Model, variant | Decode plan MiB | Decode ms (vs fp16) | Prefill 256 vs fp16 | KL (x FP8) | Prefill texts with NaN |
|---|---|---|---|---|---|
| 564M fp16 | 1583.0 | 0.963 (1.000x) | 1.00x | 0.0029 | 0 of 220 |
| 564M FP8 (shipped) | 1319.3 | 0.875 (1.100x) | 1.23x | 0.0214 (1.0x) | 0 of 220 |
| 564M NVFP4 all | 1204.0 | 0.988 (0.975x) | 1.10x | 0.332 (15.5x) | 93 of 220 |
| 564M NVFP4 mlp + FP8 attn | 1235.3 | 0.938 (1.026x) | 1.13x | 0.230 (10.7x) | 93 of 220 |
| 564M NVFP4 mlp | 1307.3 | 0.978 (0.985x) | 1.12x | 0.218 (10.2x) | 93 of 220 |
| 564M NVFP4 fc1 only | 1277.5 | 0.911 (1.057x) | 1.13x | 0.192 (9.0x) | 0 of 220 |
| 1.7B fp16 | 4315.8 | 1.512 (1.000x) | 1.00x | 0.100 | 0 of 220 |
| 1.7B FP8 (shipped) | 3260.3 | 1.284 (1.178x) | 1.42x | 0.109 (1.0x) | 0 of 220 |
| 1.7B NVFP4 all | 2798.3 | 1.265 (1.195x) | 1.41x | 0.419 (3.8x) | 93 of 220 |
| 1.7B NVFP4 mlp + FP8 attn | 2924.2 | 1.265 (1.196x) | 1.41x | 0.371 (3.4x) | 93 of 220 |
| 1.7B NVFP4 mlp | 3212.1 | 1.302 (1.162x) | 1.35x | 0.357 (3.3x) | 93 of 220 |
| 1.7B NVFP4 fc1 only | 3092.3 | 1.262 (1.198x) | 1.35x | 0.329 (3.0x) | 0 of 220 |

TensorRT 11.1 (same job layout) gives the same picture: on 564M NVFP4 decode is
0.956x to 1.015x of fp16 against 1.080x for FP8; on 1.7B 1.164x to 1.200x
against 1.182x.

Reasons:

- Speed: on 564M no NVFP4 variant reaches the FP8 decode speed, and `all` and
  `mlp` are slower than fp16. On 1.7B the best variant is 1.5% to 2% faster than
  FP8. A decode step is about 270 to 320 short kernels and is not
  weight-bandwidth bound; at one token the block-scaled FP4 GEMMs are not faster
  per kernel than the FP8 GEMV kernels. Prefill is slower than FP8 at every
  length.
- Quality: KL is 9x to 16x the FP8 KL on 564M and 3.0x to 3.8x on 1.7B.
- TensorRT problems: every variant with `fc2` in NVFP4 returns NaN from the
  prefill engine for 93 of 220 texts (texts shorter than the 256-token build
  length), on TensorRT 11.4 and 11.1. The first NaN is in the output of the
  layer-0 `fc2` FP4 GEMM; it is not an overflow (the finite maximum of that
  tensor is 530). These plans fuse the `fc1` GELU and the FP4 quantize of the
  `fc2` input into one GEMM epilogue, which the clean `fc1`-only plans do not
  have. The root cause inside TensorRT was not isolated. The decode engine has
  no NaN in any variant.

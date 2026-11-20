# OPT

This family builds OPT decoders for `text_continuation` (`facebook/opt-125m` in
the manifests: fp16 with tensor parallel size 1, fp32 with size 4, max sequence
length 256). OPT uses learned positions with offset 2, LayerNorm, a ReLU
feed-forward and biased projections. A bundle holds a one-token decode engine
and a dynamic-length prefill engine.

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
trtmc build facebook/opt-125m -o model.bundle
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
cmake --build build --target trtmc trtmc_backend_trt trtmc_model_opt test_opt_task_config
ctest --test-dir build --output-on-failure -R '^opt_task_config$'
build/test_opt_sdk_c model.bundle build text "Hello" max_new_tokens=20
build/test_opt_sdk_cpp model.bundle build text "Hello" max_new_tokens=20
```

The Config CTest target also builds both SDK consumers. Existing E2Es use
`TRTMC_NATIVE_BUILD_DIR` to locate them. The server consumer must support semantic
Tasks before serving a migrated family; legacy-interface inheritance is not kept
in this family to work around an unmigrated application.

## Quantization

`quantization=fp8` (fp16 precision, tensor parallel size 1) quantizes the
original checkpoint at build time for opt-125m. q, k, v, `fc1` and `fc2` of
every layer run as FP8 GEMMs (per-tensor weight scale amax / 448, static
activation scales from `fp8_activation_scales.json`, ModelOpt
`FP8_DEFAULT_CFG`, fp16 Dequantize). The attention output projection,
`lm_head`, embeddings and LayerNorms stay unquantized. On B300, decode is 1.08x
(TensorRT 11.4) and 1.14x (11.1) faster than fp16, prefill 1.07x to 1.10x, and
the bundle is 23.6% smaller. Other OPT sizes raise and name the calibration
tool.

### NVFP4

NVFP4 was measured on opt-125m and is not shipped. The study built real
TensorRT engines (no NVFP4 code is in the family): `fc1` and `fc2` of all 12
layers in NVFP4 with q/k/v in FP8, and a variant that keeps layers 0 and 11 in
FP8. Weights are packed FP4 with an FP8 scale per block of 16; activations are
quantized to FP4 at run time with the FP8 activation scales. All rows come from
one B300, TensorRT 11.4, max cache 256.

| opt-125m | Decode plan MiB | Decode ms (vs fp16) | Prefill 256 ms | KL to HF (x FP8) | PPL | Prefill texts with NaN |
|---|---|---|---|---|---|---|
| fp16 | 313.7 | 0.408 (1.00x) | 0.996 | 0.00024 | 70.919 | 0 of 200 |
| FP8 (shipped) | 239.6 | 0.371 (1.10x) | 0.896 | 0.01282 (1.0x) | 70.931 | 0 of 200 |
| NVFP4 `fc1` + `fc2` | 216.0 | 0.416 (0.98x) | 0.946 | 0.06313 (4.9x) | 71.738 | 107 of 200 |
| NVFP4, layers 0 and 11 FP8 | 220.0 | 0.394 (1.04x) | 0.937 | 0.04660 (3.6x) | 72.667 | 107 of 200 |
| NVFP4 `fc1` only | 227.9 | - | - | - | - | 0 of 200 |
| NVFP4 `fc2` only | build fails | - | - | - | - | - |

KL and PPL are measured through the decode engine. With a 2048-token cache the
picture is the same: decode 1.00x to 1.01x of fp16 for NVFP4 against 1.04x for
FP8.

Reasons:

- Speed: NVFP4 decode is 0.98x to 1.04x of fp16, slower than FP8 (1.10x), and
  prefill is slower than FP8 as well. A decode step of a 125M model is bound by
  many small kernels, the fp16 `lm_head` and attention, not by weight reads,
  and NVFP4 adds a run-time activation quantize in front of each of its 24
  projections. The plan is only 10% smaller than FP8.
- Quality: KL is 3.6x to 4.9x the FP8 KL.
- TensorRT problems: with `fc2` in NVFP4 the prefill engine returns NaN for
  107 of 200 texts and for all 8 short prompts; the first NaN appears right
  after the first NVFP4 feed-forward block. The decode engine with the same
  weights is finite. The `fc2`-only layout fails to build on TensorRT 11.4 (no
  implementation found for the network). The root cause was not found.

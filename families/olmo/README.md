# OLMo

This family builds OLMo v1 decoders (`allenai/OLMo-1B-hf` in the manifests) for
`text_continuation`: non-parametric LayerNorm, separate q, k, v and o
projections, SwiGLU MLP, RoPE and tied word embeddings. It builds a prefill
engine (`prefill.plan`, dynamic prompt length) and a single-token decode engine
(`engine.plan`) with a static KV cache of `max_sequence_length` tokens
(default 256). Precision is fp32, fp16 or bf16, batch 1, with optional tensor
parallelism for unquantized builds.

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
trtmc build allenai/OLMo-1B-hf -o model.bundle
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
cmake --build build --target trtmc trtmc_backend_trt trtmc_model_olmo test_olmo_task_config
ctest --test-dir build --output-on-failure -R '^olmo_task_config$'
build/test_olmo_sdk_c model.bundle build text "Hello" max_new_tokens=20
build/test_olmo_sdk_cpp model.bundle build text "Hello" max_new_tokens=20
```

The Config CTest target also builds both SDK consumers. Existing E2Es use
`TRTMC_NATIVE_BUILD_DIR` to locate them. The server consumer must support semantic
Tasks before serving a migrated family; legacy-interface inheritance is not kept
in this family to work around an unmigrated application.

## Quantization

`quantization="fp8"` self-quantizes the original checkpoint at build time
(`quantization.py`): the q, k, v, gate, up and down projections of every layer
run as FP8 W8A8 GEMMs (per-tensor weight scale `amax / 448` from the loaded
weights, static per-tensor activation scale from `fp8_activation_scales.json`,
ModelOpt `FP8_DEFAULT_CFG` calibration). The `down_proj` of layers 1, 2, 8, 9,
10 and 15 stays 16 bit (massive activation at its input, see
`EXCLUDED_PROJECTIONS`). The attention output projection, `lm_head`, the
embedding and the norms stay 16 bit. Only the model config
`olmo-h2048-l16-i8192` (OLMo-1B) has calibrated scales. FP8 requires fp16 or
bf16 and no tensor parallelism. No other quantization is accepted.

### NVFP4

NVFP4 was built and measured for OLMo-1B with a quick screen (TensorRT 11.4,
one job on B200 and one on B300). It is not shipped: decode is not faster than
the FP8 build on either GPU (0.96x to 1.00x), and the quality loss is 9x to 17x
the FP8 loss (KL against the PyTorch model), far above the 2x limit. The full
study was not run. No ModelOpt NVFP4 checkpoint of OLMo exists, so the weights
were self-quantized from the original checkpoint. The NVFP4 code was not merged.

Recipe: NVFP4 W4A4 with the weight math of the `glm` family (per-16 e4m3 block
scales, one `weight_scale_2 = amax / (6 x 448)` per tensor, E2M1 values packed
two per byte); activations through `add_dynamic_quantize` (block 16) with a
static global scale equal to the committed FP8 activation scale divided by 6
(same calibration data and statistic). Projections that are not NVFP4 keep the
FP8 layout of `quantization.py`; the excluded `down_proj` layers stay 16 bit in
every variant. The fused gate/up dual-GEMM kernel stays on for NVFP4 (the FP8
build turns it off). The plan inspector shows FP4 GEMMs for every NVFP4
projection.

| Variant | NVFP4 | FP8 W8A8 |
|---|---|---|
| NVFP4 (same projections as FP8) | q, k, v, gate, up, down | none |
| MLP only | gate, up, down | q, k, v |

Results: fp16, `max_sequence_length` 256 (manifest config), engine only, batch
1. Decode is one token at position 100 with CUDA graph replay (eager launch in
brackets), median of 7 interleaved rounds, all variants built and timed in the
same job. KL is KL(PyTorch bf16 || engine) per token, teacher-forced over 531
wikitext-2 test tokens (3 windows of 240 tokens, 64-token prefill then
one-token decode). NaN counts non-finite logits and K/V values.

| GPU | TensorRT | Decode fp16 | Decode FP8 | Decode NVFP4 (x FP8) | Decode MLP only (x FP8) | KL fp16 / FP8 / NVFP4 / MLP only | NaN |
|---|---|---|---|---|---|---|---|
| B200 | 11.4 | 1.126 ms | 0.850 ms | 0.867 ms, 0.98x (0.99x) | 0.859 ms, 0.99x (0.98x) | 0.0007 / 0.0132 / 0.2178 (16.5x) / 0.1257 (9.5x) | 0 |
| B300 | 11.4 | 0.866 ms | 0.767 ms | 0.795 ms, 0.96x (0.96x) | 0.792 ms, 0.97x (0.96x) | 0.0007 / 0.0123 / 0.1605 (13.1x) / 0.1611 (13.1x) | 0 |

- Next-token top-1 agreement with PyTorch (B200): fp16 98.1%, FP8 94.4%,
  NVFP4 81.0%, MLP only 85.5%. Perplexity (PyTorch 19.23): fp16 19.29, FP8
  19.74, NVFP4 24.93, MLP only 22.14.
- Prefill of 128 tokens: NVFP4 1.04x FP8 on B200, 0.99x on B300.
- Plan size, prefill plus decode (B200): fp16 4.39 GiB, FP8 2.71 GiB, NVFP4
  1.97 GiB, MLP only 2.13 GiB. The size gain alone is not a reason to ship.
- NaN scan: 0 non-finite logits or K/V for every variant on both GPUs (random
  tokens, one repeated token and wikitext text through the prefill and decode
  plans, mixed chunk lengths, 32 greedy steps after each).
- A second screen job on each GPU gave the same result (eager decode NVFP4
  0.96x to 1.00x FP8).

Why it does not help: OLMo-1B at batch 1 is only partly weight-bandwidth bound.
FP8 already removes most of the weight traffic (1.13x to 1.32x faster decode
than fp16); NVFP4 halves the remaining projection weights again but adds a
dynamic activation quantization per projection, which costs as much as it
saves. Earlier small-decoder studies (gpt2, opt, bloom, xglm, gpt_neox,
codegen) found the same. Revisit for a larger OLMo model, batched decode, or a
recipe that keeps the KL within 2x of FP8.

The decode plan of this family gives wrong logits when `max_sequence_length`
is 512 or more on TensorRT 11.4, also without quantization (prefill plan
correct; TensorRT 11.1 does not build that plan). Long-context results are
therefore not reported here.

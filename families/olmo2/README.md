# OLMo-2

This family builds OLMo-2 decoders (`allenai/OLMo-2-0425-1B` in the manifests)
for `text_generation`: post-norm residual layout, RMSNorm on q and k, separate
q, k, v and o projections, SwiGLU MLP, RoPE and an untied `lm_head`. It builds
a prefill engine (`prefill.plan`, dynamic prompt length) and a single-token
decode engine (`engine.plan`) with a static KV cache of `max_sequence_length`
tokens. Precision is fp32, fp16 or bf16, batch 1, with optional tensor
parallelism for unquantized builds.

## Quantization

`quantization="fp8"` self-quantizes the original BF16 checkpoint at build time
(`quantization.py`): the q, k, v, gate, up and down projections of every layer
run as FP8 W8A8 GEMMs (per-tensor weight scale `amax / 448` from the loaded
weights, static per-tensor activation scale from `fp8_activation_scales.json`,
ModelOpt `FP8_DEFAULT_CFG` calibration, fp16 Dequantize). The attention output
projection, `lm_head`, the embedding and the norms stay 16 bit. Only
OLMo-2-0425-1B has calibrated scales. FP8 requires fp16 and no tensor
parallelism. No other quantization is accepted.

### NVFP4

NVFP4 was built and measured for OLMo-2-0425-1B with a quick screen (TensorRT
11.4, one job on B200 and one on B300). It is not shipped: decode is at most
1.00x the FP8 build on B200 (eager 1.01x to 1.02x) and 0.92x to 0.95x on B300,
and the quality loss is 10x to 12x the FP8 loss (KL against the PyTorch model),
far above the 2x limit. The full study was not run. No ModelOpt NVFP4
checkpoint of OLMo-2 exists, so the weights were self-quantized from the BF16
checkpoint. The NVFP4 code was not merged.

Recipe: NVFP4 W4A4 with the weight math of the `glm` family (per-16 e4m3 block
scales, one `weight_scale_2 = amax / (6 x 448)` per tensor, E2M1 values packed
two per byte); activations through `add_dynamic_quantize` (block 16) with a
static global scale equal to the committed FP8 activation scale divided by 6
(same calibration data and statistic). Projections that are not NVFP4 keep the
FP8 layout of `quantization.py`. The fused gate/up dual-GEMM kernel stays on
for NVFP4 (the FP8 build turns it off). The plan inspector shows FP4 GEMMs for
every NVFP4 projection.

| Variant | NVFP4 | FP8 W8A8 |
|---|---|---|
| NVFP4 (same projections as FP8) | q, k, v, gate, up, down | none |
| MLP only | gate, up, down | q, k, v |

Results: fp16 (the precision FP8 requires), `max_sequence_length` 256, engine
only, batch 1. Decode is one token at position 100 with CUDA graph replay
(eager launch in brackets), median of 7 interleaved rounds, all variants built
and timed in the same job. KL is KL(PyTorch bf16 || engine) per token,
teacher-forced over 531 wikitext-2 test tokens (3 windows of 240 tokens,
64-token prefill then one-token decode). NaN counts non-finite logits and K/V
values.

| GPU | TensorRT | Decode fp16 | Decode FP8 | Decode NVFP4 (x FP8) | Decode MLP only (x FP8) | KL fp16 / FP8 / NVFP4 / MLP only | NaN |
|---|---|---|---|---|---|---|---|
| B200 | 11.4 | 1.290 ms | 0.946 ms | 0.942 ms, 1.00x (1.02x) | 0.995 ms, 0.95x (0.98x) | 0.0003 / 0.0094 / 0.1169 (12.5x) / 0.0962 (10.3x) | 0 |
| B300 | 11.4 | 0.985 ms | 0.869 ms | 0.911 ms, 0.95x (0.94x) | 0.948 ms, 0.92x (0.92x) | 0.0002 / 0.0095 / 0.1101 (11.6x) / 0.0972 (10.2x) | 0 |

- Next-token top-1 agreement with PyTorch (B200): fp16 99.2%, FP8 94.2%,
  NVFP4 84.0%, MLP only 84.4%. Perplexity (PyTorch 8.66): fp16 8.66, FP8 8.77,
  NVFP4 9.44, MLP only 9.47.
- Prefill of 128 tokens: NVFP4 1.14x FP8 on B200, 1.05x on B300. Prefill
  speed alone does not meet the decode rule.
- Plan size, prefill plus decode (B200): fp16 5.54 GiB, FP8 4.05 GiB, NVFP4
  2.85 GiB, MLP only 3.39 GiB. The size gain alone is not a reason to ship.
- NaN scan: 0 non-finite logits or K/V for every variant on both GPUs (random
  tokens, one repeated token and wikitext text through the prefill and decode
  plans, mixed chunk lengths, 32 greedy steps after each).
- A second screen job on each GPU gave the same result (eager decode NVFP4
  1.01x FP8 on B200, 0.92x on B300).

Why it does not help: OLMo-2-1B at batch 1 is only partly weight-bandwidth
bound. FP8 already removes most of the weight traffic (1.13x to 1.36x faster
decode than fp16); NVFP4 halves the remaining projection weights again but adds
a dynamic activation quantization per projection, which costs as much as it
saves. Earlier small-decoder studies (gpt2, opt, bloom, xglm, gpt_neox,
codegen) found the same. Revisit for a larger OLMo-2 model, batched decode, or
a recipe that keeps the KL within 2x of FP8.

The decode plan of this family gives wrong logits when `max_sequence_length`
is 512 or more on TensorRT 11.4, also without quantization (prefill plan
correct). Long-context results are therefore not reported here.

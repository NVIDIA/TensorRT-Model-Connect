# Granite

This family builds IBM Granite decoders (`ibm-granite/granite-3.1-2b-base` in
the manifests) for `text_generation`. Granite is a LLaMA-style decoder (RMSNorm,
grouped-query attention, RoPE, SwiGLU MLP) with four extra multipliers
(embedding, attention, residual and logits scaling); they are absorbed into the
weights at load time, so the standard decoder builder is reused. It builds a
prefill engine (`prefill.plan`, dynamic prompt length) and a single-token
decode engine (`engine.plan`) with a static KV cache of `max_sequence_length`
tokens (default 256). Precision is fp32, fp16 or bf16, batch 1, with optional
tensor parallelism for unquantized builds.

## Quantization

`quantization="fp8"` self-quantizes the original BF16 checkpoint at build time
(`quantization.py`): the q, k, v, gate, up and down projections of every layer
run as FP8 W8A8 GEMMs (per-tensor weight scale `amax / 448` taken after the
Granite multipliers are absorbed, static per-tensor activation scale from
`fp8_activation_scales.json`, ModelOpt `FP8_DEFAULT_CFG` calibration, fp32
Dequantize and matmul). The attention output projection, `lm_head`, the
embedding and the norms stay 16 bit. Only Granite-3.1-2B has calibrated scales.
FP8 requires fp16 or bf16 and no tensor parallelism. No other quantization is
accepted.

### NVFP4

NVFP4 was built and measured for Granite-3.1-2B (TensorRT 11.4 on B200 and
B300, TensorRT 11.1 on B200). It is not shipped: decode is slower than the FP8
build on both GPUs (0.95x FP8 on B200 and 0.87x on B300 at
`max_sequence_length` 4096, 0.80x to 0.88x at 256), and the quality loss is
8.7x to 8.9x the FP8 loss (KL against the PyTorch model), far above the 2x
limit. No ModelOpt NVFP4 checkpoint of Granite exists, so the weights were
self-quantized from the BF16 checkpoint. The NVFP4 code was not merged.

Recipe: NVFP4 W4A4 with the weight math of the `glm` family (per-16 e4m3 block
scales, one `weight_scale_2 = amax / (6 x 448)` per tensor, E2M1 values packed
two per byte), taken after the Granite multipliers are absorbed. Activations go
through `add_dynamic_quantize` (block 16) with a static global scale equal to
the committed FP8 activation scale divided by 6 (same calibration data and
statistic). Projections that are not NVFP4 keep the FP8 layout of
`quantization.py`. The fused gate/up dual-GEMM kernel stays on for NVFP4 (the
FP8 build turns it off). The plan inspector shows FP4 GEMMs for every NVFP4
projection.

| Variant | NVFP4 | FP8 W8A8 |
|---|---|---|
| NVFP4 (same projections as FP8) | q, k, v, gate, up, down | none |
| MLP only | gate, up, down | q, k, v |

Results: bf16, batch 1, engine only, all variants built through the family
`build()` and timed in the same job. Decode is one token at position 100
(eager enqueue, median of 7 interleaved rounds of 200 calls; position 3996
gives the same numbers). KL is KL(PyTorch bf16 || engine) per token,
teacher-forced over 5766 wikitext-2 test tokens (6 windows of 1024 tokens,
64-token prefill then one-token decode), `max_sequence_length` 4096. NaN counts
non-finite logits and K/V values.

| GPU | TensorRT | Max seq | Decode bf16 | Decode FP8 | Decode NVFP4 (x FP8) | Decode MLP only (x FP8) |
|---|---|---|---|---|---|---|
| B200 | 11.4 | 4096 | 4.839 ms | 3.929 ms | 4.120 ms, 0.95x | 4.018 ms, 0.98x |
| B300 | 11.4 | 4096 | 4.501 ms | 3.744 ms | 4.283 ms, 0.87x | 4.041 ms, 0.93x |
| B200 | 11.1 | 4096 | 4.997 ms | 3.967 ms | 4.315 ms, 0.92x | 4.252 ms, 0.93x |
| B200 | 11.4 | 256 | 2.940 ms | 2.153 ms | 2.443 ms, 0.88x | 2.243 ms, 0.96x |
| B300 | 11.4 | 256 | 2.822 ms | 2.138 ms | 2.662 ms, 0.80x | 2.391 ms, 0.89x |

| GPU | TensorRT | KL bf16 / FP8 / NVFP4 / MLP only | Top-1 bf16 / FP8 / NVFP4 / MLP only | NaN |
|---|---|---|---|---|
| B200 | 11.4 | 0.0010 / 0.0146 / 0.1297 (8.9x) / 0.0945 (6.5x) | 98.4% / 94.5% / 84.4% / 87.1% | 0 |
| B300 | 11.4 | 0.0010 / 0.0146 / 0.1270 (8.7x) / 0.0955 (6.5x) | 98.6% / 94.6% / 85.0% / 86.7% | 0 |

- Perplexity (PyTorch 5.73, B200): bf16 5.74, FP8 5.79, NVFP4 6.28, MLP only
  6.17. TensorRT 11.1 on B200 (2 windows, 1922 tokens) gives the same picture:
  KL 0.0150 / 0.1546 (10.3x) / 0.1243 (8.3x).
- Prefill of 128 tokens: NVFP4 1.01x FP8 on B200, 0.97x on B300.
- Bundle size, prefill plus decode, `max_sequence_length` 4096: bf16 9.83 GiB,
  FP8 5.62 GiB, NVFP4 3.77 GiB, MLP only 4.45 GiB. The size gain alone is not a
  reason to ship.
- NaN scan: 0 non-finite logits or K/V for every variant on both GPUs and on
  11.1 (random tokens, one repeated token and wikitext text, 4064 positions
  through the prefill and decode plans with mixed chunk lengths, then greedy
  steps; same at `max_sequence_length` 256).

Why it does not help: Granite-3.1-2B at batch 1 is only partly
weight-bandwidth bound. FP8 already gives 1.20x to 1.37x faster decode than
bf16; NVFP4 halves the remaining projection weights again but adds a dynamic
activation quantization per projection, which costs more than it saves. The
quality loss is also large for a 2B model. Revisit for a larger Granite model,
batched decode, or a recipe that keeps the KL within 2x of FP8.

# Chronos-Bolt

This family builds Chronos-Bolt forecasters (`amazon/chronos-bolt-tiny` in the
manifests; mini, small and base build from the same code path) for
`time_series_forecast` as one TensorRT engine: a T5 encoder-decoder, batch 1,
context 2048, fp32 (default) or fp16. The family has no quantized build.

## FP8 quantization study (not shipped)

FP8 self-quantization on top of the fp16 graph was built and measured on
TensorRT 11.4 (and 11.1) for the four Chronos-Bolt models. It is not shipped:
at batch 1 FP8 is 0.99x of fp16 for tiny (the manifest model) and 1.05x to
1.12x for small and base, and the forecast deviation from the reference grows
about 10x over fp16. This section records the recipe and the numbers, so that
the study does not have to be repeated. The FP8 code was not merged; this
family has no FP8 build.

### Recipe that was measured

No FP8 or NVFP4 checkpoint of Chronos-Bolt exists on the Hub (only the fp32
checkpoints and community ONNX exports), and no official quantization config
exists for it. Each quantized linear is built as
`input -> Quantize(FP8) -> Dequantize(fp16) -> MatMul`, with the weight
`FP8 constant -> Dequantize(fp16)`.

| Choice | Decision |
|---|---|
| Quantized | encoder self-attention q/k/v and feed-forward `wi`, `wo`; decoder cross-attention k/v and feed-forward `wi`, `wo` (tiny and mini 24, small 36, base 72 modules) |
| Not quantized | attention output `o` (FP8 after the fused attention does not build on TensorRT 11.4; it builds on 11.1), decoder self-attention and cross-attention q (one row), patch embedding, quantile head, norms, attention, instance normalization |
| Weight scale | per tensor, amax / 448, at build time |
| Activation scale | static, per tensor; ModelOpt `FP8_DEFAULT_CFG` (max calibration) on 512 public windows (`monash_electricity_hourly`, `m4_daily`), not the evaluation data |
| Dequantize type | fp16; no non-finite value |

### Results

TensorRT 11.4.0.106, B200 and B300, CUDA graph replay, median of interleaved
rounds; quality on 414 M4 hourly and 862 traffic held-out series (TensorRT
11.4, B200). Deviation is the mean absolute difference to the reference
`ChronosBoltPipeline` (fp32), divided by the MASE scale of each series.

| Item | fp16 | FP8 | FP8 vs fp16 |
|---|---|---|---|
| tiny, batch 1, B200 / B300 | 0.228 / 0.232 ms | 0.231 / 0.235 ms | 0.99x / 0.99x |
| mini, batch 1, B200 / B300 | 0.343 / 0.323 ms | 0.339 / 0.324 ms | 1.01x / 1.00x |
| small, batch 1, B200 / B300 | 0.437 / 0.466 ms | 0.400 / 0.418 ms | 1.09x / 1.12x |
| base, batch 1, B200 / B300 | 0.980 / 0.950 ms | 0.903 / 0.902 ms | 1.09x / 1.05x |
| base, batch 32, B200 / B300 (study-only batched graph) | 2.587 / 2.475 ms | 2.288 / 2.160 ms | 1.13x / 1.15x |
| Plan size, tiny / base (B200) | 17.7 / 394.9 MiB | 12.5 / 253.2 MiB | -29% / -36% |
| base M4 hourly MASE (reference 0.8365) | 0.8366 | 0.8383 | +0.2% |
| base M4 hourly deviation | 0.0035 | 0.033 | 9.5x |
| MASE and WQL, all models and sets | within 0.04% of the reference | within 0.7% of the reference | - |
| Non-finite forecasts (1276 held-out and 12 stress series, 4 models) | 0 | 0 | - |

TensorRT 11.1 (B200, base, batch 1): fp16 0.995 ms, FP8 0.896 ms (1.11x), same
quality within noise.

### Why it is not shipped

- No gain for tiny and mini, including the manifest model.
- 5% to 12% for small and base at batch 1. At batch 1 the encoder GEMMs have
  129 rows and the decoder GEMMs 1 row, so the time is mostly per-kernel cost
  (about 100 to 270 kernels); FP8 only cuts the GEMM share.
- The forecast deviation from the reference is about 10x the fp16 deviation.

### NVFP4 (not tried)

At batch 1 the GEMMs have 129 rows or fewer and the time is bound by kernel
count, not by weight reads. FP8 already shows that halving the weight bytes
gives 0% to 12%, so 4-bit weights cannot pay here.

### When to revisit

- The family accepts batch above 1 (FP8 gives 1.13x to 1.15x for base at
  batch 32).
- A TensorRT release after 11.4 that builds FP8 on the attention output
  projection after the fused attention (more GEMM time in FP8).
- A calibration that brings the forecast deviation close to the fp16 level.

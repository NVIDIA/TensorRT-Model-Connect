# TimesFM

This family builds `google/timesfm-2.0-500m-pytorch` (TimesFM 2.0, 500M) as one
TensorRT engine for the `series_to_point_forecast` task. The engine takes a context
of up to 2048 values (left padded, with a padding mask) and a frequency index,
and exposes the mean forecast for a horizon of 128 steps in one
forward pass; there is no decode loop. The family build accepts precision fp32
only, and the manifests build fp32 (tensor parallel size 1 and 4). Batch size is
1.

## Task SDK migration

This family builds the checkpoint's existing mean-prediction head and exposes
the `series_to_point_forecast` Task through the public C and C++ SDKs.
The Task returns the complete `[horizon, channel]` forecast with one channel and
positive horizon offsets. Channel names and units remain unspecified.

Pass history as `[time, 1]` or an unspecified flat shape. The family keeps the
newest context window, left-pads short histories, and forwards missing
observations as padding. The optional integer `frequency` Config selects the
checkpoint's frequency embedding; its default is zero and its value must be
within the loaded checkpoint's embedding range. Overrides apply to one call.

Rebuild existing `time_series_forecast` bundles after upgrading this family:
the primary Task identity has changed. The TensorRT graph, weights, supported
precision, rank-plan selection, and reference acceptance criteria are unchanged.
Only the existing point head is advertised; quantile, batch, streaming, and
stateful Tasks are not implemented by this family.

With `TRTMC_BINARY`, `TRTMC_RUNTIME_ROOT`, `TRTMC_NATIVE_BUILD_DIR`, and
`TRTMC_REFERENCE_SOURCE_DIR` set for a prepared environment, select the original
`timesfm-2.0-500m-official` or `timesfm-2.0-500m-official-tp4` E2E testcase. The
tests compare the CLI and public C/C++ SDK outputs against the official model.
The SDK consumers also cover error handling, repeated calls, and result lifetime.

## FP8 quantization study (not shipped)

FP8 self-quantization of the decoder was built and measured on TensorRT 11.4. It
is not shipped: FP8 is at most 1.03x faster than fp16 at batch 1 (0.98x on
B300), 1.18x to 1.19x at batch 32, and the forecast error (MASE) on M4 hourly
grows by 11%. NVFP4 was not tried. This section records the recipe, the method
and the numbers, so that the study does not have to be repeated. The FP8 code
was not merged; this family has no FP8 build. All numbers come from a local
study.

The study also found an fp16 overflow that is independent of FP8: an fp16
engine gives non-finite forecasts for series with large normalized values (see
[fp16 overflow found on the way](#fp16-overflow-found-on-the-way)).

### Summary and decision

| Item | fp32 (family default) | fp16 | FP8 | FP8 vs fp16 |
|---|---|---|---|---|
| Time, batch 1, B200 / B300 | 6.94 / 6.64 ms | 1.96 / 2.11 ms | 1.91 / 2.15 ms | 1.03x / 0.98x |
| Time, batch 32, B200 / B300 | 51.45 / 45.91 ms | 4.17 / 3.98 ms | 3.53 / 3.35 ms | 1.18x / 1.19x |
| Plan size (B200) | 1906.4 MiB | 956.3 MiB | 566.1 MiB | -41% |
| M4 hourly MASE (HF fp32: 0.5866) | 0.5866 | 0.5859 | 0.6490 | +11% |
| M4 hourly mean scaled deviation (reference) | 0.00002 (HF) | 0.0167 (HF) | 0.2970 (fp16) | 18x |
| M4 daily, series with non-finite forecast (200 series) | 0 | 1 | 1 | same |

Decision: do not ship FP8 (or NVFP4) for TimesFM.

- Speed: at batch 1 the GEMMs have 64 rows (2048 / patch 32) and are bound by
  fixed per-kernel cost, not by weight reads. FP8 saves 0.02 ms of GEMM time
  on B200 and adds Quantize work to the norm kernels. At batch 32 FP8 saves
  0.6 ms of 2.75 ms GEMM time; norms, transposes and attention (1.8 ms) stay.
- Quality: MASE on M4 hourly rises from 0.5859 to 0.6490 (+11%). No tested
  subset of projections keeps the fp16 deviation level (sensitivity table).
- The large speed gain is fp32 to fp16 (3.2x to 3.5x at batch 1, 11.5x to 12.4x
  at batch 32), not fp16 to FP8. fp16 first needs an overflow guard (see the
  last section).

### Model

| Item | Value |
|---|---|
| Checkpoint | `google/timesfm-2.0-500m-pytorch`, revision `dc2443792ce5516872b89b37cf1bc058c3bf0c10` (fp32 weights) |
| Model config | 50 layers, hidden 1280, intermediate 1280, 16 heads (head dim 80), context 2048, patch 32, horizon 128 |
| Tokens per forecast | 64 patches |
| Family default precision | fp32 (the network builder also has an fp16 path; the family build rejects it) |
| Execution | one forward pass per forecast, no decode loop |

### Recipe that was measured

No FP8 checkpoint exists for TimesFM, so FP8 weights were quantized from the
original checkpoint at build time (never stored). Every quantized projection is
built as:

```
activation -> Quantize(FP8) -> Dequantize(fp16) -> MatMul <- Dequantize(fp16) <- FP8 weight constant
```

| Choice | Decision |
|---|---|
| Quantized projections | `self_attn.q_proj`, `k_proj`, `v_proj`, `mlp.gate_proj`, `mlp.down_proj` of all 50 layers (250 GEMMs) |
| q, k, v input | one shared Quantize/Dequantize (`self_attn.qkv`, they read the same norm output); each weight keeps its own scale |
| Attention output projection (`o_proj`) | fp16: an FP8 Quantize after the fused attention does not build on TensorRT 11.4 (shape inference error, fp8 and half element types in one add) |
| Not quantized | input and horizon residual blocks, frequency embedding, norms, attention matmuls, de-normalization (fp16 or fp32 as in the fp16 build) |
| Weight scale | per tensor, amax / 448 at build time |
| Activation scale | static, per tensor, ModelOpt 0.46.1 `FP8_DEFAULT_CFG` (max calibration), scale = amax / 448 |
| Calibration data | 256 context windows from `monash_traffic` and `monash_electricity_hourly` (`autogluon/chronos_datasets`), fixed random positions (seed 0), context lengths 256 to 2048 |
| Outlier rule | a scale above 20x the median of the same projection in other layers stays fp16; none found, 150 of 150 scales quantized |
| Dequantize type | fp16 (bias add and ReLU fuse into the GEMM epilogue); fp32 Dequantize was not measured |
| Engine precision | fp16 (FP8 needs fp16 activations) |
| Dual-GEMM knob | turned off when TensorRT lists it |
| Variant FP8-MLP | same, but q, k, v stay fp16 (`gate_proj`, `down_proj` only) |

### Method

- GPUs: B200 and B300, one GPU per job; TensorRT 11.4.0.106, PyTorch 2.14,
  Transformers 5.17, ModelOpt 0.46.1. All variants of one GPU type were built
  and timed in one process.
- Time: median of 9 interleaved rounds (100 runs per round at batch 1, 50 at
  batch 32), CUDA events, engine only. The family engine is batch 1; batch 32
  used a study-only copy of the graph with a fixed batch of 32, and its
  forecasts equal the batch-1 forecasts (max scaled difference 0 for fp16 and
  FP8, 2.5e-5 for fp32). FP8-MLP was not timed at batch 32.
- Quality data (held out, not used for calibration), frequency index 0:
  - M4 hourly (`m4_hourly`): all 414 series, the last 48 values as truth,
    season 24.
  - M4 daily (`m4_daily`): 200 random series (seed 1234) of 4227, the last 14
    values as truth, season 7.
- Metrics: MASE = mean absolute error / mean absolute seasonal-naive error of
  the context, averaged over series. Deviation = mean absolute difference of the
  mean forecast to a reference, divided by the same per-series scale.
  Reference: Transformers `TimesFmModelForPrediction` in fp32 (HF).
- Quality numbers are the same on B200 and B300.

### Speed (TensorRT 11.4, median, ms)

| GPU | Batch | fp32 | fp16 | FP8 | FP8-MLP | FP8 vs fp16 |
|---|---|---|---|---|---|---|
| B200 | 1 | 6.94 | 1.96 | 1.91 | 1.92 | 1.03x |
| B200 | 32 | 51.45 | 4.17 | 3.53 | not measured | 1.18x |
| B300 | 1 | 6.64 | 2.11 | 2.15 | 2.10 | 0.98x |
| B300 | 32 | 45.91 | 3.98 | 3.35 | not measured | 1.19x |

Plan size (B200, batch 1): fp32 1906.4 MiB, fp16 956.3 MiB, FP8 566.1 MiB,
FP8-MLP 800.3 MiB. The plan inspector shows 250 layers with e4m3 in the FP8
plan, 150 in the FP8-MLP plan and none in the fp16 plan.

Kernel time by kind (PyTorch profiler, 20 runs, ms per forecast call):

| GPU | Batch | Variant | GEMM | Attention | Other (norms, transposes, casts) |
|---|---|---|---|---|---|
| B200 | 1 | fp16 | 1.20 | 0.23 | 0.87 |
| B200 | 1 | FP8 | 1.18 | 0.23 | 0.90 |
| B200 | 32 | fp16 | 2.75 | 0.48 | 1.33 |
| B200 | 32 | FP8 | 2.12 | 0.48 | 1.34 |
| B300 | 32 | fp16 | 2.60 | 0.45 | 1.25 |
| B300 | 32 | FP8 | 1.98 | 0.45 | 1.27 |

### Quality (B200 and B300, TensorRT 11.4)

| Variant | M4 hourly MASE | Hourly deviation vs fp16 | M4 daily MASE (200 series) | Daily non-finite series |
|---|---|---|---|---|
| HF fp32 | 0.5866 | 0.0167 | 1.1928 | 0 |
| fp32 engine | 0.5866 | 0.0167 | 1.1928 | 0 |
| fp16 engine | 0.5859 | - | not finite | 1 (all 14 values) |
| FP8 | 0.6490 | 0.2970 | not finite | 1 |
| FP8-MLP | 0.6039 | 0.2693 | not finite | 1 |

Sensitivity (B200, one engine per variant; daily MASE over the 199 series with
finite forecasts, fp16 there: 1.1368):

| Quantized | Hourly MASE | Hourly deviation vs fp16 | Daily MASE (199) |
|---|---|---|---|
| none (fp16) | 0.5859 | 0 | 1.1368 |
| all (q, k, v, gate, down) | 0.6490 | 0.2970 | 1.2476 |
| q, k, v only | 0.6445 | 0.1815 | 1.2049 |
| `gate_proj` only | 0.6093 | 0.2064 | 1.1644 |
| `down_proj` only | 0.5708 | 0.1525 | 1.1744 |
| `gate_proj` and `down_proj` (FP8-MLP) | 0.6039 | 0.2693 | 1.1914 |
| all, layers 2 to 47 only | 0.5919 | 0.1934 | 1.2149 |
| all, scales raised to the evaluation-data maximum | 0.6571 | 0.3188 | 1.1997 |

- Every subset moves the forecast 9x to 19x more than fp16 moves it from HF
  (0.0167). `down_proj` only has a lower hourly MASE than fp16 by chance; its
  deviation is 9x the fp16 level and its daily MASE is higher.
- Clipping is not the cause: on M4 hourly 0.5% of (series, layer) pairs of
  `down_proj` exceed the calibrated range and none of q, k, v or `gate_proj`;
  scales that cover the evaluation maximum (last row) are not better.
- `o_proj` in FP8 was not measured: it does not build on TensorRT 11.4.

### NVFP4 (not tried)

NVFP4 was not built (the FP8 code rejects it). Reasons:

- No decode loop and hidden size 1280: the weights are read once per forecast,
  so 4-bit weights do not remove a weight-bandwidth bottleneck.
- At batch 1 the time is fixed per-layer cost (64-row GEMMs); NVFP4 adds a
  dynamic activation-quantize kernel per projection.
- FP8 already raises the M4 hourly MASE by 11%; NVFP4 is lossier than FP8.
- NVFP4 also needs fp16 activations, so it has the same fp16 overflow.

### fp16 overflow found on the way

Independent of FP8, and not fixed by this study: an fp16 engine (the network
builder's fp16 path) gives non-finite forecasts for one of the 200 M4 daily
series. Its input to the input residual block reaches 1.53e9 and its residual
stream 8.96e8, beyond the fp16 maximum 65504 from layer 0 on. The fp32 engine
and HF fp32 are finite on all series. For comparison, the median per-series
maximum on M4 daily is 32.4 (input) and 138 (residual); on M4 hourly the
maximum over all series is 66.2 (input) and 173 (residual). TimesFM
normalizes each series by the mean and standard deviation of its first
unpadded patch, so a series with a nearly flat start can get very large
normalized values (likely cause, not verified per series). `fp32_layers`
cannot fix it, because the residual stream of every layer is out of range.

Only fp32 is safe today. fp32 to fp16 is the large speed gain (3.2x to 3.5x at
batch 1, 11.5x to 12.4x at batch 32) once an overflow guard is added, for
example input scaling before the input residual block.

### What was not tried

| Item | Note |
|---|---|
| fp32 Dequantize, per-channel weight scales, percentile calibration, SmoothQuant | not measured |
| `o_proj` in FP8 | does not build on TensorRT 11.4; retest on newer releases |
| TensorRT 11.1, H100 | not measured |
| Batch sizes other than 1 and 32; FP8-MLP at batch 32 | not measured |
| Native C++ runtime with an FP8 bundle | not run; quality used the engine from Python |
| Data beyond M4 hourly and M4 daily (200 series) | not measured |

### When to revisit

- The fp16 overflow is fixed and fp16 ships: retest FP8 against that fp16.
- A recipe holds the hourly deviation near the fp16 level (0.0167) and MASE
  within about 1% of fp16.
- Batched serving (batch 32 or more) becomes the target and 1.18x is worth the
  quality loss.
- A TensorRT release builds `o_proj` in FP8 or gives a larger FP8 gain at
  batch 1.

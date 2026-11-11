# PatchTSMixer

This family builds PatchTSMixer forecasters
(`ibm-granite/granite-timeseries-patchtsmixer`) for `time_series_forecast` as
one TensorRT engine: batch 1, context 512, 7 channels, prediction length 96.
The premerge manifest asks for fp16 but lists every stage in `fp32_layers`, so
its engine runs in fp32; the TP4 manifest is fp32. The family has no quantized
build.

## FP8 quantization study (not shipped)

FP8 self-quantization of the mixer linears was built and measured on TensorRT
11.4 (and 11.1) for `ibm-granite/granite-timeseries-patchtsmixer`. It is not
shipped: FP8 is slower than a plain fp16 engine (0.88x to 0.98x), only 1.03x to
1.11x faster than the manifest engine (plain fp16 gives 1.08x to 1.23x), and
its deviation from the reference is at least 6x above the family accuracy
gates.
This section records the recipe, the method and the numbers, so that the study
does not have to be repeated. The FP8 code was not merged; this family has no
FP8 build.

### Summary and decision

| Item | fp16 manifest (fp32 engine) | fp16 (no `fp32_layers`) | FP8 | FP8 vs manifest | FP8 vs fp16 |
|---|---|---|---|---|---|
| Latency, batch 1, B200 / B300 (TensorRT 11.4) | 0.1088 / 0.1046 ms | 0.0885 / 0.0860 ms | 0.1006 / 0.0946 ms | 1.08x / 1.11x | 0.88x / 0.91x |
| Latency, batch 32, B200 / B300 (TensorRT 11.4) | 0.3617 / 0.3523 ms | 0.3351 / 0.3275 ms | 0.3503 / 0.3329 ms | 1.03x / 1.06x | 0.96x / 0.98x |
| Plan size, B200 | 1219 KiB | 845 KiB | 861 KiB | -29% | +2% |
| ETTh1 test relative L2 vs Transformers fp32 (726 windows) | 1.0e-07 | 2.1e-04 | 3.2e-03 | - | - |
| ETTh1 test max error vs Transformers fp32 | 8e-06 | 0.015 | 0.20 | - | - |
| ETTh1 test MSE / MASE (Transformers fp32 0.44201 / 1.08854) | 0.44201 / 1.08854 | 0.44201 / 1.08853 | 0.44171 / 1.08810 | - | - |
| Non-finite outputs (1213 held-out windows, 11 stress inputs) | 0 | 0 | 0 | - | - |

Decision: do not ship FP8 for PatchTSMixer.

- Speed: the model has 0.2 M parameters and GEMMs with 32 to 224 rows and K of
  32 to 144. At these sizes the FP8 GEMM kernels are not faster than the fp16
  ones, and the Quantize kernels add work (kernel time table below). FP8 is
  slower than plain fp16 on both GPUs at batch 1 and batch 32. Against the
  manifest engine it saves 0.008 to 0.019 ms, less than plain fp16 saves.
- Quality: forecast MSE and MASE change by less than 0.1%, but the deviation
  from the reference is 15x that of fp16 (relative L2 3.2e-3, max error 0.20).
  The premerge gate (relative L2 5e-4, max pointwise error 2.5e-4) and the
  ETTh1 nightly gate (relative L2 5e-4, max pointwise error 2.5e-2) would fail.
- Size: the plan is 358 KiB smaller than the manifest engine, but slightly
  larger than plain fp16 (the Q/DQ overhead is larger than the weight saving at
  d_model 48).

### Recipe that was measured

No FP8 or NVFP4 checkpoint of this model exists on the Hub (the Hub search
finds the ibm fp32 checkpoints, ONNX exports and a small random test model),
and no official quantization config exists for it. Each quantized linear is
built as `input -> Quantize(FP8) -> Dequantize(fp16) -> MatMul -> + bias`, with
the weight `FP8 constant -> Dequantize(fp16)`.

| Choice | Decision |
|---|---|
| Quantized (`FP8`, 12 linears) | in each of the 2 mixer layers, in the patch mixer and the feature mixer: `mlp.fc1`, `mlp.fc2`, `gating_block.attn_layer`; K in {32, 48, 96, 144} |
| Variant `FP8 + head` | the above plus the forecast head `head.base_forecast_block` (largest weight, 1536 x 96) |
| Not quantized | patcher (K = 16), layer norms, softmax gating, std scaling, patchify |
| Base precision | fp16 without `fp32_layers` (FP8 needs fp16 activations) |
| Weight scale | per tensor, amax / 448 of the fp32 weight, at build time |
| Activation scale | static, per tensor, amax / 448 (ModelOpt `FP8_DEFAULT_CFG`, max calibration) over 512 windows of 512 rows from the ETTh1 and ETTh2 training split (rows 0 to 8639), seed 0 |
| Outlier rule | a scale above 20x the median of its kind stays fp16; measured scales 0.0081 to 0.0213, no outlier |
| Dequantize type | fp16; no non-finite value, fp32 Dequantize was not needed |

The plan has 20 (`FP8`) and 22 (`FP8 + head`) layers with FP8 inputs, and the
profile shows FP8 x FP8 CUTLASS GEMM kernels for the quantized linears, so the
Q/DQ pairs fuse. The missing speedup does not come from an unfused path.

### Method

- GPUs: B200 and B300, one GPU per job, all variants built and timed in the
  same job; TensorRT 11.4.0.106 (target) and 11.1.0.106 (B200, batch 1).
- Reference: the Transformers `PatchTSMixerForPrediction` model in fp32.
- Data: held-out test splits, normalized with the training-split mean and
  standard deviation: ETTh1 (from row 11520, stride 8, 726 windows), ETTh2
  (from row 11520, stride 24, 242 windows), ETTm1 (rows 46080 to 69680,
  stride 96, 245 windows). Calibration uses only the training split.
- Quality: relative L2 and max error against the reference, and forecast MSE
  and MASE against the ground truth.
- Non-finite scan: the 1213 held-out windows and 11 stress inputs (zeros,
  constant, scaled by 1e-6 to 1e6, offset 1e6, spike, random walk, noise, level
  shift, large sine), all variants.
- Speed: one forecast call, CUDA graph replay, median of 21 interleaved rounds
  of 200 calls, spread at most 1.1% (one manifest-engine B300 row 9.3%). The
  family accepts batch 1 only; batch 32 is a scratch build of one series with
  224 channels (in `common_channel` mode every channel is independent, and the
  outputs match the reference on the same 32 windows: relative L2 1.0e-07
  manifest engine, 1.9e-04 fp16, 3.0e-03 FP8).

### Results

Speed and size (ms per forecast call; x is manifest-engine time over variant
time):

| GPU | TensorRT | Batch | fp32 | manifest (fp32 engine) | fp16 | FP8 | FP8 + head | x fp16 / FP8 / FP8 + head |
|---|---|---|---|---|---|---|---|---|
| B200 | 11.4 | 1 | 0.1102 | 0.1088 | 0.0885 | 0.1006 | 0.1004 | 1.23 / 1.08 / 1.08 |
| B200 | 11.4 | 32 | 0.3637 | 0.3617 | 0.3351 | 0.3503 | 0.3513 | 1.08 / 1.03 / 1.03 |
| B300 | 11.4 | 1 | 0.1044 | 0.1046 | 0.0860 | 0.0946 | 0.0942 | 1.22 / 1.11 / 1.11 |
| B300 | 11.4 | 32 | 0.3541 | 0.3523 | 0.3275 | 0.3329 | 0.3345 | 1.08 / 1.06 / 1.05 |
| B200 | 11.1 | 1 | 0.1106 | 0.1086 | 0.0901 | 0.1024 | 0.1004 | 1.21 / 1.06 / 1.08 |

| Plan size (KiB) | fp32 | manifest | fp16 | FP8 | FP8 + head |
|---|---|---|---|---|---|
| B200, TensorRT 11.4 | 1219 | 1219 | 845 | 861 | 691 |
| B300, TensorRT 11.4 | 1220 | 1220 | 848 | 872 | 730 |
| B200, TensorRT 11.1 | 1249 | 1225 | 845 | 858 | 689 |

Quality (TensorRT 11.4; B200 and B300 give the same numbers; relative L2 and
max error against the Transformers fp32 model):

| Set (windows) | Engine | Relative L2 | Max error | MSE | MASE |
|---|---|---|---|---|---|
| ETTh1 test (726) | Transformers fp32 | - | - | 0.44201 | 1.08854 |
| | manifest (fp32 engine) | 1.0e-07 | 8e-06 | 0.44201 | 1.08854 |
| | fp16 | 2.1e-04 | 0.015 | 0.44201 | 1.08853 |
| | FP8 | 3.2e-03 | 0.20 | 0.44171 | 1.08810 |
| | FP8 + head | 5.7e-03 | 0.36 | 0.44183 | 1.08835 |
| ETTh2 test (242) | fp16 | 4.7e-05 | 0.016 | 0.21805 | 1.11030 |
| | FP8 | 9.4e-04 | 0.37 | 0.21798 | 1.11029 |
| ETTm1 test (245) | fp16 | 1.2e-04 | 0.008 | 0.81693 | 0.82392 |
| | FP8 | 3.7e-03 | 0.16 | 0.81672 | 0.82408 |

TensorRT 11.1 (B200) gives the same quality numbers for fp16 and FP8.

Kernel time per call, B300, TensorRT 11.4 (profiler, ms):

| Batch | Engine | GEMM | other (norm, softmax, scaling, patchify, Quantize) | kernels |
|---|---|---|---|---|
| 1 | manifest (fp32 engine) | 0.062 | 0.055 | 33 |
| 1 | fp16 | 0.060 | 0.050 | 28 |
| 1 | FP8 | 0.009 fp16 + 0.055 FP8 | 0.059 | 32 |
| 32 | manifest (fp32 engine) | 0.085 | 0.391 | 263 |
| 32 | fp16 | 0.082 | 0.383 | 258 |
| 32 | FP8 | 0.010 fp16 + 0.074 FP8 | 0.394 | 262 |

GEMMs are about half of the batch-1 time and a fifth of the batch-32 time. The
FP8 engine spends as long in GEMMs as the fp16 engine (0.064 vs 0.060 ms at
batch 1, 0.084 vs 0.082 ms at batch 32), and the other kernels take 0.009 to
0.011 ms longer (the Quantize kernels). At batch 32 the time is in the
patchify and element-wise part, not in the GEMMs.

### NVFP4 (not tried)

The model has 0.2 M parameters and K of 32 to 144: the engine is limited by
kernel launches and element-wise work, not by weight reads, and FP8 already
gives no GEMM gain.

### Observations on main (not changed)

- Plain fp16 (no `fp32_layers`) is 1.22x to 1.23x faster than the manifest
  engine at batch 1 and 1.08x at batch 32, with no non-finite output and the
  same MSE and MASE. Its max error against the reference on ETTh1 (0.015) is
  above the premerge threshold (2.5e-4), which is likely why the manifest keeps
  every stage in fp32.
- `add_patchify` builds the patches with one slice layer per channel and patch
  (7 x 32 at this model config). With 224 channels this part takes most of the
  time (0.39 ms of 0.47 ms of kernel time); at the family's 7 channels it does
  not matter. A reshape (the patch stride equals the patch length here) would
  remove it. Not measured.

### What was not tried

| Item | Note |
|---|---|
| Per-channel weight scales, percentile or entropy calibration | not measured; could reduce the deviation, not the time |
| Other PatchTSMixer checkpoints, H100, native runtime with an FP8 bundle | not measured |
| Batch 32 on TensorRT 11.1 | the 224-channel builds did not finish in 40 minutes and were stopped (11.4 builds them in about 1 minute) |

### When to revisit

- A larger PatchTSMixer (larger `d_model` or more mixer layers) whose GEMMs are
  large enough for FP8 kernels to be faster than fp16.
- A TensorRT release that fuses the activation Quantize into the producing
  layer (layer norm, gating) for small GEMMs.
- An accuracy gate for quantized builds based on forecast metrics (MSE, MASE)
  instead of the deviation from the fp32 reference.

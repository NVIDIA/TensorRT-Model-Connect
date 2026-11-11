# timm HRNet

This family builds `timm/hrnet_w18.ms_aug_in1k` (architecture `hrnet_w18`) for
`image_to_class_scores` as one TensorRT engine: 224 x 224 input, batch 1, fp16
or fp32 (the manifest uses fp16), builder optimization level 1. The family has
no quantized build.

## FP8 quantization study (not shipped)

FP8 self-quantization of the convolutions was built and measured on TensorRT
11.4 (and 11.1) for `timm/hrnet_w18.ms_aug_in1k`. It is not shipped: FP8 is
slower than fp16 at batch 1 (0.92x to 0.93x) and at batch 32 (0.89x to 0.94x),
and ImageNet top-1 drops by 0.7 to 0.8 point. The only gain is a 38% smaller
plan. This section records the recipe, the method and the numbers, so that the
study does not have to be repeated. The FP8 code was not merged; this family has
no FP8 build.

### Summary and decision

| Item | fp16 (manifest) | FP8 (fp16 engine) | FP8 vs fp16 |
|---|---|---|---|
| Latency, batch 1, B200 / B300 (TensorRT 11.4) | 1.298 / 1.234 ms | 1.418 / 1.334 ms | 0.92x / 0.93x |
| Latency, batch 32, B200 / B300 (TensorRT 11.4) | 3.344 / 3.019 ms | 3.554 / 3.390 ms | 0.94x / 0.89x |
| Plan size, batch 1 | 43.1 MiB | 26.9 MiB | -38% |
| ImageNet top-1, B200 / B300 (PyTorch fp32 78.53%) | 78.60% / 78.40% | 77.80% / 77.73% | -0.7 to -0.8 point |
| Non-finite outputs | 0 | 0 | none |

Decision: do not ship FP8 for timm HRNet.

- Speed: the convolutions do run in FP8 (308 FP8 convolution kernels in the
  plan), but they are narrow (18 to 144 channels per branch) and gain only 5%
  at batch 1 (2.88 to 2.73 ms of kernel time on B200). The Quantize of most
  convolution inputs is not fused into the producing layer and runs in extra
  pointwise kernels that cost more than the gain (see "Kernel time per group").
- Why the Quantize does not fuse: in HRNet most convolution inputs are the
  output of a residual add + ReLU, or of a multi-resolution fuse sum, that also
  feeds an fp16 shortcut or another branch. That fp16 tensor must still be
  written, so TensorRT adds a second pointwise kernel (or a second output) for
  the FP8 copy.
- The only gain is plan size (16 MiB less). Plan size alone is not a reason to
  ship.

### Recipe that was measured

No FP8 or NVFP4 checkpoint of timm HRNet exists on the Hub (one INT8 Lite-HRNet
for a mobile runtime, a different model), and no official quantization config
exists for it. Each quantized convolution is built as
`input -> Quantize -> Dequantize(fp16) -> Convolution(+bias)`, with the kernel
`FP8 constant -> Dequantize(fp16)`.

| Choice | Decision |
|---|---|
| Quantized (324 of 325 convolutions) | every convolution except the stem convolution (3 input channels) |
| fp16 | the stem convolution, the classifier |
| Weight scale | per output channel, amax / 448 of the batch-norm-folded kernel, at build time (per tensor measured as a variant) |
| Activation scale | static, per tensor, from ModelOpt `FP8_DEFAULT_CFG` (max calibration) on the timm PyTorch model, 500 calibration images |
| Dequantize type | fp16; no non-finite value, fp32 Dequantize was not needed |

### Method

- GPUs: B200 and B300, one GPU per job, all variants built and timed in the same
  job; TensorRT 11.4.0.106 (target) and 11.1.0.106 (B200 check).
- Graph: the family's own graph with the family builder settings, with only the
  Q/DQ convolution wrapper added. Batch 32 is a scratch build (the family builds
  batch 1 only).
- Data: 2000 ImageNet validation images from `clip-benchmark/wds_imagenet1k`
  (every 25th image of each shard, shuffled with seed 0): every 4th of them
  (500) for calibration, the other 1500 held out. timm preprocessing of the
  checkpoint.
- Quality: top-1 against the labels, top-1 agreement and KL divergence against
  the timm PyTorch fp32 model on the 1500 held-out images, batch 1. Batch 32 is
  checked for top-1 and non-finite outputs.
- Non-finite scan: all outputs of all variants, batch 1 and 32, B200 and B300.
- Speed: CUDA graph replay, median of 5 interleaved rounds (200 calls per round
  at batch 1, 50 at batch 32), spread at most 0.1%.

### Results

Speed and size (TensorRT 11.4, ms per engine call; x is fp16 time over FP8
time):

| GPU | Batch | fp32 ms | fp16 ms | FP8 per channel ms | x | FP8 per tensor ms | x | plan MiB fp16 to FP8 |
|---|---|---|---|---|---|---|---|---|
| B200 | 1 | 1.669 | 1.298 | 1.418 | 0.92 | 1.429 | 0.91 | 43.1 to 26.9 |
| B200 | 32 | 4.422 | 3.344 | 3.554 | 0.94 | 3.543 | 0.94 | 47.6 to 33.2 |
| B300 | 1 | 1.650 | 1.234 | 1.334 | 0.93 | 1.317 | 0.94 | 42.7 to 26.7 |
| B300 | 32 | 4.528 | 3.019 | 3.390 | 0.89 | 3.420 | 0.88 | 48.0 to 33.6 |

TensorRT 11.1.0.106 check (B200, FP8 per channel, same method):

| GPU | Batch | fp16 ms | FP8 ms | x | plan MiB fp16 to FP8 | Top-1 fp16 / FP8 |
|---|---|---|---|---|---|---|
| B200 | 1 | 1.300 | 1.441 | 0.90 | 43.1 to 26.9 | 78.60% / 77.80% |
| B200 | 32 | 3.441 | 3.561 | 0.97 | 47.5 to 33.2 | - |

Quality (1500 held-out images, batch 1, TensorRT 11.4):

| Engine | Top-1 B200 / B300 | Agreement with PyTorch top-1 | KL mean | KL p99 |
|---|---|---|---|---|
| PyTorch fp32 | 78.53% | - | - | - |
| fp16 | 78.60% / 78.40% | 99.93% / 99.73% | < 0.0001 | < 0.001 |
| FP8, per-channel weights | 77.80% / 77.73% | 95.67% / 95.60% | 0.021 / 0.020 | 0.20 |
| FP8, per-tensor weights | 77.73% / 78.20% | 95.33% / 95.53% | 0.022 / 0.023 | 0.22 to 0.26 |

Kernel time per group (TensorRT 11.4, FP8 per channel, layer profiler, ms; the
profiler adds time per layer, so the sums are larger than the latencies above):

| Kernel group | B200 fp16 b1 | B200 FP8 b1 | B300 fp16 b1 | B300 FP8 b1 | B300 fp16 b32 | B300 FP8 b32 |
|---|---|---|---|---|---|---|
| convolutions (309 layers) | 2.877 | 2.729 | 2.951 | 2.766 | 4.440 | 4.421 |
| pointwise kernels that contain a Quantize | - | 0.948 (142 layers) | - | 0.936 (142) | - | 1.305 (142) |
| of these, Quantize only (no add or ReLU) | - | 0.714 (107) | - | 0.693 (107) | - | 0.847 (107) |
| other (add, ReLU, resize, pooling, classifier) | 0.450 (66) | 0.219 (31) | 0.460 (70) | 0.216 (31) | 0.809 (94) | 0.258 (31) |
| layers in the plan | 375 | 482 | 379 | 482 | 403 | 482 |

### NVFP4 (not tried)

TensorRT does not support NVFP4 on convolutions (block-scaled FP4 needs a 2-D
GEMM), FP8 already loses 0.7 to 0.8 top-1 point, and the engine is not limited
by weight reads.

### What was not tried

| Item | Note |
|---|---|
| Builder optimization level 3 | the family builds at level 1; FP8 was measured at level 1 only |
| Percentile or entropy calibration, quantization-aware training | not measured; could recover accuracy, not speed |
| Wider HRNet models (W30 to W64), H100, native runtime with an FP8 bundle | not measured; the family qualifies only `hrnet_w18` |

### When to revisit

- A TensorRT release that fuses the Quantize into a residual add or fuse sum
  whose fp16 output has a second consumer (one kernel that writes both the fp16
  and the FP8 tensor).
- The family qualifies the wide HRNet models (W48, W64), where the convolutions
  are wider and take a larger share of the time.

# timm RegNet

This family builds `timm/regnety_040.ra3_in1k` (RegNetY-4.0GF, 224 x 224) for
`image_to_class_scores` as one TensorRT engine at batch 1. The manifest builds
fp16 (fp32 is also supported). The family has no quantized build.

## FP8 quantization study (not shipped)

FP8 self-quantization of the convolutions was built and measured on TensorRT
11.4 (and 11.1) for `timm/regnety_040.ra3_in1k`. It is not shipped: FP8 is
slower than fp16 at batch 1 (0.94x to 0.96x) and at batch 32 (0.94x to 0.96x)
on B200 and B300, and FP8 grouped convolutions are split into one convolution
per group (0.53x to 0.72x). This section records the recipe, the method and
the numbers, so that the study does not have to be repeated. The FP8 code was
not merged; this family has no FP8 build.

### Summary and decision

| Item | fp16 (manifest) | FP8 (fp16 engine) | FP8 vs fp16 |
|---|---|---|---|
| Latency, batch 1, B200 / B300 (TensorRT 11.4) | 0.520 / 0.510 ms | 0.541 / 0.538 ms | 0.96x / 0.95x |
| Latency, batch 32, B200 / B300 (TensorRT 11.4) | 1.258 / 1.188 ms | 1.320 / 1.264 ms | 0.95x / 0.94x |
| Plan size, batch 1, B200 / B300 | 40.5 / 40.0 MiB | 30.0 / 29.5 MiB | -26% |
| ImageNet top-1, B200 (PyTorch fp32 82.67%) | 82.73% | 82.40% | -0.3 points |
| Top-1 agreement with PyTorch, B200 | 99.93% | 96.87% | -3.1 points |
| Non-finite outputs | 0 | 0 | none |

Decision: do not ship FP8 for timm RegNet.

- Speed: the 48 FP8 1x1 convolutions are small at batch 1 (7 x 7 to 112 x 112
  maps) and save less kernel time than the stand-alone Quantize kernels cost.
  Each block input feeds both the shortcut add and the first 1x1 convolution,
  and the squeeze-excite gate multiplies the 3x3 convolution output, so
  TensorRT cannot write the FP8 activation from the producing kernel: it adds
  separate multiply-and-cast kernels (165 to 187 kernels per call).
- Grouped convolutions: in FP8, TensorRT 11.4 splits each of the 22 grouped
  3x3 convolutions (group width 64) into one convolution per group (152 FP8
  convolution kernels instead of 22; plan 167 to 337 layers), and the engine
  is about 2x slower at batch 1. So the grouped convolutions stay fp16.
- Plan size: the 26% smaller plan (38% with the grouped convolutions in FP8)
  comes with no speed gain, so it is not a reason to ship.

### Recipe that was measured

No FP8 or NVFP4 checkpoint of this model exists on the Hub, and no official
quantization config exists for it. Each quantized convolution is built as
`input -> Quantize -> Dequantize(fp16) -> Convolution(+bias)`, with the kernel
`FP8 constant -> Dequantize(fp16)`.

| Choice | Decision |
|---|---|
| Quantized (48 of 115 convolutions) | every dense convolution on a spatial map larger than 1x1 (the 1x1 convolutions of each block and the downsample shortcuts) |
| fp16 | the 3x3 stem (3 input channels), the 22 grouped 3x3 convolutions, the 44 squeeze-excite convolutions (on a 1x1 map), the classifier |
| Weight scale | per tensor (amax / 448) of the batch-norm-folded kernel at build time; per output channel measured as a variant |
| Activation scale | static, per tensor, max \|input\| / 448 over 500 calibration images (the statistic of ModelOpt max calibration), read from an fp32 engine of the family graph |
| Outlier rule | a scale above 20x the median of its kind stays fp16; measured: no outlier (largest ratio 2.1x) |
| Dequantize type | fp16; no non-finite value, fp32 Dequantize was not needed |

### Method

- GPUs: B200 and B300, one GPU per job, all variants built and timed in the
  same job; TensorRT 11.4.0.106 (target) and 11.1.0.106.
- Data: 2000 ImageNet validation images from `clip-benchmark/wds_imagenet1k`:
  500 for calibration, the other 1500 held out. Family preprocessing.
- Quality: top-1 against the labels, top-1 agreement and KL divergence against
  the timm PyTorch fp32 model on the 1500 held-out images.
- Non-finite scan: the 2000 images and 29 stress inputs (noise with standard
  deviation 1, 3 and 10, constants), all variants, B200 and B300.
- Speed: CUDA graph replay, median of 9 interleaved rounds of 100 calls,
  spread at most 2.7%. Batch 32 is a scratch build (the family builds batch 1
  only). The engines use the family graph and builder settings (builder
  optimization level 1); the family code did not change between the study
  and this note.

### Results

Speed and size, TensorRT 11.4 (ms per engine call; x is fp16 time over FP8
time):

| GPU | Batch | fp32 ms | fp16 ms | FP8 ms | x | FP8 per-channel ms (x) | FP8 + grouped ms (x) |
|---|---|---|---|---|---|---|---|
| B200 | 1 | 0.814 | 0.520 | 0.541 | 0.96 | 0.543 (0.96) | 0.964 (0.54) |
| B200 | 32 | 2.176 | 1.258 | 1.320 | 0.95 | 1.316 (0.96) | 1.759 (0.72) |
| B300 | 1 | 0.779 | 0.510 | 0.538 | 0.95 | 0.542 (0.94) | 0.970 (0.53) |
| B300 | 32 | 2.088 | 1.188 | 1.264 | 0.94 | 1.263 (0.94) | 1.693 (0.70) |

Plan size and layers (batch 1, B300): fp16 40.0 MiB and 167 layers, FP8
29.5 MiB and 189 layers, FP8 + grouped 24.8 MiB and 337 layers.

TensorRT 11.1 sanity check (B200, same method, latest main):

| Batch | fp16 ms | FP8 ms | x | FP8 + grouped ms (x) | Top-1 fp16 / FP8 |
|---|---|---|---|---|---|
| 1 | 0.498 | 0.539 | 0.92 | 0.935 (0.53) | 82.67% / 82.33% |
| 32 | 1.254 | 1.310 | 0.96 | 1.750 (0.72) | 82.73% / 82.73% |

TensorRT 11.1 gives the same kernel counts as 11.4 (165 fp16, 187 FP8, 335
FP8 + grouped), so the grouped convolutions are also split per group on 11.1.

Quality (1500 held-out images; TensorRT 11.4):

| Engine | Top-1 B200 | Top-1 B300 | Agreement with PyTorch top-1 (B200) | KL mean (B200) | KL p99 (B200) |
|---|---|---|---|---|---|
| PyTorch fp32 | 82.67% | - | - | - | - |
| fp16 engine | 82.73% | 82.67% | 99.93% | < 0.0001 | 0.0001 |
| FP8 | 82.40% | 82.87% | 96.87% | 0.0107 | 0.162 |
| FP8 per-channel weights | 83.00% | 82.60% | 96.87% | 0.0091 | 0.108 |
| FP8 + grouped | 82.60% | 82.73% | 96.20% | 0.0139 | 0.170 |

No variant gives a NaN or Inf on the 2000 images and 29 stress inputs (B200,
B300, batch 1 and 32). The FP8 KL is about 3x the KL of the same recipe on
timm ConvNeXt (0.0107 against 0.0032, B200, batch 1).

Kernel time per group, TensorRT 11.4, batch 1 (CUDA profiler, sum of kernel
time per call, ms):

| GPU | Kernel group | fp16 | FP8 | FP8 + grouped |
|---|---|---|---|---|
| B200 | convolutions and GEMMs | 0.508 | 0.498 | 1.133 |
| B200 | other (pointwise, Quantize, pooling) | 0.127 | 0.188 | 0.241 |
| B200 | kernels per call | 165 | 187 | 335 |
| B300 | convolutions and GEMMs | 0.598 | 0.558 | 1.166 |
| B300 | other (pointwise, Quantize, pooling) | 0.053 | 0.107 | 0.171 |
| B300 | kernels per call | 165 | 187 | 335 |

The FP8 convolutions save 0.01 to 0.04 ms; the added Quantize kernels cost
0.05 to 0.06 ms.

### Builder optimization level 3

The numbers above use builder optimization level 1, the family setting when
the study was made. Level 3 makes the fp16 engine faster (batch 1, TensorRT
11.4, family benchmark: 0.501 to 0.448 ms on B200 and 0.506 to 0.453 ms on
B300, 1.12x). The same FP8 recipe built at level 3 is still slower than fp16
at level 3 (TensorRT 11.4, same method as above, one job per GPU):

| GPU | Batch | fp16 ms | FP8 ms | x | FP8 + grouped ms (x) |
|---|---|---|---|---|---|
| B200 | 1 | 0.453 | 0.482 | 0.94 | 0.888 (0.51) |
| B200 | 32 | 1.175 | 1.209 | 0.97 | 1.669 (0.70) |
| B300 | 1 | 0.472 | 0.490 | 0.96 | 0.874 (0.54) |
| B300 | 32 | 1.116 | 1.159 | 0.96 | 1.583 (0.71) |

At level 3 FP8 top-1 is 82.60% (B200) and 82.67% (B300) against 82.67% for
fp16, with no NaN or Inf, and the FP8 + grouped engine still runs 331 kernels
per call against 164 for fp16 (the grouped convolutions are still split per
group).

### NVFP4 (not tried)

TensorRT does not support NVFP4 on convolutions (block-scaled FP4 needs a 2-D
GEMM), FP8 already loses to fp16, and the engine is not limited by weight
reads (plan 40 MiB).

### What was not tried

| Item | Note |
|---|---|
| Percentile or entropy calibration, quantization-aware training | not measured; could lower the KL, not the latency |
| Other RegNet sizes, H100, native runtime with an FP8 bundle | not measured |

### When to revisit

- A TensorRT release that fuses the Quantize into the producing layer after
  residual adds and squeeze-excite products.
- A TensorRT FP8 kernel for grouped convolutions that keeps one layer per
  convolution.
- A larger RegNet in this family, or larger batches, where the 1x1
  convolutions are limited by math rather than by activation traffic.

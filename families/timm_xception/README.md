# timm Xception

This family builds the timm Xception-41 image classifier
(`timm/xception41.tf_in1k` in the test manifest) as one TensorRT engine:
299 x 299 input, batch 1, fp16 in the manifest (fp32 is also supported),
builder optimization level 3. The block layout (separable convolutions, strides,
shortcut projections) is read from the checkpoint; see `model.py`. The output
contains every class logit in checkpoint order. The family has no quantized
build.

## FP8 quantization study (not shipped)

FP8 self-quantization of the convolutions was built and measured on TensorRT
11.4 (and 11.1) for `timm/xception41.tf_in1k`. It is not shipped: with the
family's optimization level 3, FP8 is slower than fp16 at batch 1 (0.90x on
B200, 0.95x on B300) and at batch 32 (0.92x to 0.94x). Putting the depthwise
convolutions in FP8 as well is still slower at batch 1 (0.86x to 0.87x), equal
at batch 32 (1.01x), and costs 0.5 ImageNet top-1 points. This section records
the recipe, the method and the numbers, so that the study does not have to be
repeated. The FP8 code was not merged; this family has no FP8 build.

### Summary and decision

| Item | fp16 (manifest) | FP8 (fp16 engine) | FP8 vs fp16 |
|---|---|---|---|
| Latency, batch 1, B200 / B300 (TensorRT 11.4) | 0.394 / 0.373 ms | 0.438 / 0.392 ms | 0.90x / 0.95x |
| Latency, batch 32, B200 / B300 (TensorRT 11.4) | 2.345 / 2.259 ms | 2.499 / 2.448 ms | 0.94x / 0.92x |
| Plan size, batch 1, B200 | 54.4 MiB | 30.7 MiB | -44% |
| ImageNet top-1, B200 (PyTorch fp32 79.07%) | 79.07% | 79.07% | 0.0 points |
| Agreement with PyTorch top-1 | 99.87% | 98.13% | -1.7 points |
| Non-finite outputs | 0 | 0 | none |

Decision: do not ship FP8 for timm Xception.

- Speed: FP8 is slower on both GPUs at batch 1 and batch 32. The depthwise
  3x3 half of every separable convolution stays fp16, so every FP8 pointwise
  convolution needs its own Quantize after the depthwise convolution. The plan
  grows from 101 to 145 layers, and the kernels outside the convolutions take
  more time than the FP8 convolutions save (see the kernel table below).
- Size alone (-44% plan) with slower engines does not justify a second build
  path.
- Depthwise convolutions in FP8: faster than the FP8 recipe above only at
  batch 32, never faster than fp16 at batch 1, and the quality loss is large
  (KL 0.103, 13x the recipe above, top-1 -0.5 points, 7.7% of top-1 classes
  change). A 3x3 depthwise filter has 9 weights per channel and does not
  tolerate e4m3 rounding.

### Recipe that was measured

No FP8 or NVFP4 checkpoint of this model exists on the Hub, and no official
quantization config exists for it. Each quantized convolution is built as
`input -> Quantize -> Dequantize(fp16) -> Convolution(+bias)`, with the kernel
`FP8 constant -> Dequantize(fp16, axis 0)`.

| Choice | Decision |
|---|---|
| Quantized (43 of 84 convolutions) | every dense convolution: the 1x1 pointwise half of each separable convolution and the 1x1 shortcut projections |
| fp16 | the two stem convolutions (`stem.0`, `stem.1`), the 3x3 depthwise half of each separable convolution, the classifier |
| Weight scale | per output channel, amax / 448 of the batch-norm-folded kernel, at build time |
| Activation scale | static, per tensor, from ModelOpt `FP8_DEFAULT_CFG` (max calibration) on 500 calibration images; one Quantize per tensor, shared by the convolutions that read it |
| Outlier rule | an activation scale above 20x the median of the dense convolutions stays fp16; measured: no outlier (largest ratio to the median 3.7) |
| Dequantize type | fp16; no non-finite value, fp32 Dequantize was not needed |

### Method

- GPUs: B200 and B300, one GPU per job, all variants built and timed in the
  same job; TensorRT 11.4.0.106 (target) and 11.1.0.106 (B200 check).
- Build: the family's own build function at optimization level 3 (the family
  default); the FP8 layers are added to the same network.
- Data: 2000 ImageNet validation images from `clip-benchmark/wds_imagenet1k`
  (every 25th image): every 4th image (500) for calibration, the other 1500
  held out. Family preprocessing.
- Quality: top-1 against the labels, top-1 agreement and KL divergence against
  the timm PyTorch fp32 model on the 1500 held-out images.
- Non-finite scan: the 1500 held-out images, the 500 calibration images and 29
  stress inputs (noise with standard deviation 1, 3 and 10, constants), every
  batch-1 engine, B200 and B300.
- Speed: CUDA graph replay, median of 7 interleaved rounds of 100 calls.
  Batch 32 is a scratch build (the family builds batch 1 only).

### Results

Speed and size (TensorRT 11.4 unless noted, ms per engine call; x is fp16 time
over FP8 time):

| GPU | TensorRT | Batch | fp32 ms | fp16 ms | FP8 ms | x | plan MiB fp16 to FP8 |
|---|---|---|---|---|---|---|---|
| B200 | 11.4 | 1 | 0.588 | 0.394 | 0.438 | 0.90 | 54.4 to 30.7 |
| B200 | 11.4 | 32 | - | 2.345 | 2.499 | 0.94 | 58.0 to 37.5 |
| B300 | 11.4 | 1 | 0.557 | 0.373 | 0.392 | 0.95 | 54.5 to 30.9 |
| B300 | 11.4 | 32 | - | 2.259 | 2.448 | 0.92 | 57.9 to 37.3 |
| B200 | 11.1 | 1 | - | 0.393 | 0.433 | 0.91 | 53.1 to 30.7 |

Quality (1500 held-out images, batch 1, TensorRT 11.4):

| Engine | Top-1 | Agreement with PyTorch top-1 | KL mean | KL p99 |
|---|---|---|---|---|
| PyTorch fp32 | 79.07% | - | - | - |
| fp16 engine (B200) | 79.07% | 99.87% | < 0.0001 | < 0.001 |
| FP8 engine (B200) | 79.07% | 98.13% | 0.0077 | 0.093 |
| FP8 engine (B300) | 79.07% | 98.13% | 0.0077 | 0.093 |
| FP8 engine (B200, TensorRT 11.1) | 79.27% | 97.93% | 0.0078 | 0.103 |
| + depthwise in FP8 (B200 and B300) | 78.53% | 92.33% | 0.103 | 1.34 |

Kernel time per group, TensorRT 11.4 (layer profiler, ms; the profiler adds
time per layer, so the sums are larger than the latencies above; at level 3
TensorRT fuses part of the convolutions into generated kernels, which are
counted under "other"):

| Kernel group | fp16 batch 1 | FP8 batch 1 | fp16 batch 32 | FP8 batch 32 |
|---|---|---|---|---|
| B200 convolutions | 0.325 | 0.308 fp16 + 0.069 FP8 = 0.376 | 2.122 | 1.404 fp16 + 0.376 FP8 = 1.780 |
| B200 other (fused kernels, Quantize, pooling, classifier) | 0.519 | 0.727 | 0.544 | 1.242 |
| B200 sum | 0.844 | 1.103 | 2.666 | 3.021 |
| B300 sum | 0.839 | 1.073 | 2.582 | 2.981 |
| layers in the plan | 101 | 145 | 101 | 144 |

Variants (TensorRT 11.4; x is fp16 time over variant time):

| Variant | B200 x batch 1 / 32 | B300 x batch 1 / 32 | plan MiB batch 1 | Top-1 | KL mean |
|---|---|---|---|---|---|
| recipe above | 0.90 / 0.94 | 0.95 / 0.92 | 30.7 | 79.07% | 0.0077 |
| + depthwise convolutions in FP8 | 0.87 / 1.01 | 0.86 / 1.01 | 30.7 | 78.53% | 0.103 |
| fp16 at optimization level 1 (before #2123) | 0.77 | 0.74 | 52.9 | 79.07% | < 0.0001 |

An earlier run of the same recipe at optimization level 1 (TensorRT 11.4,
when the family built at level 1) gave the same picture: FP8 0.97x to 0.98x
(B200) and 1.02x (B300) of the level-1 fp16 engine at batch 1; with the stem in
FP8 too, KL rose to 0.029 (agreement 95.5%); with depthwise in FP8, 1.09x to
1.13x at batch 1, but with the quality loss above. The fp16 engine at level 3
is faster than every FP8 variant at level 1.

### NVFP4 (not tried)

TensorRT does not support NVFP4 on convolutions (block-scaled FP4 needs a 2-D
GEMM), and the only GEMM, the classifier (2048 x 1000), holds about 8% of the
weights.

### What was not tried

| Item | Note |
|---|---|
| Percentile or entropy calibration, quantization-aware training | not measured; could recover depthwise FP8 accuracy, not the batch-1 speed |
| Other Xception sizes (65, 71), H100, native runtime with an FP8 bundle | not measured |

### When to revisit

- A TensorRT release that fuses the Quantize into the depthwise convolution
  before each FP8 pointwise convolution, or that makes the FP8 1x1 convolutions
  faster than fp16 at this size.
- An FP8 depthwise recipe that keeps KL near the pointwise-only recipe
  (for example quantization-aware training) on a TensorRT version where it is
  faster than fp16 at batch 1.

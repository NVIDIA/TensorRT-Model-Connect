# timm MobileNetV4

This family implements `image_to_class_scores` through `IImageToClassScores`.
The family binds the interface on its loaded model; the shared C ABI and the
header-only C++ `ImageToClassScores` wrapper discover that binding without a
family-specific shared registry.

The input is contiguous host RGB float32 in `[0, 1]`. The family owns the
checkpoint's resize, crop, normalization and engine call. The output contains
every class logit in checkpoint order, without softmax or top-k truncation.
`label_names` and `vocabulary_id` are retained when explicitly supplied by the
checkpoint. Missing metadata is returned as empty labels/identity: the class
indices remain model-local ordinals. There are no runtime Config options;
unsupported overrides are rejected.

Build a new bundle with this family version; the old `classification` bundle
mode is not retained. The existing CLI command remains simple:

```sh
trtmc build timm/mobilenetv4_conv_small_050.e3000_r224_in1k -o mobilenetv4.bundle
trtmc classify mobilenetv4.bundle --image photo.jpg
```

## Validation

The existing official-checkpoint test still compares the native top-1 class
against the timm reference using the original image and preprocessing. It also
checks the complete logits and class metadata exposed by the semantic Task.
That same E2E executes both public SDK consumers, using the existing
`TRTMC_NATIVE_BUILD_DIR` to locate their built binaries. They receive identical
decoded RGB input, must agree on every score, and retain the original top-1
reference check. No new CI selector or environment variable is introduced.

The CPU contract test checks binding, unknown and explicit class identity,
normalization, complete owned output, and rejection before inference of invalid
input or unsupported Config. It also builds the two public SDK consumers:

```sh
cmake --build build --target test_timm_mobilenetv4_task_contract test_timm_mobilenetv4_image_preprocess
ctest --test-dir build --output-on-failure -R '^timm_mobilenetv4_(task_contract|image_preprocess)$'
```

`tests/sdk_consumer.c` and `tests/sdk_consumer.cpp` use only the public SDK. For
an already-built bundle, supply an unprocessed RGB float32 HWC file and its
original height and width:

```sh
build/test_timm_mobilenetv4_sdk_c mobilenetv4.bundle build image.rgb.f32 480 640
build/test_timm_mobilenetv4_sdk_cpp mobilenetv4.bundle build image.rgb.f32 480 640
```

Each prints all scores as JSON and reads the result after releasing its model
handle. These commands perform real inference and require the matching runtime
and checkpoint bundle; the CPU contract test alone does not qualify a checkpoint.

## Benchmark timing

`tests/performance.yaml` takes over the existing `timm_mobilenetv4.classify`
entry through the benchmark's family-owned reference protocol. The workload,
precision, 3 warmups, 10 measurements, 5% margin and top-class oracle are unchanged.
The reference times inference, complete float32 host logits and synchronization;
argmax, finite checks and JSON reporting happen after timing, as in the semantic
SDK benchmark worker. The existing reference policy still excludes input
preparation (`task-model-call-wall`), while the native public Task call includes
family preprocessing. This fixes reduction/reporting placement, not that existing
scope difference, and does not establish a performance improvement.

## FP8 quantization study (not shipped)

FP8 self-quantization of the pointwise convolutions was built and measured on
TensorRT 11.4 (and 11.1) for `timm/mobilenetv4_conv_large.e500_r256_in1k`. It is
not shipped: the plan is 43% smaller, but at batch 1 FP8 is only 1.01x (B200)
and 1.04x (B300) the fp16 speed, it is slower than fp16 at batch 32 and 128
(0.92x to 0.96x), and with TensorRT optimization level 3 it is slower than fp16
at batch 1 too (0.93x to 0.94x). A smaller plan with no real speed gain is not a
reason to ship FP8. This section records the recipe, the method and the
numbers, so that the study does not have to be repeated. The FP8 code was not
merged; this family has no FP8 build.

### Summary and decision

| Item | fp16 (manifest) | FP8 (fp16 engine) | FP8 vs fp16 |
|---|---|---|---|
| Latency, batch 1, B200 / B300 (TensorRT 11.4, optimization level 1) | 0.480 / 0.465 ms | 0.475 / 0.449 ms | 1.01x / 1.04x |
| Latency, batch 1, B200 / B300 (TensorRT 11.4, optimization level 3) | 0.360 / 0.334 ms | 0.386 / 0.354 ms | 0.93x / 0.94x |
| Latency, batch 32, B200 / B300 (TensorRT 11.4, level 1) | 1.092 / 1.038 ms | 1.160 / 1.111 ms | 0.94x / 0.93x |
| Latency, batch 128, B200 / B300 (TensorRT 11.4, level 1) | 2.908 / 2.711 ms | 3.030 / 2.940 ms | 0.96x / 0.92x |
| Plan size, batch 1 | 63.6 MiB | 36.0 MiB | -43% |
| ImageNet top-1, B200 / B300 (PyTorch fp32 81.53%) | 81.53% / 81.47% | 81.40% / 81.60% | -0.13 / +0.13 points |
| Non-finite outputs | 0 | 0 | none |

Decision: do not ship FP8 for timm MobileNetV4.

- Speed: at batch 1 the engine is bound by launches and small kernels (170
  layers, about 0.47 ms). The FP8 1x1 convolutions are small (48 to 2048
  channels on 8x8 to 64x64 maps) and take about the same kernel time as the fp16
  ones, and 12 to 15 Quantize layers do not fuse into the producing layer (for
  example after a residual add). The 1% to 4% gain at batch 1 is within the
  spread between jobs (B200: 1.004x to 1.018x over three jobs). At batch 32 and
  128 the extra Quantize layers cost more than the convolutions save.
- Optimization level 3 (#2123, open: build this family at level 3) changes the
  fp16 baseline. It makes the fp16 engine 1.36x (B200, 0.354 ms) and 1.38x
  (B300, 0.332 ms) faster at batch 1 in the #2123 measurement, 1.36x and 1.40x in
  the job below, and removes about 60 layers. FP8 at level 3 is 1.26x to 1.30x
  faster than FP8 at level 1, but slower than fp16 at level 3 (0.93x to 0.94x).
- Quality is not the problem: top-1 stays at the fp16 level (2 of 1500 images
  change), KL 0.011 against PyTorch. PyTorch fake quantization of the same recipe
  gives the same KL, so the small loss comes from the recipe, not from TensorRT.
- `conv_medium` and `conv_small` with the same recipe are slower than fp16 at
  every batch (0.86x to 0.97x), and `conv_small` loses 3.2 top-1 points.

### Recipe that was measured

No FP8 or NVFP4 checkpoint of these models exists on the Hub, and no official
quantization config exists for them. Each quantized convolution is built as
`input -> Quantize -> Dequantize(fp16) -> Convolution(+bias)`, with the kernel
`FP8 constant -> Dequantize(fp16, axis 0)`.

| Choice | Decision |
|---|---|
| Quantized (54 of 102 convolutions in `conv_large`) | every 1x1 convolution inside the blocks: `blocks.0.0.conv_pwl`, `pw_exp.conv` and `pw_proj.conv` of every universal inverted bottleneck, `blocks.4.0.conv` |
| fp16 | the 3x3 stem, the 3x3 `conv_exp` of the edge residual block, the 45 depthwise convolutions, the head convolution (on the pooled 1x1 map), the classifier |
| Weight scale | per output channel, amax / 448 of the batch-norm-folded kernel, at build time |
| Activation scale | static, per tensor, ModelOpt `FP8_DEFAULT_CFG` (max calibration) on the timm PyTorch model in fp32 over 500 calibration images: amax / 448 (54 scales, 0.0085 to 0.30, median 0.050) |
| Outlier rule | a scale above 20x the median stays fp16; measured: no outlier |
| Dequantize type | fp16; no non-finite value, fp32 Dequantize was not needed |

### Method

- GPUs: B200 and B300, one GPU per job, all variants of one GPU type built and
  timed in the same job; TensorRT 11.4.0.106 (target) and 11.1.0.106.
- Data: 2000 ImageNet validation images from `clip-benchmark/wds_imagenet1k`
  (every 25th image of each shard, shuffled with seed 0): the first 500 for
  calibration, the other 1500 held out. Family preprocessing (256x256, bicubic,
  crop 0.95).
- Quality: top-1 against the labels, top-1 agreement and KL(PyTorch || engine)
  of the softmax against the timm PyTorch fp32 model on the 1500 held-out
  images, batch 1.
- Non-finite scan: the 2000 images and 29 stress inputs (Gaussian noise with
  standard deviation 1, 3 and 10, constants 0, +-1, +-10), B200 and B300.
- Speed: CUDA graph replay, median of 9 interleaved rounds of 200 calls, spread
  at most 0.8%. The family builds batch 1 only; batch 32 and 128 are scratch
  builds of the same family graph with the batch dimension patched in. The
  optimization level 3 rows come from a separate job per GPU that built fp16
  and FP8 at levels 1 and 3 and timed them together.

### Results

Speed and size, `conv_large` (ms per engine call; x is fp16 time over FP8 time;
optimization level 1 unless noted):

| GPU | TensorRT | Batch | fp32 ms | fp16 ms | FP8 ms | x | plan MiB fp16 to FP8 |
|---|---|---|---|---|---|---|---|
| B200 | 11.4 | 1 | 0.639 | 0.480 | 0.475 | 1.01 | 63.6 to 36.0 |
| B200 | 11.4 | 32 | 2.064 | 1.092 | 1.160 | 0.94 | 66.6 to 39.3 |
| B200 | 11.4 | 128 | 7.407 | 2.908 | 3.030 | 0.96 | 78.6 to 51.2 |
| B300 | 11.4 | 1 | 0.632 | 0.465 | 0.449 | 1.04 | 63.2 to 35.6 |
| B300 | 11.4 | 32 | 1.983 | 1.038 | 1.111 | 0.93 | 66.7 to 39.3 |
| B300 | 11.4 | 128 | 7.147 | 2.711 | 2.940 | 0.92 | 78.7 to 51.3 |
| B200 | 11.1 | 1 | - | 0.476 | 0.479 | 0.99 | 63.2 to 35.8 |

Optimization level 3, batch 1 (one job per GPU; level 1 measured again in the
same job):

| GPU | TensorRT | fp16 level 1 | FP8 level 1 | fp16 level 3 | FP8 level 3 | x at level 3 | layers fp16 / FP8 at level 3 |
|---|---|---|---|---|---|---|---|
| B200 | 11.4 | 0.490 | 0.485 | 0.360 | 0.386 | 0.93 | 109 / 154 |
| B300 | 11.4 | 0.467 | 0.445 | 0.334 | 0.354 | 0.94 | 109 / 154 |
| B200 | 11.1 | 0.489 | 0.495 | 0.354 | 0.381 | 0.93 | 111 / 155 |

At level 3 the FP8 plan is 37.3 MiB against 64.6 MiB for fp16, and FP8 top-1 is
81.40% (fp16 81.53%).

Quality (1500 held-out images, batch 1, level 1):

| Engine | Top-1 | Agreement with PyTorch top-1 | KL mean | KL p99 |
|---|---|---|---|---|
| PyTorch fp32 | 81.53% | - | - | - |
| fp16 engine (B200 / B300) | 81.53% / 81.47% | 100.00% / 99.93% | 0.00001 | 0.0001 |
| FP8 engine, TensorRT 11.4 (B200 / B300) | 81.40% / 81.60% | 97.13% / 97.33% | 0.0107 / 0.0105 | 0.127 / 0.125 |
| FP8 engine, TensorRT 11.1 (B200) | 81.00% | 97.27% | 0.0104 | 0.116 |
| PyTorch fake quantization of the same recipe | 81.53% | 97.33% | 0.0110 | 0.126 |

Kernel time per group, B200, TensorRT 11.4, level 1 (layer profiler, ms; the
profiler adds time per layer, so the sums are larger than the latencies above):

| Kernel group | fp16 batch 1 | FP8 batch 1 | fp16 batch 128 | FP8 batch 128 |
|---|---|---|---|---|
| convolutions | 0.871 | 0.372 fp16 + 0.488 FP8 = 0.860 | 3.398 | 1.992 fp16 + 1.161 FP8 = 3.153 |
| other (pointwise, Quantize, pooling, classifier) | 0.428 | 0.530 | 0.083 | 0.685 |
| layers in the plan | 170 | 185 | 108 | 155 |

Variants, `conv_large` (TensorRT 11.4, B200; x is fp16 time over variant time):

| Variant | x batch 1 / 32 / 128 | plan MiB batch 1 | Top-1 | KL mean |
|---|---|---|---|---|
| recipe above | 1.01 / 0.94 / 0.96 | 36.0 | 81.40% | 0.0104 |
| + 3x3 stem and edge residual `conv_exp` in FP8 | 1.02 / 0.88 / 0.86 | 36.2 (139.2 at batch 128) | 81.53% | 0.0217 |
| per-tensor weight scale | 1.02 / 0.94 / 0.96 | 36.0 | 81.40% | 0.0124 |
| batch norm kept as a scale layer | 1.01 / 0.94 / 0.96 | 36.0 | 81.33% | 0.0107 |

Other widths with the same recipe (TensorRT 11.4, level 1; x at batch 1 / 32 /
128; quality on the same 1500 images, B200):

| Model | x B200 | x B300 | plan MiB fp16 to FP8 (B200) | Top-1 fp16 to FP8 | KL mean |
|---|---|---|---|---|---|
| `mobilenetv4_conv_medium.e500_r256_in1k` | 0.94 / 0.91 / 0.94 | 0.97 / 0.92 / 0.91 | 19.7 to 13.5 | 79.67% to 80.33% | 0.0150 |
| `mobilenetv4_conv_small.e2400_r224_in1k` | 0.92 / 0.86 / 0.89 | 0.92 / 0.87 / 0.88 | 8.2 to 7.3 | 73.40% to 70.20% | 0.179 |

The depthwise convolutions were not quantized: in FP8, TensorRT 11.4 splits a
depthwise convolution into one convolution per group (measured on EfficientNet-B0,
see the timm EfficientNet README: 157 to 2462 layers, 15x slower).

### NVFP4 (not tried)

TensorRT does not support NVFP4 on convolutions (block-scaled FP4 needs a 2-D
GEMM), and the engine is not limited by weight reads (36 MiB FP8 plan, about
0.45 ms per image).

### What was not tried

| Item | Note |
|---|---|
| Percentile or entropy calibration, quantization-aware training | not measured; quality is already at the fp16 level |
| FP8 at batch 32 and 128 with optimization level 3 | not measured; FP8 is already slower at level 1 |
| `conv_small_050`, H100, native runtime with an FP8 bundle | not measured |

### When to revisit

- A TensorRT release with FP8 1x1 convolution kernels that are clearly faster
  than fp16 for small channel counts, and that fuses the Quantize into the
  producing layer after residual adds, so that FP8 beats fp16 at the family's
  optimization level.
- A TensorRT FP8 kernel for depthwise (grouped) convolutions that keeps one
  layer per convolution.
- A use case where only the plan size matters (43% smaller at the fp16 top-1
  level); the speed rule does not allow it today.

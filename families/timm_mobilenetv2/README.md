# timm MobileNetV2

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
trtmc build timm/mobilenetv2_100.ra_in1k -o mobilenetv2.bundle
trtmc classify mobilenetv2.bundle --image photo.jpg
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
cmake --build build --target test_timm_mobilenetv2_task_contract test_timm_mobilenetv2_image_preprocess
ctest --test-dir build --output-on-failure -R '^timm_mobilenetv2_(task_contract|image_preprocess)$'
```

`tests/sdk_consumer.c` and `tests/sdk_consumer.cpp` use only the public SDK. For
an already-built bundle, supply an unprocessed RGB float32 HWC file and its
original height and width:

```sh
build/test_timm_mobilenetv2_sdk_c mobilenetv2.bundle build image.rgb.f32 480 640
build/test_timm_mobilenetv2_sdk_cpp mobilenetv2.bundle build image.rgb.f32 480 640
```

Each prints all scores as JSON and reads the result after releasing its model
handle. These commands perform real inference and require the matching runtime
and checkpoint bundle; the CPU contract test alone does not qualify a checkpoint.

## Benchmark timing

`tests/performance.yaml` takes over the existing `timm_mobilenetv2.classify`
entry through the benchmark's family-owned reference protocol. The workload,
precision, 3 warmups, 10 measurements, 5% margin and top-class oracle are unchanged.
The reference times inference, complete float32 host logits and synchronization;
argmax, finite checks and JSON reporting happen after timing, as in the semantic
SDK benchmark worker. The existing reference policy still excludes input
preparation (`task-model-call-wall`), while the native public Task call includes
family preprocessing. This fixes reduction/reporting placement, not that existing
scope difference, and does not establish a performance improvement.

## FP8 quantization study (not shipped)

FP8 self-quantization of the convolutions was built and measured on TensorRT
11.4 (and 11.1) for `timm/mobilenetv2_100.ra_in1k`. It is not shipped: FP8 is
slower than fp16 at batch 1 (0.77x to 0.79x) and at batch 32 and 128 (0.83x to
0.90x), and ImageNet top-1 drops by 1.6 points. This section records the
recipe, the method and the numbers, so that the study does not have to be
repeated. The FP8 code was not merged; this family has no FP8 build.

### Summary and decision

| Item | fp16 (manifest) | FP8 (fp16 engine) | FP8 vs fp16 |
|---|---|---|---|
| Latency, batch 1, B200 / B300 (TensorRT 11.4) | 0.168 / 0.164 ms | 0.219 / 0.206 ms | 0.77x / 0.79x |
| Latency, batch 32, B200 / B300 (TensorRT 11.4) | 0.510 / 0.477 ms | 0.614 / 0.576 ms | 0.83x / 0.83x |
| Latency, batch 128, B200 / B300 (TensorRT 11.4) | 1.541 / 1.438 ms | 1.717 / 1.657 ms | 0.90x / 0.87x |
| Plan size, batch 1, B200 | 7.5 MiB | 5.8 MiB | -23% |
| ImageNet top-1, B200 (PyTorch fp32 73.73%) | 73.73% | 72.13% | -1.6 points |
| Non-finite outputs | 0 | 0 | none |

Decision: do not ship FP8 for timm MobileNetV2.

- Speed: the FP8 1x1 convolutions take about the same kernel time as the fp16
  ones. They are small (16 to 1280 channels) and limited by activation traffic,
  not by math. The fp16 plan fuses almost every elementwise operation into the
  convolutions (58 layers). In FP8, the Quantize after each ReLU6 and after each
  residual add does not fuse into the producing convolution: the plan has 90
  layers, and these layers cost more than the convolutions save.
- Quality: per-tensor 8-bit activations cost 1.6 top-1 points on this mobile
  network. PyTorch fake quantization of the same recipe loses the same, so the
  loss comes from the recipe, not from TensorRT.
- The numbers above are at the family's current builder optimization level 1.
  The open change that moves this family to level 3 (#2123) makes the fp16
  engine faster at batch 1: 0.146 ms on B200 and 0.135 ms on B300 (1.16x and
  1.21x over level 1, TensorRT 11.4, a separate run with 15 rounds of 100
  calls). FP8 at level 3 was not measured; at level 1 it is already slower than
  the level-1 fp16 engine.

### Recipe that was measured

No FP8 or NVFP4 checkpoint of this model exists on the Hub, and no official
quantization config exists for it. Each quantized convolution is built as
`input -> Quantize -> Dequantize(fp16) -> Convolution(+bias)`, with the kernel
`FP8 constant -> Dequantize(fp16, axis 0)`.

| Choice | Decision |
|---|---|
| Quantized (34 of 52 convolutions) | every pointwise (1x1) convolution on a spatial map larger than 1x1 |
| fp16 | the 3x3 stem (3 input channels), the 17 depthwise convolutions, the classifier |
| Weight scale | per output channel, amax / 448 of the batch-norm-folded kernel, at build time |
| Activation scale | static, per tensor, max \|input\| / 448 over 500 calibration images (the statistic of ModelOpt max calibration), read from an fp32 engine of the family graph |
| Outlier rule | a scale above 20x the median of its kind stays fp16; measured: no outlier |
| Dequantize type | fp16; no non-finite value, fp32 Dequantize was not needed |

### Method

- GPUs: B200 and B300, one GPU per job; TensorRT 11.4.0.106 (target) and
  11.1.0.106 (B200, batch 1). Engines built through the family build at its
  current builder optimization level 1.
- Data: 2000 ImageNet validation images from `clip-benchmark/wds_imagenet1k`
  (every 25th image of each shard, shuffled with seed 0): the first 500 for
  calibration, the other 1500 held out. Family preprocessing.
- Quality: top-1 against the labels, top-1 agreement and KL divergence against
  the timm PyTorch fp32 model on the 1500 held-out images, batch 1.
- Non-finite scan: the 2000 images and 29 stress inputs (noise with standard
  deviation 1, 3 and 10, constants), all variants, B200 and B300.
- Speed: CUDA graph replay, all variants built and timed in one job per GPU
  type, median of 7 interleaved rounds of 200 calls; spread at most 1.4%
  (4.2% for one variant, stem in FP8, batch 1 on B200). Batch 32 and 128 are
  scratch builds (the family builds batch 1 only).

### Results

Speed and size (ms per engine call; x is fp16 time over FP8 time):

| GPU | TensorRT | Batch | fp32 ms | fp16 ms | FP8 ms | x | plan MiB fp16 to FP8 |
|---|---|---|---|---|---|---|---|
| B200 | 11.4 | 1 | 0.250 | 0.168 | 0.219 | 0.77 | 7.5 to 5.8 |
| B200 | 11.4 | 32 | 1.028 | 0.510 | 0.614 | 0.83 | 10.1 to 9.1 |
| B200 | 11.4 | 128 | 3.412 | 1.541 | 1.717 | 0.90 | 19.3 to 20.7 |
| B300 | 11.4 | 1 | 0.231 | 0.164 | 0.206 | 0.79 | 7.0 to 5.3 |
| B300 | 11.4 | 32 | 1.014 | 0.477 | 0.576 | 0.83 | 10.1 to 9.3 |
| B300 | 11.4 | 128 | 3.216 | 1.438 | 1.657 | 0.87 | 19.4 to 20.7 |
| B200 | 11.1 | 1 | - | 0.168 | 0.219 | 0.77 | 7.5 to 5.8 |

Quality (1500 held-out images, batch 1):

| Engine | Top-1 | Agreement with PyTorch top-1 | KL mean | KL p99 |
|---|---|---|---|---|
| PyTorch fp32 | 73.73% | - | - | - |
| fp32 engine (B200) | 73.67% | 99.93% | < 0.0001 | < 0.0001 |
| fp16 engine (B200) | 73.73% | 99.80% | < 0.0001 | < 0.001 |
| FP8 engine (B200) | 72.13% | 88.87% | 0.0830 | 0.547 |
| FP8 engine (B300) | 71.67% | 88.67% | 0.0846 | 0.565 |
| FP8 engine (B200, TensorRT 11.1) | 72.20% | 88.73% | 0.0847 | 0.556 |
| PyTorch fake quantization of the same recipe | 72.00% | 89.47% | 0.0828 | 0.564 |

Kernel time per group, B200, TensorRT 11.4 (layer profiler, ms; the profiler
adds time per layer, so the sums are larger than the latencies above):

| Kernel group | fp16 batch 1 | FP8 batch 1 | fp16 batch 128 | FP8 batch 128 |
|---|---|---|---|---|
| convolutions | 0.401 | 0.125 fp16 + 0.295 FP8 = 0.420 | 1.764 | 0.962 fp16 + 0.774 FP8 = 1.736 |
| other (pointwise, Quantize, pooling, classifier) | 0.038 | 0.253 | 0.086 | 0.441 |
| layers in the plan | 58 | 90 | 58 | 90 |

Variants (TensorRT 11.4, B200; x is fp16 time over variant time):

| Variant | x batch 1 / 32 / 128 | plan MiB batch 1 | Top-1 | KL mean |
|---|---|---|---|---|
| recipe above | 0.77 / 0.83 / 0.90 | 5.8 | 72.13% | 0.0830 |
| + 3x3 stem in FP8 | 0.76 / 0.73 / 0.75 | 5.9 (88.0 at batch 128) | 72.07% | 0.0947 |
| per-tensor weight scale | 0.77 / 0.83 / 0.90 | 5.8 | 71.27% | 0.1055 |
| batch norm kept as a scale layer | 0.76 / 0.83 / 0.90 | 5.8 | 72.27% | 0.0844 |
| + depthwise convolutions in FP8 (PyTorch fake quantization only) | - | - | 71.20% | 0.1619 |

The depthwise convolutions were not built in FP8 for this family: in the
timm_efficientnet study, TensorRT 11.4 split each FP8 depthwise (grouped)
convolution into one convolution per group, and that engine was 15x slower.

### NVFP4 (not tried)

TensorRT does not support NVFP4 on convolutions (block-scaled FP4 needs a 2-D
GEMM), FP8 already loses 1.6 top-1 points, and the engine is not limited by
weight reads (plan under 12 MiB at batch 1).

### What was not tried

| Item | Note |
|---|---|
| FP8 at builder optimization level 3 (#2123) | not measured; the level-3 fp16 engine is 1.16x to 1.21x faster than the level-1 fp16 engine used above |
| Percentile or entropy calibration, quantization-aware training | not measured; could recover accuracy, not speed |
| Other MobileNetV2 widths, H100, native runtime with an FP8 bundle | not measured |

### When to revisit

- A TensorRT release with FP8 1x1 convolution kernels that are faster than
  fp16 for small channel counts, and that fuses the Quantize into the producing
  layer after ReLU6 and residual adds.
- A TensorRT FP8 kernel for depthwise (grouped) convolutions that keeps one
  layer per convolution.
- A wider, compute-bound model in this family: at batch 128 FP8 reaches only
  0.87x to 0.90x, so the gain, if any, would come from larger convolutions.

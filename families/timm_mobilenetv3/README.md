# timm MobileNetV3

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
trtmc build timm/mobilenetv3_large_100.ra_in1k -o mobilenetv3.bundle
trtmc classify mobilenetv3.bundle --image photo.jpg
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
cmake --build build --target test_timm_mobilenetv3_task_contract test_timm_mobilenetv3_image_preprocess
ctest --test-dir build --output-on-failure -R '^timm_mobilenetv3_(task_contract|image_preprocess)$'
```

`tests/sdk_consumer.c` and `tests/sdk_consumer.cpp` use only the public SDK. For
an already-built bundle, supply an unprocessed RGB float32 HWC file and its
original height and width:

```sh
build/test_timm_mobilenetv3_sdk_c mobilenetv3.bundle build image.rgb.f32 480 640
build/test_timm_mobilenetv3_sdk_cpp mobilenetv3.bundle build image.rgb.f32 480 640
```

Each prints all scores as JSON and reads the result after releasing its model
handle. These commands perform real inference and require the matching runtime
and checkpoint bundle; the CPU contract test alone does not qualify a checkpoint.

## Benchmark timing

`tests/performance.yaml` takes over the existing `timm_mobilenetv3.classify`
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
11.4 (and 11.1) for `timm/mobilenetv3_large_100.ra_in1k`. It is not shipped:
FP8 is slower than fp16 at batch 1 (0.89x to 0.90x) and at batch 32 and 128
(0.91x to 0.94x), and ImageNet top-1 drops by 0.9 points. This section records
the recipe, the method and the numbers, so that the study does not have to be
repeated. The FP8 code was not merged; this family has no FP8 build. The same
study for timm EfficientNet is in `families/timm_efficientnet/README.md`.

### Summary and decision

| Item | fp16 (manifest) | FP8 (fp16 engine) | FP8 vs fp16 |
|---|---|---|---|
| Latency, batch 1, B200 / B300 (TensorRT 11.4) | 0.272 / 0.276 ms | 0.304 / 0.308 ms | 0.89x / 0.90x |
| Latency, batch 32, B200 / B300 (TensorRT 11.4) | 0.618 / 0.589 ms | 0.682 / 0.640 ms | 0.91x / 0.92x |
| Latency, batch 128, B200 / B300 (TensorRT 11.4) | 1.510 / 1.404 ms | 1.614 / 1.550 ms | 0.94x / 0.91x |
| Plan size, batch 1, B200 | 11.7 MiB | 10.7 MiB | -9% |
| ImageNet top-1, B200 (PyTorch fp32 77.00%) | 77.07% | 76.20% | -0.9 points |
| Non-finite outputs | 0 | 0 | none |

Decision: do not ship FP8 for timm MobileNetV3.

- Speed: the FP8 1x1 convolutions take about the same kernel time as the fp16
  ones. They are small (16 to 960 channels) and limited by activation traffic,
  not by math. The Quantize layers that do not fuse into the producing layer
  (after residual adds, activations and squeeze-excite products) add 17 to 18
  layers and make the FP8 engine slower.
- The 15 depthwise convolutions stay fp16. In FP8, TensorRT 11.4 splits each
  depthwise (grouped) convolution into one convolution per group: the plan grows
  from 112 to 2092 layers, the build takes 146 s instead of 19 s, and batch 1 is
  21x slower (5.82 ms instead of 0.274 ms on B200).
- Quality: per-tensor 8-bit activations cost 0.9 top-1 points on this mobile
  network. PyTorch fake quantization of the same recipe loses about the same, so
  the loss comes from the recipe, not from TensorRT.
- TensorRT builder optimization level: the numbers here use the family's
  current level 1. At level 3 the fp16 engine is faster (batch 1: 0.219 ms on
  B200 and 0.226 ms on B300, 1.23x and 1.22x over level 1, measured for the
  level change in #2123), so the FP8 gap would only grow. FP8 at level 3 was not
  measured.

### Recipe that was measured

No FP8 or NVFP4 checkpoint of this model exists on the Hub, and no official
quantization config exists for it. Each quantized convolution is built as
`input -> Quantize -> Dequantize(fp16) -> Convolution(+bias)`, with the kernel
`FP8 constant -> Dequantize(fp16, axis 0)`.

| Choice | Decision |
|---|---|
| Quantized (30 of 63 convolutions) | every pointwise (1x1) convolution on a spatial map larger than 1x1 |
| fp16 | the 3x3 stem (3 input channels), the 15 depthwise convolutions, the 16 squeeze-excite convolutions and the head convolution (on a 1x1 map), the classifier |
| Weight scale | per output channel, amax / 448 of the batch-norm-folded kernel, at build time |
| Activation scale | static, per tensor, max \|input\| / 448 over 500 calibration images (the statistic of ModelOpt max calibration), read from an fp32 engine of the family graph |
| Outlier rule | a scale above 20x the median of its kind stays fp16; measured: no outlier |
| Dequantize type | fp16; no non-finite value, fp32 Dequantize was not needed |

### Method

- GPUs: B200 and B300, one GPU per job; TensorRT 11.4.0.106 (target) and
  11.1.0.106.
- Data: 2000 ImageNet validation images from `clip-benchmark/wds_imagenet1k`
  (every 25th image of each shard, shuffled with seed 0): the first 500 for
  calibration, the other 1500 held out. Family preprocessing.
- Quality: top-1 against the labels, top-1 agreement and KL divergence against
  the timm PyTorch fp32 model on the 1500 held-out images, batch 1.
- Non-finite scan: the 2000 images and 29 stress inputs (noise with standard
  deviation 1, 3 and 10, constants), all variants, B200 and B300.
- Speed: CUDA graph replay, median of 7 interleaved rounds of 200 calls, all
  variants built and timed in one job per GPU type, spread at most 1.4% (1.7%
  in the depthwise job). Batch
  32 and 128 are scratch builds (the family builds batch 1 only).

### Results

Speed and size (ms per engine call; x is fp16 time over FP8 time):

| GPU | TensorRT | Batch | fp32 ms | fp16 ms | FP8 ms | x | plan MiB fp16 to FP8 |
|---|---|---|---|---|---|---|---|
| B200 | 11.4 | 1 | 0.343 | 0.272 | 0.304 | 0.89 | 11.7 to 10.7 |
| B200 | 11.4 | 32 | 0.992 | 0.618 | 0.682 | 0.91 | 14.3 to 14.3 |
| B200 | 11.4 | 128 | 2.995 | 1.510 | 1.614 | 0.94 | 23.5 to 26.5 |
| B300 | 11.4 | 1 | 0.324 | 0.276 | 0.308 | 0.90 | 11.3 to 10.2 |
| B300 | 11.4 | 32 | 0.953 | 0.589 | 0.640 | 0.92 | 14.5 to 14.5 |
| B300 | 11.4 | 128 | 2.850 | 1.404 | 1.550 | 0.91 | 23.8 to 26.6 |
| B200 | 11.1 | 1 | - | 0.277 | 0.306 | 0.90 | 11.7 to 10.8 |

Quality (1500 held-out images, batch 1):

| Engine | Top-1 | Agreement with PyTorch top-1 | KL mean | KL p99 |
|---|---|---|---|---|
| PyTorch fp32 | 77.00% | - | - | - |
| fp16 engine (B200) | 77.07% | 99.73% | < 0.0001 | < 0.001 |
| FP8 engine (B200) | 76.20% | 93.47% | 0.0385 | 0.325 |
| FP8 engine (B300) | 76.27% | 93.67% | 0.0388 | 0.316 |
| FP8 engine (B200, TensorRT 11.1) | 76.53% | 93.27% | 0.0386 | 0.308 |
| PyTorch fake quantization of the same recipe | 76.33% | 93.27% | 0.0374 | 0.327 |

Kernel time per group, B200, TensorRT 11.4 (layer profiler, ms; the profiler
adds time per layer, so the sums are larger than the latencies above):

| Kernel group | fp16 batch 1 | FP8 batch 1 | fp16 batch 128 | FP8 batch 128 |
|---|---|---|---|---|
| convolutions | 0.497 | 0.256 fp16 + 0.261 FP8 = 0.516 | 1.495 | 0.817 fp16 + 0.656 FP8 = 1.473 |
| other (pointwise, Quantize, pooling, classifier) | 0.330 | 0.453 | 0.584 | 0.780 |
| layers in the plan | 112 | 130 | 107 | 124 |

Variants (TensorRT 11.4, B200; x is fp16 time over variant time):

| Variant | x batch 1 / 32 / 128 | plan MiB batch 1 | Top-1 | KL mean |
|---|---|---|---|---|
| recipe above | 0.89 / 0.91 / 0.94 | 10.7 | 76.20% | 0.0385 |
| + 3x3 stem in FP8 | 0.89 / 0.81 / 0.77 | 10.8 (93.9 at batch 128) | 76.20% | 0.0476 |
| per-tensor weight scale | 0.90 / 0.91 / 0.93 | 10.7 | 75.47% | 0.0683 |
| batch norm kept as a scale layer | 0.90 / 0.91 / 0.94 | 10.7 | 76.33% | 0.0382 |
| + depthwise convolutions in FP8 (separate job) | 0.047 / 0.070 / - | 23.1 (2092 layers) | 74.73% | 0.1065 |

The depthwise row comes from a separate B200 and B300 job (batch 1 and 32, with
fp16 and the recipe above rebuilt and timed in the same job: 0.90x and 0.90x on
B200, 0.91x and 0.92x on B300). On B300 the depthwise FP8 engine is 0.051x at
batch 1 and 0.073x at batch 32, with 75.00% top-1. PyTorch fake quantization
with the depthwise convolutions also in FP8 gives 75.20% top-1.

### NVFP4 (not tried)

TensorRT does not support NVFP4 on convolutions (block-scaled FP4 needs a 2-D
GEMM), FP8 already loses 0.9 top-1 points, and the engine is not limited by
weight reads (plan under 12 MiB).

### What was not tried

| Item | Note |
|---|---|
| Percentile or entropy calibration, quantization-aware training | not measured; could recover accuracy, not speed |
| FP8 at TensorRT builder optimization level 3 | not measured; fp16 gains 1.22x to 1.23x at level 3 |
| Other MobileNetV3 sizes, H100, native runtime with an FP8 bundle | not measured |

### When to revisit

- A TensorRT release with FP8 1x1 convolution kernels that are faster than
  fp16 for small channel counts, and that fuses the Quantize into the producing
  layer after residual adds, activations and squeeze-excite products.
- A TensorRT FP8 kernel for depthwise (grouped) convolutions that keeps one
  layer per convolution.

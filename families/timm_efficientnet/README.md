# timm EfficientNet

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
trtmc build timm/efficientnet_b0.ra_in1k -o efficientnet.bundle
trtmc classify efficientnet.bundle --image photo.jpg
```

## Validation

The existing official-checkpoint test still compares the native top-1 class
against the timm reference using the original image, preprocessing and existing
runner-up margin allowance. It also
checks the complete logits and class metadata exposed by the semantic Task.
That same E2E executes both public SDK consumers, using the existing
`TRTMC_NATIVE_BUILD_DIR` to locate their built binaries. They receive identical
decoded RGB input, must agree on every score, and retain the original top-1 / runner-up-margin
reference check. No new CI selector or environment variable is introduced.

The CPU contract test checks binding, unknown and explicit class identity,
normalization, complete owned output, and rejection before inference of invalid
input or unsupported Config. It also builds the two public SDK consumers:

```sh
cmake --build build --target test_timm_efficientnet_task_contract test_timm_efficientnet_image_preprocess
ctest --test-dir build --output-on-failure -R '^timm_efficientnet_(task_contract|image_preprocess)$'
```

`tests/sdk_consumer.c` and `tests/sdk_consumer.cpp` use only the public SDK. For
an already-built bundle, supply an unprocessed RGB float32 HWC file and its
original height and width:

```sh
build/test_timm_efficientnet_sdk_c efficientnet.bundle build image.rgb.f32 480 640
build/test_timm_efficientnet_sdk_cpp efficientnet.bundle build image.rgb.f32 480 640
```

Each prints all scores as JSON and reads the result after releasing its model
handle. These commands perform real inference and require the matching runtime
and checkpoint bundle; the CPU contract test alone does not qualify a checkpoint.

## Benchmark timing

`tests/performance.yaml` takes over the existing `timm_efficientnet.classify`
entry through the benchmark's family-owned reference protocol. The workload,
precision, 3 warmups, 10 measurements, 5% margin and benchmark top-class oracle are unchanged.
The reference times inference, complete float32 host logits and synchronization;
argmax, finite checks and JSON reporting happen after timing, as in the semantic
SDK benchmark worker. The existing reference policy still excludes input
preparation (`task-model-call-wall`), while the native public Task call includes
family preprocessing. This fixes reduction/reporting placement, not that existing
scope difference, and does not establish a performance improvement.

## FP8 quantization study (not shipped)

FP8 self-quantization of the convolutions was built and measured on TensorRT
11.4 (and 11.1) for `timm/efficientnet_b0.ra_in1k`. It is not shipped: FP8 is
slower than fp16 at batch 1 (0.94x to 0.95x) and at batch 32 and 128 (0.95x to
0.99x), and ImageNet top-1 drops by 1.6 points. This section records the
recipe, the method and the numbers, so that the study does not have to be
repeated. The FP8 code was not merged; this family has no FP8 build.

### Summary and decision

| Item | fp16 (manifest) | FP8 (fp16 engine) | FP8 vs fp16 |
|---|---|---|---|
| Latency, batch 1, B200 / B300 (TensorRT 11.4) | 0.414 / 0.419 ms | 0.439 / 0.441 ms | 0.94x / 0.95x |
| Latency, batch 32, B200 / B300 (TensorRT 11.4) | 1.041 / 0.994 ms | 1.091 / 1.034 ms | 0.95x / 0.96x |
| Latency, batch 128, B200 / B300 (TensorRT 11.4) | 2.807 / 2.688 ms | 2.830 / 2.723 ms | 0.99x / 0.99x |
| Plan size, batch 1 | 11.6 MiB | 8.9 MiB | -23% |
| ImageNet top-1, B200 (PyTorch fp32 79.13%) | 79.20% | 77.60% | -1.6 points |
| Non-finite outputs | 0 | 0 | none |

Decision: do not ship FP8 for timm EfficientNet.

- Speed: the FP8 1x1 convolutions take about the same kernel time as the fp16
  ones. They are small (16 to 1280 channels) and limited by activation traffic,
  not by math. The Quantize layers that do not fuse into the producing layer
  (after residual adds and squeeze-excite products) add 11 to 14 layers and make
  the FP8 engine slower.
- The 16 depthwise convolutions stay fp16. In FP8, TensorRT 11.4 splits each
  depthwise (grouped) convolution into one convolution per group: the plan grows
  from 157 to 2462 layers, the build takes 144 s instead of 16 s, and batch 1 is
  15x slower (6.43 ms instead of 0.42 ms).
- Quality: per-tensor 8-bit activations cost 1.6 top-1 points on this mobile
  network. PyTorch fake quantization of the same recipe loses the same, so the
  loss comes from the recipe, not from TensorRT.

### Recipe that was measured

No FP8 or NVFP4 checkpoint of this model exists on the Hub, and no official
quantization config exists for it. Each quantized convolution is built as
`input -> Quantize -> Dequantize(fp16) -> Convolution(+bias)`, with the kernel
`FP8 constant -> Dequantize(fp16, axis 0)`.

| Choice | Decision |
|---|---|
| Quantized (32 of 81 convolutions) | every pointwise (1x1) convolution on a spatial map larger than 1x1 |
| fp16 | the 3x3 stem (3 input channels), the 16 depthwise convolutions, the 32 squeeze-excite convolutions (on a 1x1 map), the classifier |
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
- Speed: CUDA graph replay, median of 7 interleaved rounds of 200 calls, spread
  at most 1.4%. Batch 32 and 128 are scratch builds (the family builds batch 1
  only).

### Results

Speed and size (ms per engine call; x is fp16 time over FP8 time):

| GPU | TensorRT | Batch | fp32 ms | fp16 ms | FP8 ms | x | plan MiB fp16 to FP8 |
|---|---|---|---|---|---|---|---|
| B200 | 11.4 | 1 | 0.522 | 0.414 | 0.439 | 0.94 | 11.6 to 8.9 |
| B200 | 11.4 | 32 | 1.831 | 1.041 | 1.091 | 0.95 | 14.1 to 12.4 |
| B200 | 11.4 | 128 | 5.897 | 2.807 | 2.830 | 0.99 | 23.4 to 24.6 |
| B300 | 11.4 | 1 | 0.496 | 0.419 | 0.441 | 0.95 | 11.5 to 9.0 |
| B300 | 11.4 | 32 | 1.793 | 0.994 | 1.034 | 0.96 | 14.3 to 12.5 |
| B300 | 11.4 | 128 | 5.709 | 2.688 | 2.723 | 0.99 | 23.5 to 24.6 |
| B200 | 11.1 | 1 | - | 0.420 | 0.432 | 0.97 | 11.5 to 8.9 |

Quality (1500 held-out images, batch 1):

| Engine | Top-1 | Agreement with PyTorch top-1 | KL mean | KL p99 |
|---|---|---|---|---|
| PyTorch fp32 | 79.13% | - | - | - |
| fp16 engine (B200) | 79.20% | 99.87% | < 0.0001 | < 0.001 |
| FP8 engine (B200) | 77.60% | 93.13% | 0.0424 | 0.358 |
| FP8 engine (B300) | 77.67% | 93.27% | 0.0420 | - |
| PyTorch fake quantization of the same recipe | 77.93% | 93.40% | 0.0417 | 0.346 |

Kernel time per group, B200, TensorRT 11.4 (layer profiler, ms; the profiler
adds time per layer, so the sums are larger than the latencies above):

| Kernel group | fp16 batch 1 | FP8 batch 1 | fp16 batch 128 | FP8 batch 128 |
|---|---|---|---|---|
| convolutions | 0.674 | 0.399 fp16 + 0.282 FP8 = 0.681 | 2.284 | 1.332 fp16 + 0.926 FP8 = 2.258 |
| other (pointwise, Quantize, pooling, classifier) | 0.540 | 0.619 | 1.238 | 1.359 |
| layers in the plan | 159 | 170 | 136 | 150 |

Variants (TensorRT 11.4, B200; x is fp16 time over variant time):

| Variant | x batch 1 / 32 / 128 | plan MiB batch 1 | Top-1 | KL mean |
|---|---|---|---|---|
| recipe above | 0.94 / 0.95 / 0.99 | 8.9 | 77.60% | 0.0424 |
| + 3x3 stem in FP8 | 0.96 / 0.89 / 0.88 | 9.1 (92.0 at batch 128) | 77.53% | 0.0558 |
| per-tensor weight scale | 0.95 / 0.95 / 0.99 | 8.9 | 77.20% | 0.0576 |
| batch norm kept as a scale layer | 0.95 / 0.95 / 0.99 | 8.9 | 77.67% | 0.0421 |
| + depthwise convolutions in FP8 (B300, 100 images) | 0.07 (batch 1 and 8) | 23.4 | - | 0.117 |

### NVFP4 (not tried)

TensorRT does not support NVFP4 on convolutions (block-scaled FP4 needs a 2-D
GEMM), FP8 already loses 1.6 top-1 points, and the engine is not limited by
weight reads (plan under 12 MiB).

### What was not tried

| Item | Note |
|---|---|
| Percentile or entropy calibration, quantization-aware training | not measured; could recover accuracy, not speed |
| Other EfficientNet sizes (b1 to b7), H100, native runtime with an FP8 bundle | not measured |

### When to revisit

- A TensorRT release with FP8 1x1 convolution kernels that are faster than
  fp16 for small channel counts, and that fuses the Quantize into the producing
  layer after residual adds and squeeze-excite products.
- A TensorRT FP8 kernel for depthwise (grouped) convolutions that keeps one
  layer per convolution.
- A larger, compute-bound model in this family: at batch 128 FP8 is already at
  0.99x, so the gain, if any, would come from larger convolutions.

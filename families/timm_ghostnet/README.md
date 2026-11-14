# timm GhostNet

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
trtmc build timm/ghostnet_100.in1k -o timm_ghostnet.bundle
trtmc classify timm_ghostnet.bundle --image photo.jpg
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
cmake --build build --target test_timm_ghostnet_task_contract test_timm_ghostnet_image_preprocess
ctest --test-dir build --output-on-failure -R '^timm_ghostnet_(task_contract|image_preprocess)$'
```

`tests/sdk_consumer.c` and `tests/sdk_consumer.cpp` use only the public SDK. For
an already-built bundle, supply an unprocessed RGB float32 HWC file and its
original height and width:

```sh
build/test_timm_ghostnet_sdk_c timm_ghostnet.bundle build image.rgb.f32 480 640
build/test_timm_ghostnet_sdk_cpp timm_ghostnet.bundle build image.rgb.f32 480 640
```

Each prints all scores as JSON and reads the result after releasing its model
handle. These commands perform real inference and require the matching runtime
and checkpoint bundle; the CPU contract test alone does not qualify a checkpoint.

## Benchmark timing

`tests/performance.yaml` takes over the existing `timm_ghostnet.classify`
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
11.4 (and 11.1) for `timm/ghostnet_100.in1k`. It is not shipped: FP8 is slower
than fp16 at batch 1 (0.89x to 0.91x) and at batch 32 and 128 (0.86x to 0.90x),
and ImageNet top-1 drops by 0.1 to 0.3 points, with only 93% top-1 agreement
with PyTorch. This section records the recipe, the method and the numbers, so
that the study does not have to be repeated. The FP8 code was not merged; this
family has no FP8 build.

### Summary and decision

| Item | fp16 (manifest) | FP8 (fp16 engine) | FP8 vs fp16 |
|---|---|---|---|
| Latency, batch 1, B200 / B300 (TensorRT 11.4) | 0.364 / 0.362 ms | 0.407 / 0.397 ms | 0.89x / 0.91x |
| Latency, batch 32, B200 / B300 (TensorRT 11.4) | 0.828 / 0.711 ms | 0.918 / 0.822 ms | 0.90x / 0.86x |
| Latency, batch 128, B200 / B300 (TensorRT 11.4) | 1.824 / 1.734 ms | 2.047 / 2.022 ms | 0.89x / 0.86x |
| Plan size, batch 1, B200 | 11.4 MiB | 10.7 MiB | -7% |
| ImageNet top-1, B200 (PyTorch fp32 74.00%) | 74.00% | 73.67% | -0.33 points |
| Top-1 agreement with PyTorch, B200 | 99.73% | 93.60% | -6.1 points |
| Non-finite outputs | 0 | 0 | none |

Decision: do not ship FP8 for timm GhostNet.

- Speed: the FP8 1x1 convolutions are not faster than the fp16 ones. They are
  small (8 to 960 output channels) and limited by activation traffic, not by
  math. The Quantize layers that do not fuse into the producing layer (after
  the Ghost module concatenations, residual adds and squeeze-excite products)
  add 22 to 28 layers and make the FP8 engine slower at every batch size.
- Size: the plan shrinks by 0.7 MiB at batch 1; the weights are small, so FP8
  gives no useful size gain.
- Quality: per-tensor 8-bit activations change the top-1 class of 6% to 7% of
  the images on this mobile network. PyTorch fake quantization of the same
  recipe gives the same loss, so the loss comes from the recipe, not from
  TensorRT.
- The 41 depthwise convolutions stay fp16. With them in FP8, the TensorRT 11.4
  build fails on B300 at batch 1 and 8: the autotuner finds no tactic for a
  Ghost module concatenation that feeds an FP8 depthwise convolution ("no
  tactics to implement operation", then "Could not find any implementation for
  node").

TensorRT builder optimization level 3 (proposed for this family in #2123, not
merged when this study was done) makes the fp16 engine faster: at batch 1,
0.379 to 0.338 ms (1.12x) on B200 and 0.368 to 0.319 ms (1.15x) on B300 with
TensorRT 11.4, measured with a separate harness, so its absolute numbers
differ slightly from the tables here. FP8 was not measured at level 3; the
fp16 baseline that FP8 would have to beat only gets faster.

### Recipe that was measured

No FP8 or NVFP4 checkpoint of this model exists on the Hub, and no official
quantization config exists for it. Each quantized convolution is built as
`input -> Quantize -> Dequantize(fp16) -> Convolution(+bias)`, with the kernel
`FP8 constant -> Dequantize(fp16, axis 0)`.

| Choice | Decision |
|---|---|
| Quantized (38 of 95 convolutions) | every pointwise (1x1) convolution on a spatial map larger than 1x1: the primary convolutions of the Ghost modules and the shortcut 1x1 convolutions |
| fp16 | the 3x3 stem (3 input channels), the 41 depthwise convolutions (Ghost cheap operations, strided and shortcut depthwise), the 15 squeeze-excite and head convolutions (on a 1x1 map), the classifier |
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
  variants built and timed in one job per GPU type, spread at most 3.1%
  (B200 fp16 batch 128; all others at most 1.5%). Batch 32 and 128 are scratch
  builds (the family builds batch 1 only).

### Results

Speed and size (ms per engine call; x is fp16 time over FP8 time):

| GPU | TensorRT | Batch | fp32 ms | fp16 ms | FP8 ms | x | plan MiB fp16 to FP8 | layers fp16 to FP8 |
|---|---|---|---|---|---|---|---|---|
| B200 | 11.4 | 1 | 0.443 | 0.364 | 0.407 | 0.89 | 11.4 to 10.7 | 168 to 190 |
| B200 | 11.4 | 32 | 1.272 | 0.828 | 0.918 | 0.90 | 14.9 to 14.2 | 151 to 174 |
| B200 | 11.4 | 128 | 3.525 | 1.824 | 2.047 | 0.89 | 27.2 to 26.5 | 151 to 174 |
| B300 | 11.4 | 1 | 0.423 | 0.362 | 0.397 | 0.91 | 10.8 to 10.2 | 158 to 186 |
| B300 | 11.4 | 32 | 1.148 | 0.711 | 0.822 | 0.86 | 15.0 to 14.4 | 152 to 175 |
| B300 | 11.4 | 128 | 3.315 | 1.734 | 2.022 | 0.86 | 27.3 to 26.6 | 152 to 175 |
| B200 | 11.1 | 1 | - | 0.363 | 0.407 | 0.89 | 11.4 to 10.6 | 162 to 182 |

Quality (1500 held-out images, batch 1):

| Engine | Top-1 | Agreement with PyTorch top-1 | KL mean | KL p99 |
|---|---|---|---|---|
| PyTorch fp32 | 74.00% | - | - | - |
| fp16 engine (B200) | 74.00% | 99.73% | < 0.0001 | < 0.001 |
| fp16 engine (B300) | 73.93% | 99.73% | < 0.0001 | < 0.001 |
| FP8 engine (B200) | 73.67% | 93.60% | 0.0297 | 0.228 |
| FP8 engine (B300) | 73.87% | 93.20% | 0.0297 | 0.240 |
| FP8 engine (B200, TensorRT 11.1) | 73.33% | 93.60% | 0.0299 | 0.220 |
| PyTorch fake quantization of the same recipe | 73.73% | 93.47% | 0.0294 | 0.228 |

Kernel time per group, B200, TensorRT 11.4 (layer profiler, ms; the profiler
adds time per layer, so the sums are larger than the latencies above):

| Kernel group | fp16 batch 1 | FP8 batch 1 | fp16 batch 128 | FP8 batch 128 |
|---|---|---|---|---|
| convolutions | 0.708 | 0.411 fp16 + 0.354 FP8 = 0.766 | 1.885 | 1.174 fp16 + 0.760 FP8 = 1.934 |
| other (pointwise, Quantize, concatenation, pooling, classifier) | 0.472 | 0.595 | 0.711 | 1.011 |
| layers in the plan | 168 | 190 | 151 | 174 |

Variants (TensorRT 11.4, B200; x is fp16 time over variant time):

| Variant | x batch 1 / 32 / 128 | plan MiB batch 1 (batch 128) | Top-1 | KL mean |
|---|---|---|---|---|
| recipe above | 0.89 / 0.90 / 0.89 | 10.7 (26.5) | 73.67% | 0.0297 |
| + 3x3 stem in FP8 | 0.89 / 0.82 / 0.76 | 10.8 (93.9) | 73.07% | 0.0552 |
| per-tensor weight scale | 0.89 / 0.90 / 0.89 | 10.6 (26.5) | 73.53% | 0.0421 |
| batch norm kept as a scale layer | 0.89 / 0.90 / 0.89 | 10.6 (26.5) | 73.33% | 0.0297 |
| + depthwise convolutions in FP8 (B300) | build fails at batch 1 and 8 | - | - | - |
| + depthwise convolutions, PyTorch fake quantization | - | - | 72.47% | 0.0923 |

### NVFP4 (not tried)

TensorRT does not support NVFP4 on convolutions (block-scaled FP4 needs a 2-D
GEMM), FP8 already loses accuracy, and the engine is not limited by weight
reads (plan under 12 MiB).

### What was not tried

| Item | Note |
|---|---|
| Percentile or entropy calibration, quantization-aware training | not measured; could recover accuracy, not speed |
| FP8 at builder optimization level 3 (#2123) | not measured |
| Other GhostNet sizes or versions (GhostNetV2, V3), H100, native runtime with an FP8 bundle | not measured |

### When to revisit

- A TensorRT release with FP8 1x1 convolution kernels that are faster than
  fp16 for small channel counts, and that fuses the Quantize into the producing
  layer after concatenations, residual adds and squeeze-excite products.
- A TensorRT FP8 kernel for depthwise (grouped) convolutions that builds after
  a concatenation and keeps one layer per convolution.
- A larger, compute-bound model in this family.

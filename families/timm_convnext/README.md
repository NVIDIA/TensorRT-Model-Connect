# timm ConvNeXt

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
trtmc build timm/convnext_tiny.in12k_ft_in1k -o timm_convnext.bundle
trtmc classify timm_convnext.bundle --image photo.jpg
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
cmake --build build --target test_timm_convnext_task_contract test_timm_convnext_image_preprocess
ctest --test-dir build --output-on-failure -R '^timm_convnext_(task_contract|image_preprocess)$'
```

`tests/sdk_consumer.c` and `tests/sdk_consumer.cpp` use only the public SDK. For
an already-built bundle, supply an unprocessed RGB float32 HWC file and its
original height and width:

```sh
build/test_timm_convnext_sdk_c timm_convnext.bundle build image.rgb.f32 480 640
build/test_timm_convnext_sdk_cpp timm_convnext.bundle build image.rgb.f32 480 640
```

Each prints all scores as JSON and reads the result after releasing its model
handle. These commands perform real inference and require the matching runtime
and checkpoint bundle; the CPU contract test alone does not qualify a checkpoint.

## Benchmark timing

`tests/performance.yaml` takes over the existing `timm_convnext.classify`
entry through the benchmark's family-owned reference protocol. The workload,
precision, 3 warmups, 10 measurements, 5% margin and top-class oracle are unchanged.
The reference times inference, complete float32 host logits and synchronization;
argmax, finite checks and JSON reporting happen after timing, as in the semantic
SDK benchmark worker. The existing reference policy still excludes input
preparation (`task-model-call-wall`), while the native public Task call includes
family preprocessing. This fixes reduction/reporting placement, not that existing
scope difference, and does not establish a performance improvement.

## FP8

`--quantization fp8` self-quantizes the original checkpoint at build time; no FP8
checkpoint exists for these models. FP8 needs the fp16 work precision.

```sh
trtmc build timm/convnext_tiny.in12k_ft_in1k --precision fp16 --quantization fp8 -o timm_convnext_fp8.bundle
```

- Quantized: the two MLP projections of every block (`mlp.fc1`, `mlp.fc2`,
  built as 1x1 convolutions) and the 2x2 downsample convolution of every stage
  after the first. Each runs as an e4m3 convolution with a static per-tensor
  activation scale and a per-tensor weight scale (`max(abs(weight)) / 448`).
- Kept in fp16: the stem convolution, the 7x7 depthwise convolutions, the layer
  norms, the layer scale, the GELU and the classifier head.
- Activation scales are calibrated once with ModelOpt `FP8_DEFAULT_CFG` (max
  calibration, 500 ImageNet validation images) and stored in
  `fp8_activation_scales.json`, keyed by model config and two weight sums.
  Calibrated: `convnext_tiny.in12k_ft_in1k`, `convnext_small.in12k_ft_in1k`,
  `convnext_base.fb_in22k_ft_in1k` at 224x224 (automatic family resolution
  claims `convnext_tiny` only; the other two were built through the family
  builder). Another model config is refused;
  run `python -m families.timm_convnext.calibrate_fp8_scales` (`fetch`, then
  `calibrate`) to add it. FP8 weights are never stored.
- An already quantized checkpoint, `--precision fp32` and any quantization other
  than `fp8` are refused.

The `convnext-tiny-in12k-ft-in1k-fp8` manifest runs the same top-1 E2E check as
the fp16 case.

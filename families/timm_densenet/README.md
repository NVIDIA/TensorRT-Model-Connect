# timm DenseNet

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
trtmc build timm/densenet121.ra_in1k -o timm_densenet.bundle
trtmc classify timm_densenet.bundle --image photo.jpg
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
cmake --build build --target test_timm_densenet_task_contract test_timm_densenet_image_preprocess
ctest --test-dir build --output-on-failure -R '^timm_densenet_(task_contract|image_preprocess)$'
```

`tests/sdk_consumer.c` and `tests/sdk_consumer.cpp` use only the public SDK. For
an already-built bundle, supply an unprocessed RGB float32 HWC file and its
original height and width:

```sh
build/test_timm_densenet_sdk_c timm_densenet.bundle build image.rgb.f32 480 640
build/test_timm_densenet_sdk_cpp timm_densenet.bundle build image.rgb.f32 480 640
```

Each prints all scores as JSON and reads the result after releasing its model
handle. These commands perform real inference and require the matching runtime
and checkpoint bundle; the CPU contract test alone does not qualify a checkpoint.

## FP8 quantization

`quantization=fp8` (precision fp16 only) self-quantizes the original checkpoint at
build time; no FP8 checkpoint exists for these classifiers. The 1x1 and 3x3
convolutions of every dense layer and the 1x1 convolution of every transition
run as FP8 convolutions: weights are quantized per output channel
(`max(abs(weight)) / 448`), and each convolution input has a static per-tensor
scale read from `fp8_activation_scales.json`. The stem convolution, batch norms,
pooling and the classifier stay fp16. TensorRT fuses each batch norm and ReLU
into the Quantize, or into the epilogue of the FP8 convolution before it.

```sh
trtmc build timm/densenet121.ra_in1k -o timm_densenet_fp8.bundle --quantization fp8
```

Scales are committed for `densenet121.ra_in1k`, `densenet121.tv_in1k`,
`densenet161.tv_in1k`, `densenet169.tv_in1k` and `densenet201.tv_in1k` at
224x224. Another checkpoint or input size raises and names the developer tool
`calibrate_fp8_scales.py` (ModelOpt `FP8_DEFAULT_CFG`, max calibration on 500
ImageNet validation images). A checkpoint that is already quantized is refused.

## Benchmark timing

`tests/performance.yaml` takes over the existing `timm_densenet.classify`
entry through the benchmark's family-owned reference protocol. The workload,
precision, 3 warmups, 10 measurements, 5% margin and top-class oracle are unchanged.
The reference times inference, complete float32 host logits and synchronization;
argmax, finite checks and JSON reporting happen after timing, as in the semantic
SDK benchmark worker. The existing reference policy still excludes input
preparation (`task-model-call-wall`), while the native public Task call includes
family preprocessing. This fixes reduction/reporting placement, not that existing
scope difference, and does not establish a performance improvement.

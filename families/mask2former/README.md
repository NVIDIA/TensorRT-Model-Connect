# Mask2Former

This family builds `Mask2FormerForUniversalSegmentation` checkpoints with a Swin backbone, validated on
`facebook/mask2former-swin-large-ade-semantic` (Swin-L, 216 M parameters, 150 ADE20K classes), and serves them through
the `image_to_semantic_segmentation` Task (`IModel` + `IImageToSemanticSegmentation`). The CLI command is `segment`.

## What runs where

- **Engine** (one TensorRT plan, float32, strongly typed, batch 1, static 384 x 384 input): Swin backbone, the pixel
  decoder (six multi-scale deformable-attention layers written with TensorRT grid sample, one FPN level, mask
  projection), the nine masked-attention decoder layers, the class probabilities and the semantic scores: the einsum of
  the class probabilities (no-object class removed) with the sigmoid of the final mask logits after a bilinear resize to
  384 x 384. Its only output is `semantic_scores` with shape (1, classes, 384, 384).
- **Runtime** (`runtime/`): the checkpoint processor's preprocessing (Pillow-exact 8-bit bilinear resize to the
  processor size, rescale, normalization), the engine run, then the bilinear resize of the scores to the original image
  size (as `torch.nn.functional.interpolate`, `align_corners=False`) and the argmax over classes.

The result carries one class ID per pixel at the original resolution, the class IDs `0..N-1` and the checkpoint's
`id2label` names. The vocabulary ID stays empty (the checkpoint declares none), no ignore or background label is
claimed, and the score maps are not returned: the Task accepts only logit or probability maps, and these sums of
class probability times mask probability over the queries are neither. There are no runtime Config fields.

Inputs: contiguous RGB, 8-bit or float32 in `[0, 1]`. Float32 values are quantized to the nearest 8-bit value, which
recovers the exact bytes of an 8-bit image decoded as `value / 255` (the CLI's image loader); the checkpoint's
processor resizes 8-bit images.

## Supported checkpoints

The builder reads `config.json` and `preprocessor_config.json` and accepts only what the graph reproduces: a Swin
backbone with four stages and all four outputs, stage resolutions that are whole multiples of the window size (no
padding), the post-norm decoder with ReLU, `feature_size == hidden_dim`, one extra FPN level, and a processor with a
fixed `{"height", "width"}` size, bilinear resampling, rescaling and normalization. Anything else is rejected with an
error that names the setting. Weights must be float32 as released.

The preprocessing follows `Mask2FormerImageProcessor`, the processor class the checkpoint's
`preprocessor_config.json` names, which resizes with PIL. Transformers' `AutoImageProcessor` may instead select the
torchvision-based fast processor, whose antialiased resize gives slightly different pixels.

No model weights are included in this repository. The validated checkpoint carries no license file and its Hugging
Face license field is `other`; users obtain it separately under its publisher's terms.

Build options: `precision=fp32` only (the checkpoint is float32), `backend=trt`, batch size 1, no quantization,
parallelism or KV cache; `image_height` / `image_width` may only repeat the processor size.

## Not supported

- FP16 / BF16 builds, quantization and batch sizes above 1.
- Instance and panoptic segmentation (the reference's other post-processors), and the per-class score maps.
- Swin windows that need padding (for example window 7 at a 384 x 384 input) and processors that resize by the
  shortest edge.

## Validate

```sh
cmake --build build --target trtmc trtmc_backend_trt trtmc_c trtmc_model_mask2former test_mask2former_task_contract
ctest --test-dir build --output-on-failure -R '^mask2former_task_contract$'
TRTMC_NATIVE_BUILD_DIR=build python -m pytest families/mask2former/tests -m "not gpu and not trt" \
  --ignore=families/mask2former/tests/test_e2e.py -q
TRTMC_BINARY=build/trtmc TRTMC_RUNTIME_ROOT=build TRTMC_NATIVE_BUILD_DIR=build \
  python -m pytest families/mask2former/tests/test_e2e.py --e2e-testcase mask2former-swin-large-ade-semantic -q
```

The CTest target also builds the C and C++ SDK consumers and `mask2former_image_ops_tool`, which
`tests/test_image_ops.py` uses to compare the runtime's image operations with Pillow (bit-exact resize), the
checkpoint's image processor (bit-exact pixel values) and torch (resize and argmax).

The E2E builds the bundle from the cached checkpoint and compares the label maps with the Transformers reference:
`Mask2FormerImageProcessor` and `Mask2FormerForUniversalSegmentation` in float32 on the GPU with TF32 disabled,
post-processed by `post_process_semantic_segmentation` at the original image size. The test image is a PNG, so the
CLI's image loader, the two SDK consumers and the reference read identical 8-bit pixels (JPEG decoders differ: on the
JPEG this PNG was decoded from, the CLI's loader and PIL differ by up to 3 in 4.7 % of the values). Every label map
must reach the pixel accuracy and mean IoU in `tests/thresholds/`; on a B300 all three maps equal the reference on
every pixel.

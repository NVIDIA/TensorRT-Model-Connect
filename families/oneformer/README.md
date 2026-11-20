# OneFormer

This family builds OneFormer universal-segmentation checkpoints with a Swin
backbone (`shi-labs/oneformer_cityscapes_swin_large`, 219 M parameters, FP32
weights, 19 Cityscapes classes) for `image_to_semantic_segmentation` as one
TensorRT engine: batch 1, dynamic image height and width (32 to 2048 pixels per
side, optimized for 1024 x 2048), fp32 with TF32 disabled. The family has no
fp16 or quantized build (see "FP16 study" below).

## What runs where

| Stage | Where | Matches |
|---|---|---|
| uint8 conversion, resize so the shortest edge is 1024 and the longest at most 2048, normalization | native runtime (C++) | ATen's uint8 antialiased bilinear kernel (int16 weights, horizontal pass first) and the fast `OneFormerImageProcessor`'s fused `mean * 255` / `std * 255` normalization, bit for bit; torchvision 0.27 and later use that kernel for uint8 CPU tensors on AVX2 / AVX-512 x86 and on aarch64, so the native pixels equal the processor's there (verified with torchvision 0.27 and 0.30 on x86-64 with AVX2) |
| Swin-L backbone, pixel decoder (multi-scale deformable attention, FPN), task-conditioned transformer decoder, semantic score map at stride 4 | TensorRT engine | `OneFormerForUniversalSegmentation` with the task prompt "the task is semantic" |
| resize of the score map to the input image and argmax | native runtime (C++) | `post_process_semantic_segmentation` (ATen float antialiased bilinear, first maximum) |

The task prompt is tokenized at build time with the checkpoint's CLIP tokenizer
and its ids enter the task MLP as values, as in the reference. The engine's
`pixel_values` input is the processed image (no padding for one image); its
`semantic_scores` output is
`einsum("qc,qhw->chw", softmax(class_logits)[:, :-1], sigmoid(mask_logits))`.

The Task returns the label map at the input resolution, the class ids and the
class names from `config.json`. It returns no score maps: the semantic scores
are sums over queries, neither logits nor probabilities, and the shared result
contract accepts only those two kinds.

Images are RGB, uint8 or float32 in [0, 1]; float pixels are quantized to the
same 8-bit values the reference processor resizes. An image whose processed
size falls outside the engine profile (an aspect ratio beyond about 64:1, or a
smaller `image_height` / `image_width` build capacity) is rejected.

The resampling code is adapted from PyTorch v2.12.0 (BSD-3-Clause, derived from
Pillow and Pillow-SIMD, MIT-CMU) and the size rule from Transformers v5.2.0
(Apache-2.0); see `NOTICE`.

## Limits

- Instance and panoptic segmentation, which the checkpoint also performs with
  other task prompts, need an instance / panoptic segments Task that the shared
  API does not define; they are not exposed.
- DiNAT backbones (`shi-labs/oneformer_*_dinat_*`) and training checkpoints (text
  mapper) are rejected by the builder.

## FP16 study (not shipped)

A strongly typed fp16 engine of the same graph was built and compared with the
FP32 reference on the fixture (B300, TensorRT 11.4): 32 ms per image at
1024 x 1715 against 117 ms for fp32 (both engines with the stage tensors as
extra outputs, mean of 5 warm runs), but the Swin-L stage-3 and stage-4 outputs
move by up to 7 (their range is 26 to 28; stage 1 and 2 move by 0.15),
mask logits by up to 114 and the stride-4 labels agree on 99.92 % of the
pixels. Computing every LayerNorm and GroupNorm in float32 does not change
this, so the loss arises inside the fp16 stage-3 blocks (residual stream,
attention or MLP); it was not narrowed further. An fp16 build would need that
fixed and an oracle calibrated against the reference's own fp16 drift; until
then the builder rejects fp16.

## Validation

```bash
# CPU: support, configuration and request policy; with a native build, the
# pre- and post-processing against the official processor (needs torchvision).
TRTMC_NATIVE_BUILD_DIR=<build> python -m pytest families/oneformer/tests -k "not e2e" -rs

# GPU: E2E against the official pipeline (fixture, portrait, downscaled, 2048 x 1024
# frame, wide) and the stage-by-stage graph check.
TRTMC_BINARY=<build>/trtmc TRTMC_RUNTIME_ROOT=<build> TRTMC_NATIVE_BUILD_DIR=<build> \
  python -m pytest families/oneformer/tests/test_e2e.py --e2e-model oneformer -vv
```

The E2E reads the same lossless PNG on both sides and requires at least 99.99 %
label agreement, with every differing pixel within 1e-3 of a tie in the
reference's own scores. The graph check compares the backbone stages, the
pixel-decoder levels, the mask features and the class and mask logits of an
engine built with them as extra outputs against the reference's tensors.

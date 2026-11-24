# DINOv3

This family builds DINOv3 image-feature extractors (ViT-S, ViT-B, ViT-L, ViT-H+
and ConvNeXt backbones) as one TensorRT engine for the `image_features` task:
`facebook/dinov3-vits16-pretrain-lvd1689m`,
`facebook/dinov3-convnext-tiny-pretrain-lvd1689m` and
`timm/vit_small_patch16_dinov3_qkvb.lvd1689m` in the manifests, all fp16.
Batch is 1. ViT-L and ViT-H+ must be built in fp32: their residual stream
reaches about 1.6e5, above the fp16 maximum 65504, so their fp16 engines return
NaN, and the family has no bf16 path.

## Quantization

`--quantization fp8` quantizes the original checkpoint at build time (#1918),
with `--precision fp16` for ViT-S, ViT-B and ConvNeXt and `--precision fp32` for
ViT-L and ViT-H+. The fused q/k/v, `o_proj`, `up_proj`, `down_proj` (and
`gate_proj` of the gated MLP of ViT-H+) of every ViT layer and both pointwise
projections of every ConvNeXt block get per-tensor FP8 weights (amax / 448) and
static activation scales from `fp8_activation_scales.json` (200 ImageNet
validation images, one table per model config and resolution). Dequantize and
the matmul run in the engine precision. The patch embedding, convolutions, norms,
RoPE and attention stay unquantized. The plan is about 2x (fp16) or 4x (fp32)
smaller.

The ConvNeXt-T FP8 criteria (`tests/thresholds/dinov3-convnext-tiny-pretrain-lvd1689m-fp8.json`)
hold the full-feature cosine to 0.98, the cosine their relative Frobenius limit 0.2
allows at equal norms (1 - 0.2^2 / 2); ViT-S keeps 0.99. The recipe's own error spreads
over every block: on the 200 held-out ImageNet validation images (200..399 of the
calibration set, TensorRT 11.4, GB300, against Transformers fp16) the full-feature cosine
has median 0.992 and 1st percentile 0.980, so 60 of the 200 images missed 0.99. A
PyTorch emulation of the recipe misses the same 60, so the error is the quantization's,
not the engine's. With 0.98, 7 of the 200 miss a criterion (each within twice its
budget). The features keep their task accuracy: 20-NN accuracy on a 100-class
Tiny-ImageNet subset (3000 gallery, 1000 queries) is 0.732 for the FP8 engine, 0.729
for Transformers fp32 and fp16.

### NVFP4

NVFP4 was not tried; no NVFP4 engine was built. FP8 speed against the base
engine of the same model, batch 1 (the only batch the family builds), B300
(numbers from #1918):

| Model | Resolution | Base engine | FP8 vs base, TensorRT 11.1 / 11.4 | FP8 plan, TensorRT 11.1 (MiB) |
|---|---|---|---|---|
| ViT-S/16 | 224 | fp16 | 0.99x / 0.99x | 22.8 |
| ConvNeXt-T | 224 | fp16 | 1.04x / 1.04x | 29.6 |
| ViT-B/16 | 224 | fp16 | 1.05x / 1.11x | 85.8 |
| ViT-B/16 | 512 | fp16 | 1.06x / 1.08x | - |
| ViT-L/16 | 224 | fp32 | 1.26x / 1.30x | 295.1 |
| ViT-L/16 | 512 | fp32 | 1.10x / 1.12x | - |
| ViT-H+/16 | 224 | fp32 | 1.36x / 1.38x | 810.8 |
| ViT-H+/16 | 512 | fp32 | 1.19x / 1.23x | - |

Reasons:

- ViT-S, ViT-B and ConvNeXt-T: FP8 itself gives only 0.99x to 1.11x at batch 1,
  and the FP8 plans are small (23 to 86 MiB).
- ViT-L and ViT-H+: the FP8 gain is measured against fp32 (TF32 GEMMs) engines,
  because fp16 overflows and the family has no bf16 path. A bf16 baseline would
  be the better reference before a 4-bit format is studied; it needs a new
  precision path first.

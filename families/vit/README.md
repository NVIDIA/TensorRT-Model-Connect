# ViT

This family builds the transformers ViT classifier
(`ViTForImageClassification`) as one TensorRT engine for the
`image_to_class_scores` task (`google/vit-base-patch16-224` in the manifests).
Batch is 1. The default precision is fp16 (#2160); `--precision fp32` builds
the fp32 network. The manifests build fp32, fp16 and FP8.

## Quantization

`--quantization fp8` with the default fp16 precision quantizes the original
checkpoint at build time (#1974, moved onto the fp16 graph in #2059). Query,
key, value, the attention output, `intermediate.dense` and `output.dense` of
every layer get per-tensor FP8 weights (amax / 448) and static activation
scales from `fp8_activation_scales.json` (ModelOpt `FP8_DEFAULT_CFG`, 500
ImageNet validation images). Dequantize and the GEMM run in fp16. The patch
convolution, embeddings, layer norms, attention matmuls and classifier stay
fp16. FP8 with `--precision fp32` or `bf16` raises. Scales ship for
`vit-base-patch16-224`. TensorRT 11.4, batch 1: FP8 is 1.05x (B200) and 1.26x
(B300) faster than fp16, the plan is 48% smaller (175 MB to 91 MB), and top-1
agrees with the fp32 PyTorch model on 98.1% of 1500 held-out ImageNet images
(accuracy 80.5% against 81.0%).

### NVFP4

NVFP4 was not tried; no NVFP4 engine was built. ViT at batch 1 is not bound by
weight reads (numbers from #2059, TensorRT 11.4):

| Item | vit-base-patch16-224 |
|---|---|
| FP8 projection weights | 86 MB |
| Time to read them at about 8 TB/s | about 0.011 ms |
| FP8 engine call, batch 1 | about 0.35 ms |
| Weight reads as share of the call | about 3% |
| FP8 vs fp16, batch 1 | 1.05x (B200), 1.26x (B300) |

4-bit weights could save at most the weight-read time; the call is bound by
compute and launches. The larger FP8 gain on B300 comes from a slower fp16
engine there (0.434 ms against 0.382 ms on B200), not from weight reads. NVFP4
also adds dynamic quantize kernels and is lossier than FP8.

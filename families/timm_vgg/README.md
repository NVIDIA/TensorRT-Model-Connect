# timm VGG

This family builds timm VGG classifiers (`timm/vgg16.tv_in1k` in the manifests;
VGG11, VGG13 and VGG19 build from the same code path) as one TensorRT engine for
the `classification` task: checkpoint input size, batch 1, fp16 or fp32 (the
manifests use fp16), builder optimization level 1. The convolution and pooling
layout is read from the checkpoint. The output contains every class logit in
checkpoint order.

## Quantization

`quantization=fp8` (fp16 only) self-quantizes the original checkpoint at build
time; FP8 weights are never stored. Every 3x3 convolution after the first and
the two convolutional head layers `pre_logits.fc1` (7x7) and `pre_logits.fc2`
(1x1) run as FP8 convolutions with one weight scale per tensor and a static
per-tensor input scale from `fp8_activation_scales.json`. The first convolution
and the classifier `head.fc` stay fp16. On B200 and B300 with TensorRT 11.4, FP8
is 1.24x to 1.47x faster than fp16 at batch 1 and the plan is 48% smaller, at
the fp16 top-1 floor. VGG16 and VGG19 are calibrated; VGG11 and VGG13 are
refused until `calibrate_fp8_scales.py` is run for them.

### NVFP4

NVFP4 is not used. TensorRT has no FP4 convolution (block-scaled FP4 needs a
2-D GEMM input), and every layer that FP8 quantizes is a convolution, including
the head layers `fc1` and `fc2`. Only these two head layers (120M weights) could
be rewritten as GEMMs and moved to FP4:

| Item (VGG16, B300, batch 1, TensorRT 11.4) | Value |
|---|---|
| Head time in the FP8 engine | 0.017 ms of 0.199 ms |
| Plan size saving from an FP4 head | about 57 MiB |

The gain would be at most a few percent of the time, and NVFP4 had 5x to 14x the
FP8 KL divergence on timm Swin. NVFP4 was not built.

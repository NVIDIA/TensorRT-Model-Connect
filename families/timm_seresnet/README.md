# timm SE-ResNet

This family builds timm SE-ResNet and SE-ResNeXt classifiers
(`timm/seresnet50.a1_in1k` in the manifests) as one TensorRT engine for the
`classification` task: checkpoint input size, batch 1, fp16 or fp32 (the
manifests use fp16). Depth, block type and grouping are read from the
checkpoint; a checkpoint without the squeeze-excitation gate is refused (it
belongs to `timm_resnet`). The output contains every class logit in checkpoint
order.

## Quantization

`quantization=fp8` (fp16 only) self-quantizes the original checkpoint at build
time; FP8 weights are never stored. The convolutions of the stages `layer2` to
`layer4` (42 convolutions of SE-ResNet-50) run as FP8 convolutions with the
batch norm folded in, one weight scale per output channel and a static
per-tensor input scale from `fp8_activation_scales.json`. The stem, the first
stage `layer1`, the squeeze-excitation gates and the classifier head stay fp16.
On B200 and B300 with TensorRT 11.4, FP8 is 1.04x to 1.08x faster than fp16 and
the plan is 38% to 40% smaller, for 0.0 to 0.5 point of top-1. Only
`seresnet50.a1_in1k` is calibrated; other checkpoints are refused until
`calibrate_fp8_scales.py` is run for them.

### NVFP4

NVFP4 is not used. TensorRT has no FP4 convolution (block-scaled FP4 needs a
2-D GEMM input), and every layer that FP8 quantizes is a convolution. The only
GEMMs are the squeeze-excitation gate projections, which run on the pooled 1x1
map (too small to gain even from FP8), and the 2048 x 1000 classifier head.
NVFP4 was not built.

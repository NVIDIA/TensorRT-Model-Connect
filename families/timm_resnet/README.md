# timm ResNet

This family builds timm ResNet and ResNeXt classifiers
(`timm/resnet50.a1_in1k` in the manifests) as one TensorRT engine for the
`image_to_class_scores` task: checkpoint input size, batch 1, fp16 or fp32 (the
manifests use fp16). The stage and block layout is read from the checkpoint, so
ResNet-18 to ResNet-152 and the grouped ResNeXt variants build from the same
code path. The output contains every class logit in checkpoint order.

## Quantization

`quantization=fp8` (fp16 only) self-quantizes the original checkpoint at build
time; FP8 weights are never stored. The convolutions of the stages `layer2` to
`layer4` (42 convolutions of ResNet-50) run as FP8 convolutions with the batch
norm folded in, one weight scale per output channel and a static per-tensor
input scale from `fp8_activation_scales.json`. The stem, the first stage
`layer1` and the classifier head stay fp16. On B200 and B300 with TensorRT 11.4,
FP8 is 1.07x to 1.12x faster than fp16 and the plan is 42% to 44% smaller, for
0.1 to 1.1 point of top-1. Only `resnet50.a1_in1k` is calibrated; other
checkpoints are refused until `calibrate_fp8_scales.py` is run for them.

### NVFP4

NVFP4 is not used. TensorRT has no FP4 convolution (block-scaled FP4 needs a
2-D GEMM input), and every layer that FP8 quantizes is a convolution. The only
GEMM is the 2048 x 1000 classifier head (4 MB in fp16): in the FP8 engine on
B300 at batch 1, GEMM kernels take 0.013 ms of 0.303 ms of GPU kernel time.
NVFP4 was not built.

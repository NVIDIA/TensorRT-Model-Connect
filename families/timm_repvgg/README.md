# timm RepVGG

This family builds timm RepVGG classifiers (`timm/repvgg_a2.rvgg_in1k` in the
manifests) as one TensorRT engine for the `classification` task: checkpoint
input size, batch 1, fp16 or fp32 (the manifests use fp16). The build fuses the
3x3, 1x1 and identity branches of each block (with their batch norms) into one
dense 3x3 convolution, so the engine is a plain stack of 3x3 convolutions. The
output contains every class logit in checkpoint order.

## Quantization

`quantization=fp8` (fp16 only) self-quantizes the original checkpoint at build
time; FP8 weights are never stored. The fused 3x3 convolution of every block
runs as an FP8 convolution with one weight scale per output channel and a static
per-tensor input scale from `fp8_activation_scales.json`. The stem (3 input
channels), the global pool and the head stay fp16. On B200 and B300 with
TensorRT 11.4, FP8 is 1.28x to 1.45x faster than fp16 at batch 1 and the plan is
46% to 49% smaller, with top-1 within 0.4 point of the fp16 engine. Another
model config is refused until `calibrate_fp8_scales.py` is run for it.

### NVFP4

NVFP4 is not used. TensorRT has no FP4 convolution (block-scaled FP4 needs a
2-D GEMM input), and it cannot run a 3x3 convolution as an FP4 GEMM. Every layer
that FP8 quantizes is a 3x3 convolution, and the convolutions are 91% to 97% of
the GPU time (B300, RepVGG-A2 at batch 1 and RepVGG-B3 at batch 32); the only
GEMM is the head. The FP8 KL divergence of RepVGG is already 4x the ConvNeXt and
VGG FP8 KL divergence, so a 4-bit format would also need a new recipe for
quality. NVFP4 was not built.

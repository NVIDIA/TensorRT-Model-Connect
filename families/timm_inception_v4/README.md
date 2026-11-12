# timm Inception-v4

This family builds timm Inception-v4 classifiers (`timm/inception_v4.tf_in1k`
in the manifests) as one TensorRT engine for the `image_to_class_scores` task:
checkpoint input size (299 x 299), batch 1, fp16 or fp32 (the manifests use
fp16). Every block is identified from the weight names the checkpoint carries
for it, not from a depth table. The output contains every class logit in
checkpoint order.

## Quantization

`quantization=fp8` (fp16 only) self-quantizes the original checkpoint at build
time; FP8 weights are never stored. Every convolution of the Inception-A,
Reduction-A, Inception-B, Reduction-B and Inception-C blocks (`features.6`
onward) runs as an FP8 convolution with the batch norm folded in, one weight
scale per output channel and a static per-tensor input scale from
`fp8_activation_scales.json`. The stem and the classifier stay fp16. On B200 and
B300 with TensorRT 11.4, FP8 is 1.15x to 1.21x faster than fp16 at batch 1 and
the plan is 46% smaller, at the fp16 top-1 floor.

### NVFP4

NVFP4 is not used. TensorRT has no FP4 convolution (block-scaled FP4 needs a
2-D GEMM input), and every layer that FP8 quantizes is a convolution. The only
GEMM in the model is the 1536 x 1000 classifier, which holds 3.6% of the
weights. NVFP4 was not built.

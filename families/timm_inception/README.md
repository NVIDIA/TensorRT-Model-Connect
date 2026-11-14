# timm Inception

This family builds timm Inception-v3 classifiers (`timm/inception_v3.tv_in1k` in
the manifests) as one TensorRT engine for the `image_to_class_scores` task:
checkpoint input size (299 x 299), batch 1, fp16 or fp32 (the manifests use
fp16). The graph is written with TensorRT Network API calls; each of the five
Inception block topologies is identified from the branch names in the
checkpoint. The output contains every class logit in checkpoint order.

## Quantization

`quantization=fp8` (fp16 only) self-quantizes the original checkpoint at build
time; FP8 weights are never stored. Every convolution of the 17x17 and 8x8
stages (`Mixed_6a` onward) runs as an FP8 convolution with the batch norm
folded in, one weight scale per output channel and a static per-tensor input
scale from `fp8_activation_scales.json`. The stem, the 35x35 InceptionA blocks
and the classifier stay fp16. On B200 and B300 with TensorRT 11.4, FP8 is 1.06x
to 1.07x faster than fp16 at batch 1 and the plan is 42% smaller, at the fp16
top-1 floor.

### NVFP4

NVFP4 is not used. TensorRT has no FP4 convolution (block-scaled FP4 needs a
2-D GEMM input), and every layer that FP8 quantizes is a convolution. The only
GEMM in the model is the 2048 x 1000 classifier, which holds 8.6% of the
weights. NVFP4 was not built.

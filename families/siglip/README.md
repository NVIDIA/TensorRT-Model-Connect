# SigLIP

This family builds SigLIP (`SiglipModel`) as a vision engine and a text engine
for the `image_to_embedding`, `text_to_embedding` and
`text_image_to_relevance` tasks (`google/siglip-base-patch16-224` in the
manifests). Batch is 1. The family builds fp32 only: an all-fp16 engine moves
the embeddings by more than the family's E2E tolerance.

## Quantization

`--precision fp32 --quantization fp8` quantizes the original checkpoint at build
time (#1855). The q, k, v, out, fc1 and fc2 projections of every encoder layer
in both towers get per-tensor FP8 weights (amax / 448) and static activation
scales from `fp8_activation_scales.json` (ModelOpt `FP8_DEFAULT_CFG`, ImageNet
images and `a photo of <class>` prompts). Dequantize and the matmul run in fp32;
the patch convolution, embeddings, layer norms, pooling head and final text
projection stay fp32. Scales ship for `siglip-base-patch16-224` and
`siglip-large-patch16-384`. Against fp32, the bundle goes from 779 to 295 MiB
(base) and from 2494 to 771 MiB (large), and 93% to 97% of 200 ImageNet images
get the same zero-shot top-1 class as the PyTorch model.

### NVFP4

NVFP4 was not tried; no NVFP4 engine was built. No FP8 or NVFP4 checkpoint of
SigLIP exists on the Hub, and NVFP4 was out of scope in #1855. The models are
small encoders run at batch 1 (`siglip-base-patch16-224` has about 0.2B
parameters). The only measured FP8 gain is against fp32 math (TF32 off), the
family's only precision; the gain over fp16 was not measured and is smaller
(#1855). So the numbers do not show that weight reads limit the call (B300,
TensorRT 11.4, batch 1; the ratio is computed from the times in #1855):

| Model | Engine | fp32 | FP8 | FP8 vs fp32 (computed) |
|---|---|---|---|---|
| base-224 | vision | 1.89 ms | 0.63 ms | 3.00x |
| base-224 | text | 1.31 ms | 0.39 ms | 3.36x |
| large-384 | vision | 9.67 ms | 2.30 ms | 4.20x |
| large-384 | text | 3.17 ms | 0.81 ms | 3.91x |
| base-224 | bundle | 779 MiB | 295 MiB | |
| large-384 | bundle | 2494 MiB | 771 MiB | |

If the family gets an fp16 build, compare FP8 with it before trying NVFP4.

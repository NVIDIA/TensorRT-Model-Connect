# OWLv2

This family builds the OWLv2 open-vocabulary detector
(`google/owlv2-base-patch16-ensemble` in the manifest) for
`image_text_to_boxes`, fp16 or fp32 (the manifest uses fp16), batch 1. A bundle
holds two engines: an image engine (CLIP ViT-B/16 at 960 x 960, 3601 tokens,
plus the box and class heads) that runs once per image, and a text engine (12
layers, 16 tokens) that runs once per query phrase.

## Precision

fp16 engines keep fp16 weights and activations but accumulate every matmul and
convolution in fp32: the operands are cast to fp32 and the result back to fp16,
which is TensorRT's documented request for wider accumulation
(`graph._widen_accumulation`). Without it TensorRT may pick fp16-accumulating
kernels, and which kernel wins depends on build-time timing. The class head
multiplies each logit by a scale of about 31, so detection scores then moved by
0.02 to 0.105 against PyTorch fp32 between builds of one revision, and one build
replaced a detection with its neighbouring patch. With fp32 accumulation the worst
score delta on the E2E case is 0.008 in every build, the same as PyTorch fp16 on
the GPU. On GB300 the image engine takes the same time within noise (2.17 ms
before, 2.18 to 2.20 ms after) and its plan grows from 185 to 188 MB.

## Quantization

`quantization=fp8` quantizes the text tower only: q/k/v/`out_proj`, `fc1` and
`fc2` of its 12 layers, per-tensor weight scale amax / 448, static activation
scales from `fp8_activation_scales.json` (`text_model.encoder.layers.0.mlp.fc2`
stays fp16 as an outlier). It is a size option: the bundle goes from 299.6 to
266.2 MiB and the text engine is 1.00x to 1.07x of fp16. The image tower and
the heads are supported by the code but not enabled: FP8 on the image tower
moves the detections far from the fp16 engine (logit correlation 0.8875 against
0.9989 for fp16, both vs PyTorch) for at most 1.1x.

### NVFP4

NVFP4 was measured as GEMM timing only and is not shipped. No NVFP4 engine of
the model was built, so there is no quality or NaN measurement. The benchmark is
a stack of the 12 image-tower MLP pairs (`fc1` and `fc2`, 3601 rows, 768 and
3072 wide) with random weights and FP4 activations quantized at run time,
TensorRT 11.4.

| GEMM stack | B200 ms | B300 ms |
|---|---|---|
| Image-tower MLPs, fp16 | 0.406 | 0.385 |
| Image-tower MLPs, FP8 | 0.228 | 0.222 |
| Image-tower MLPs, NVFP4 | 0.191 | 0.202 |
| Text-tower MLPs (16 rows), fp16 / FP8 / NVFP4 | 0.100 / 0.079 / 0.105 | - |

Reasons:

- Speed: on the image tower NVFP4 saves only 0.02 to 0.04 ms more than FP8,
  about 2% of the image engine. The image engine is compute and attention bound
  at 3601 tokens: the fused attention, which stays fp16, is 46% (B300) to 52%
  (B200) of its time. On the text tower NVFP4 is slower than fp16.
- Quality: FP8 on the image tower already fails the detection check (logit
  correlation 0.8875), so the small NVFP4 time saving would come on top of a
  quality that is not usable.

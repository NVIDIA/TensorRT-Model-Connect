# DINOv2

This family builds DINOv2 image-feature extractors (`facebook/dinov2-base` in
the manifests; dinov2-small and -large build from the same code path) as one
TensorRT engine for the `image_to_token_and_pooled_features` task. One forward
pass returns the final hidden rows: the patch rows are the token features and
the class row is the pooled vector, both unnormalized. Batch is 1, input 224
pixels. The default precision is fp16; `--precision fp32` builds full float32
GEMMs (TF32 off).

## Quantization

`--precision fp32 --quantization fp8` quantizes the original checkpoint at build
time (#1975); FP8 builds the fp32 graph only. Query, key, value, the attention
output projection, `mlp.fc1` and `mlp.fc2` of every layer get per-tensor FP8
weights (amax / 448) and static activation scales from
`fp8_activation_scales.json` (ModelOpt `FP8_DEFAULT_CFG`). The patch
convolution, embeddings, layer norms, LayerScale, GELU and attention matmuls
stay float32. B200, batch 1: FP8 is 2.7x to 4.4x faster than the fp32 engine
and the plan is about 3.8x smaller, but an fp16 engine is still faster (FP8 on
fp32 work runs at 0.6x to 0.7x of fp16). kNN top-1 is within 0.7 point of fp32
PyTorch.

### NVFP4

NVFP4 was not built (#1975). DINOv2 at 197 to 257 tokens is not bound by weight
reads; this is a bound from measured plan sizes and times, not a measured
NVFP4 engine (B200, batch 1):

| Item | Value |
|---|---|
| fp16 plan, dinov2-large | 581 MiB |
| Time to read it at 8 TB/s | about 0.08 ms |
| fp16 engine call, dinov2-large, TensorRT 11.4 | 0.97 ms |
| FP8 on fp16 work vs fp16, batch 1 (measurement-only build) | 1.01x to 1.12x |

4-bit weights could save at most the weight-read time, and FP8 on top of fp16
already adds only 1.0x to 1.1x at batch 1.

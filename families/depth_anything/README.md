# Depth Anything

This family builds Depth Anything V2 monocular depth models
(`depth-anything/Depth-Anything-V2-Small-hf` in the manifests, fp16; base and
large build from the same code path) as one TensorRT engine for the
`monocular_depth` task. The graph is a DINOv2 ViT backbone that feeds a DPT
convolution neck and head. Precision fp32 (default) or fp16.

## Quantization

`quantization=fp8` (fp16 recommended, fp32 accepted) quantizes the original
checkpoint at build time. Only the ViT backbone is quantized: q, k, v, the
attention output projection and both MLP projections of every layer (per-tensor
weight scale amax / 448, static activation scales from
`fp8_activation_scales.json`, ModelOpt `FP8_DEFAULT_CFG`). The patch
convolution, norms, GELU and the whole DPT neck and head stay in the work
precision. On TensorRT 11.4, batch 1, FP8 is 1.03x to 1.38x faster than fp16
(B200 and B300, base and large).

### NVFP4

NVFP4 was measured on V2 base and large and is not shipped. The study built
real TensorRT engines (no NVFP4 code is in the family) with four layouts on top
of the FP8 recipe. TensorRT 11.4, batch 1, family graph, one job per GPU.

Speed vs FP8 (below 1.00x is slower than FP8):

| Layout | B200 base | B200 large | B300 base | B300 large |
|---|---|---|---|---|
| FP8 (shipped), vs fp16 | 1.03x | 1.14x | 1.20x | 1.38x |
| NVFP4 fc1 + fc2, q/k/v/o FP8 | 0.92x | 0.92x | 0.91x | 0.91x |
| NVFP4 all six projections | 0.83x | 0.83x | 0.81x | 0.83x |
| NVFP4 fc1 only, rest FP8 | 0.95x | 0.93x | 0.96x | 0.94x |
| NVFP4 fc2 only, rest FP8 | 0.93x | 0.90x | 0.93x | 0.91x |

| Criterion | Result |
|---|---|
| Speed vs FP8 | 0.81x to 0.96x at batch 1; best case fc1 only at 1.00x to 1.01x on B200 with a 3-D activation graph (noise level); 0.81x to 1.04x at batch 8 |
| Speed vs fp16 | 0.85x to 1.06x at batch 1 on B200 |
| Plan size | -10% (fc1 only) to -34% (all six) vs FP8 |
| Quality vs FP8 | depth error (normalized RMSE) 1.6x (fc1 only) to 3.4x (all six) the FP8 error |
| NaN | none at batch 1 in the family graph; 19 of 64 NVFP4 engines return NaN or wrong depth (batch 8 and the 3-D graph), none of the fc1-only engines |

Reasons:

- Speed: the FP4 GEMMs are faster than the FP8 GEMMs they replace, but each
  needs a separate dynamic-quantize kernel, and GELU and layer norm no longer
  fuse into the next GEMM. These added kernels cost more than the GEMMs save.
- Quality: 1.6x to 3.4x the FP8 depth error.
- TensorRT problem: the block-scaled FP4 GEMM that TensorRT picks for some
  batch, graph and GPU combinations returns wrong or NaN rows (seen on
  TensorRT 11.1 and 11.4). The cause was not isolated.

# Depth Anything

This family builds Depth Anything V2 monocular depth models
(`depth-anything/Depth-Anything-V2-Small-hf` in the manifests, fp16, and
`Depth-Anything-V2-Base-hf` for NVFP4; base and large build from the same code
path) as one TensorRT engine for the
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

`quantization=nvfp4` is shipped for the V2 Base and Large model configs only
(backbone hidden size 768 and 1024). Small (hidden size 384) is rejected with an
error that names the allowed model configs; use FP8 for Small.

`trtmc build <checkpoint> -o <bundle> --precision fp16 --quantization nvfp4`
(Base or Large checkpoint; fp16 recommended, fp32 accepted; Blackwell GPU)
builds the FP8 recipe above with one change: the first MLP projection
(`mlp.fc1`) of every layer is NVFP4 W4A4. The other five projections stay FP8
exactly as above.

- Weight: quantized at build time from the original checkpoint, the ModelOpt
  NVFP4 format (`weight_scale_2 = amax / (6 * 448)`, one e4m3 scale per 16
  inputs, E2M1 values packed two per byte). The bytes match ModelOpt
  `NVFP4QTensor` on real fc1 weights.
- Activation: `add_dynamic_quantize` (per-16 e4m3 block scales at run time)
  with a static global scale from `nvfp4_activation_scales.json`. The scale is
  the FP8 calibration amax / (6 * 448), that is the FP8 scale divided by 6;
  `calibrate_fp8_scales.py` writes both tables (`nvfp4` command: from the
  committed FP8 table). The table also has a Small entry; the build does not
  use it.
- Why only fc1: NVFP4 on q/k/v/o or fc2 was slower and some of those layouts
  returned NaN or wrong depth at batch 8 or with a 3-D activation graph
  (TensorRT 11.1 and 11.4). fc1-only had no NaN in any measured case.

Ship rule, checked per model config: NVFP4 at least 0.8x the FP8 speed on B200
and B300 (TensorRT 11.4), depth error against the PyTorch fp32 model at most
about 2x the FP8 error, no NaN.

Measured on TensorRT 11.4, batch 1, fp16, family build, one job per GPU type
with all builds timed together (median of 9 interleaved rounds of 100 runs).
Speedup in brackets is vs FP8 (below 1.00x is slower than FP8):

| Model | GPU | fp16 ms | FP8 ms | NVFP4 ms | Plan fp16 / FP8 / NVFP4 (MiB) |
|---|---|---|---|---|---|
| V2 Small (not shipped) | B200 | 0.844 | 0.812 | 0.862 (0.94x) | 51.0 / 32.5 / 29.9 |
| V2 Small (not shipped) | B300 | 0.780 | 0.730 | 0.763 (0.96x) | 52.3 / 32.4 / 29.2 |
| V2 Base | B200 | 1.106 | 1.042 | 1.090 (0.96x) | 189.4 / 111.0 / 99.3 |
| V2 Base | B300 | 1.125 | 0.930 | 1.010 (0.92x) | 190.9 / 111.0 / 98.5 |
| V2 Large | B200 | 2.350 | 2.054 | 2.193 (0.94x) | 643.0 / 357.3 / 316.3 |
| V2 Large | B300 | 2.595 | 1.896 | 2.045 (0.93x) | 645.8 / 358.1 / 314.7 |

Depth error against the PyTorch fp32 model (normalized RMSE, mean over 200
held-out ImageNet validation images, B300; B200 within 0.0001). fp32 work
precision in the last column:

| Model | FP8 | NVFP4 | NVFP4 / FP8 | Worst-image correlation FP8 / NVFP4 | NVFP4 / FP8, fp32 |
|---|---|---|---|---|---|
| V2 Small (not shipped) | 0.0284 | 0.0561 | 1.97x | 0.914 / 0.723 | 2.08x |
| V2 Base | 0.0208 | 0.0334 | 1.60x | 0.973 / 0.969 | 1.60x |
| V2 Large | 0.0125 | 0.0196 | 1.56x | 0.980 / 0.941 | 1.73x |

Why Small is not shipped: its speed passes (0.94x and 0.96x of FP8), but its
quality does not. The NVFP4 depth error is 1.97x the FP8 error at fp16 and
2.08x at fp32, at or above the 2x limit, while Base and Large stay at 1.56x to
1.73x. Its worst held-out image drops from 0.914 (FP8) to 0.723 correlation and
the mean delta1 against the fp16 engine from 0.993 to 0.942 (Base: 0.997 to
0.990).

| Criterion (Base and Large) | Result |
|---|---|
| Speed vs FP8 | 0.92x to 0.96x (B200 and B300); NVFP4 is slower than FP8 but within the 0.8x ship rule |
| Speed vs fp16 | 0.98x to 1.27x |
| Plan size vs FP8 | -10% (Base) to -12% (Large) |
| Quality vs FP8 | 1.56x to 1.73x the FP8 depth error |
| NaN | none: 400 real images and 16 stress inputs per engine, every model, B200 and B300, batch 1 (fp16 and fp32) and batch 8 (fp16, a test copy of the graph), TensorRT 11.4; also Small and Base on TensorRT 11.1 |

Why it is slower than FP8: each NVFP4 GEMM needs a separate dynamic-quantize
kernel, and the layer norm before fc1 and the GELU after it no longer fuse into
the neighbouring GEMMs as in the FP8 build. Use NVFP4 for the smaller plan; use
FP8 for speed. TensorRT 11.1 gives the same picture (Base, 1.00x to 1.02x of
FP8). With fp32 work precision a batch-8 test copy of the graph does not build
(no TensorRT tactic); the family builds batch 1 only.

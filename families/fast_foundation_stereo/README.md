# Fast-FoundationStereo

This family builds `nvidia/c-fast-foundationstereo` with the pinned
`NVlabs/Fast-FoundationStereo` source for `stereo_disparity` as two TensorRT
engines: a feature engine (EdgeNeXt-style encoder, 704 x 704 input, fp32
convolution accumulation) and a post engine (cost volume through the family
plugins, 3-D cost aggregation, 8 GRU refinement iterations at 1/4 resolution,
upsampling). The build is fp16, batch 1, at a fixed 704 x 704 input. The family
has no quantized build.

## FP8 quantization study (not shipped)

FP8 self-quantization of the convolutions was built and measured on TensorRT
11.4 (B200 and B300, and a TensorRT 11.1 check on B200). It is not shipped: every
variant is within 0.6% of the fp16 pipeline (0.998x to 1.002x on B200 and B300
with TensorRT 11.4, 0.998x to 1.006x on B200 with 11.1).
This section records the recipe, the method and the numbers, so that the study
does not have to be repeated. The FP8 code was not merged; this family has no
FP8 build.

### Summary and decision

| Item | fp16 (manifest) | Best FP8 variant | FP8 vs fp16 |
|---|---|---|---|
| Feature + post latency, B200 (TensorRT 11.4) | 17.214 ms | 17.175 ms (feature engine in FP8) | 1.002x |
| Feature + post latency, B300 (TensorRT 11.4) | 17.435 ms | 17.408 ms (feature engine in FP8) | 1.002x |
| Plan size, feature + post, B200 | 71.4 MiB | 65.7 to 71.9 MiB | -8% to +0.7% |
| EPE, 7 held-out Middlebury-Q scenes, B200 (PyTorch 0.343 px) | 0.342 px | 0.341 to 0.350 px | -0.001 to +0.008 px |
| Non-finite outputs | 0 | 0 | none |

Decision: do not ship FP8 for Fast-FoundationStereo.

- Speed: the convolutions that TensorRT runs as convolution kernels are about
  12% of the pipeline time (1.7 of 14.0 ms in the layer profile), and only
  part of them can use FP8. The rest of the time is in fused element-wise and
  recurrent kernels (GRU iterations, cost-volume processing) and in the family
  plugins, which FP8 Q/DQ on convolutions does not change.
- Quality is not the reason: the post-engine variants are within 0.002 px EPE
  of the fp16 engine. The feature-engine variant loses 0.008 px.
- Plan size gain is small (at most 5.6 MiB, feature engine only) and comes with
  no speed gain.

### Recipe that was measured

No FP8 or NVFP4 checkpoint of this model exists, and no official quantization
config exists for it. Each quantized convolution is built as
`input -> Quantize -> Dequantize(fp16) -> Convolution(+bias)`, with the kernel
`FP8 constant -> Dequantize(fp16, axis 0)`, in place of the family's fp16
convolution (batch norm folded first, as in the fp16 build).

| Choice | Decision |
|---|---|
| Eligible convolutions | 2-D (and, in one variant, 3-D) convolutions with groups = 1 and input and output channels that are multiples of 16 |
| fp16 | depthwise and grouped convolutions, convolutions with fewer than 16 or non-multiple-of-16 channels, deconvolutions, the convolutions inside the family plugins, GEMMs |
| Weight scale | per output channel, amax / 448 of the batch-norm-folded kernel, at build time |
| Activation scale | static, per tensor, max \|input\| / 448 (the statistic of ModelOpt max calibration), from forward pre-hooks on the official PyTorch model under fp16 autocast |
| Dequantize type | fp16; no non-finite value, fp32 Dequantize was not needed |

Variants:

| Variant | Engine | FP8 convolutions | Not eligible (fp16) |
|---|---|---|---|
| feature 2-D | feature | 11 | 27 |
| post 2-D | post | 7 | 54 |
| post 2-D and 3-D | post | 12 | 49 |

The engine inspector shows FP8 inputs on the quantized layers of each variant.

### Method

- GPUs: B200 and B300, one GPU per job, all variants built and timed in the
  same job; TensorRT 11.4.0.106 (target) and 11.1.0.106 (B200 check); torch
  2.14.0 (CUDA 13).
- Data: Middlebury v3 trainingQ, the 15 scenes of the 700 x 700 profile of
  `tests/prepare_middlebury_q.py`. Calibration: the first 8 scenes by name
  (Adirondack to Pipes). Held out: Playroom, Playtable, PlaytableP, Recycle,
  Shelves, Teddy, Vintage.
- Quality: end-point error (EPE) and bad-2 (error above 2 px) against the
  ground truth on non-occluded pixels, and the mean absolute disparity
  difference against the official PyTorch model (fp16 autocast, 8 iterations).
  Non-finite check on the 7 held-out scenes.
- Speed: feature engine, post engine and both back to back, median of 7
  interleaved rounds of 20 calls (3 warmups), batch 1. The copy between the
  two engines is not timed.

### Results

Speed (ms per call; x is fp16 time over variant time for feature + post):

| GPU | TensorRT | Variant | feature | post | feature + post | x |
|---|---|---|---|---|---|---|
| B200 | 11.4 | fp16 | 4.268 | 12.893 | 17.214 | 1.000 |
| B200 | 11.4 | feature 2-D | 4.253 | 12.885 | 17.175 | 1.002 |
| B200 | 11.4 | post 2-D | 4.270 | 12.921 | 17.246 | 0.998 |
| B200 | 11.4 | post 2-D and 3-D | 4.264 | 12.904 | 17.226 | 0.999 |
| B300 | 11.4 | fp16 | 4.435 | 12.953 | 17.435 | 1.000 |
| B300 | 11.4 | feature 2-D | 4.413 | 12.956 | 17.408 | 1.002 |
| B300 | 11.4 | post 2-D | 4.434 | 12.977 | 17.462 | 0.998 |
| B300 | 11.4 | post 2-D and 3-D | 4.432 | 12.982 | 17.461 | 0.998 |
| B200 | 11.1 | fp16 | 4.337 | 11.824 | 16.201 | 1.000 |
| B200 | 11.1 | feature 2-D | 4.228 | 11.824 | 16.107 | 1.006 |
| B200 | 11.1 | post 2-D | 4.334 | 11.864 | 16.236 | 0.998 |
| B200 | 11.1 | post 2-D and 3-D | 4.335 | 11.830 | 16.219 | 0.999 |

On B200 the fp16 post engine is faster with TensorRT 11.1 than with 11.4
(11.824 against 12.893 ms, separate jobs); FP8 does not change the result on
either version. Quality with TensorRT 11.1 matches 11.4 within 0.004 px EPE.

Plan size (MiB, TensorRT 11.4):

| GPU | Engine | fp16 | FP8 variant |
|---|---|---|---|
| B200 | feature | 35.1 | 29.5 (feature 2-D) |
| B200 | post | 36.3 | 36.8 (post 2-D), 35.4 (post 2-D and 3-D) |
| B300 | feature | 35.3 | 29.5 (feature 2-D) |
| B300 | post | 37.4 | 36.8 (post 2-D), 36.3 (post 2-D and 3-D) |

Quality (7 held-out scenes, non-occluded pixels, TensorRT 11.4, B200; B300
within 0.001 px; PyTorch EPE 0.343 px):

| Engine | EPE (px) | bad-2 | Mean abs difference vs PyTorch (px) | Finite |
|---|---|---|---|---|
| fp16 | 0.342 | 2.24% | 0.011 | yes |
| feature 2-D | 0.350 | 2.52% | 0.075 | yes |
| post 2-D | 0.344 | 2.29% | 0.023 | yes |
| post 2-D and 3-D | 0.341 | 2.22% | 0.033 | yes |

Kernel time per group, B200, TensorRT 11.4, fp16 (layer profiler, ms; the
profiler serializes layers, so the sums differ from the latencies above):

| Engine | Fused element-wise kernels | Convolutions | Plugins | Deconvolution | GEMM | Other |
|---|---|---|---|---|---|---|
| feature | 1.94 | 0.43 | 0.02 | 0.05 | 0.10 | 0.10 (pooling) |
| post | 8.34 | 1.30 | 1.05 | 0.37 | 0.10 | 0.24 (reformat, pooling) |

With FP8, the post 2-D variant moves about 0.2 ms of convolution time into
0.10 ms of FP8 kernels, and the fused element-wise time grows by about the
same amount (8.34 to 8.47 ms); the feature 2-D variant runs 0.32 ms in FP8
kernels and 1.63 ms in the other fused kernels (1.94 ms before). Net: no change.

### NVFP4 (not tried)

TensorRT does not support NVFP4 on convolutions, and the GEMMs are about 1% of
the time.

### What was not tried

| Item | Note |
|---|---|
| FP8 on the plugin-owned convolutions or inside the plugins | needs plugin changes; plugins are about 9% of the post-engine profile |
| Percentile or entropy calibration, more calibration scenes | not measured; could change quality, not speed |
| Non-finite scan on many inputs (only the 7 held-out scenes), H100, native runtime with an FP8 bundle | not measured |

### When to revisit

- A larger input or batch in this family, where convolutions take a larger
  share of the time.
- A TensorRT release that runs the GRU refinement and cost-volume element-wise
  work in FP8, or fuses the Quantize into the producing kernels.

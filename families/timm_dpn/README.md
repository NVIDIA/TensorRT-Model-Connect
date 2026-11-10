# timm DPN

This family builds timm Dual Path Network classifiers (`timm/dpn68b.ra_in1k`,
`timm/dpn92.mx_in1k` in the manifests; dpn68, 98, 107 and 131 build from the
same code path) as one TensorRT engine: 224 x 224 input, batch 1, fp16 or fp32
(the manifests use fp16), builder optimization level 1. The output contains
every class logit in checkpoint order. The family has no quantized build.

## FP8 quantization study (not shipped)

FP8 self-quantization of the convolutions was built and measured on TensorRT
11.4 (and 11.1) for `timm/dpn68b.ra_in1k` and `timm/dpn92.mx_in1k`. It is not
shipped: with every convolution in FP8, dpn68b is 7x slower than fp16, because
TensorRT 11.4 runs each FP8 grouped 3x3 convolution as one convolution per
group. With the 1x1 convolutions only in FP8, dpn68b runs at 0.95x to 0.99x of
fp16, and ImageNet top-1 drops by 2.2 to 3.0 points on both models. This
section records the recipe, the method and the numbers, so that the study does
not have to be repeated. The FP8 code was not merged; this family has no FP8
build.

### Summary and decision

| Item | fp16 (manifest) | FP8, all convolutions | FP8, 1x1 only | FP8 1x1 only vs fp16 |
|---|---|---|---|---|
| dpn68b latency, batch 1, B200 / B300 (TensorRT 11.4) | 0.424 / 0.416 ms | 3.073 / 2.817 ms | 0.436 / 0.422 ms | 0.97x / 0.99x |
| dpn68b latency, batch 32, B200 / B300 (TensorRT 11.4) | 1.109 / 1.036 ms | 7.180 / 6.635 ms | 1.140 / 1.087 ms | 0.97x / 0.95x |
| dpn68b plan size, batch 1, B200 | 25.4 MiB | 17.0 MiB | 16.1 MiB | -37% |
| dpn68b ImageNet top-1, B200 (PyTorch fp32 78.60%) | 78.60% | 75.73% | 76.20% | -2.4 points |
| dpn92 latency, batch 1, B200 / B300 (TensorRT 11.4) | 5.512 / 4.919 ms | - | 4.396 / 3.876 ms | 1.25x / 1.27x |
| dpn92 ImageNet top-1, B200 (PyTorch fp32 80.93%) | 80.87% | - | 77.93% | -2.9 points |
| Non-finite outputs | 0 | 0 | 0 | none |

Decision: do not ship FP8 for timm DPN.

- dpn68b (the premerge model): no speed gain in any layout. With the grouped
  3x3 convolutions in FP8, TensorRT 11.4 lowers each of them (32 groups) into
  one FP8 convolution per group: the plan grows from 131 to 1059 layers (774
  FP8 convolution kernels) and batch 1 is 7x slower. With the 1x1 convolutions
  only, FP8 is 0.95x to 0.99x of fp16 (0.67x to 0.87x on TensorRT 11.1).
- dpn92: the 1x1-only FP8 engine is 1.25x faster at batch 1 only because the
  fp16 baseline is slow (see "fp16 grouped convolutions in dpn92" below), and it
  loses 2.6 to 2.9 top-1 points. At batch 32 on B300 it is at 1.00x.
- Quality: 2.2 to 3.0 top-1 points lost against PyTorch fp32 with per-channel
  weights (up to 3.7 with per-tensor weights), KL divergence 0.09 to 0.13.
  The input of each 1x1 convolution is batch norm + ReLU of the concatenated
  dual-path state, whose channels span a wide range; one static per-tensor
  activation scale per convolution loses the small channels.
- The only gain is plan size.

### Recipe that was measured

No FP8 or NVFP4 checkpoint of any timm DPN exists on the Hub, and no official
quantization config exists for one. Each quantized convolution is built as
`input -> Quantize -> Dequantize(fp16) -> Convolution(+bias)`, with the kernel
`FP8 constant -> Dequantize(fp16, axis 0)`, on the family's own graph.

| Choice | Decision |
|---|---|
| Variant A, all convolutions (92 in dpn68b) | every convolution except the stem: 1x1 `c1x1_a`, projections `c1x1_w_*`, grouped 3x3 `c3x3_b` (32 groups), block outputs `c1x1_c` / `c1x1_c1` / `c1x1_c2` |
| Variant B, 1x1 only (70 in dpn68b, 64 in dpn92) | the 1x1 convolutions; the grouped 3x3 convolutions stay fp16 |
| fp16 | the 7x7 stem (3 input channels), batch norm + ReLU, pooling, the classifier |
| Weight scale | per output channel, amax / 448 (per tensor measured as a variant) |
| Activation scale | static, per tensor; ModelOpt `FP8_DEFAULT_CFG` (max calibration) on the timm PyTorch model, 500 ImageNet validation images |
| Dequantize type | fp16; no non-finite value, fp32 Dequantize was not needed |

### Method

- GPUs: B200 and B300, one GPU per job, all variants of a job built and timed
  together; TensorRT 11.4.0.106 (target) and 11.1 (sanity check, B200).
  Family builder settings (optimization level 1).
- Data: 2000 ImageNet validation images: 500 for calibration, the other 1500
  held out. Family preprocessing.
- Quality: top-1 against the labels, top-1 agreement and KL divergence against
  the timm PyTorch fp32 model on the 1500 held-out images, batch 1.
- Non-finite scan: the 1500 held-out images at batch 1 and at batch 32, all
  variants, B200 and B300.
- Speed: CUDA graph replay, median of 5 interleaved rounds of 200 calls, spread
  at most 0.6%. Batch 32 is a scratch build (the family builds batch 1 only).

### Results

Speed and size, dpn68b (ms per engine call; fp16 and variant B from one job,
fp32 and variant A from another job, whose fp16 is in brackets):

| GPU | TensorRT | Batch | fp32 ms | fp16 ms | FP8 A ms | FP8 B ms | B vs fp16 | plan MiB fp16 / A / B |
|---|---|---|---|---|---|---|---|---|
| B200 | 11.4 | 1 | 0.629 | 0.424 (0.437) | 3.073 | 0.436 | 0.97x | 25.4 / 17.0 / 16.1 |
| B200 | 11.4 | 32 | 2.703 | 1.109 (1.103) | 7.180 | 1.140 | 0.97x | 29.2 / 22.4 / 19.4 |
| B300 | 11.4 | 1 | 0.608 | 0.416 (0.418) | 2.817 | 0.422 | 0.99x | 24.9 / 16.7 / 15.7 |
| B300 | 11.4 | 32 | 2.706 | 1.036 (1.035) | 6.635 | 1.087 | 0.95x | 29.3 / 22.6 / 19.5 |
| B200 | 11.1 | 1 | - | 0.451 | - | 0.519 | 0.87x | 25.4 / - / 16.3 |
| B200 | 11.1 | 32 | - | 1.464 | - | 2.195 | 0.67x | 29.6 / - / 19.9 |

Speed and size, dpn92 (variant B only):

| GPU | TensorRT | Batch | fp16 ms | FP8 B ms | B vs fp16 | plan MiB fp16 / B | layers fp16 / B |
|---|---|---|---|---|---|---|---|
| B200 | 11.4 | 1 | 5.512 | 4.396 | 1.25x | 78.7 / 46.6 | 2515 / 1784 |
| B200 | 11.4 | 32 | 7.926 | 7.020 | 1.13x | 82.1 / 50.0 | 2515 / 1784 |
| B300 | 11.4 | 1 | 4.919 | 3.876 | 1.27x | 77.8 / 46.1 | 2259 / 1656 |
| B300 | 11.4 | 32 | 5.277 | 5.265 | 1.00x | 77.6 / 46.8 | 979 / 1016 |
| B200 | 11.1 | 1 | 5.561 | 4.431 | 1.25x | 78.8 / 46.8 | 2515 / 1811 |
| B200 | 11.1 | 32 | 8.916 | 8.714 | 1.02x | 82.6 / 50.6 | 2515 / 1811 |

TensorRT 11.1 gives the same top-1 as 11.4 within 0.1 point (dpn68b fp16
78.67%, FP8 B 76.20%; dpn92 fp16 80.87%, FP8 B 78.20%, B200).

Layers in the plan, dpn68b, batch 1, TensorRT 11.4 (B200 / B300):

| Engine | Layers | Convolution kernels | FP8 convolution kernels |
|---|---|---|---|
| fp16 | 131 / 130 | 93 | 0 |
| FP8 A (all convolutions) | 1059 / 1058 | 775 | 774 |
| FP8 B (1x1 only) | 153 / 152 | 93 | 70 |

Quality (1500 held-out images, batch 1; B200 / B300):

| Model | Engine | Top-1 | Agreement with PyTorch top-1 | KL mean | KL p99 |
|---|---|---|---|---|---|
| dpn68b | PyTorch fp32 | 78.60% | - | - | - |
| dpn68b | fp16 | 78.60% / 78.53% | 99.6% | 0.00006 | 0.0004 |
| dpn68b | FP8 A, per-channel weights | 75.73% / 76.33% | 86.2% / 86.8% | 0.095 | 0.81 |
| dpn68b | FP8 A, per-tensor weights | 74.87% / 75.53% | 84.4% / 85.1% | 0.107 | 0.81 |
| dpn68b | FP8 B, per-channel weights | 76.20% / 76.07% | 86.6% / 86.6% | 0.091 | 0.74 |
| dpn92 | PyTorch fp32 | 80.93% | - | - | - |
| dpn92 | fp16 | 80.87% / 80.87% | 99.5% / 99.7% | 0.00006 | 0.0009 |
| dpn92 | FP8 B, per-channel weights | 77.93% / 78.27% | 89.5% / 90.3% | 0.133 | 1.5 |

### fp16 grouped convolutions in dpn92 (TensorRT 11.4 behavior)

Independent of FP8: at builder optimization level 1, TensorRT 11.4 splits each
fp16 grouped 3x3 convolution of dpn92 into per-group convolutions with slice
and concatenation kernels (2515 layers on B200, 2259 on B300; 740 group
convolutions, 640 slice and 608 concatenation kernels). Batch 1 is launch bound:
5.51 ms (B200) and 4.92 ms (B300), almost as long as batch 32 (7.93 / 5.28 ms).
dpn68b, also with 32 groups but fewer channels per group, builds 131 layers and
runs batch 1 in 0.42 ms. TensorRT 11.1 builds the same 2515-layer fp16 plan on
B200. The FP8 B speedup on dpn92 comes from this slow baseline, not from FP8
math.

Builder optimization level 3 (proposed in #2123, not merged with this note)
changes the fp16 baseline. Measured on TensorRT 11.4, batch 1, fp16 (separate
job, 500 held-out images):

| Model | GPU | Level 1 ms (layers) | Level 3 ms (layers) | Level 3 vs level 1 |
|---|---|---|---|---|
| dpn68b | B200 | 0.436 (131) | 0.377 (128) | 1.16x |
| dpn68b | B300 | 0.418 (130) | 0.358 (128) | 1.17x |
| dpn92 | B200 | 5.685 (2515) | 3.932 (1232) | 1.45x |
| dpn92 | B300 | 4.881 (2259) | 3.721 (1233) | 1.31x |

At level 3 the fp16 engines are faster than the level-1 FP8 B engines measured
above on dpn68b, and close to them on dpn92 (3.93 vs 4.40 ms on B200, 3.72 vs
3.88 ms on B300; FP8 was not rebuilt at level 3).

### NVFP4 (not tried)

TensorRT does not support NVFP4 on convolutions (block-scaled FP4 needs a 2-D
GEMM), FP8 already loses 2.2 to 3.0 top-1 points, and the engines are small
(plan under 80 MiB).

### What was not tried

| Item | Note |
|---|---|
| Per-channel activation scales, percentile or entropy calibration, quantization-aware training | not measured; could recover accuracy, not speed |
| FP8 at builder optimization level 3 | not measured |
| dpn68, dpn98, dpn107, dpn131, H100, native runtime with an FP8 bundle | not measured |

### When to revisit

- A TensorRT release with an FP8 kernel for grouped convolutions that keeps
  one layer per convolution (variant A).
- A calibration that keeps the small channels of the concatenated dual-path
  state (for example per-channel activation scales), so that the loss drops
  well below 1 top-1 point.
- A TensorRT release that builds the fp16 grouped convolutions of dpn92 as one
  layer each; then compare FP8 with that baseline.

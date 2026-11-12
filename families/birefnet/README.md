# BiRefNet

This family builds BiRefNet (`ZhengPeng7/BiRefNet_lite`) for `segmentation` as
one TensorRT engine: a Swin-T backbone run at full and half resolution, then a
convolutional decoder with deformable-convolution ASPP blocks and gating
convolutions. Input 1024 x 1024, batch 1, fp16 or fp32 (the manifest uses
fp16). The output is one mask logit per pixel. The family has no quantized
build.

## FP8 quantization study (not shipped)

FP8 self-quantization of the Swin backbone linears was built and measured on
TensorRT 11.4 (and 11.1) on B200 and B300. It is not shipped: the engine is
1.01x (B200) to 1.05x (B300) faster than fp16 and the plan is 31% to 35%
smaller, but the mean mask IoU against the fp16 engine drops to 0.985 and the
worst held-out image drops to 0.75 IoU (the fp16 engine is at 0.999 / 0.99
against PyTorch). This section records the recipe, the method and the numbers,
so that the study does not have to be repeated. The FP8 code was not merged;
this family has no FP8 build. All numbers come from a local study.

### Summary and decision

| Item | fp16 (manifest) | FP8 (fp16 engine) | FP8 vs fp16 |
|---|---|---|---|
| Latency, batch 1, B200 / B300 (TensorRT 11.4) | 6.338 / 6.102 ms | 6.262 / 5.838 ms | 1.01x / 1.05x |
| Plan size, B200 / B300 (TensorRT 11.4) | 154.3 / 148.4 MiB | 100.9 / 102.9 MiB | -35% / -31% |
| Mask IoU vs PyTorch fp32, mean / min, B200 | 0.9990 / 0.9920 | 0.9844 / 0.7450 | -1.5 points mean |
| Mask IoU vs PyTorch fp32, mean / min, B300 | 0.9989 / 0.9899 | 0.9845 / 0.7444 | -1.4 points mean |
| Non-finite outputs | 0 | 0 | none |

Decision: do not ship FP8 for BiRefNet.

- Speed: the Swin linears are only about 11% of the engine time. FP8 makes
  them about 13% faster (1.26 to 1.10 ms in the B200 layer profile) and nothing
  else changes. The decoder (convolutions, deformable-convolution sampling,
  fused pointwise kernels) holds most of the time and stays fp16.
- Quality: the FP8 masks differ from fp16 far more than fp16 differs from
  PyTorch, and single images lose up to a quarter of the mask IoU. The
  manifest bar (`min_mask_iou` 0.9) would not hold on every image.
- The plan size gain alone does not justify the quality loss.

### Recipe that was measured

No FP8 or NVFP4 checkpoint of BiRefNet exists on the Hub, and no official
quantization config exists for it. The recipe follows the public ViT and Swin
recipes and the ModelOpt vision example: quantize the transformer linears,
keep convolutions, normalization and attention math in fp16.

| Choice | Decision |
|---|---|
| Quantized (51 projections) | `attn.qkv`, `attn.proj`, `mlp.fc1`, `mlp.fc2` of the 12 Swin blocks and the 3 patch-merging `reduction` layers |
| fp16 | patch-embedding convolution, layer norms, window-attention matmuls and softmax, GELU, all decoder convolutions, the deformable-convolution sampling, the gating convolutions |
| Two backbone passes | the full- and half-resolution passes share weights, so one scale per projection covers both (204 FP8 layers in the plan) |
| Weight scale | per tensor, amax / 448, at build time |
| Activation scale | static, per tensor; ModelOpt 0.46.1 `FP8_DEFAULT_CFG` (max calibration), restricted to the Swin projections, on the checkpoint's own PyTorch code, 128 COCO 2017 validation images at 1024 x 1024 with the checkpoint's preprocessing |
| Outlier rule | a scale above 20x the median of its kind stays fp16; measured: no outlier |
| Dequantize type | fp16 (the GEMM writes fp16 and the bias add fuses into it); no non-finite value, fp32 Dequantize was not needed |
| Builder | the fused dual-GEMM path was turned off for FP8 builds (it failed to compile with scalar-scale FP8 weights) |

### Method

- GPUs: B200 and B300, one GPU per job; TensorRT 11.4.0.106 (target) and
  11.1.0.106. torch 2.14.0 (CUDA 13), ModelOpt 0.46.1.
- Checkpoint: `ZhengPeng7/BiRefNet_lite` at the manifest revision.
- Data: the first 300 validation images of `detection-datasets/coco`; 128
  evenly spread images for calibration, 40 of the others held out for
  evaluation.
- Quality: sigmoid mask at 0.5; mask IoU and mean absolute probability error
  against the fp32 PyTorch model (the checkpoint's own code) and against the
  fp16 engine. No ground-truth masks were used.
- Speed: all variants built and timed in one process on the same GPU; 5
  interleaved rounds of 100 calls (plain enqueue), median.

### Results

Speed and size, batch 1 (x is fp16 time over FP8 time):

| GPU | TensorRT | fp32 ms | fp16 ms | FP8 ms | x | plan MiB fp16 to FP8 |
|---|---|---|---|---|---|---|
| B200 | 11.4 | 9.904 | 6.338 | 6.262 | 1.01 | 154.3 to 100.9 |
| B300 | 11.4 | 9.402 | 6.102 | 5.838 | 1.05 | 148.4 to 102.9 |
| B200 | 11.1 | - | 6.035 | 5.989 | 1.01 | 170.3 to 102.0 |
| B300 | 11.1 | - | 5.732 | 5.623 | 1.02 | 141.2 to 102.3 |

The 11.1 rows come from a separate job per GPU (fp32 not built). In those
jobs the fp16 engine is 5% (B200) to 6% (B300) faster on 11.1 than in the 11.4
jobs; this was not checked on the same node.

Quality, 40 held-out images, TensorRT 11.4:

| GPU | Engine | mask IoU vs PyTorch fp32 (mean / min) | mask IoU vs fp16 engine (mean / min) | probability error vs fp16 engine (mean / max image) |
|---|---|---|---|---|
| B200 | fp16 | 0.9990 / 0.9920 | - | - |
| B200 | fp32 | 0.9998 / 0.9980 | 0.9990 / 0.9935 | 0.00018 / 0.0012 |
| B200 | FP8 | 0.9844 / 0.7450 | 0.9845 / 0.7491 | 0.0027 / 0.0207 |
| B300 | fp16 | 0.9989 / 0.9899 | - | - |
| B300 | fp32 | 0.9998 / 0.9980 | 0.9989 / 0.9916 | 0.00017 / 0.0007 |
| B300 | FP8 | 0.9845 / 0.7444 | 0.9847 / 0.7492 | 0.0027 / 0.0214 |

TensorRT 11.1 gives the same picture: FP8 mask IoU vs PyTorch fp32 0.9843 /
0.7442 (B200) and 0.9846 / 0.7460 (B300), fp16 0.9990 / 0.9909 and 0.9990 /
0.9948; no non-finite value.

Kernel time per group, batch 1, TensorRT 11.4 (layer profiler; the sum is
larger than the engine time because profiling serializes layers), ms:

| Kernel group | B200 fp16 | B200 FP8 | B300 fp16 | B300 FP8 |
|---|---|---|---|---|
| Swin linears | 1.255 | 1.097 | 1.353 | 1.051 |
| window attention | 0.493 | 0.479 | 0.478 | 0.467 |
| convolutions | 0.713 | 0.722 | 0.668 | 0.648 |
| deformable-convolution sampling and gating | 1.074 | 1.076 | 1.020 | 1.020 |
| other fused pointwise and decoder kernels | 7.538 | 7.581 | 7.489 | 7.622 |
| sum | 11.07 | 10.95 | 11.01 | 10.81 |

### NVFP4 (not tried)

The only matmuls are the Swin linears, about 11% of the engine time, so even
a perfect 2x over FP8 on them would give a few percent end to end, with more
quality loss than FP8. TensorRT does not support NVFP4 on convolutions, where
most of the time is.

### What was not tried

| Item | Note |
|---|---|
| FP8 decoder convolutions or deformable-convolution taps (about 80% of the time) | not measured |
| Excluding the most sensitive Swin projections, percentile or entropy calibration | not measured |
| Batch greater than 1 | the family builds batch 1 only |
| `ZhengPeng7/BiRefNet` (Swin-L) | not supported by the family |
| Matting or DIS benchmark metrics with ground-truth masks, H100, native runtime with an FP8 bundle | not measured |

### When to revisit

- An FP8 recipe for the decoder convolutions that TensorRT runs faster than
  fp16 without breaking the fused pointwise kernels: that is where the time is.
- A calibration or a selective recipe that keeps the worst-image mask IoU near
  the fp16 engine (above 0.99); the Swin-only recipe above gains too little to
  justify its loss.

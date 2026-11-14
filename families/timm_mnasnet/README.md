# timm MnasNet

This family builds `timm/mnasnet_100.rmsp_in1k` as one fp16 image
classification engine at batch 1 (task `classification`).

## FP8 quantization study (not shipped)

FP8 self-quantization of the convolutions was built and measured on TensorRT
11.4 (and 11.1) for `timm/mnasnet_100.rmsp_in1k`. It is not shipped: FP8 is
slower than fp16 at batch 1 (0.90x) and at batch 32 and 128 (0.86x to 0.90x),
and ImageNet top-1 drops by 0.2 to 0.7 points. This section records the
recipe, the method and the numbers, so that the study does not have to be
repeated. The FP8 code was not merged; this family has no FP8 build.

### Summary and decision

| Item | fp16 (manifest) | FP8 (fp16 engine) | FP8 vs fp16 |
|---|---|---|---|
| Latency, batch 1, B200 / B300 (TensorRT 11.4) | 0.221 / 0.213 ms | 0.244 / 0.236 ms | 0.90x / 0.90x |
| Latency, batch 32, B200 / B300 (TensorRT 11.4) | 0.526 / 0.503 ms | 0.615 / 0.581 ms | 0.86x / 0.87x |
| Latency, batch 128, B200 / B300 (TensorRT 11.4) | 1.414 / 1.314 ms | 1.576 / 1.508 ms | 0.90x / 0.87x |
| Plan size, batch 1, B200 | 9.4 MiB | 6.9 MiB | -27% |
| ImageNet top-1, B200 / B300 (PyTorch fp32 75.07%) | 75.00% / 75.00% | 74.27% / 74.80% | -0.73 / -0.20 points |
| Non-finite outputs | 0 | 0 | none |

Decision: do not ship FP8 for timm MnasNet.

- Speed: the FP8 1x1 convolutions take about the same kernel time as the fp16
  ones. They are small (16 to 1280 channels) and limited by activation traffic,
  not by math. The Quantize layers that do not fuse into the producing layer
  add 20 layers at batch 1 and 32 layers at batch 32 and 128, and make the FP8
  engine slower.
- The 17 depthwise convolutions stay fp16. On TensorRT 11.4, FP8 depthwise
  (grouped) convolutions are split into one convolution per group (measured on
  timm EfficientNet: 15x slower at batch 1, see that family's README), and in
  PyTorch fake quantization, FP8 on all 52 convolutions (stem and depthwise
  included) drops top-1 to 71.27%.
- Quality: per-tensor 8-bit activations lower top-1 agreement with PyTorch
  from 99.73% to 92.6% on this mobile network. PyTorch fake quantization of the
  same recipe gives the same top-1 and KL as the B200 engine, so the loss comes
  from the recipe, not from TensorRT.
- The open change that builds this family at TensorRT optimization level 3
  (#2123) makes the fp16 baseline faster still: batch 1 0.167 ms (B200) and
  0.158 ms (B300) on TensorRT 11.4, against 0.224 / 0.211 ms at level 1 in the
  same measurement. The FP8 engines above were built at level 1, so the gap to
  that baseline would be larger.

### Recipe that was measured

No FP8 or NVFP4 checkpoint of this model exists on the Hub, and no official
quantization config exists for it. Each quantized convolution is built as
`input -> Quantize -> Dequantize(fp16) -> Convolution(+bias)`, with the kernel
`FP8 constant -> Dequantize(fp16, axis 0)`.

| Choice | Decision |
|---|---|
| Quantized (34 of 52 convolutions) | every pointwise (1x1) convolution on a spatial map larger than 1x1 |
| fp16 | the 3x3 stem (3 input channels), the 17 depthwise convolutions, the classifier |
| Weight scale | per output channel, amax / 448 of the batch-norm-folded kernel, at build time |
| Activation scale | static, per tensor, max \|input\| / 448 over 500 calibration images (the statistic of ModelOpt max calibration), read from an fp32 engine of the family graph |
| Outlier rule | a scale above 20x the median of its kind stays fp16; measured: no outlier |
| Dequantize type | fp16; no non-finite value, fp32 Dequantize was not needed |

### Method

- GPUs: B200 and B300, one GPU per job, all variants built and timed in the
  same job; TensorRT 11.4.0.106 (target) and 11.1.0.106 (B200).
- Data: 2000 ImageNet validation images from `clip-benchmark/wds_imagenet1k`
  (every 25th image of each shard, shuffled with seed 0): the first 500 for
  calibration, the other 1500 held out. Family preprocessing.
- Quality: top-1 against the labels, top-1 agreement and KL divergence against
  the timm PyTorch fp32 model on the 1500 held-out images, batch 1.
- Non-finite scan: the 2000 images and 29 stress inputs (noise with standard
  deviation 1, 3 and 10, constants), all variants, B200 and B300.
- Speed: CUDA graph replay, median of 7 interleaved rounds of 200 calls, spread
  at most 0.6%. Batch 32 and 128 are scratch builds (the family builds batch 1
  only). Builder optimization level 1 (the family setting on main).

### Results

Speed and size (ms per engine call; x is fp16 time over FP8 time):

| GPU | TensorRT | Batch | fp32 ms | fp16 ms | FP8 ms | x | plan MiB fp16 to FP8 | layers fp16 to FP8 |
|---|---|---|---|---|---|---|---|---|
| B200 | 11.4 | 1 | 0.252 | 0.221 | 0.244 | 0.90 | 9.4 to 6.9 | 82 to 102 |
| B200 | 11.4 | 32 | 0.987 | 0.526 | 0.615 | 0.86 | 11.9 to 10.5 | 58 to 90 |
| B200 | 11.4 | 128 | 3.283 | 1.414 | 1.576 | 0.90 | 21.1 to 22.5 | 58 to 90 |
| B300 | 11.4 | 1 | 0.242 | 0.213 | 0.236 | 0.90 | 9.0 to 6.4 | 84 to 102 |
| B300 | 11.4 | 32 | 0.950 | 0.503 | 0.581 | 0.87 | 12.0 to 10.6 | 58 to 90 |
| B300 | 11.4 | 128 | 3.019 | 1.314 | 1.508 | 0.87 | 21.3 to 22.5 | 58 to 90 |
| B200 | 11.1 | 1 | - | 0.221 | 0.250 | 0.88 | 9.4 to 6.9 | 84 to 102 |

Quality (1500 held-out images, batch 1):

| Engine | Top-1 | Agreement with PyTorch top-1 | KL mean | KL p99 |
|---|---|---|---|---|
| PyTorch fp32 | 75.07% | - | - | - |
| fp16 engine (B200) | 75.00% | 99.73% | < 0.0001 | < 0.001 |
| FP8 engine (B200) | 74.27% | 92.60% | 0.0354 | 0.265 |
| FP8 engine (B300) | 74.80% | 92.80% | 0.0356 | 0.257 |
| FP8 engine (B200, TensorRT 11.1) | 74.20% | 92.60% | 0.0355 | 0.261 |
| PyTorch fake quantization of the same recipe | 74.27% | 92.40% | 0.0354 | 0.254 |

Kernel time per group, B200, TensorRT 11.4 (layer profiler, ms; the profiler
adds time per layer, so the sums are larger than the latencies above):

| Kernel group | fp16 batch 1 | FP8 batch 1 | fp16 batch 128 | FP8 batch 128 |
|---|---|---|---|---|
| convolutions | 0.427 | 0.143 fp16 + 0.293 FP8 = 0.436 | 1.643 | 0.884 fp16 + 0.703 FP8 = 1.587 |
| other (pointwise, Quantize, pooling) | 0.190 | 0.323 | 0.076 | 0.450 |
| classifier | 0.009 | 0.009 | 0.011 | 0.011 |
| layers in the plan | 82 | 102 | 58 | 90 |

Variants (TensorRT 11.4, B200; x is fp16 time over variant time):

| Variant | x batch 1 / 32 / 128 | plan MiB batch 1 | Top-1 | KL mean |
|---|---|---|---|---|
| recipe above | 0.90 / 0.86 / 0.90 | 6.9 | 74.27% | 0.0354 |
| + 3x3 stem in FP8 | 0.88 / 0.75 / 0.74 | 7.0 (89.9 at batch 128) | 75.27% | 0.0557 |
| per-tensor weight scale | 0.89 / 0.85 / 0.90 | 6.9 | 73.73% | 0.0544 |
| batch norm kept as a scale layer | 0.89 / 0.85 / 0.90 | 6.9 | 74.40% | 0.0356 |
| + stem and depthwise convolutions in FP8 (PyTorch fake quantization only) | - | - | 71.27% | 0.167 |

### NVFP4 (not tried)

TensorRT does not support NVFP4 on convolutions (block-scaled FP4 needs a 2-D
GEMM), FP8 already loses accuracy, and the engine is not limited by weight
reads (plan under 12 MiB at batch 1).

### What was not tried

| Item | Note |
|---|---|
| Percentile or entropy calibration, quantization-aware training | not measured; could recover accuracy, not speed |
| FP8 at builder optimization level 3, H100, native runtime with an FP8 bundle | not measured |

### When to revisit

- A TensorRT release with FP8 1x1 convolution kernels that are faster than
  fp16 for small channel counts, and that fuses each Quantize into the
  producing layer.
- A TensorRT FP8 kernel for depthwise (grouped) convolutions that keeps one
  layer per convolution.

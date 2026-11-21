# timm Inception-ResNet

This family builds the timm Inception-ResNet-v2 classifier
(`timm/inception_resnet_v2.tf_in1k`) for `image_to_class_scores` as one
TensorRT engine: 299 x 299 input, batch 1, fp16 or fp32 (the manifest uses
fp16), builder optimization level 3. The batch norms are folded into the
convolutions at build time. The family has no quantized build.

## FP8 quantization study (not shipped)

FP8 self-quantization of the convolutions was built and measured on TensorRT
11.4 (and 11.1) for `timm/inception_resnet_v2.tf_in1k`. It is not shipped: the
recipe that keeps the fp16 accuracy is not reliably faster than the fp16 engine
(1.01x on B200, 1.07x on B300 at batch 1, 0.95x to 0.96x at batch 32); only the
plan is 43% smaller. The variants that are faster (1.12x to 1.21x) round the
residual stream to FP8 and have 2x to 4x the output error. This section records
the recipe, the method and the numbers, so that the study does not have to be
repeated. The FP8 code was not merged; this family has no FP8 build.

### Summary and decision

All numbers in this table are from one TensorRT 11.4 job per GPU at builder
optimization level 3 (the family's level), batch 1 unless noted.

| Item | fp16 (manifest) | FP8 (fp16 engine) | FP8 vs fp16 |
|---|---|---|---|
| Latency, batch 1, B200 / B300 | 0.905 / 0.866 ms | 0.898 / 0.807 ms | 1.01x / 1.07x |
| Latency, batch 32, B200 / B300 | 2.324 / 2.210 ms | 2.428 / 2.329 ms | 0.96x / 0.95x |
| Plan size, batch 1 | 111.9 MiB | 64.3 MiB | -43% |
| ImageNet top-1, B200 / B300 (PyTorch fp32 80.80%) | 80.73% / 80.73% | 80.80% / 80.53% | -0.2 to +0.1 point |
| Top-1 agreement with PyTorch fp32 | 99.9% | 96.7% | -3.2 points |
| Non-finite outputs | 0 | 0 | none |

Decision: do not ship FP8 for timm Inception-ResNet.

- Speed: the FP8 convolution kernels are only about 6% faster than the fp16
  ones at batch 1, and the residual blocks add Quantize kernels that take the
  gain back (see "Why there is no gain"). The result moves between 0.98x and
  1.07x from job to job and GPU to GPU, and batch 32 is slower in FP8.
- Size: the plan is 43% smaller (111.9 to 64.3 MiB). A size gain without a
  speed gain is not a reason to ship.
- The faster variants put the residual stream in FP8. They are 1.12x to 1.21x
  faster, but the KL divergence to PyTorch grows from 0.008 to 0.016 to 0.033,
  2x to 4x the error of the FP8 recipes shipped for Inception-v3 and
  Inception-v4 (0.008 to 0.009), and top-1 agreement drops to 94% to 96%.

### Recipe that was measured

No FP8 or NVFP4 checkpoint of this model exists on the Hub (only fp32 ONNX
exports), and no official quantization config exists for it. The recipe is the
one shipped for `timm_inception` and `timm_inception_v4`. Each quantized
convolution is built as `input -> Quantize -> Dequantize(fp16) ->
Convolution(+bias)`, with the kernel `FP8 constant -> Dequantize(fp16, axis 0)`.

| Choice | Decision |
|---|---|
| Quantized | 238 of 244 convolutions: every convolution of `mixed_5b`, the 40 residual blocks (`repeat`, `repeat_1`, `repeat_2`, `block8`), `mixed_6a`, `mixed_7a` and `conv2d_7b`, including the 1x1 projection (with bias) of each residual block |
| fp16 | the stem (`conv2d_1a` to `conv2d_4a`, five convolutions), the classifier (1536 x 1000) and every pointwise layer |
| Weight scale | per output channel, amax / 448 of the batch-norm-folded kernel, at build time |
| Activation scale | static, per tensor; ModelOpt `FP8_DEFAULT_CFG` (max calibration) on the timm PyTorch model, 500 ImageNet validation images |
| Shared Quantize | one Quantize per tensor; the pooling branches of `mixed_5b`, `mixed_6a` and `mixed_7a` read the dequantized FP8 input |
| Residual add | reads the fp16 block input (the "residual in FP8" variants read the FP8 copy instead) |
| Outlier rule | an input scale above 20x the median stays fp16; measured: no outlier (median amax 24.3, largest 103.6) |
| Dequantize type | fp16; no non-finite value, fp32 Dequantize was not needed |

### Method

- GPUs: B200 and B300, one GPU per job; all variants of a table were built and
  timed in the same job. TensorRT 11.4.0.106 (target) and 11.1.0.106.
- Engines: the family's own build function, with the convolutions replaced by
  the Q/DQ form above from outside the repository.
- Quality: 2000 ImageNet validation images (`clip-benchmark/wds_imagenet1k`,
  every 25th image); 500 for calibration, 1500 others held out. Reference: timm
  PyTorch fp32 with the checkpoint's preprocessing (bicubic, crop 0.8975).
  Agreement is the share of held-out images with the same top-1 class as
  PyTorch; KL is the mean KL divergence of the softmax to PyTorch.
- Non-finite scan: the 1500 held-out images for every engine; the FP8 engines
  also over the 500 calibration images and 29 stress inputs (noise, constants).
- Speed: CUDA graph replay, median of 7 interleaved rounds of 100 runs. Batch
  32 is a separate study build (the family builds batch 1 only).

### Results

Speed, size and quality at batch 1, builder optimization level 3, TensorRT
11.4 (x is fp16 time over variant time; B300 in brackets where it differs):

| Variant | B200 ms (x) | B300 ms (x) | layers | plan MiB | top-1 B200 / B300 | agreement | KL mean (p99) |
|---|---|---|---|---|---|---|---|
| fp16 (manifest) | 0.905 | 0.866 | 241 | 111.9 | 80.73% / 80.73% | 99.9% | 0.00001 (0.0002) |
| fp16, optimization level 1 | 1.026 (0.88) | 0.997 (0.87) | 254 | 108.1 (107.3) | 80.73% / 80.80% | 99.9% | 0.00001 to 0.00002 (0.0002) |
| FP8, recipe above | 0.898 (1.01) | 0.807 (1.07) | 291 | 64.3 | 80.80% / 80.53% | 96.7% | 0.008 (0.11 to 0.12) |
| FP8, residual in FP8 in blocks 10 to 39 | 0.807 (1.12) | 0.757 (1.14) | 260 | 65.0 (64.9) | 81.27% / 80.47% | 96.1% (95.8%) | 0.016 to 0.017 (0.22) |
| FP8, residual in FP8 in all 40 blocks | 0.779 (1.16) | 0.716 (1.21) | 248 | 63.1 (64.1) | 80.27% / 80.73% | 94.1% (94.7%) | 0.031 to 0.033 (0.43 to 0.51) |

Blocks 10 to 39 are the 20 Block17 (`repeat_1`) and the 10 Block8 (`repeat_2`,
`block8`) blocks. Batch 32: fp16 2.324 / 2.210 ms, FP8 recipe 2.428 / 2.329 ms
(0.96x / 0.95x), top-1 80.80% and 80.93%.

TensorRT 11.1, B200, optimization level 3: fp16 0.853 ms, FP8 recipe 0.876 ms
(0.97x), plan 111.7 to 64.6 MiB, top-1 80.73% and 80.53%, agreement 96.9%.

Earlier jobs at optimization level 1 (TensorRT 11.4, three jobs per GPU) gave
the same picture: FP8 recipe 0.98x to 1.02x (B200) and 1.02x to 1.03x (B300)
at batch 1, 0.97x at batch 32; residual in FP8 in all blocks 1.15x / 1.17x.
With the stem in FP8 too (every convolution but the first), FP8 was 1.02x /
1.01x and the KL grew to 0.028 to 0.029 (top-1 79.87% to 79.93%), so the stem
stays fp16.

### Why there is no gain

Kernel time per group, batch 1, B300, TensorRT 11.4, optimization level 1
(layer profiler, ms (layers); level 3 fuses convolutions with pointwise layers,
so its split is not by kernel type):

| Kernel group | fp16 | FP8 recipe | FP8, residual in FP8 |
|---|---|---|---|
| convolutions | 1.894 (190) | 1.716 FP8 + 0.063 fp16 (185 + 6) | 1.692 FP8 + 0.063 fp16 (184 + 6) |
| other (Quantize, add, concat, pooling, classifier) | 0.462 (64) | 0.766 (113) | 0.452 (66) |
| sum | 2.356 | 2.545 | 2.206 |

At level 3 the profiled sums are 2.188 (fp16), 2.419 (FP8 recipe) and 2.096
(residual in FP8) ms on B200, and 2.147, 2.395 and 2.060 ms on B300.

- The FP8 convolutions save about 0.12 ms of profiled time.
- Each of the 40 residual blocks writes its output once and needs it twice: in
  fp16 for the residual add of the next block and in FP8 for the convolutions
  of the next block. TensorRT does not write both from the add kernel; it adds
  one Quantize kernel per block: the plan grows from 241 to 291 layers (254 to
  304 at level 1), and the extra kernels take the convolution gain back.
- Feeding the residual add from the FP8 copy removes those kernels (248
  layers) but rounds the residual stream to e4m3 in every block, and the error
  adds up over the blocks.
- PyTorch fake quantization of the convolutions with ModelOpt (stem
  included) gives KL 0.028 to 0.030, the same as the engine with the stem in
  FP8 (0.028 to 0.029), so the error comes from the recipe, not from
  TensorRT.

### NVFP4 (not tried)

TensorRT does not support NVFP4 on convolutions (4-D tensors), and the only
matmul (the 1536 x 1000 classifier) holds about 2.8% of the weights.

### What was not tried

| Item | Note |
|---|---|
| Percentile or entropy calibration, quantization-aware training | not measured |
| A residual add that writes both fp16 and FP8 in one kernel | needs TensorRT support |
| H100, native runtime with an FP8 bundle | not measured |

### When to revisit

- A TensorRT release that fuses the Quantize of a residual block output into
  the residual add while keeping the fp16 output: the FP8 recipe would then
  run at about the speed of the residual-in-FP8 variants (1.12x to 1.21x) with
  KL 0.008.
- A decision to accept a 2x to 4x higher output error (agreement 94% to 96%,
  top-1 within 0.5 point of fp16) for 1.12x to 1.21x at batch 1.

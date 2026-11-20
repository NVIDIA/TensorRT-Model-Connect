# timm Res2Net

This family builds timm Res2Net and Res2NeXt image classifiers
(`timm/res2net50_26w_4s.in1k`, `timm/res2net50d.in1k`, `timm/res2next50.in1k`
in the test manifests) as one TensorRT engine per checkpoint: batch 1, fp16 in
the manifests (fp32 is also supported). The network layout (stage depths,
scale, chunk width, groups, stem) is read from the checkpoint; see
`model.py`.

## FP8 quantization study (not shipped)

FP8 self-quantization of the convolutions was built and measured on TensorRT
11.4 (and 11.1) for `timm/res2net50_26w_4s.in1k`. It is not shipped: with the
family's builder optimization level 1, FP8 is slower than fp16 at batch 1
(0.92x to 0.97x) and at batch 32 (0.93x to 0.97x). With builder optimization
level 3, FP8 is 0.98x to 0.99x of fp16 on B200 and 0.99x to 1.01x on B300, so
it is still not faster on both GPUs. This section records the recipe, the
method and the numbers, so that the study does not have to be repeated. The
FP8 code was not merged; this family has no FP8 build.

### Summary and decision

Family builder optimization level 1, TensorRT 11.4, recipe "all" (see
below):

| Item | fp16 (manifest) | FP8 (fp16 engine) | FP8 vs fp16 |
|---|---|---|---|
| Latency, batch 1, B200 / B300 | 0.517 / 0.493 ms | 0.560 / 0.517 ms | 0.92x / 0.95x |
| Latency, batch 32, B200 / B300 | 1.401 / 1.317 ms | 1.489 / 1.411 ms | 0.94x / 0.93x |
| Plan size, batch 1 | 50.6 MiB | 29.1 MiB | -42% |
| ImageNet top-1, B200 (PyTorch fp32 79.27%) | 79.27% | 79.00% | -0.27 points |
| Non-finite outputs | 0 | 0 | none |

Decision: do not ship FP8 for timm Res2Net. A smaller plan with no speed gain
is not a reason to ship.

- Speed: the FP8 convolutions take less kernel time than the fp16 ones (0.42
  vs 0.46 ms per call at batch 1 on B300, convolution and GEMM kernels), but
  the Quantize work around the Res2Net chunk chain adds more (other kernels
  0.31 vs 0.21 ms, 228 vs 184 kernels). Each block splits the 1x1 output into
  4 chunks, adds each chunk to the previous result before its own 3x3, and
  concatenates the chunks again; every one of these 3x3 convolutions needs its
  own quantized input, so the Quantize cannot fuse into one producer.
- The chunks are small: 26, 52, 104 and 208 channels in the 4 stages (only 208
  is a multiple of 16), so each FP8 3x3 convolution is short and the FP8 math
  gain per convolution is small.
- Quality: FP8 "all" costs 0.3 to 0.5 top-1 points, and top-1 agreement with
  PyTorch drops from 99.8% to 96%. Keeping stage `layer1` in fp16 (the shipped
  `timm_resnet` layout) recovers the accuracy, but it is not faster either.

### Recipe that was measured

No FP8 or NVFP4 checkpoint of this model exists on the Hub, and no official
quantization config exists for it. Each quantized convolution is built as
`input -> Quantize -> Dequantize(fp16) -> Convolution(+bias)`, with the kernel
`FP8 constant -> Dequantize(fp16, axis 0)`. This is the recipe of the shipped
`timm_resnet` FP8 build.

| Choice | Decision |
|---|---|
| Quantized ("all", 84 of 85 convolutions) | every convolution after the stem: the 1x1 `conv1`, the three 3x3 `convs.N` of each block, the 1x1 `conv3`, the downsample convolutions |
| Variant "no layer1" (68 of 85) | as "all", with the 16 convolutions of stage `layer1` in fp16 |
| fp16 | the 7x7 stem (3 input channels), the classifier |
| Weight scale | per output channel, amax / 448 of the batch-norm-folded kernel, at build time |
| Activation scale | static, per tensor, max \|input\| / 448 over 500 calibration images (the statistic of ModelOpt max calibration), from the timm PyTorch model |
| Dequantize type | fp16; no non-finite value, fp32 Dequantize was not needed |

This family has no squeeze-excite or attention-gate convolutions, so the
`timm_resnet` rule that keeps those in fp16 changes nothing here.

### Method

- GPUs: B200 and B300, one GPU per job, all variants built and timed in the
  same job; TensorRT 11.4.0.106 (target) and 11.1.0.106 (B200 sanity check).
- Builds: the family's own graph code with the FP8 convolution wrapped in,
  family builder settings (optimization level 1) unless the table says level 3.
  Batch 32 is a scratch build (the family builds batch 1 only).
- Data: 2000 ImageNet validation images from `clip-benchmark/wds_imagenet1k`,
  taken at a fixed stride of 25 (the set is sorted by class): every 4th of
  them (500) for calibration, the other 1500 held out. The checkpoint's own
  timm preprocessing.
- Revisions: batch 1 rows on upstream main at `eabf8250`; batch 32 rows and
  the kernel table from an earlier run on upstream main at `a17640d2` (no
  change to this family or the builder between the two).
- Quality: top-1 against the labels, top-1 agreement and KL divergence against
  the timm PyTorch fp32 model on the 1500 held-out images.
- Non-finite scan: the 2000 images and 29 stress inputs (noise with standard
  deviation 1, 3 and 10, constants), all variants.
- Speed: CUDA graph replay, median of 9 interleaved rounds of 100 calls,
  spread at most 1.3% (one FP8 row 3.4%).

### Results

Speed and size, family builder optimization level 1 (ms per engine call; x is
fp16 time over FP8 time):

| GPU | TensorRT | Batch | fp16 ms | FP8 "all" ms (x) | FP8 "no layer1" ms (x) | plan MiB fp16 / "all" |
|---|---|---|---|---|---|---|
| B200 | 11.4 | 1 | 0.517 | 0.560 (0.92) | 0.549 (0.94) | 50.6 / 29.1 |
| B300 | 11.4 | 1 | 0.493 | 0.517 (0.95) | 0.509 (0.97) | 50.7 / 29.4 |
| B200 | 11.4 | 32 | 1.401 | 1.489 (0.94) | 1.449 (0.97) | 55.4 / 35.0 |
| B300 | 11.4 | 32 | 1.317 | 1.411 (0.93) | 1.355 (0.97) | 55.5 / 35.1 |
| B200 | 11.1 | 1 | 0.517 | 0.557 (0.93) | 0.555 (0.93) | 50.6 / 29.2 |

Builder optimization level 3 (batch 1, TensorRT 11.4; x is level-3 fp16 time
over the variant time):

| GPU | fp16 level 1 ms | fp16 level 3 ms | FP8 "all" level 3 ms (x) | FP8 "no layer1" level 3 ms (x) | build s fp16 / FP8, level 3 |
|---|---|---|---|---|---|
| B200 | 0.517 | 0.486 | 0.498 (0.98) | 0.491 (0.99) | 97 / 134 |
| B300 | 0.493 | 0.459 | 0.462 (0.99) | 0.454 (1.01) | 94 / 135 |

The family builds at level 1 today; #2123 proposes level 3. Level 3 makes the
fp16 engine 1.06x (B200) and 1.07x (B300) faster, so it moves the fp16
baseline further from the level-1 FP8 numbers above, and FP8 built at level 3
only matches it. At level 3 the FP8 plan is 32.4 MiB instead of 29.1 MiB, and
the build takes 134 s instead of 27 s.

Quality (1500 held-out images, batch 1, TensorRT 11.4):

| Engine | Top-1 B200 / B300 | Agreement with PyTorch top-1 | KL mean |
|---|---|---|---|
| PyTorch fp32 | 79.27% | - | - |
| fp16 engine, level 1 | 79.27% / 79.33% | 99.80% to 99.93% | < 0.0001 |
| fp16 engine, level 3 | 79.27% / 79.27% | 99.87% | < 0.0001 |
| FP8 "all", level 1 | 79.00% / 78.80% | 96.00% to 96.47% | 0.0176 to 0.0178 |
| FP8 "all", level 3 | 78.73% / 78.73% | 96.07% | 0.0174 |
| FP8 "no layer1", level 1 | 79.33% / 79.27% | 97.87% to 98.00% | 0.0046 to 0.0049 |
| FP8 "no layer1", level 3 | 79.27% / 79.33% | 97.47% to 97.87% | 0.0047 to 0.0050 |
| PyTorch fake quantization, "all" | 78.93% | 95.60% | 0.0177 |
| PyTorch fake quantization, "no layer1" | 79.40% | 97.73% | 0.0050 |

TensorRT 11.1 (B200, level 1) gives the same quality: fp16 79.27%, FP8 "all"
79.00% (KL 0.0178), FP8 "no layer1" 79.40% (KL 0.0050). No non-finite output
in any engine. Other calibration statistics in PyTorch fake quantization are
worse than max: percentile 99.99 gives KL 0.0239, percentile 99.9 gives KL
0.0586; FP8 weights alone give KL 0.0054.

Kernel time per group, batch 1, TensorRT 11.4, level 1 (profiler, ms; the
profiler adds time per kernel, so the sums are larger than the latencies
above):

| Kernel group | fp16 B200 | FP8 "all" B200 | fp16 B300 | FP8 "all" B300 |
|---|---|---|---|---|
| convolution and GEMM | 0.470 | 0.424 | 0.457 | 0.416 |
| other (slice, add, concat, Quantize, pooling) | 0.231 | 0.334 | 0.213 | 0.310 |
| sum | 0.701 | 0.758 | 0.671 | 0.726 |
| kernels | 184 | 227 | 184 | 228 |

### NVFP4 (not tried)

TensorRT does not support NVFP4 on convolutions (block-scaled FP4 needs a 2-D
GEMM); the only GEMM in this network is the classifier.

### What was not tried

| Item | Note |
|---|---|
| Entropy calibration, quantization-aware training | not measured; could recover accuracy, not speed |
| Batch 32 at builder optimization level 3 | not measured |
| `res2net50d` and `res2next50`, H100, native runtime with an FP8 bundle | not measured |

### When to revisit

- A TensorRT release that fuses the Quantize into the slice, add and concat
  kernels of the Res2Net chunk chain, so that FP8 does not add kernels.
- Builder optimization level 3 or higher as the family default together with
  a TensorRT release where FP8 at that level is faster than fp16 on both B200
  and B300 (at level 3 on TensorRT 11.4 it is 0.98x to 1.01x).
- A wider Res2Net (larger chunks), where the FP8 convolution time would be a
  larger share of the call.

# timm SENet

This family builds the timm SENet-154 image classifier
(`timm/senet154.gluon_in1k` in the test manifest) as one TensorRT engine:
batch 1, 224 x 224, fp16 in the manifest (fp32 is also supported), builder
optimization level 3. The network layout (stem, stage depths, grouped 3x3
convolutions, squeeze-excitation gates, 3x3 projection shortcuts) is read from
the checkpoint; see `model.py`.

## FP8 quantization study (not shipped)

FP8 self-quantization of the convolutions was built and measured on TensorRT
11.4 (and 11.1) for `timm/senet154.gluon_in1k`. It is not shipped: FP8 is
slower than fp16 in every layout, at batch 1 and at batch 32, on B200 and
B300 (0.53x to 0.79x). Quality is fine (top-1 within 0.2 points of PyTorch
fp32); speed is the blocker. This section records the recipe, the method and
the numbers, so that the study does not have to be repeated. The FP8 code was
not merged; this family has no FP8 build.

### Summary and decision

Builder optimization level 3 (the family default), batch 1, TensorRT 11.4,
best FP8 layout ("no SE, no layer1", see below):

| Item | fp16 (manifest) | FP8 (fp16 engine) | FP8 vs fp16 |
|---|---|---|---|
| Latency, batch 1, B200 / B300 | 11.40 / 10.84 ms | 14.75 / 13.66 ms | 0.77x / 0.79x |
| Plan size, batch 1, B200 / B300 | 232.7 / 229.7 MiB | 234.0 / 232.2 MiB | +1% |
| Build time, B200 / B300 | 293 / 220 s | 955 / 860 s | 3.3x / 3.9x |
| ImageNet top-1, B200 / B300 (PyTorch fp32 82.67%) | 82.67% / 82.67% | 82.60% / 82.80% | -0.07 / +0.13 points |
| Non-finite outputs | 0 | 0 | none |

Decision: do not ship FP8 for timm SENet.

- Speed: the fp16 engine is already limited by kernel launches. TensorRT 11.4
  runs each of the 50 grouped 3x3 convolutions (64 groups) as one convolution
  kernel per group: 3343 to 3859 kernels per image for 257 convolutions. In
  FP8, TensorRT keeps the per-group split and adds one concatenation kernel
  per group (2240 + 448 + 192 calls), so the FP8 engine runs about 6200 to
  6630 kernels per image and is slower in every layout.
- At builder optimization level 3, TensorRT runs most of the FP8 grouped
  convolutions with fp16 kernels (B300: 12.6 ms of fp16 convolution kernels,
  0.9 ms of FP8), so the FP8 plan is not smaller than the fp16 plan (237 vs
  230 MiB on B300) and the build takes 3.3 to 4.4 times longer. At level 1 the
  FP8 plan is 43% smaller (130 vs 228 MiB), but FP8 is still 0.53x to 0.79x
  of fp16.
- Quality is not the problem: FP8 keeps top-1 within 0.2 points of PyTorch
  fp32 (KL divergence 0.003 to 0.006).

### Recipe that was measured

No FP8 or NVFP4 checkpoint of this model exists on the Hub, and no official
quantization config exists for it. Each quantized convolution is built as
`input -> Quantize -> Dequantize(fp16) -> Convolution(+bias)`, with the kernel
`FP8 constant -> Dequantize(fp16, axis 0)`. This is the recipe of the shipped
`timm_resnet` FP8 build.

| Choice | Decision |
|---|---|
| "all" (254 of 257 convolutions) | every convolution after the stem: 1x1 `conv1`, grouped 3x3 `conv2` (64 groups), 1x1 `conv3`, the squeeze-excitation `se.fc1` / `se.fc2`, the downsample convolutions |
| "no SE" (154 of 257) | as "all", with the 100 squeeze-excitation convolutions (on a 1x1 map) in fp16 |
| "no SE, no layer1" (144 of 257) | as "no SE", with the 10 convolutions of stage `layer1` also in fp16 (the shipped `timm_resnet` layout) |
| fp16 | the three 3x3 stem convolutions (3 input channels), the classifier |
| Weight scale | per output channel, amax / 448 of the batch-norm-folded kernel, at build time |
| Activation scale | static, per tensor, max \|input\| / 448 over 500 calibration images (the statistic of ModelOpt max calibration), from the timm PyTorch model |
| Dequantize type | fp16; no non-finite value, fp32 Dequantize was not needed |

### Method

- GPUs: B200 and B300, one GPU per job, all variants of a job built and timed
  together; TensorRT 11.4.0.106 (target) and 11.1.0.106 (B200 sanity check).
- Builds: the family's own graph code with the FP8 convolution wrapped in.
  Level-3 table, TensorRT 11.1 numbers and the 2307-call count below:
  upstream main at `d39e1ec9` (family default level 3, from #2123; level 1
  set by the script for the "fp16 level 1" column). Level-1 table and the
  level-1 rows of the kernel table: an earlier run on upstream main at
  `a17640d2`, when the family built at level 1.
  Batch 32 is a scratch build (the family builds batch 1 only).
- Data: 2000 ImageNet validation images from `clip-benchmark/wds_imagenet1k`,
  taken at a fixed stride of 25 (the set is sorted by class): every 4th of
  them (500) for calibration, the other 1500 held out. The checkpoint's own
  timm preprocessing.
- Quality: top-1 against the labels, top-1 agreement and KL divergence against
  the timm PyTorch fp32 model on the 1500 held-out images.
- Non-finite scan: the 2000 images and 29 stress inputs (noise with standard
  deviation 1, 3 and 10, constants), all variants.
- Speed: CUDA graph replay, median of 9 interleaved rounds of 100 calls,
  spread at most 0.2%.

### Results

Builder optimization level 3, batch 1, TensorRT 11.4 (upstream main at
`d39e1ec9`; x is level-3 fp16 time over the variant time):

| GPU | fp16 level 3 ms | fp16 level 1 ms | FP8 "all" ms (x) | FP8 "no SE, no layer1" ms (x) | plan MiB fp16 / "all" / "no SE, no layer1" | build s fp16 / "all" |
|---|---|---|---|---|---|---|
| B200 | 11.40 | 12.63 | 15.56 (0.73) | 14.75 (0.77) | 232.7 / 238.0 / 234.0 | 293 / 1058 |
| B300 | 10.84 | 10.87 | 14.52 (0.75) | 13.66 (0.79) | 229.7 / 237.4 / 232.2 | 220 / 976 |

TensorRT 11.1, B200, level 3, batch 1: fp16 10.86 ms, FP8 "no SE, no layer1"
14.18 ms (0.77x), plan 232.3 / 231.5 MiB.

The level-3 fp16 speed varies between builds: the builder optimization level
study for #2123 measured 10.65 ms (B200, 1.17x over level 1) and 10.32 ms
(B300, 1.07x); the job above measured 11.40 ms (1.11x) and 10.84 ms (1.00x).

Builder optimization level 1, TensorRT 11.4 (upstream main at `a17640d2`; x is
fp16 time over the variant time):

| GPU | Batch | fp16 ms | FP8 "all" ms (x) | FP8 "no SE" ms (x) | FP8 "no SE, no layer1" ms (x) | plan MiB fp16 / "all" |
|---|---|---|---|---|---|---|
| B200 | 1 | 12.53 | 17.02 (0.74) | 16.84 (0.74) | 15.84 (0.79) | 227.8 / 129.9 |
| B300 | 1 | 10.81 | 16.25 (0.67) | 15.96 (0.68) | 15.13 (0.71) | 225.6 / 129.7 |
| B200 | 32 | 16.02 | 28.41 (0.56) | 29.27 (0.55) | 25.30 (0.63) | 228.6 / 135.9 |
| B300 | 32 | 14.52 | 27.40 (0.53) | 27.08 (0.54) | 22.70 (0.64) | 228.9 / 136.2 |

Quality (1500 held-out images, batch 1, TensorRT 11.4):

| Engine | Top-1 B200 / B300 | Agreement with PyTorch top-1 | KL mean |
|---|---|---|---|
| PyTorch fp32 | 82.67% | - | - |
| fp16 engine, level 1 | 82.67% / 82.73% | 99.93% to 100% | < 0.0001 |
| fp16 engine, level 3 | 82.67% / 82.67% | 100% | < 0.0001 |
| FP8 "all", level 1 | 82.47% / 82.73% | 98.93% to 99.20% | 0.0051 to 0.0053 |
| FP8 "all", level 3 | 82.60% / 82.67% | 98.93% to 99.00% | 0.0052 to 0.0055 |
| FP8 "no SE", level 1 | 82.73% / 82.73% | 98.67% to 98.73% | 0.0046 |
| FP8 "no SE, no layer1", level 1 | 82.73% / 82.73% | 98.67% to 99.07% | 0.0027 to 0.0029 |
| FP8 "no SE, no layer1", level 3 | 82.60% / 82.80% | 98.67% to 99.00% | 0.0027 to 0.0028 |
| PyTorch fake quantization, "all" | 82.53% | - | 0.0076 |

TensorRT 11.1 (B200, level 3) gives the same quality: fp16 82.67%, FP8 "no SE,
no layer1" 82.67% (KL 0.0027). No non-finite output in any engine.

Kernels per call, batch 1, TensorRT 11.4 (profiler, ms; the profiler adds time
per kernel, so the sums are larger than the latencies above):

| Engine | GPU | Kernels | Convolution kernels ms (fp16 / FP8) | GEMM and other kernels ms | Sum ms |
|---|---|---|---|---|---|
| fp16, level 1 | B200 | 3859 | 13.83 / - | 2.19 | 16.03 |
| fp16, level 1 | B300 | 3347 | 13.44 / - | 1.25 | 14.68 |
| FP8 "all", level 1 | B300 | 6630 | 0.02 / 15.33 | 5.88 | 21.23 |
| fp16, level 3 | B200 | 3343 | 12.55 / - | 0.81 | 13.36 |
| fp16, level 3 | B300 | 3343 | 11.78 / - | 0.63 | 12.40 |
| FP8 "all", level 3 | B200 | 6627 | 13.72 / 0.95 | 10.25 | 24.91 |
| FP8 "all", level 3 | B300 | 6627 | 12.63 / 0.89 | 8.92 | 22.44 |

In the fp16 level-1 plan on B300, 2307 calls of one fp16 implicit-GEMM
convolution kernel take about 4.7 us each (10.98 ms). In the FP8 "all" plan
at level 3 on B300, 2240 calls of one per-group concatenation kernel take
6.1 ms of the 8.92 ms "GEMM and other" time.

### fp16 grouped convolutions (TensorRT 11.4 behavior)

Independent of FP8: TensorRT 11.4 runs SENet-154 at batch 1 in fp16 at 10.3 to
12.6 ms (B300 and B200, level 1 and 3), while PyTorch fp16 (cuDNN,
channels_last, CUDA graph) takes 4.36 ms on B300, 2.5x faster. At batch 32 both
are about equal (TensorRT 14.52 ms at level 1, PyTorch 14.32 ms on B300). The
cause is the per-group lowering above: 3343 to 3859 kernels per image for 257
convolutions. Builder optimization level 3 does not change it (3343 kernels).
This is recorded here and not fixed; a family workaround (for example, a
blocked form of the grouped convolution) or a TensorRT change would be needed.

### NVFP4 (not tried)

TensorRT does not support NVFP4 on convolutions (block-scaled FP4 needs a 2-D
GEMM); the only GEMM in this network is the classifier.

### What was not tried

| Item | Note |
|---|---|
| FP8 for the 1x1 convolutions only (grouped 3x3 in fp16) | not measured; the fp16 grouped convolutions alone take most of the time |
| Batch 32 at builder optimization level 3 | not measured |
| Entropy or percentile calibration, quantization-aware training | not measured; quality is already fine |
| H100, native runtime with an FP8 bundle | not measured |

### When to revisit

- A TensorRT release that runs a grouped convolution (fp16 and FP8) as one
  kernel instead of one kernel per group. Then the fp16 baseline gets faster
  first, and FP8 has to be measured again against it.
- A TensorRT release where FP8 grouped convolutions at builder optimization
  level 3 use FP8 kernels and give a smaller plan.

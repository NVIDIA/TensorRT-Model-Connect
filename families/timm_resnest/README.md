# timm ResNeSt

This family builds `timm/resnest50d.in1k` as one image classification engine
(224 x 224, batch 1, fp16 in the manifest, TensorRT builder optimization level
3) and implements `image_to_class_scores`. The family has no quantized build.

## FP8 quantization study (not shipped)

FP8 self-quantization of the convolutions was built and measured on TensorRT
11.4 (and 11.1) for `timm/resnest50d.in1k`. It is not shipped: at batch 1 no
FP8 layout is faster than fp16 on both B200 and B300 (0.89x to 1.02x against
fp16 at the same optimization level, level 3 as on this branch or the earlier
level 1), and
quantizing all convolutions costs about 1 ImageNet top-1 point. The plan is
about 40% smaller, but a size gain without a speed gain is not a reason to
ship. This section records the recipe, the method and the numbers, so that
the study does not have to be repeated. The FP8 code was not merged; this
family has no FP8 build.

### Summary and decision

Batch 1, TensorRT 11.4, optimization level 3 (the family default), x is fp16
time over FP8 time:

| Item | fp16 (manifest) | FP8, all convolutions | FP8, layer1 and gates fp16 |
|---|---|---|---|
| Latency, B200 | 0.552 ms | 0.604 ms (0.91x) | 0.556 ms (0.99x) |
| Latency, B300 | 0.558 ms | 0.555 ms (1.01x) | 0.547 ms (1.02x) |
| Plan size | 55.9 MiB | 32.6 MiB (-42%) | 34.0 MiB (-39%) |
| ImageNet top-1, B200 / B300 (PyTorch fp32 82.47%) | 82.47% / 82.47% | 81.20% / 81.20% | 82.33% / 81.93% |
| Non-finite outputs | 0 | 0 | 0 |

Decision: do not ship FP8 for timm ResNeSt.

- Speed: the convolutions are only about 60% of the batch 1 kernel time
  (0.46 of 0.76 ms on B300), and the FP8 convolutions are not faster than
  the fp16 ones at this size. The rest is the split-attention reshape, the
  radix softmax, sums, pooling and multiplies, which FP8 does not change, and
  the Quantize layers add 13 to 23 kernels. The only gain measured is 1.04x to
  1.07x at batch 32 with all convolutions in FP8, which the family does not
  build and which costs about 1 top-1 point.
- Quality: all convolutions in FP8 lose 0.9 to 1.3 top-1 points (KL 0.013).
  Keeping stage `layer1` and the attention gate convolutions in fp16 brings the
  loss to 0.1 to 0.5 points, but that layout is at 0.99x to 1.02x.

### Recipe that was measured

No FP8 or NVFP4 checkpoint of this model exists on the Hub, and no official
quantization config exists for it. The recipe is the one of the shipped
`timm_resnet` FP8 build, applied to the ResNeSt blocks. Each quantized convolution is built
as `input -> Quantize -> Dequantize(fp16) -> Convolution(+bias)`, with the
kernel `FP8 constant -> Dequantize(fp16, axis 0)`.

| Choice | Decision |
|---|---|
| Quantized ("all", 84 of 87 convolutions) | block `conv1`, the split-attention grouped `conv2.conv`, the attention gate 1x1 convolutions `conv2.fc1` and `conv2.fc2`, `conv3`, downsample |
| fp16 | the three deep-stem convolutions, the classifier |
| Variant "nose" (52 FP8) | as "all", gate convolutions `conv2.fc1` / `conv2.fc2` (on a pooled 1x1 map) fp16 |
| Variant "nose-L1" (42 FP8) | as "nose", stage `layer1` fp16 |
| Weight scale | per output channel, amax / 448 of the batch-norm-folded kernel, at build time |
| Activation scale | static, per tensor, max \|input\| / 448 over 500 calibration images (the statistic of ModelOpt max calibration), from forward hooks on the timm PyTorch model |
| Dequantize type | fp16; no non-finite value, fp32 Dequantize was not needed |

### Method

- GPUs: B200 and B300, one GPU per job, all variants built and timed in the
  same job; TensorRT 11.4.0.106 (target) and 11.1.0.106 (B200).
- Two runs: the first study used optimization level 1 (the family default at
  that time) at batch 1 and 32; a later check on this main branch, after the
  family moved to level 3, built every variant at level 1 and level 3 at
  batch 1.
- Data: 2000 ImageNet validation images from `clip-benchmark/wds_imagenet1k`
  (fixed stride): every 4th image (500) for calibration, the other 1500 held
  out. timm preprocessing of the checkpoint (224 x 224).
- Quality: top-1 against the labels, top-1 agreement and KL divergence against
  the timm PyTorch fp32 model on the 1500 held-out images.
- Non-finite scan: 2029 inputs per variant (the 2000 images and 29 stress
  inputs: noise with standard deviation 1, 3 and 10, constants), B200 and B300.
- Speed: CUDA graph replay, median of 9 interleaved rounds of 100 calls, spread
  at most 1.2%. Batch 32 is a scratch build (the family builds batch 1 only).

### Results

Speed at batch 1 on this main branch (ms per engine call; x is fp16 level 3
time over variant time):

| GPU | TensorRT | Level | fp16 | FP8 all | FP8 nose-L1 |
|---|---|---|---|---|---|
| B200 | 11.4 | 3 | 0.552 | 0.604 (0.91x) | 0.556 (0.99x) |
| B200 | 11.4 | 1 | 0.601 (0.92x) | 0.676 (0.82x) | 0.596 (0.93x) |
| B300 | 11.4 | 3 | 0.558 | 0.555 (1.01x) | 0.547 (1.02x) |
| B300 | 11.4 | 1 | 0.586 (0.95x) | 0.628 (0.89x) | 0.581 (0.96x) |
| B200 | 11.1 | 3 | 0.538 | 0.548 (0.98x) | 0.540 (1.00x) |
| B200 | 11.1 | 1 | 0.572 (0.94x) | 0.631 (0.85x) | 0.589 (0.91x) |

Plan size at level 3: fp16 55.7 to 55.9 MiB, FP8 all 32.6 MiB, FP8 nose-L1
34.0 to 34.1 MiB. Build time at level 3: 108 to 120 s for every variant
(17 to 26 s at level 1).

Speed of the first study, optimization level 1 (ms; x is fp16 time over
variant time):

| GPU | Batch | fp16 | FP8 all | FP8 nose | FP8 nose-L1 |
|---|---|---|---|---|---|
| B200 | 1 | 0.582 | 0.638 (0.91x) | 0.604 (0.96x) | 0.583 (1.00x) |
| B300 | 1 | 0.587 | 0.622 (0.94x) | 0.599 (0.98x) | 0.584 (1.01x) |
| B200 | 32 | 2.791 | 2.681 (1.04x) | 2.905 (0.96x) | 2.863 (0.97x) |
| B300 | 32 | 2.777 | 2.596 (1.07x) | 2.788 (1.00x) | 2.777 (1.00x) |

Plan size at batch 1, level 1: fp16 53.4 to 53.9 MiB, FP8 all 30.0 to 30.5
MiB, nose 31.7 to 32.1 MiB, nose-L1 31.8 to 32.2 MiB.

Quality (1500 held-out images, batch 1, TensorRT 11.4, B200 / B300):

| Engine | Level | Top-1 | Agreement with PyTorch top-1 | KL mean |
|---|---|---|---|---|
| PyTorch fp32 | - | 82.47% | - | - |
| fp16 engine | 3 | 82.47% / 82.47% | 100% / 100% | < 0.0001 |
| fp16 engine | 1 | 82.47% / 82.33% | 100% / 99.8% | < 0.0001 |
| FP8 all | 3 | 81.20% / 81.20% | 96.8% / 96.8% | 0.0131 / 0.0131 |
| FP8 all | 1 | 81.40% / 81.53% | 97.1% / 97.3% | 0.0133 / 0.0130 |
| FP8 nose | 1 | 81.40% / 81.47% | 97.4% / 97.5% | 0.0112 / 0.0111 |
| FP8 nose-L1 | 3 | 82.33% / 81.93% | 98.6% / 98.6% | 0.0043 / 0.0045 |
| FP8 nose-L1 | 1 | 82.20% / 82.07% | 98.5% / 98.7% | 0.0045 / 0.0045 |

TensorRT 11.1 (B200, level 3): FP8 all 81.47%, nose-L1 82.20%. No non-finite
output in any run.

PyTorch fake quantization of the same scales on the timm model (1500 held-out
images): all convolutions 81.27% (KL 0.0152), weights only 81.73% (KL 0.0099),
activations only 82.00% (KL 0.0067). Percentile activation scales are worse:
99.99th 81.00%, 99.9th 79.47%. The TensorRT FP8 loss matches the recipe's
loss, so it does not come from TensorRT.

Kernel time per group, B300, TensorRT 11.4, level 1 (CUDA kernel sums in
ms; the sums are larger than the latencies above because kernels are timed
one by one):

| Kernel group | fp16 batch 1 | FP8 all batch 1 | fp16 batch 32 | FP8 all batch 32 |
|---|---|---|---|---|
| convolutions and GEMMs | 0.457 | 0.485 | 1.076 | 1.043 |
| other (split-attention reshape, radix softmax, sums, pooling, multiply, Quantize) | 0.298 | 0.362 | 1.896 | 1.847 |
| total | 0.755 | 0.847 | 2.971 | 2.891 |
| kernels | 185 | 198 | 185 | 198 |

At level 3 the same split holds (B300, batch 1): fp16 0.743 ms in 181
kernels, FP8 all 0.768 ms in 195 kernels, with 0.30 to 0.35 ms in the
non-convolution kernels for every variant.

### NVFP4 (not tried)

TensorRT does not support NVFP4 on convolutions (block-scaled FP4 needs a 2-D
GEMM), the only GEMM is the classifier, and FP8 already loses about 1 top-1
point with all convolutions quantized.

### What was not tried

| Item | Note |
|---|---|
| Entropy calibration, quantization-aware training | not measured; could recover accuracy, not speed |
| Other ResNeSt sizes (resnest14d to resnest269e), H100, native runtime with an FP8 bundle | not measured |
| FP8 with batch above 1 at level 3 | not measured; the family builds batch 1 only |

### When to revisit

- The family adds batch sizes above 1: FP8 of all convolutions was 1.04x to
  1.07x at batch 32 (level 1), but at a cost of about 1 top-1 point.
- A TensorRT release that fuses the split-attention reshape, radix softmax and
  sums into fewer kernels, so that the convolutions become the larger share
  of the time, or that fuses the Quantize into the producing layer.
- A larger, compute-bound ResNeSt (resnest101e and up), where the convolution
  share is higher.

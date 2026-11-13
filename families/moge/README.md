# MoGe

This family builds MoGe-2 ViT-L (`Ruicheng/moge-2-vitl`) for
`monocular_geometry` as one TensorRT engine: the DINOv2 encoder, the
convolutional neck and the heads. The engine takes one RGB image of any size
from 64 to 2048 pixels per side, resamples it to 1800 patch tokens and returns
`affine_depth`, `valid`, `focal_samples` and `metric_scale`; the runtime
recovers focal length and shift and builds the depth map and the point map.
The builder supports fp32 (the manifest) and fp16, batch 1 only. The family
has no quantized build.

## FP8 quantization study (not shipped)

FP8 self-quantization of the DINOv2 encoder linears was built and measured on
TensorRT 11.4 (B200, B300). It is not shipped: depth and focal error is about 8
times the fp16 noise floor and above the family's own E2E thresholds, and the
speed gain over an fp16 build is only 1.08x on B200 (1.25x on B300). The
smallest variant (`mlp.fc1` only) is still 4.7 times the noise floor and gives
no speed gain on B200. This section records the recipe, the method and the
numbers, so that the study does not have to be repeated. The FP8 code was not
merged; this family has no FP8 build.

### Summary and decision

Image 720 x 960, batch 1, TensorRT 11.4. Error is the mean over 109 held-out
images against the fp32 engine (the manifest build).

| Item | fp32 (manifest) | fp16 | FP8 (fp16 engine) | FP8 (fp32 engine) |
|---|---|---|---|---|
| Latency B200 / B300 | 55.41 / 53.28 ms | 4.174 / 4.619 ms | 3.882 / 3.681 ms | 34.18 / 32.99 ms |
| Speed vs fp32, B200 / B300 | 1.00x | 13.3x / 11.5x | 14.3x / 14.5x | 1.62x / 1.61x |
| Speed vs fp16, B200 / B300 | - | 1.00x | 1.08x / 1.25x | - |
| Plan size | 1253 MiB | 637 MiB | 379 MiB | 467 MiB |
| Depth AbsRel vs fp32 engine, B200 / B300 | - | 0.0017 / 0.0022 | 0.0137 / 0.0137 | 0.0138 / 0.0138 |
| Focal relative error vs fp32 engine | - | 0.0008 / 0.0011 | 0.0070 / 0.0069 | 0.0075 / 0.0075 |
| Non-finite outputs | 0 | 0 | 0 | 0 |

Decision: do not ship FP8 for MoGe.

- Quality: the family's E2E thresholds against the official PyTorch fp32
  reference are depth AbsRel 0.005, depth and point relative L2 0.02 and
  intrinsics error 0.002. The FP8 mean is 2.7 times over the AbsRel bar and 3.5
  times over the intrinsics bar; the worst image is 4 to 14 times over. The
  fp16 engine stays inside these bars on the mean (AbsRel 0.0017 against the
  fp32 engine).
- Speed: almost all of the 14x over the fp32 manifest comes from fp16, which
  needs no quantization. On top of fp16, FP8 gives 1.08x on B200 and 1.25x on
  B300. FP8 on the fp32 work precision is 1.61x to 1.62x faster than fp32,
  with the same error, and 8 times slower than fp16.
- Variants that quantize fewer projections lose speed faster than they gain
  quality (see "Variants").

### Recipe that was measured

No ModelOpt FP8 or NVFP4 checkpoint and no official quantization config exists
for MoGe-2. The recipe follows public ViT recipes and the ModelOpt vision
example: quantize the encoder linears, keep the rest in the work precision.

| Choice | Decision |
|---|---|
| Quantized (72 projections) | `attn.qkv`, `mlp.fc1`, `mlp.fc2` of the 24 DINOv2 blocks |
| Work precision (fp16 or fp32) | attention output projection `attn.proj`, attention matmuls and softmax, patch convolution, embeddings, layer norms, layer scales, all convolutions of the neck and heads, scale head |
| Weight scale | per tensor, max(abs(weight)) / 448, one projection at a time at build time; FP8 weights are not stored |
| Activation scale | static, per tensor; ModelOpt 0.46.1 `FP8_DEFAULT_CFG` (max calibration) on the official MoGe PyTorch model in fp32 at 1800 tokens, 248 COCO validation images |
| Outlier rule | a scale above 20x the median of its kind stays unquantized; measured: no outlier (largest ratio `mlp.fc2`, 19x) |
| Graph | `input -> Quantize -> Dequantize -> MatMul`, weight `FP8 constant -> Dequantize`; Dequantize and MatMul in the work precision |
| Scale table key | model config plus a numeric fingerprint of the first `qkv` weight, so MoGe-2 checkpoints that share the ViT-L config do not share scales |

`attn.proj` stays in the work precision because an FP8 Quantize right after the
attention layer does not build on TensorRT 11.4: the softmax gets an FP8
operand and the builder reports "Could not find any implementation for node"
for both fp16 and fp32 work precision.

### Method

- GPUs: B200 and B300, one job per GPU type; TensorRT 11.4.0.106, CUDA 13.0,
  torch 2.14.0, ModelOpt 0.46.1.
- Data: validation shard 0 of `detection-datasets/coco`. Image index mod 10 = 3
  for calibration (248 images), index mod 25 = 7 for evaluation (99 images,
  disjoint), plus the 10 example images of the official MoGe repository: 109
  held-out images. Each image is center-cropped to an aspect ratio between 0.6
  and 1.8 and resized to 720 pixels on the short side.
- Quality: geometry is recovered from the raw engine outputs with the official
  focal and shift solver, the same steps as `runtime/pipeline.cpp`. Errors are
  computed over pixels that are valid in both engines and reported as mean over
  images / worst image. The fp16 engine against the fp32 engine is the noise
  floor. Quality is engine to engine; no ground-truth depth was used.
- Non-finite scan: every output of every engine on the 109 images.
- Speed: all variants built and timed in one process per GPU. 20 warm-up calls,
  then the best of 3 medians of 100 CUDA-event timed calls, at input images of
  720 x 960 and 1080 x 1920 (the engine resamples both to 1800 tokens).

### Results

Speed (ms; x is fp16 time over FP8 time):

| GPU | Image | fp32 | fp16 | FP8 | x | FP8 (fp32 engine) | x vs fp32 |
|---|---|---|---|---|---|---|---|
| B200 | 720 x 960 | 55.41 | 4.174 | 3.882 | 1.08 | 34.18 | 1.62 |
| B200 | 1080 x 1920 | 55.73 | 3.838 | 3.519 | 1.09 | 34.48 | 1.62 |
| B300 | 720 x 960 | 53.28 | 4.619 | 3.681 | 1.25 | 32.99 | 1.61 |
| B300 | 1080 x 1920 | 53.56 | 4.286 | 3.337 | 1.28 | 33.28 | 1.61 |

Plan size (MiB, B200; B300 within 3 MiB): fp32 1253, fp16 637, FP8 379 (-40%
vs fp16), FP8 on fp32 467 (-63% vs fp32). Build time (B200): fp32 56 s, fp16
81 s, FP8 126 s. The FP8 plan holds 216 FP8 tensors (72 projections x Quantize,
Dequantize and weight).

Quality against the fp32 engine, B200 (B300 in brackets where it differs by
more than 0.0005), mean / worst image:

| Engine | depth AbsRel | depth rel. L2 | points rel. L2 | focal rel. error | mask IoU mean / min |
|---|---|---|---|---|---|
| fp16 (noise floor) | 0.0017 / 0.042 (0.0022) | 0.0021 / 0.042 (0.0026) | 0.0021 / 0.044 (0.0027) | 0.0008 / 0.011 | 0.9999 / 0.997 |
| FP8 qkv + fc1 + fc2 | 0.0137 / 0.049 | 0.0181 / 0.076 | 0.0183 / 0.074 | 0.0070 / 0.027 | 0.9996 / 0.988 |
| FP8 on fp32 engine | 0.0138 / 0.048 | 0.0181 / 0.050 | 0.0182 / 0.054 | 0.0075 / 0.027 | 0.9997 / 0.992 |
| E2E threshold (vs PyTorch fp32) | 0.005 | 0.02 | 0.02 | 0.002 (intrinsics) | 0.999 |

### Variants

Same job as the tables above (fp16 work precision; x is fp16 time over FP8
time at 720 x 960):

| Variant | x B200 / B300 | plan MiB | depth AbsRel mean B200 / B300 | focal rel. error | vs noise floor |
|---|---|---|---|---|---|
| qkv + fc1 + fc2 (recipe) | 1.08 / 1.25 | 379 | 0.0137 / 0.0137 | 0.0070 | 8.1x |
| fc1 + fc2 | 1.05 / 1.23 | 449 | 0.0115 / 0.0113 | 0.0062 | 6.8x |
| fc1 only | 1.01 / 1.19 | 543 | 0.0080 / 0.0080 | 0.0040 | 4.7x |

The error is spread over all three kinds (fc1 0.0080, plus fc2 0.0035, plus
qkv 0.0022) and no single projection is an outlier, so leaving out a few
layers does not bring the error near the floor. The B200 gain per kind is
0.04 ms (fc1), 0.16 ms (fc2) and 0.09 ms (qkv) at 720 x 960, so every layer left
out also removes most of the small B200 gain. Even fc1 alone, with no B200
gain, is 1.6 times over the AbsRel bar and 2 times over the intrinsics bar.

### NVFP4 (not tried)

FP8 is already outside the accuracy bars, and NVFP4 has a larger quantization
error than FP8. No NVFP4 checkpoint or official config exists for MoGe-2.

### What was not tried

| Item | Note |
|---|---|
| Per-block sensitivity (keeping single DINOv2 blocks in fp16) | not measured; the per-kind variants above show the error is spread |
| Per-channel or SmoothQuant activation scales, percentile or entropy calibration, quantization-aware training | not measured |
| FP8 `attn.proj` and FP8 attention | does not build on TensorRT 11.4 (see the recipe) |
| TensorRT 11.1, H100, native runtime E2E with an FP8 bundle, ground-truth depth metrics | not measured |

### When to revisit

- A recipe that keeps depth AbsRel under 0.005 and intrinsics error under 0.002
  against the PyTorch fp32 reference (for example finer activation scales or
  quantization-aware training).
- A TensorRT release that builds FP8 `attn.proj` (Quantize after the attention
  layer) or gives a larger FP8 gain over fp16 on B200.

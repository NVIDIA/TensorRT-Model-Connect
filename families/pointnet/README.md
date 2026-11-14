# PointNet

This family builds the yanx27 PointNet semantic-segmentation checkpoint
(S3DIS, 13 classes, 9 input features per point) for
`points_to_semantic_segmentation` as one TensorRT engine. The checkpoint is
exported to ONNX and imported with the TensorRT ONNX parser. Input `[1, 9, N]`
with N up to 4096 points, batch 1, fp32 (default) or fp16 (the manifest uses
fp16). The family has no quantized build.

## FP8 quantization study (not shipped)

FP8 self-quantization of the convolutions and fully connected layers was built
and measured on TensorRT 11.4 (and 11.1) for the S3DIS checkpoint. It is not
shipped: latency stays at the fp16 level (0.97x to 1.04x at batch 1), and
point accuracy on the held-out S3DIS area drops by 7.5 to 9.9 points. The only
gain is plan size (7.4 to about 5 MiB). This section records the recipe, the
method and the numbers, so that the study does not have to be repeated. The FP8 code was not
merged; this family has no FP8 build.

### Summary and decision

| Item | fp16 (manifest) | FP8 per-tensor (fp16 engine) | FP8 vs fp16 |
|---|---|---|---|
| Latency, 4096 points, B200 / B300 (TensorRT 11.4) | 136.3 / 133.0 us | 137.0 / 128.3 us | 0.99x / 1.04x |
| Latency, 1024 points, B200 / B300 (TensorRT 11.4) | 112.6 / 122.2 us | 112.2 / 119.2 us | 1.00x / 1.03x |
| Plan size | 7.4 MiB | 4.9 to 5.0 MiB | -33% |
| Point accuracy, Area 5 (PyTorch fp32 83.17%) | 83.18% | 75.63% | -7.5 points |
| mIoU, Area 5 (PyTorch fp32 40.6) | 40.6 | 36.1 | -4.5 |
| Non-finite outputs (TensorRT 11.4) | 0 | 0 | none; TensorRT 11.1 FP8 engines give non-finite outputs |

Decision: do not ship FP8 for PointNet.

- Speed: the engine is launch bound. The FP8 plan has 40 layers (13
  convolutions, 6 GEMMs, 20 generated pointwise kernels), each a few
  microseconds. The convolutions are small GEMMs (M = N points, K up to 1088,
  N up to 1024, about 10 GFLOP per 4096-point block), so the math is a small
  share of the latency. fp16 is 0.96x to 1.07x of fp32 speed, and FP8 adds
  nothing over fp16.
- The family builds batch 1 only, so there is no batched point where the GEMMs
  would dominate.
- Quality: the measured recipe loses 7.5 to 9.9 accuracy points. A narrower
  recipe may recover quality, but it cannot add speed, so it was not tried.
- A size-only gain (7.4 to about 5 MiB on a small model) is not a reason to
  ship.

### Recipe that was measured

No FP8 or NVFP4 checkpoint of this model exists, and no official quantization
config exists for it. FP8 Q/DQ (ONNX `QuantizeLinear` / `DequantizeLinear`,
e4m3) was inserted into the family's fp16 ONNX graph. Each quantized layer is
`input -> Quantize -> Dequantize(fp16) -> Conv or Gemm`, with the weight
`FP8 constant -> Dequantize(fp16)`.

| Choice | Decision |
|---|---|
| Quantized (14 layers) | every Conv1d (1x1) and fully connected layer with a nonzero fp16 weight: the input T-Net (3 convolutions, 3 fully connected), the last fully connected layer of the feature T-Net, the shared MLP (3 convolutions) and the segmentation head (4 convolutions) |
| fp16 | the first 5 layers of the feature T-Net (their weights are zero in fp16, see below), batch norm, ReLU, max pooling, the two transform matmuls, log-softmax |
| Weight scale | per tensor, amax / 448 (variant: per output channel) |
| Activation scale | static, per tensor, max \|input\| / 448 (the statistic of ModelOpt max calibration), from forward pre-hooks on the PyTorch fp32 reference |
| Dequantize type | fp16; no non-finite value with TensorRT 11.4, fp32 Dequantize was not needed |
| Builder optimization level | 1, as the family builder (variant: 3) |

### Method

- GPUs: B200 and B300, one GPU per job; TensorRT 11.4.0.106 (target) and
  11.1.0.106. Repository base at the time of the study: `a17640d2` (the
  PointNet family code is unchanged since then).
- Data: the official `indoor3d_sem_seg_hdf5_data` blocks (4096 points, 9
  features). Calibration: 419 blocks of Areas 1 to 4 and 6 (every 40th block).
  Held out: 400 blocks of Area 5 (every 10th block, 1.64 M points).
- Quality: point accuracy and mIoU against the labels, argmax agreement and KL
  divergence against the PyTorch fp32 model on the 400 held-out blocks,
  batch 1.
- Non-finite scan: all outputs of the 400 held-out blocks, every variant, B200
  and B300, TensorRT 11.4 and 11.1.
- Speed: one engine call at batch 1, median of 7 interleaved rounds of 300
  calls, all variants built and timed in the same job.

### Results

Speed and size (us per engine call, TensorRT 11.4; x is fp16 time over FP8
per-tensor time):

| GPU | Points | fp32 | fp16 | FP8 per-channel | FP8 per-tensor | x | fp16 level 3 | FP8 per-channel level 3 |
|---|---|---|---|---|---|---|---|---|
| B200 | 4096 | 146.3 | 136.3 | 140.0 | 137.0 | 0.99 | 132.5 | 133.1 |
| B200 | 1024 | 118.1 | 112.6 | 114.7 | 112.2 | 1.00 | 110.8 | 110.0 |
| B300 | 4096 | 138.2 | 133.0 | 131.6 | 128.3 | 1.04 | 125.0 | 121.9 |
| B300 | 1024 | 117.3 | 122.2 | 122.1 | 119.2 | 1.03 | 115.7 | 112.7 |

Plan size (B200 and B300, TensorRT 11.4): fp32 14.0 MiB, fp16 7.4 MiB, FP8
4.9 to 5.0 MiB.

TensorRT 11.1 sanity check (same script and variants):

| GPU | Points | fp32 | fp16 | FP8 per-channel | FP8 per-tensor | x | fp16 level 3 | FP8 per-channel level 3 |
|---|---|---|---|---|---|---|---|---|
| B200 | 4096 | 146.0 | 135.2 | 139.0 | 135.6 | 1.00 | 129.6 | 129.0 |
| B200 | 1024 | 129.2 | 122.2 | 128.0 | 125.2 | 0.98 | 118.2 | 120.3 |
| B300 | 4096 | 138.7 | 126.1 | 130.9 | 126.3 | 1.00 | 124.8 | 122.3 |
| B300 | 1024 | 121.9 | 115.6 | 120.5 | 117.3 | 0.99 | 115.5 | 109.3 |

With TensorRT 11.1 every FP8 engine (per-tensor, per-channel, level 1 and 3)
returns non-finite logits on the held-out blocks on B200 and B300 (argmax
agreement 23%); the fp32 and fp16 engines match the TensorRT 11.4 quality. The
same ONNX graphs give finite outputs with TensorRT 11.4. The cause was not
isolated; the speed of these 11.1 FP8 engines is shown only for comparison.
Plan sizes match TensorRT 11.4 within 0.1 MiB.

Quality (400 held-out Area 5 blocks, B200; B300 is within 0.1 point):

| Engine | Point accuracy | mIoU | Argmax agreement with PyTorch | KL mean | KL p99 |
|---|---|---|---|---|---|
| PyTorch fp32 | 83.17% | 40.6 | - | - | - |
| fp32 engine | 83.17% | 40.6 | 99.90% | 1.9e-5 | 2.7e-4 |
| fp16 engine | 83.18% | 40.6 | 99.77% | 6.7e-5 | 1.4e-3 |
| FP8 per-tensor | 75.63% | 36.1 | 82.3% | 0.475 | 6.0 |
| FP8 per-channel | 73.31% | 33.6 | 81.2% | 0.476 | 5.3 |

Builder optimization level 3 gives the same FP8 quality as level 1.

### Notes on the checkpoint and the recipe

- The feature T-Net (`feat.fstn`) of this checkpoint is inactive: the weights
  and biases of `conv1` to `conv3`, `fc1`, `fc2` and their batch norms are
  about 9e-41 in fp32 (zero in fp16), so the 64x64 feature transform is a
  constant matrix. These 5 layers stay
  fp16 in the measured recipe.
- The quality loss above is from the measured recipe and was not split per
  layer.

### NVFP4 (not tried)

TensorRT does not support NVFP4 on convolutions, FP8 already gives no speedup,
and the engine is not limited by weight reads (all weights fit in a few MiB).

### What was not tried

| Item | Note |
|---|---|
| Narrower recipe (T-Nets or head kept fp16), percentile or entropy calibration, quantization-aware training | not measured; could recover accuracy, not speed |
| Batched engines | the family builds batch 1 only |
| H100, the native C++ runtime with an FP8 bundle, other point counts | not measured |

### When to revisit

- The family adds batched inference (many blocks per call), so that the GEMMs,
  not kernel launches, set the latency.
- A larger point-cloud model in this family with compute-bound GEMMs.

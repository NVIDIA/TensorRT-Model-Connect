# RT-DETR v2

This family builds RT-DETR v2 detectors (`PekingU/rtdetr_v2_r18vd`) for
`object_detection` as one TensorRT engine: 640 x 640 input, batch 1, fp16 or
fp32 (the manifest uses fp16). The engine holds the ResNet-D backbone, the
hybrid encoder (one AIFI transformer layer on the stride-32 level, then the CCFM
fusion of RepVGG convolutions) and the decoder (query selection on 8400 anchors,
3 layers with self-attention, deformable cross-attention and a feed-forward, box
and class heads). The family has no quantized build.

## FP8 quantization study (not shipped)

FP8 self-quantization of the linear layers was built and measured on TensorRT
11.4 (and 11.1) for `PekingU/rtdetr_v2_r18vd`. It is not shipped: no part and no
combination of parts is faster than fp16 beyond the run-to-run spread (0.98x to
1.03x on B200 and B300), the plan is only 3% smaller with the default parts,
and each quantized part moves 2 to 3 percent of the detections at the runtime
score threshold (COCO AP stays at the fp16 level). This section records the
recipe, the method and the numbers, so that the study does not have to be
repeated. The FP8 code was not merged; this family has no FP8 build.

### Summary and decision

| Item | fp16 (manifest) | FP8 default, 12 linears (fp16 engine) | FP8 vs fp16 |
|---|---|---|---|
| Latency, batch 1, B200 / B300 (TensorRT 11.4) | 0.727 / 0.705 ms | 0.723 / 0.703 ms | 1.01x / 1.00x |
| Latency, all 44 linears, B200 / B300 (TensorRT 11.4) | 0.727 / 0.705 ms | 0.726 / 0.698 ms | 1.00x / 1.01x |
| Plan size, default / all 44 linears | 42.8 MiB | 41.6 / 39.0 MiB | -3% / -9% |
| COCO-style AP, B200 / B300 (PyTorch fp32 0.5080) | 0.5077 / 0.5077 | 0.5085 / 0.5094 | at the fp16 level |
| Detections at score 0.3 matching PyTorch (prec / rec), B200 | 0.992 / 0.990 | 0.976 / 0.974 | 2 percent move |
| Non-finite outputs | 0 | 0 (sampling pair on 11.4: all) | see Results |

Numbers of this table: one TensorRT 11.4 job per GPU on the current family code
(the spot check in Results). The per-part tables come from earlier jobs on the
same family code and the same TensorRT versions.

Decision: do not ship FP8 for RT-DETR v2.

- Speed: the engine runs in about 0.7 ms as 182 small kernels. The linear GEMMs
  (the FP8 candidates) are 22% of the kernel time, about 9 microseconds each,
  close to the launch floor. FP8 makes them 4% faster (355 to 342
  microseconds), and the added Quantize and cast kernels (about 45
  microseconds) take that back.
- The convolutions (42% of the kernel time) have no FP8 path in this family and
  stay fp16.
- Size: the weights are 42 MiB; reading them takes about 5 microseconds, under
  1% of the latency, so the engine is not weight-bandwidth bound. The plan is
  3% smaller with the default parts and 9% smaller with all 44 linears.
- Quality: AP is kept, but the detection set at the 0.3 threshold drifts by 2
  to 3 percent per part and 5 percent with all parts, for no speed gain.

### Recipe that was measured

No FP8 or NVFP4 checkpoint of RT-DETR v2 exists on the Hub, and no official
quantization config exists for it. The layer list follows the usual guidance
for small vision encoders and detectors: attention, MLP and head linears with
static per-tensor activation scales; convolutions, norms and position tables
stay fp16. Each quantized linear is built as
`input -> Quantize -> Dequantize(fp16) -> MatMul (+bias)`, with the weight
`FP8 constant -> Dequantize(fp16)`. Linears that read the same tensor with the
same scale share one Quantize/Dequantize pair.

| Choice | Decision |
|---|---|
| Parts (55 linears in total) | `aifi` (6), `decoder_self` (q, k, v, out of 3 layers; 12), `decoder_value` (value and output projection of the deformable attention; 6), `decoder_sampling` (sampling offsets and attention weights; 6), `decoder_ffn` (6), `query_select` (encoder output projection, class and box heads; 4), `refine` (query position head, per-layer box deltas and class heads; 10) |
| Measured default | `aifi` + `decoder_value` (12 linears), the two parts that move the detections least together |
| fp16 | all convolutions (backbone, CCFM, input projections), norms, attention matmuls and softmax, grid sample, top-k; a linear whose input or output width is not a multiple of 16 (position head input of 4, last box-head layers with 4 outputs) |
| Weight scale | per tensor, amax / 448, at build time |
| Activation scale | static, per tensor; ModelOpt 0.46.1 `FP8_DEFAULT_CFG` (max calibration) on the transformers `RTDetrV2ForObjectDetection` model, 256 COCO validation images resized like the runtime |
| Outlier rule | a scale above 20x the median of its kind stays fp16; measured: no outlier |
| Dequantize type | fp16; no non-finite value except the sampling pair on 11.4 (below), fp32 Dequantize was not needed |
| Scale table key | model config (hidden size, layers, feed-forward sizes, queries, labels, input size) and a numeric weight fingerprint |

### Method

- GPUs: B200 and B300, one GPU per job, all variants of a table built and timed
  in that job; TensorRT 11.4.0.106 (target) and 11.1.0.106. Builder
  optimization level 3 (the family default), strongly typed networks.
- Reference: PyTorch fp32 `RTDetrV2ForObjectDetection` (transformers 5.17.0) on
  the same preprocessed pixels.
- Quality: COCO validation images 256 to 555 of `detection-datasets/coco`
  (300 held out; 0 to 255 are the calibration set), resized to 640 x 640 without
  normalization, as the runtime does. Detections are the transformers
  post-processing (sigmoid, top 300 of queries x classes). AP is a COCO-style AP
  (101-point, IoU 0.50 to 0.95) from a small numpy implementation on these 300
  images, comparable only inside these tables. Agreement: detections at score
  0.3 (the runtime threshold) matched greedily to the reference detections (same
  class, IoU at least 0.5); `prec` is the share of the engine's detections that
  match, `rec` the share of the reference's.
- Non-finite scan: the 300 images and 3 stress inputs (noise, all white, all
  black).
- Speed: `execute_async_v3` loop, 400 calls after 50 warm-up, 5 interleaved
  rounds; the table shows the fastest round (the median round differs by at
  most 0.0004 ms). x is fp16 time over FP8 time.

### Results

Spot check, TensorRT 11.4 (one job per GPU, all variants built and timed in
that job; x is fp16 time over variant time):

| GPU | fp32 ms | fp16 ms | FP8 default ms (x) | FP8 all 44 ms (x) | plan MiB fp16 / default / 44 | AP fp16 / default / 44 |
|---|---|---|---|---|---|---|
| B200 | 1.085 | 0.727 | 0.723 (1.01) | 0.726 (1.00) | 42.8 / 41.6 / 39.0 | 0.5077 / 0.5085 / 0.5085 |
| B300 | 1.046 | 0.705 | 0.703 (1.00) | 0.698 (1.01) | 42.8 / 41.8 / 39.2 | 0.5077 / 0.5094 / 0.5073 |

Default build against its fp16 control, earlier jobs (TensorRT 11.4 and 11.1):

| GPU | TensorRT | fp32 ms | fp16 ms | FP8 default ms | x | plan MiB fp16 / default | prec / rec vs PyTorch, default | AP default |
|---|---|---|---|---|---|---|---|---|
| B200 | 11.4 | 1.086 | 0.734 | 0.726 | 1.01 | 42.7 / 41.7 | 0.974 / 0.973 | 0.5092 |
| B300 | 11.4 | 1.030 | 0.688 | 0.682 | 1.01 | 42.7 / 41.5 | 0.972 / 0.973 | 0.5094 |
| B200 | 11.1 | 1.081 | 0.723 | 0.719 | 1.01 | 42.4 / 41.5 | 0.973 / 0.975 | 0.5080 |
| B300 | 11.1 | 1.034 | 0.688 | 0.687 | 1.00 | 42.6 / 41.3 | 0.971 / 0.975 | 0.5089 |

The fp16 engine of the same GPU and TensorRT version varied by 1.5% between
jobs by up to 2.4% (B200, 11.4: 0.727 to 0.744 ms in five jobs), which is as
large as every FP8 gain measured.

Each part alone and all parts together, x against the fp16 engine of the same
job:

| FP8 linears | B200 11.4 | B300 11.4 | B200 11.1 | B300 11.1 |
|---|---|---|---|---|
| fp16 ms | 0.744 | 0.689 | 0.730 | 0.686 |
| `aifi` (6) | 1.010 | 0.999 | 0.996 | 1.002 |
| `decoder_self` (12) | 1.016 | 0.998 | 0.995 | 0.995 |
| `decoder_value` (6) | 1.010 | 0.991 | 1.013 | 0.992 |
| `decoder_ffn` (6) | 1.017 | 0.998 | 0.998 | 0.998 |
| `query_select` (4) | 1.013 | 0.991 | 0.995 | 0.978 |
| `refine` (10) | 1.004 | 0.994 | 1.000 | 0.989 |
| all except `decoder_sampling` (44) | 0.995 | 0.994 | 1.001 | 0.988 |
| `decoder_sampling` (6) | non-finite | non-finite | 0.984 | 0.982 |
| all 50 | non-finite | not run | 0.983 | 0.992 |

Combinations, B200, TensorRT 11.4, one more job (fp16 0.741 ms):
`decoder_value` + `aifi` 1.012, `decoder_value` + `decoder_ffn` 1.025,
`decoder_value` + `decoder_ffn` + `aifi` 1.015, the same plus `decoder_self` (30
linears) 1.028, the best seen, with 4.5 percent of the detections moved.

Quality of each part, B200, TensorRT 11.4 (300 held-out images; the first row
is the floor):

| Engine (FP8 linears) | AP | prec / rec vs PyTorch | prec / rec vs fp16 engine | box IoU vs fp16 | score error vs fp16 |
|---|---|---|---|---|---|
| PyTorch fp32 | 0.5080 | - | - | - | - |
| fp16 engine (floor) | 0.5077 | 0.991 / 0.991 | - | - | - |
| `aifi` (6) | 0.5098 | 0.979 / 0.980 | 0.978 / 0.979 | 0.994 | 0.0056 |
| `decoder_self` (12) | 0.5088 | 0.971 / 0.971 | 0.972 / 0.972 | 0.991 | 0.0067 |
| `decoder_value` (6) | 0.5088 | 0.982 / 0.982 | 0.982 / 0.983 | 0.993 | 0.0050 |
| `decoder_ffn` (6) | 0.5090 | 0.975 / 0.976 | 0.975 / 0.976 | 0.992 | 0.0064 |
| `query_select` (4) | 0.5068 | 0.975 / 0.976 | 0.977 / 0.979 | 0.990 | 0.0055 |
| `refine` (10) | 0.5088 | 0.980 / 0.972 | 0.981 / 0.973 | 0.992 | 0.0066 |
| all except `decoder_sampling` (44) | 0.5090 | 0.951 / 0.946 | 0.950 / 0.946 | 0.983 | 0.0140 |
| `decoder_sampling` (6), TensorRT 11.1 | 0.5087 | 0.987 / 0.986 | 0.987 / 0.988 | 0.993 | 0.0034 |
| `decoder_sampling` (6), TensorRT 11.4 | 0 (non-finite) | | | | |

B300 and TensorRT 11.1 give the same picture (AP 0.507 to 0.510 for every
part). Each part moves 2 to 3 percent of the detections at the 0.3 threshold
and the effect adds up; the boxes stay (IoU 0.98 to 0.99).

Non-finite scan: 0 non-finite outputs on the 300 images and the 3 stress inputs
for every variant, except `decoder_sampling` (and every set that contains it) on
TensorRT 11.4: the outputs are non-finite on every input, on B200 and B300,
while each of the two sampling linears alone is finite and TensorRT 11.1 is
finite with both. This part is not in the default.

Kernel time per group of the fp16 engine and of the FP8 engine with 44 linears,
B200, TensorRT 11.4 (layer profiler; the profiler adds overhead, so the sums are
above the wall time; microseconds (count)):

| Kernel group | fp16 | FP8, 44 linears |
|---|---|---|
| convolutions | 686 (65) | 673 (65) |
| linear GEMMs | 355 (41) | 342 (41) |
| norms, softmax, casts, adds, resizes, Quantize | 347 (49) | 392 (54) |
| deformable attention tail | 78 (9) | 87 (10) |
| grid sample | 76 (9) | 77 (9) |
| attention matmuls (AIFI) | 39 (4) | 40 (4) |
| pooling, top-k | 59 (5) | 59 (5) |
| sum | 1640 | 1668 |

B300 and TensorRT 11.1 show the same split (linear GEMMs 20% to 22% of the
kernel time). The GEMMs are small (M = 300 queries or 400 tokens for most, M =
8400 for the value projections and the query-selection heads).

### NVFP4 (not tried)

The engine is launch and latency bound, not weight-bandwidth bound, and FP8 on
all linears gives no gain. NVFP4 cannot save more than the linears' own time
(22%) while adding dynamic-quantize kernels, and TensorRT has no NVFP4
convolution for the 42% of the time spent in convolutions.

### What was not tried

| Item | Note |
|---|---|
| FP8 convolutions (backbone, CCFM) | no FP8 convolution path in this family; the largest share of the time |
| FP8 attention matmuls or softmax, percentile calibration, per-channel weights | not measured |
| Larger checkpoints (r34vd, r50vd, r101vd), batch greater than 1 | not measured; the family builds batch 1 only |
| Full 5000-image COCO validation with `pycocotools`, H100, native runtime with an FP8 bundle | not measured |
| Isolating the TensorRT 11.4 non-finite output of the sampling pair | a tiny engine with two FP8 linears of 192 and 96 outputs on one activation does not reproduce it; separate Quantize/Dequantize pairs do not avoid it |

### When to revisit

- An FP8 convolution path for this family: the convolutions are 42% of the
  kernel time, so this is the only part large enough to give a real speed-up.
- Larger RT-DETR v2 checkpoints or larger batches, where the linears become a
  larger share of the time.
- A TensorRT release that fuses the Quantize into the producer of each FP8
  linear, so the added kernels no longer take back the GEMM gain.

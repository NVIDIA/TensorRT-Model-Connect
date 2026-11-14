# Grounding DINO

This family builds `IDEA-Research/grounding-dino-tiny` for
`image_text_to_boxes` as one TensorRT engine: a Swin-T image backbone, a BERT
text tower, six fusion and enhancer encoder layers, six cross-modality decoder
layers, and the box and contrastive heads. The engine takes the image (one
square 800 x 800 input, batch 1) and the tokenized prompt (padded to 256 tokens
and masked), in fp16 or fp32 (the manifest uses fp16). The host side of the text
inputs (phrase mask, position ids) is built by the runtime. The family has no
quantized build.

## FP8 quantization study (not shipped)

FP8 self-quantization of the linear projections was built and measured on
TensorRT 11.4 (and 11.1) for `grounding-dino-tiny`. It is not shipped: the only
part that keeps the detections at the fp16 level (the BERT text tower) makes the
plan 23% smaller but is not faster (1.00x on B200, 1.01x on B300, TensorRT 11.4).
Quantizing all 319 projections gives at most 1.03x and loses 4 to 5 points of
precision and recall. This section records the recipe, the method and the
numbers, so that the study does not have to be repeated. The FP8 code was not
merged; this family has no FP8 build.

### Summary and decision

| Item | fp16 (manifest) | FP8 text tower | FP8 all projections |
|---|---|---|---|
| Latency, batch 1, B200 / B300 (TensorRT 11.4) | 8.727 / 8.576 ms | 8.711 / 8.491 ms | 8.581 / 8.315 ms |
| Speedup vs fp16, B200 / B300 | - | 1.00x / 1.01x | 1.02x / 1.03x |
| Plan size, B200 | 352.0 MiB | 272.0 MiB (-23%) | 221.6 MiB (-37%) |
| Precision / recall vs PyTorch fp32, B200 | 0.987 / 0.985 | 0.984 / 0.991 | 0.940 / 0.943 |
| Non-finite outputs | 0 | 0 | 0 |

Decision: do not ship FP8 for Grounding DINO.

- Speed: the GEMMs that lower to FP8 kernels (the 768-wide BERT projections and
  part of Swin) are a small share of the 8.6 to 8.7 ms. Most of the time is in
  the six deformable-attention sampling kernels and in many small fused kernels
  of the 256-wide encoder and decoder.
- The 256-wide encoder, decoder and head projections get no FP8 GEMM kernel
  (none found in the plan), so quantizing them costs quality and gives no speed.
- Quality: Swin, the encoder deformable attention, the decoder self- and
  cross-attention and the heads each lose precision or recall when quantized
  alone.
- The only gain of the quality-neutral text tower is plan size, and a size-only
  gain is not a reason to ship.

### Recipe that was measured

No FP8 or NVFP4 checkpoint of Grounding DINO exists on the Hub, and no official
quantization config exists for it. The recipe follows the ModelOpt and Edge-LLM
guidance for encoders and detectors: all attention and MLP linears are
candidates; patch embedding, convolutions, norms and attention matmuls stay
fp16. Each part was then measured alone against the fp16 engine. Each quantized
projection is built as `input -> Quantize -> Dequantize(fp16) -> MatMul`, with
the weight `FP8 constant -> Dequantize(fp16)`; one Quantize per shared input.

| Choice | Decision |
|---|---|
| Candidate parts (319 projections) | `text` (72: query, key, value, attention output, intermediate, output of the 12 BERT layers), `swin` (72), `fusion` (36), `enhancer` (36), `enc_deformable` (24), `dec_self` (24), `dec_text` (24), `dec_deformable` (12), `dec_ffn` (12), `heads` (7: box refinement, encoder output, reference points) |
| Best part | `text` only |
| fp16 always | text projection, deformable `sampling_offsets` and `attention_weights`, the 4-output box layers, patch embedding and merging, input-projection convolutions and group norms, embeddings, layer norms, attention matmuls, contrastive logits |
| Weight scale | per tensor, amax / 448, at build time (FP8 weights never stored) |
| Activation scale | static, per tensor; ModelOpt `FP8_DEFAULT_CFG` (max calibration) on PyTorch (`transformers` 5.17.0), 200 COCO validation images at 800 x 800 with one prompt each |
| Outlier rule | a scale above 20x the median of its kind stays fp16; measured: no outlier |
| Dequantize type | fp16; no non-finite value, fp32 Dequantize was not needed |

### Method

- GPUs: B200 and B300, one GPU per job, all variants built and timed in the same
  job; TensorRT 11.4.0.106 (target) and 11.1.0.106.
- Data: `detection-datasets/coco` validation split. Images 0 to 199 calibrate;
  images 200 to 319 are held out (120 images on 11.4, the first 60 on 11.1). The
  prompt of an image lists the largest object categories present (at most five)
  plus one absent category.
- Quality: detections at box score 0.3 matched to the PyTorch fp32 detections
  (same phrase, IoU at least 0.5): precision, recall, mean box IoU of the
  matches, mean absolute score error, and the correlation of the logits of the
  detected queries. The fp16 engine against PyTorch is the floor.
- Non-finite scan: the held-out images plus noise, all-white and all-black
  images.
- Speed: batch 1, same image and prompt for all variants; per variant 20
  warm-up runs, then the mean of 100 back-to-back runs; best of 3 interleaved
  rounds.

### Results

Speed and size, default parts (x is fp16 time over FP8 time):

| GPU | TensorRT | fp16 ms | FP8 text ms (x) | FP8 swin ms (x) | FP8 all ms (x) | plan MiB fp16 / text / all |
|---|---|---|---|---|---|---|
| B200 | 11.4 | 8.727 | 8.711 (1.00) | 8.674 (1.01) | 8.581 (1.02) | 352.0 / 272.0 / 221.6 |
| B300 | 11.4 | 8.576 | 8.491 (1.01) | 8.470 (1.01) | 8.315 (1.03) | 355.3 / 273.5 / 222.0 |
| B200 | 11.1 | 8.778 | 8.741 (1.00) | 8.765 (1.00) | 8.637 (1.02) | 352.1 / 272.4 / 228.6 |
| B300 | 11.1 | 8.548 | 8.438 (1.01) | 8.516 (1.00) | 8.264 (1.03) | 355.2 / 274.1 / 223.6 |

Quality vs PyTorch fp32 (TensorRT 11.4, 120 images, 1061 reference
detections; B300 in brackets):

| Engine | Precision | Recall | Box IoU | Score error (max) | Logit corr. (detected) |
|---|---|---|---|---|---|
| fp16 engine (floor) | 0.987 (0.986) | 0.985 (0.990) | 0.991 (0.991) | 0.0051 (0.0050) | 0.910 (0.913) |
| FP8 text | 0.984 (0.987) | 0.991 (0.986) | 0.991 (0.991) | 0.0053 (0.0052) | 0.902 (0.905) |
| FP8 swin | 0.975 (0.976) | 0.979 (0.975) | 0.987 (0.986) | 0.0098 (0.0097) | 0.886 (0.879) |
| FP8 all | 0.940 (0.940) | 0.943 (0.944) | 0.981 (0.982) | 0.0241 (0.0231) | 0.848 (0.851) |

On TensorRT 11.1 (60 images, 571 reference detections) the picture is the same:
fp16 0.988 / 0.991 (B200) and 0.991 / 0.991 (B300) precision / recall, FP8 text
0.986 / 0.984 and 0.984 / 0.993, FP8 all 0.952 / 0.939 and 0.951 / 0.942.

Per-part sensitivity (B200, TensorRT 11.4, one job, 120 images; FP8 GEMM
kernels = CUTLASS e4m3 GEMM kernels found in the plan):

| Part | Projections | Plan MiB | FP8 GEMM kernels | ms | Precision | Recall | Box IoU | Score error (max) | Result |
|---|---|---|---|---|---|---|---|---|---|
| fp16 engine | 0 | 352.2 | 0 | 8.688 | 0.987 | 0.987 | 0.991 | 0.0050 (0.057) | floor |
| text | 72 | 272.0 | 12 | 8.683 | 0.984 | 0.988 | 0.991 | 0.0053 (0.062) | at floor |
| fusion | 36 | 344.5 | 0 | 8.646 | 0.986 | 0.983 | 0.991 | 0.0054 (0.066) | at floor, no FP8 GEMM |
| enhancer | 36 | 348.5 | 0 | 8.668 | 0.988 | 0.989 | 0.991 | 0.0061 (0.074) | at floor, no FP8 GEMM |
| dec_ffn | 12 | 346.7 | 0 | 8.663 | 0.983 | 0.981 | 0.990 | 0.0070 (0.059) | slightly below, no FP8 GEMM |
| dec_deformable | 12 | 351.6 | 0 | 8.692 | 0.990 | 0.981 | 0.990 | 0.0055 (0.082) | slightly below, no FP8 GEMM |
| swin | 72 | 328.5 | 6 | 8.636 | 0.972 | 0.980 | 0.986 | 0.0094 (0.115) | fails |
| enc_deformable | 24 | 351.5 | 0 | 8.636 | 0.982 | 0.973 | 0.988 | 0.0085 (0.220) | fails |
| dec_self | 24 | 351.3 | 0 | 8.690 | 0.973 | 0.961 | 0.989 | 0.0092 (0.118) | fails |
| dec_text | 24 | 351.1 | 0 | 8.686 | 0.957 | 0.988 | 0.991 | 0.0149 (0.105) | fails (extra detections) |
| heads | 7 | 351.8 | 0 | 8.700 | 0.982 | 0.967 | 0.988 | 0.0088 (0.226) | fails |
| encoder (fusion, enhancer, enc_deformable) | 96 | 334.6 | 6 | 8.594 | 0.983 | 0.977 | 0.987 | 0.0095 (0.168) | fails |
| decoder (dec_self, dec_text, dec_deformable, dec_ffn) | 72 | 344.2 | 0 | 8.689 | 0.954 | 0.966 | 0.987 | 0.0176 (0.132) | fails |
| all parts | 319 | 222.5 | 18 | 8.540 | 0.942 | 0.943 | 0.981 | 0.0232 (0.292) | fails |

On TensorRT 11.1 the plan inspector found no CUTLASS e4m3 GEMM kernel for the
text part (the plan still shrinks to 272 MiB) and 2 to 4 for the Swin part.

Where the time goes (layer profiler, batch 1, TensorRT 11.4; the profiler sum is
larger than the end-to-end time): the six deformable-attention sampling kernels
take 0.445 ms each (2.67 ms of 11.89 ms on B200). Quantizing all 319 projections
lowers the profiler sum by 0.14 ms on B200 (11.89 to 11.75 ms) and 0.25 ms on
B300 (11.87 to 11.63 ms).

### NVFP4 (not tried)

The share of time in weight-bound GEMMs is the same small share that FP8 already
targets, so NVFP4 cannot give a larger speedup at batch 1, and the parts that
fail in FP8 would lose more at 4 bits. No NVFP4 engine was built.

### What was not tried

| Item | Note |
|---|---|
| Batch above 1 | the family builds batch 1 only |
| fp32 engine as the control | not measured; the manifest uses fp16 |
| Other checkpoints (Swin-B, MM Grounding DINO) | not measured |
| Percentile or entropy calibration, per-channel weight scales | not measured |
| Native runtime with an FP8 bundle, H100 | not measured |

### When to revisit

- A TensorRT release that gives the 256-wide projections FP8 GEMM kernels and
  makes the deformable-attention sampling faster, so that the GEMMs become a
  larger share of the time.
- A larger checkpoint (Swin-B backbone) or larger batches, where the GEMM share
  is larger.
- A plan-size target on its own: the text tower alone gives 23% with no quality
  loss.

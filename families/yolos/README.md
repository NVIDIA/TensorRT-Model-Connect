# YOLOS

This family builds YOLOS detectors (`YolosForObjectDetection`, for example
`hustvl/yolos-small`) for `object_detection` as one TensorRT engine: a ViT
encoder over the image patches plus 100 detection tokens, followed by the class
and box heads. The engine takes the native input size of the checkpoint
(512 x 864 for yolos-small, 1728 patches + 1 + 100 tokens = 1829 tokens),
batch 1, fp16 or fp32 (the manifest uses fp16). The family has no quantized
build.

## FP8 quantization study (not shipped)

FP8 self-quantization of the encoder linears was built and measured on
TensorRT 11.4 (and 11.1) for `hustvl/yolos-small`. It is not shipped: FP8 is
1.11x (B200) and 1.17x (B300) faster than fp16 at batch 1, but about 11% of the
detections at score 0.5 change against the PyTorch fp32 reference (fp16: 0.4%),
and the test image drifts by 3x the score tolerance of the premerge case. This
section records the recipe, the method and the numbers, so that the study does
not have to be repeated. The FP8 code was not merged; this family has no FP8
build.

### Summary and decision

| Item | fp16 (manifest) | FP8, all 72 linears (fp16 engine) | FP8 vs fp16 |
|---|---|---|---|
| Latency, batch 1, B200 / B300 (TensorRT 11.4) | 0.706 / 0.655 ms | 0.635 / 0.561 ms | 1.11x / 1.17x |
| Plan size (B200, TensorRT 11.4) | 64.6 MB | 44.6 MB | -31% |
| COCO AP, B200 / B300 (PyTorch fp32 0.3341) | 0.3344 / 0.3343 | 0.3301 / 0.3302 | -0.4 point |
| Detections at score 0.5 matched to PyTorch fp32, precision / recall (B200) | 0.996 / 0.997 | 0.885 / 0.891 | 11% of detections change |
| Mean IoU of matched boxes (B200) | 0.991 | 0.888 | |
| Mean class KL per detection token (B200) | 5.3e-05 | 0.0236 | about 450x |
| Test image of the premerge case, largest score difference (tolerance 0.05) | 0.030 | 0.147 | 3x over |
| Non-finite outputs | 0 | 0 | none |

Decision: do not ship FP8 for YOLOS.

- Quality: the detections at the usual 0.5 score threshold change for about 1
  in 9 objects, and matched boxes move (IoU 0.888 vs 0.991 for fp16). AP moves
  less (-0.4 point), but the detection agreement and the per-token class
  distribution are far from the fp16 floor.
- The loss is the per-tensor 8-bit recipe itself, not TensorRT: the same recipe
  as fake quantization in PyTorch (ModelOpt) gives the same numbers (KL 0.0229
  vs 0.0236 for the engine).
- No subset of the linears keeps the detection agreement near fp16 (see the
  variants below). The best one (MLP only) keeps AP but still changes 7% of the
  detections for a 1.05x to 1.15x speedup.
- The speedup is small at batch 1 (the only batch the family builds):
  attention over 1829 tokens is about 45% of the FLOPs and stays in fp16.

### Recipe that was measured

No FP8, NVFP4, INT8 or ModelOpt checkpoint of any YOLOS model exists on the Hub
(the YOLOS ONNX repositories cannot be read by this family), and no official
quantization config exists for one. The recipe follows the ViT guideline: FP8
W8A8 on the encoder linears, everything else fp16. Each quantized linear is
built as `input -> Quantize -> Dequantize(fp16) -> MatMul(+bias)`, with the
weight `FP8 constant -> Dequantize(fp16)`.

| Choice | Decision |
|---|---|
| Quantized (72 linears) | query, key, value, attention output, intermediate and output dense of all 12 encoder layers |
| fp16 | patch convolution, CLS and detection tokens, position and mid-position embeddings, layer norms, attention matmuls and softmax, class and box heads |
| Weight scale | per tensor, amax / 448 of each weight at build time (FP8 bytes are never stored) |
| Activation scale | static, per tensor; ModelOpt 0.46.1 `FP8_DEFAULT_CFG` (max calibration) on the transformers model, 300 COCO 2017 validation images resized to 512 x 864 |
| Scale key | model config and weight fingerprint of the checkpoint; an already quantized checkpoint is refused |
| Outlier rule | a scale above 20x the median of its kind stays fp16; measured: no outlier (activation amax 4 to 18) |
| Dequantize type | fp16; no non-finite value, fp32 Dequantize was not needed |
| Precision | `quantization=fp8` on the fp16 build |

The attention is written as explicit matmuls and softmax; TensorRT still fuses
it into 12 attention blocks in both the fp16 and the FP8 plan. The FP8
attention output projection after it builds on TensorRT 11.4 and 11.1.

### Method

- GPUs: B200 and B300, one GPU per job, all variants of a row built and timed in
  the same job; TensorRT 11.4.0.106 (target) and 11.1.0.106; transformers
  5.17.0, torch 2.14.0 (CUDA 13).
- Data: the first 800 COCO 2017 validation images of `detection-datasets/coco`
  with their annotations. Calibration: the even images 0 to 598 (300). Held out:
  the other 500 (odd images 1 to 599 and images 600 to 799).
- Reference: transformers fp32 `YolosForObjectDetection` with the same 512 x 864
  preprocessing as the engine.
- Quality: AP is `pycocotools` COCO AP (IoU 0.50 to 0.95) of the 100 detections
  per image, comparable only inside these tables. Precision / recall: detections
  at score 0.5 greedily matched to the reference detections (same class, IoU at
  least 0.5). KL: KL(reference || engine) of the 92-class softmax per detection
  token (50,000 tokens). Top-1: per-token class agreement. Box L1: mean absolute
  difference of the normalized boxes.
- Non-finite scan: the 800 images, the test image and 14 stress inputs (Gaussian
  noise at 1x, 3x and 10x, constants -2.2 and 2.7).
- Speed: one engine call, batch 1, CUDA graph replay, median of 9 interleaved
  rounds of 200 calls. x is fp16 time over FP8 time.

### Results

Speed (ms per call, batch 1):

| GPU | TensorRT | fp32 | fp16 | FP8 all 72 | x | FP8 without attention output (60) | x | FP8 MLP only (24) | x |
|---|---|---|---|---|---|---|---|---|---|
| B200 | 11.4 | 2.600 | 0.706 | 0.635 | 1.11 | 0.649 | 1.09 | 0.649 | 1.09 |
| B300 | 11.4 | 2.497 | 0.655 | 0.561 | 1.17 | 0.557 | 1.18 | 0.570 | 1.15 |
| B200 | 11.1 | 2.634 | 0.719 | 0.673 | 1.07 | 0.686 | 1.05 | 0.688 | 1.05 |
| B300 | 11.1 | 2.548 | 0.647 | 0.579 | 1.12 | 0.580 | 1.12 | 0.591 | 1.09 |

A second build of the FP8 engine in each job timed within 0.006 ms of the
first. Plan size (B200, TensorRT 11.4): fp32 125.9 MB, fp16 64.6 MB, FP8 all 72
44.6 MB, FP8 without attention output 46.0 MB, FP8 MLP only 51.2 MB.

Quality, 500 held-out images, B200, TensorRT 11.4 (B300 and TensorRT 11.1 are
within 0.002 AP of these rows, except FP8 without attention output on 11.1 at
0.3306; FP8 all 72 precision / recall: B300 0.886 / 0.892, 11.1 0.886 / 0.898 to 0.899):

| Engine | AP | AP50 | prec / rec vs PyTorch | matched IoU | KL mean | KL p99 | top-1 | box L1 |
|---|---|---|---|---|---|---|---|---|
| PyTorch fp32 | 0.3341 | 0.5132 | - | - | - | - | - | - |
| fp32 engine | 0.3344 | 0.5130 | 0.998 / 0.999 | 0.997 | 6.0e-06 | 5.2e-05 | 0.9997 | 0.00015 |
| fp16 engine | 0.3344 | 0.5136 | 0.996 / 0.997 | 0.991 | 5.3e-05 | 4.6e-04 | 0.9992 | 0.0005 |
| FP8 all 72 | 0.3301 | 0.5102 | 0.885 / 0.891 | 0.888 | 0.0236 | 0.366 | 0.9736 | 0.0113 |
| FP8 without attention output (60) | 0.3339 | 0.5096 | 0.904 / 0.913 | 0.898 | 0.0190 | 0.284 | 0.9770 | 0.0103 |
| FP8 MLP only (24) | 0.3342 | 0.5117 | 0.930 / 0.935 | 0.919 | 0.0125 | 0.162 | 0.9828 | 0.0077 |
| ModelOpt fake quant in PyTorch, all 72 | 0.3298 | 0.5071 | 0.892 / 0.897 | 0.888 | 0.0229 | 0.354 | 0.9742 | 0.0112 |
| ModelOpt fake quant in PyTorch, MLP only | 0.3327 | 0.5117 | 0.929 / 0.933 | 0.919 | 0.0114 | 0.159 | 0.9826 | 0.0077 |

Variants (B200, TensorRT 11.4, one separate job; x vs the fp16 engine of that
job, 0.705 ms):

| Variant | x | prec / rec vs PyTorch | KL mean | AP |
|---|---|---|---|---|
| FP8 all 72 | 1.11 | 0.886 / 0.892 | 0.0236 | 0.3304 |
| query, key, value only (36) | 1.01 | 0.937 / 0.934 | 0.0114 | 0.3344 |
| attention output only (12) | 1.01 | 0.954 / 0.951 | 0.0066 | 0.3328 |
| intermediate dense only (12) | 1.03 | 0.949 / 0.948 | 0.0067 | 0.3350 |
| output dense only (12) | 1.04 | 0.955 / 0.956 | 0.0057 | 0.3342 |
| all 72, layers 0 and 11 kept fp16 | 1.09 | 0.901 / 0.902 | 0.0196 | 0.3324 |
| all 72, layers 10 and 11 kept fp16 | 1.08 | 0.897 / 0.898 | 0.0226 | 0.3322 |
| all 72, layers 0, 1, 10 and 11 kept fp16 | 1.06 | 0.917 / 0.915 | 0.0176 | 0.3349 |
| MLP only, layers 0 and 11 kept fp16 | 1.06 | 0.939 / 0.943 | 0.0083 | 0.3344 |

Every single kind of linear alone already changes 4.5% to 6.5% of the
detections at score 0.5, and keeping the edge layers in fp16 recovers little:
the loss is spread over the whole encoder.

Test image of the premerge case (native runtime, B200, TensorRT 11.4; the case
requires the same detections at score 0.5, IoU at least 0.5 and a score
difference of at most 0.05 against the fp32 reference):

| Engine | detections | min IoU | largest score difference |
|---|---|---|---|
| fp16 | 2 of 2 | 0.992 | 0.030 |
| FP8 all 72 | 2 of 2 | 0.906 | 0.147 (person 0.736 to 0.589) |

An FP8 premerge case with this check would need a score tolerance of about 0.45
(3x margin), and the person score is close to the 0.5 detection threshold, so
the count check would be fragile.

### Why the gain is small

Per encoder layer at 1829 tokens, the linears are about 55% of the FLOPs and the
attention matmuls about 45%; only the linears run in FP8. At batch 1 the call is
also bound by launches and the fp16 parts. Single kinds of linear give 1.01x to
1.04x each.

### NVFP4 (not tried)

The 72 quantized linears hold 21 M weights (21 MB in FP8, about 3 us of reads
at 8 TB/s) against a 0.56 to 0.64 ms call, so the call is not bound by weight
reads and a 4-bit weight format cannot help much. NVFP4 is also lossier than FP8,
which already fails the detection agreement.

### What was not tried

| Item | Note |
|---|---|
| FP8 attention matmuls (query x key, attention x value), percentile or entropy calibration, per-channel weight scales | not measured |
| Larger yolos checkpoints (base), batch above 1 (the family builds batch 1 only) | not measured |
| Full 5000-image COCO validation, H100 | not measured |

### When to revisit

- Batched serving: the family builds batch 1 only; at larger batches the
  linears weigh more and the FP8 gain may grow.
- A quality policy based on COCO AP alone (FP8 MLP only keeps AP at 0.3342 vs
  0.3344 for fp16).
- A recipe that keeps the detection agreement near fp16 (for example a
  calibration or scale granularity not measured here).

Noted, not a problem: the fp32 engine (TF32 on) is 3.7x to 3.9x slower than
fp16 at the same quality (AP 0.3344 for both).

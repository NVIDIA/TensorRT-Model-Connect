# DETR

This family builds DETR object detectors (`facebook/detr-resnet-50`) for
`image_to_boxes` as one TensorRT engine: ResNet-50 backbone, 1x1 input
projection, sine position embeddings, a 6-layer encoder and a 6-layer decoder
(d_model 256, 100 queries) and the class and box heads. The manifest builds
batch 1, fp16, with image capacity 1333 x 1333. The family has no quantized
build.

## FP8 quantization study (not shipped)

FP8 self-quantization of the transformer linears was built and measured on
TensorRT 11.4 (and 11.1) for `facebook/detr-resnet-50`. It is not shipped: the
best recipe (the encoder and decoder MLPs) is only 1.01x to 1.03x faster than
fp16 at batch 1, and detection agreement with PyTorch drops from 0.995 to 0.975
(precision and recall), although COCO AP does not change. This section records
the recipe, the method and the numbers, so that the study does not have to be
repeated. The FP8 code was not merged; this family has no FP8 build.

### Summary and decision

Recipe: FP8 for fc1 and fc2 of the 6 encoder and 6 decoder layers (24 linears).
TensorRT 11.4, batch 1, 800 x 1067 input (the range over 800 x 800, 800 x 1067
and 800 x 1333 is in brackets).

| Item | fp16 (manifest) | FP8 MLPs (fp16 engine) | FP8 vs fp16 |
|---|---|---|---|
| Latency, B200 | 1.223 ms | 1.192 ms | 1.03x (1.026x to 1.033x) |
| Latency, B300 | 1.113 ms | 1.098 ms | 1.01x (1.013x to 1.014x) |
| Plan size, B200 / B300 | 82.06 / 82.10 MB | 70.16 / 70.24 MB | -14.5% / -14.4% |
| COCO AP, B200 / B300 (PyTorch fp32 0.4241) | 0.4243 / 0.4248 | 0.4250 / 0.4242 | unchanged |
| Precision / recall vs PyTorch, B200 | 0.9951 / 0.9961 | 0.9764 / 0.9726 | -0.02 |
| Precision / recall vs PyTorch, B300 | 0.9953 / 0.9965 | 0.9756 / 0.9743 | -0.02 |
| Non-finite outputs | 0 | 0 | none |

Decision: do not ship FP8 for DETR.

- Speed: 1 to 3 percent on B200 and about 1 percent on B300. The model is
  launch bound: about 250 small plan layers, the GEMMs have at most 1050 rows, and
  the ResNet-50 convolutions (a third of the time) have no FP8 path in this
  family.
- Quality: AP is unchanged, but 2 to 3 percent of the detections at score 0.5
  differ from PyTorch (the fp16 engine: 0.5 percent).
- Every other part either loses quality (encoder attention, decoder
  cross-attention, box head), gives no speed (decoder self-attention, class
  head) or does not build on TensorRT 11.4 (attention output projection).
- The plan is 14.5 percent smaller; that alone is not a reason to ship.

### Recipe that was measured

No FP8 or NVFP4 checkpoint of DETR exists on the Hub, and no official
quantization config exists for it. The start was the guideline for small
encoders and vision models: all attention and MLP linears, per-tensor static
activation scales, ModelOpt max calibration. Each part was then measured alone,
and every part that loses quality or gives no speed was put back to fp16.

| Choice | Decision |
|---|---|
| Quantized (24 of 100 linears) | `fc1`, `fc2` of the 6 encoder and 6 decoder layers |
| fp16 | ResNet-50 backbone and 1x1 input projection (convolutions), attention q, k, v and `out_proj`, class and box heads, layer norms, position embedding, attention matmuls and softmax |
| Graph | `input -> Quantize -> Dequantize(fp16) -> MatMul`; weight `FP8 constant -> Dequantize(fp16)` |
| Weight scale | per tensor, amax / 448 of each linear, at build time (FP8 weights not stored) |
| Activation scale | static, per tensor; ModelOpt 0.46.1 `FP8_DEFAULT_CFG` (max calibration) on the transformers PyTorch model, 300 COCO validation images at native size, batch 1 |
| Outlier rule | a scale above 20x the median of its kind stays fp16; measured: none. The decoder layer-0 self-attention `v_proj` reads an all-zero input (the queries start at zero) and has no scale |
| Dequantize type | fp16; no non-finite value, fp32 Dequantize was not needed |

TensorRT picks real FP8 GEMMs for the MLPs (on B300:
`cutlass3x_sm100 ... gemm_e4m3_e4m3_f32_f16_f16 ... bias_f16_relu`, 36 FP8
layers in the plan).

### Method

- GPUs: B200 and B300, one GPU per job, all variants of one TensorRT version
  built and timed in the same job; TensorRT 11.4.0.106 (target) and 11.1.0.106.
  Family code at `94fcf143`, before the Task SDK migration (#2019).
- Data: the first 800 validation images of `detection-datasets/coco` with their
  COCO 2017 boxes. Calibration: every second image of the first 600 (300
  images). Quality: 500 other images (held out).
- Reference: transformers `DetrForObjectDetection` in fp32 with the same
  preprocessed input. AP is `pycocotools` COCO box AP on the 500 images
  (comparable only inside these tables). Precision and recall: detections at
  score 0.5 matched to the PyTorch detections (same class, IoU at least 0.5).
  Box IoU is the mean IoU of the matched boxes.
- Non-finite scan: the 500 images and noise, white and black inputs, every
  variant.
- Speed: batch 1, random input of the given size, 30 warm-up runs, then 200
  runs timed with CUDA events; 3 interleaved rounds over all variants, minimum
  per variant.

### Results

Speed and size, both MLPs (x is fp16 time over FP8 time):

| GPU | TensorRT | size | fp32 ms | fp16 ms | FP8 ms | x | plan MB fp16 to FP8 |
|---|---|---|---|---|---|---|---|
| B200 | 11.4 | 800x1067 | 2.160 | 1.223 | 1.192 | 1.026 | 82.06 to 70.16 |
| B200 | 11.4 | 800x1333 | 2.491 | 1.338 | 1.296 | 1.033 | |
| B200 | 11.4 | 800x800 | 1.734 | 1.113 | 1.078 | 1.032 | |
| B300 | 11.4 | 800x1067 | 2.012 | 1.113 | 1.098 | 1.013 | 82.10 to 70.24 |
| B300 | 11.4 | 800x1333 | 2.341 | 1.205 | 1.189 | 1.014 | |
| B300 | 11.4 | 800x800 | 1.617 | 1.013 | 1.000 | 1.013 | |
| B200 | 11.1 | 800x1067 | 2.154 | 1.253 | 1.223 | 1.025 | 82.09 to 70.24 |
| B300 | 11.1 | 800x1067 | 2.055 | 1.153 | 1.139 | 1.013 | 82.16 to 70.25 |

Each part alone (TensorRT 11.4, 500 held-out images; quality on B300, speed x
at 800 x 1067 on B200 / B300):

| Engine | FP8 linears | AP | prec / rec vs PyTorch | box IoU | x B200 / B300 |
|---|---|---|---|---|---|
| PyTorch fp32 | - | 0.4241 | - | - | - |
| fp16 engine | 0 | 0.4248 | 0.9953 / 0.9965 | 0.991 | 1 / 1 |
| encoder MLP | 12 | 0.4246 | 0.979 / 0.977 | 0.976 | 1.011 / 1.010 |
| decoder MLP | 12 | 0.4245 | 0.988 / 0.988 | 0.985 | 1.022 / 1.007 |
| **both MLPs (the recipe)** | 24 | **0.4242** | 0.976 / 0.974 | 0.974 | **1.026 / 1.013** |
| decoder self-attention q, k, v | 17 | 0.4248 | 0.988 / 0.990 | 0.984 | 0.995 / 0.996 |
| both MLPs + decoder self-attention | 41 | 0.4241 | 0.975 / 0.970 | 0.971 | 1.007 / 1.000 |
| encoder self-attention q, k, v | 18 | 0.4178 | 0.969 / 0.961 | 0.936 | 0.992 / 0.989 |
| decoder cross-attention q, k, v | 18 | 0.4177 | 0.945 / 0.943 | 0.917 | 0.997 / 0.995 |
| class head | 1 | 0.4233 | 0.981 / 0.988 | 0.989 | 1.004 / 1.003 |
| box head (3 linears) | 3 | 0.3760 | 0.921 / 0.922 | 0.845 | 1.005 / 1.000 |
| every part that builds (`out_proj` excluded) | 81 | 0.3666 | 0.861 / 0.849 | 0.835 | 0.999 / 0.990 |
| attention `out_proj` (TensorRT 11.1 only) | 18 | 0.4231 | 0.981 / 0.983 | 0.968 | 0.983 / 0.989 |

B200 quality is within 0.003 AP and 0.006 precision or recall of the B300 row;
11.1 gives the same picture. All variants were finite on every input.

Time split (B300, TensorRT 11.4, fp16, 800 x 1067, layer profiler, which adds
overhead): 2.37 ms in total, of which 0.80 ms (34 percent) is the fused ResNet
convolutions; the rest is spread over about 250 other plan layers of 10 to 50
microseconds each. The GEMMs
have 850 rows (800 x 1067) to 1050 rows (800 x 1333) and 256 or 2048 columns,
too small to keep the GPU busy, so FP8 saves only a few microseconds per GEMM.

### NVFP4 (not tried)

The transformer has 17.3 M weights (34.6 MB in fp16, a few microseconds of
reads), against a 1.0 to 1.4 ms run, so the model is not limited by weight
reads. FP8 gives 1 to 3 percent, and NVFP4 would lose more quality than FP8.

### What was not tried

| Item | Note |
|---|---|
| FP8 for the backbone convolutions | no FP8 convolution path in this family |
| Batch above 1 | the family builds batch 1 only |
| Percentile or entropy calibration, per-channel weight scales | not measured |
| Full 5000-image COCO validation, H100, native runtime with an FP8 bundle | not measured |

### When to revisit

- A larger batch or a larger DETR (wider d_model, more queries), where the
  transformer GEMMs are large enough for FP8 to pay.
- FP8 convolutions for the ResNet-50 backbone, which is a third of the time.
- A TensorRT release that builds FP8 `out_proj` behind the fused attention (11.4
  fails with "Could not find any implementation for node"; 11.1 builds it but
  gives no speed).

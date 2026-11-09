# SAM3

This family builds `facebook/sam3` for `text_prompted_segmentation` as a bundle
of TensorRT plans: the CLIP text encoder (24 layers, hidden 1024, 32 tokens),
the vision plan (ViT-L, 32 blocks, 1008 x 1008 image, plus the FPN necks), the
detector core (DETR encoder and decoder, mask head) and the video tracker plans.
Batch size is 1. The family builds `fp32` only: the text plan is fp32, and the
vision and core plans keep the residual stream, norms and activations in fp32
and run most GEMMs and attention in fp16. The family has no quantized build.

## FP8 quantization study (not shipped)

FP8 self-quantization was built and measured on TensorRT 11.4 (B200, B300) and
11.1 (B300). It is not shipped: only the text encoder keeps the mask quality,
and the text encoder is about 8% of the request time, so a request is only
1.02x faster. The vision plan, where the time is, loses mask quality with FP8.
This section records the recipe, the method and the numbers, so that the study
does not have to be repeated. The FP8 code was not merged; this family has no
FP8 build.

### Summary and decision

Measured FP8 build: q/k/v, `out_proj`, `fc1` and `fc2` of the 24 text encoder
blocks in FP8; vision plan, core and tracker plans unchanged.

| Item | default bundle | FP8 text encoder | FP8 vs default |
|---|---|---|---|
| Request time (text + vision + core), B200 / B300, TensorRT 11.4 | 18.601 / 17.118 ms | 18.215 / 16.759 ms | 1.02x / 1.02x |
| Text encoder time, B200 / B300, TensorRT 11.4 | 1.385 / 1.314 ms | 1.005 / 0.932 ms | 1.38x / 1.41x |
| Text plan size | 1348.7 MiB | 485.5 MiB | -64% |
| Bundle size (all plans) | 2769.3 MiB | 1905.9 MiB | -31% |
| Mask IoU vs PyTorch fp32, B200 / B300 (mean) | 0.9992 / 0.9975 | 0.9981 / 0.9973 | -0.001 / -0.0002 |
| Non-finite outputs | 0 | 0 | none |

Decision: do not ship FP8 for SAM3.

- Speed: the text encoder is 1.4x faster, but a request is 1.02x faster on
  B200 and B300 (11.1: 1.03x). This is the same level as the yolov10 study,
  which was not shipped.
- The text encoder runs 32 tokens per GEMM and is launch bound. A plain fp16
  text plan (no FP8) is faster than the FP8 one (0.806 ms against 1.005 ms on
  B200) and keeps the default quality. The family cannot build it today:
  `precision` is one value for the bundle and the vision plan refuses fp16.
- Vision plan: FP8 of its projections gives at most 1.11x for that plan and
  lowers the mask IoU from 0.9975 to 0.96. No subset of the projections keeps
  the quality.
- Detector core: about 470 small kernels, no kernel above 0.14 ms; not worth
  quantizing.
- The only clear gain is size: the bundle is 31% smaller.

### Recipe that was measured

No FP8 or NVFP4 checkpoint of SAM3 exists, and no official quantization config
exists for it. The recipe follows the public ModelOpt and Edge-LLM guidelines
for text encoders (attention and MLP linears quantized; embeddings, norms and
the output projection kept) and for multi-tower models (the image tower kept in
high precision unless it holds quality).

| Choice | Decision |
|---|---|
| Quantized (text encoder) | q, k, v, `out_proj`, `fc1`, `fc2` of the 24 blocks: 96 modules, 144 weight tensors; q, k and v share one Quantize on the layer-norm output |
| Not quantized | token and position embeddings, layer norms, attention matmuls, `text_projection`; the vision plan, the core and the tracker plans |
| Graph | `fp32 activation -> Cast fp16 -> Quantize -> Dequantize(fp16) -> MatMul -> Cast fp32`; residual stream, norms, GELU and bias adds stay fp32 |
| Weight scale | per tensor, amax / 448, at build time (nothing FP8 stored) |
| Activation scale | static, per tensor; ModelOpt 0.46.1 `FP8_DEFAULT_CFG` (max calibration) on the PyTorch `Sam3Model` text encoder with 304 prompts (80 COCO object names, each also with an adjective and with a context phrase, and 40 longer noun phrases), padded to 32 tokens |
| Outlier rule | a scale above 20x the median of its kind stays fp32; measured: no outlier (largest ratio 12.4x, `layers.0.mlp.fc2`) |
| Scale key | model config plus two weight sums, so another checkpoint does not reuse the table |
| Dequantize type | fp16; no non-finite value, fp32 Dequantize was not needed |

### Method

- GPUs: B200 and B300, one GPU per job, all variants built and timed in that
  job through the family's `build()`; TensorRT 11.4.0.106 (target) and
  11.1.0.106 (B300 only). Builder optimization level 1, precision fp32 (the
  family default).
- Reference: PyTorch fp32 `Sam3Model` (Transformers 5.17.0) on the same
  preprocessed pixels and tokens. The default bundle is the quality floor.
- Data: `detection-datasets/coco` validation, 400 images at a fixed stride;
  100 held-out images give 361 prompt and image pairs: 181 present-category
  prompts, 100 absent-category prompts and 80 pairs with noun phrases that are
  not in the calibration set. The calibration used no image.
- Quality: 288 x 288 mask logits, sigmoid, bilinear to the image size,
  threshold 0.5; score = sigmoid(logit) x sigmoid(presence), kept above 0.5;
  instances compared by query slot.
- Non-finite scan: the 361 pairs, the 304 calibration prompts plus the held-out
  phrases through the text plan, and five stress images (noise with std 1, 3 and
  10, all black, all white).
- Speed: the three plans of one request captured in one CUDA graph, median of
  9 interleaved rounds (default, FP8, fp16 text control). The fp16 text control
  is a scratch build of the text plan with `precision="fp16"`.

### Results

Speed, batch 1, ms (x is default time over FP8 time):

| GPU | TensorRT | request default | request FP8 | x | text default | text FP8 | text fp16 control | request with fp16 text control |
|---|---|---|---|---|---|---|---|---|
| B200 | 11.4 | 18.601 | 18.215 | 1.02 | 1.385 | 1.005 | 0.806 | 18.029 |
| B300 | 11.4 | 17.118 | 16.759 | 1.02 | 1.314 | 0.932 | 0.767 | 16.594 |
| B300 | 11.1 | 17.181 | 16.735 | 1.03 | 1.352 | 0.913 | 0.786 | 16.604 |

Parts on their own (CUDA graph, median of 5 rounds): vision plan 13.718 ms
(B200) and 12.585 ms (B300), core 3.732 ms and 3.347 ms, TensorRT 11.4.

Size, MiB (same on all three runs within 1 MiB): text plan 1348.7 to 485.5
(the 193 MiB token table stays fp32), bundle 2769.3 to 1905.9.

Quality against PyTorch fp32, 361 pairs:

| GPU, TensorRT | Build | mask IoU mean (p5) | semantic IoU | box IoU | score abs err | same instance count | top-10 logit abs err | text feature cos (min) |
|---|---|---|---|---|---|---|---|---|
| B200, 11.4 | default | 0.9992 (0.9964) | 0.9994 | 0.9985 | 0.0011 | 0.992 | 0.011 | 1.0000 (1.0000) |
| B200, 11.4 | FP8 text | 0.9981 (0.9959) | 0.9990 | 0.9973 | 0.0027 | 0.986 | 0.032 | 0.9988 (0.9968) |
| B200, 11.4 | fp16 text control | 0.9993 (0.9964) | 0.9995 | 0.9985 | 0.0011 | 0.992 | 0.010 | 1.0000 (1.0000) |
| B300, 11.4 | default | 0.9975 (0.9955) | 0.9994 | 0.9970 | 0.0020 | 0.995 | 0.014 | 1.0000 (1.0000) |
| B300, 11.4 | FP8 text | 0.9973 (0.9951) | 0.9992 | 0.9964 | 0.0034 | 0.997 | 0.036 | 0.9988 (0.9968) |
| B300, 11.4 | fp16 text control | 0.9975 (0.9954) | 0.9993 | 0.9971 | 0.0020 | 0.992 | 0.015 | 1.0000 (1.0000) |
| B300, 11.1 | default | 0.9988 (0.9952) | 0.9993 | 0.9983 | 0.0027 | 0.989 | 0.017 | 1.0000 (1.0000) |
| B300, 11.1 | FP8 text | 0.9987 (0.9946) | 0.9990 | 0.9983 | 0.0033 | 0.986 | 0.037 | 0.9989 (0.9971) |

FP8 text against the default engine: mask IoU 0.9986 (B200), 0.9979 (B300),
0.9980 (B300, 11.1); the fp16 text control: 0.9998 to 0.9999. The text feature
cosine on the held-out phrases is 0.9988, as on the calibration prompts. No
non-finite output in any build.

Kernels (plan inspector, text plan, B200): TensorRT 11.4 runs 96 e4m3 x e4m3
CUTLASS GEMMs (q, k and v fused into one GEMM, 4 per block); TensorRT 11.1 runs
144 (q, k and v separate). The 48 attention matmuls stay fp16.

Vision plan, FP8 of q/k/v, `fc1` and `fc2` of all 32 blocks (scratch, B300,
TensorRT 11.4, 281 pairs, text plan default): 11.24 ms against 12.48 ms (1.11x),
653 MiB against 1044 MiB, mask IoU 0.9606 against 0.9975. TensorRT 11.1: 11.55
ms against 12.53 ms (1.08x), mask IoU 0.9634 against 0.9990. Subsets (same job,
TensorRT 11.4):

| FP8 projections of the vision plan | vision ms | mask IoU | IoU >= 0.9 | score abs err |
|---|---|---|---|---|
| none (default) | 12.48 | 0.9975 | 0.9986 | 0.0020 |
| q/k/v | 12.19 | 0.9797 | 0.9788 | 0.0243 |
| `fc1` | 12.04 | 0.9745 | 0.9774 | 0.0249 |
| `fc2` | 11.86 | 0.9891 | 0.9929 | 0.0161 |
| `fc1` + `fc2` | 11.45 | 0.9709 | 0.9774 | 0.0237 |
| q/k/v + `fc1` + `fc2` | 11.21 | 0.9607 | 0.9605 | 0.0346 |
| all, first 4 blocks kept | 11.35 | 0.9729 | 0.9746 | 0.0285 |
| all, the 4 global-attention blocks kept | 11.36 | 0.9638 | 0.9676 | 0.0293 |

A fake-quant run of the same recipe on the PyTorch model (16 images) gives a
relative error of the deepest FPN map of 0.111 (the engine: 0.111; weights only
0.060, activations only 0.101; half the activation scale 0.158), so TensorRT
runs the recipe as written and the loss comes from static per-tensor FP8 on
this ViT-L.

### Why there is no gain

- The text encoder is about 8% of a request (1.3 ms of 17 to 18.6 ms). Even a
  free text encoder would give at most 1.08x.
- At 32 tokens the text GEMMs are small and the text plan is launch bound. The
  fp32 to fp16 change alone gives more (0.77 to 0.81 ms) than FP8 (0.91 to
  1.01 ms).
- The vision plan is about 73% of a request but does not hold mask quality in
  FP8, and a plain fp16 vision graph (scratch, no fp32 residual, norms or GELU)
  is 9.40 ms against 11.11 ms (1.18x) at mask IoU 0.9974, more than FP8 gives.

### NVFP4

Measured (scratch, B300, TensorRT 11.4, 281 pairs): NVFP4 MLP projections
(block-16 scales) with FP8 attention projections. Text encoder 1.037 ms against
0.927 ms for FP8 (slower), mask IoU 0.9954 against 0.9975; vision plan 11.39
ms against 11.24 ms for FP8, mask IoU 0.9337 (5th percentile 0.63). All outputs
finite. Not pursued: the text encoder is not weight-bandwidth bound at 32 rows,
and the vision plan is compute and elementwise bound at 5184 tokens.

### TensorRT note

On TensorRT 11.4 an FP8 text graph built in fp16 (scratch) fails to build when
`out_proj` is quantized (error `Input 0's element type (fp8_e4m3) differs from
input 1's element type (half)`); it builds on 11.1 (0.676 ms on B300) and on
11.4 without `out_proj` (0.745 ms on B200, 0.694 ms on B300). The fp32 graph of
the measured recipe builds on both versions.

### What was not tried

| Item | Note |
|---|---|
| Percentile or entropy calibration, per-channel weights for the vision plan | not measured |
| Batch greater than 1 (the family builds batch 1 only), the video tracker plans | not measured |
| H100, the native runtime with an FP8 bundle, other SAM3 checkpoints | not measured |
| Segmentation benchmark against ground truth | not measured; quality is against PyTorch fp32 |

### When to revisit

- The family gets an fp16 text plan or fp16 vision plan: FP8 would then have
  to beat those, not the fp32 default.
- A recipe that keeps the vision plan at the floor (mask IoU about 0.997) with
  a clear speedup; that plan is where the request time is.
- A caller that encodes many prompts per image: the text encoder share grows,
  and so does the FP8 gain.

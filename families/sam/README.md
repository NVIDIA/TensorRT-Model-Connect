# SAM

This family builds `facebook/sam-vit-base` (Segment Anything, ViT-B) as two
TensorRT engines: the image encoder and the mask decoder. The prompt encoder
runs on the CPU in the native runtime. The family default precision is fp32 and
the manifest builds fp16. Batch size is 1 and the encoder input is 1024x1024.

## FP8 quantization study (not shipped)

FP8 quantization of the image encoder was built and measured on TensorRT 11.4
and 11.1. It is not shipped: the encoder is only 1.04x to 1.05x faster than
fp16, and the masks get worse. This section records the recipe, the method and
every number, so that the study does not have to be repeated. The FP8 code was
not merged. All numbers come from a local study; this repository has no FP8
build for SAM.

### 1. Summary and decision

| Item | fp16 (manifest build) | FP8, all 4 projections | Change |
|---|---|---|---|
| Encoder time, B200, TensorRT 11.4 | 4.215 ms | 4.027 ms | 1.05x faster |
| Encoder time, B300, TensorRT 11.4 | 4.012 ms | 3.856 ms | 1.04x faster |
| Mask IoU vs PyTorch fp32, point prompts (mean) | 0.9881 | 0.9681 | -0.020 |
| Mask IoU vs PyTorch fp32, box prompts (mean) | 0.9891 | 0.9791 | -0.010 |
| Embedding relative L2 error vs PyTorch fp32 | 0.0037 | 0.0327 | 8.8x larger (12x on B300, TensorRT 11.4: 0.0027 vs 0.0326) |
| Best-score mask same as PyTorch, point prompts | 97.6% | 94.8% | -2.8 points |
| Encoder plan size, B200 | 190.0 MiB | 108.6 MiB | -43% |

Decision: do not ship FP8 for the SAM image encoder.

- Speed: 1.04x to 1.05x over fp16 on B200 and B300, TensorRT 11.1 and 11.4.
  The 1.5x gain over the fp32 family default comes from fp16, not from FP8.
- Quality: the embedding error is 9x to 12x larger than fp16 (B200, B300), the mean mask IoU
  is 0.01 to 0.02 lower, and the 5th percentile IoU against the fp16 engine is
  0.87 (point) and 0.92 (box).
- No subset of the projections reaches the fp16 quality (section 5).
- The only clear gain is the 43% smaller encoder plan.
- fp16 alone keeps the quality floor and is already 1.5x faster than the fp32
  default.

### 2. Model and engines

| Item | Value |
|---|---|
| Checkpoint | `facebook/sam-vit-base`, revision `70c1a07f894ebb5b307fd9eaaee97b9dfc16068f` |
| Engines | image encoder, mask decoder |
| Prompt encoder | CPU, native runtime (numpy port in the study) |
| Family default precision | fp32 |
| Manifest precision | fp16 |
| Batch size | 1 (the family supports batch 1 only) |
| Encoder input | 1024x1024 |
| Encoder model config | hidden 768, 12 layers, MLP 3072, 12 heads, patch 16, image 1024, window 14 |
| Embedding output | 256x64x64 |

### 3. The recipe and the design choices

No official FP8 config exists for SAM. The recipe follows the ModelOpt and
Edge-LLM guidelines for small vision encoders and the other ViT families in
this repository: quantize the linear layers of the encoder blocks, keep
everything else in fp16.

Every quantized projection is built as:

```
activation -> Quantize(FP8) -> Dequantize(fp16) -> MatMul <- Dequantize(fp16) <- FP8 weight constant
```

| Choice | Decision | Reason |
|---|---|---|
| Quantized projections | q, k, v, attention output (`attn.proj`), `mlp.lin1`, `mlp.lin2` in all 12 blocks: 48 per model | These are the GEMMs of the encoder. They have 4096 to 4900 rows. |
| Patch embedding, position embedding | fp16 | Public ViT recipes (ModelOpt) exclude the patch embedding. It is a convolution, not a linear projection. |
| Layer norms | fp16 | Not a GEMM. A Q/DQ on the layer-norm output was not tried (section 7). |
| Attention matmuls and relative-position bias | fp16 | Not weight GEMMs. A fixed softmax-output scale was not tried (section 7). |
| Neck convolutions | fp16 | Convolutions, not linear projections. Not measured separately. |
| Mask decoder | fp16, not quantized | Its GEMMs have 7 rows or 128 to 256 columns, and the decoder is about 0.3 ms (7% of one image plus one prompt). FP8 cannot speed it up. |
| Prompt encoder | CPU, not quantized | Runs outside TensorRT. |
| Activation scale | static, per tensor, one per projection input | Max calibration with ModelOpt `FP8_DEFAULT_CFG`, 300 COCO images, scale = amax / 448. A static scale needs no amax computation at run time. |
| Weight scale | per tensor, amax / 448 at build time | 448 is the largest finite e4m3 value. Each weight keeps its own scale. |
| Weight storage | FP8 weights are quantized from the original checkpoint at build time and never stored | No FP8 checkpoint exists for SAM. A checkpoint that is already quantized is rejected. |
| Dequantize type | fp16 | The FP8 GEMM then writes fp16 and the bias add fuses into its epilogue. No non-finite output was found in the NaN scan (section 4). fp32 Dequantize was not measured for SAM. |
| q, k, v input scale | one shared scale, one Quantize/Dequantize pair | q, k and v read the same layer-norm output. Each of the three weights keeps its own weight scale. |
| Attention output projection | quantized | On TensorRT 11.4 and 11.1 the encoder attention is built from explicit matmul and softmax layers, so the projection input is a normal GEMM input and it builds as FP8. Other families with fused attention cannot quantize this projection on 11.4. Retest on newer releases. |
| Outlier rule | a scale above 20x the median of the same projection in the other blocks is stored as `null` and stays fp16 | A massive activation loses most values with one 8-bit scale. Measured: no outlier, 48 of 48 projections quantized. |
| Missing scale, unknown model config | build error | A partial table would build a half-quantized engine without any error. |
| Precision | FP8 needs an fp16 engine | The Q/DQ scales and the Dequantize layers are fp16 and the activation must be fp16. The fp32 default build has no FP8 variant. |
| Tensor parallelism | not supported with FP8 | Not built. |

The activation-scale table is keyed by the model config (the values above) plus
two numeric weight sums (patch embedding and block 0 query), so a different
checkpoint of the same architecture needs its own calibration.

### 4. How the results were measured

Speed: all encoder variants were built and timed in one process per job, on
the same GPU, batch 1, 1024x1024 input. Time is the median of 5 runs of 40
replays of a CUDA graph. "fp32 default" is the family default build, "fp16" is
the manifest build and the control.

Quality data and prompts:

| Item | Value |
|---|---|
| Source | COCO 2017 validation split, first shard of `detection-datasets/coco`, 800 images at a fixed stride |
| Calibration | 300 images |
| Evaluation | 350 held-out images, disjoint from the 300 calibration images |
| Objects | 543: the largest and a mid-size object per image (area at least 2500 pixels and at most 90% of the image; the mid-size object only when the image has 3 or more such objects) |
| Prompts | each object prompted twice: a foreground point at the box center, and the box |
| Reference | PyTorch fp32 `SamModel` (not ground truth) |
| Masks | the 3 multimask outputs, upsampled to the original image size, threshold 0 |
| Decoder | encoder variants use the fp16 mask decoder; the "fp32" row uses the fp32 decoder |

Metrics (each prompt gives 3 masks; values are means over prompts unless noted):

| Metric | Definition |
|---|---|
| Mask IoU vs PyTorch | IoU of each binary mask against the PyTorch fp32 mask, averaged over the 3 masks. Mean and 5th percentile over prompts. |
| Mask IoU vs fp16 engine | Same, with the fp16 engine masks as reference. |
| Logit relative error | norm(engine logits - PyTorch logits) / norm(PyTorch logits), at 256x256. |
| Score absolute difference | mean absolute difference of the 3 predicted IoU scores. |
| Best-score mask agreement | fraction of prompts where the highest-scored of the 3 masks is the same index as in PyTorch. |
| Embedding relative L2 error | norm(engine embedding - PyTorch embedding) / norm(PyTorch embedding), 256x64x64, mean over 343 images that have at least one prompted object. |
| Non-finite scan | count of NaN and Inf in the encoder output on the 350 real images, and on random noise at std 1, 3 and 10 and two constant images. |

Numerical checks (embedding error, logit error) show how far the engine moves
from PyTorch. Semantic checks (mask IoU, best-score agreement) show whether the
masks a user sees change. Both are reported.

Limits:

- The reference is PyTorch fp32, not ground truth. Mask IoU against COCO
  annotations was not measured.
- Only the largest and mid-size objects were prompted. Small objects were not
  evaluated.
- The native runtime supports point prompts only. Box prompts were evaluated by
  feeding the box embeddings to the same decoder engine in a Python harness.
  The harness used the PyTorch prompt encoder; the CPU prompt encoder of the
  native runtime matches it within 1.2e-7 (max absolute difference of the point
  embedding).
- The native runtime with an FP8 bundle was not run.

### 5. Results

#### Speed

| GPU | TensorRT | fp32 default (ms) | fp16 (ms) | FP8 (ms) | FP8 vs fp16 | FP8 vs fp32 default |
|---|---|---|---|---|---|---|
| B200 | 11.4 | 6.302 | 4.215 | 4.027 | 1.05x | 1.56x |
| B300 | 11.4 | 5.939 | 4.012 | 3.856 | 1.04x | 1.54x |
| B200 | 11.1 | 6.607 | 4.723 | 4.533 | 1.04x | 1.46x |
| B300 | 11.1 | 6.311 | 4.441 | 4.267 | 1.04x | 1.48x |

The speedup over the fp32 default comes from fp16 (fp16 alone is 1.5x). Mask
decoder, B200, TensorRT 11.4: 0.326 ms (fp32 default), 0.315 ms (fp16).

#### Ablations

B200, TensorRT 11.4. Quality is point prompts, mask IoU against PyTorch.

| Variant | Quantized projections | Encoder (ms) | Mask IoU mean (5th pct) | Embedding rel. L2 error | e4m3 mentions in plan | Plan size (MiB) |
|---|---|---|---|---|---|---|
| fp16 | none | 4.215 | 0.9881 (0.954) | 0.0037 | 0 | 190.0 |
| FP8, all 4 | q, k, v, attn.proj, lin1, lin2 | 4.027 | 0.9681 (0.887) | 0.0327 | 120 | 108.6 |
| FP8, no attn proj | q, k, v, lin1, lin2 | 4.052 | 0.9706 (0.905) | 0.0303 | 96 | 115.4 |
| FP8, MLP only | lin1, lin2 | 4.083 | 0.9707 (0.893) | 0.0286 | 72 | 136.2 |

Dropping the attention output projection, or all attention projections (MLP only), costs
0.025 ms to 0.056 ms and does not bring the embedding error near fp16. No
subset of the projections reaches the fp16 floor.

#### Quality, point prompts

B200, TensorRT 11.4, 543 prompts x 3 masks. The B300 run agrees within 0.004 in
mean IoU, and so do the TensorRT 11.1 runs (100 images).

| Encoder build | Mask IoU vs PyTorch, mean (5th pct) | Mask IoU vs fp16 engine, mean (5th pct) | Logit rel. error vs PyTorch | Score abs. diff | Best-score mask same as PyTorch |
|---|---|---|---|---|---|
| fp32 default | 0.9957 (0.986) | 0.9877 (0.953) | 0.0027 | 0.0008 | 99.8% |
| fp16 (floor) | 0.9881 (0.954) | - | 0.0090 | 0.0035 | 97.6% |
| FP8, all 4 | 0.9681 (0.887) | 0.9665 (0.874) | 0.0225 | 0.0078 | 94.8% |
| FP8, no attn proj | 0.9706 (0.905) | 0.9695 (0.891) | 0.0211 | 0.0070 | 95.9% |
| FP8, MLP only | 0.9707 (0.893) | 0.9686 (0.889) | 0.0208 | 0.0068 | 95.2% |

#### Quality, box prompts

Same setup.

| Encoder build | Mask IoU vs PyTorch, mean (5th pct) | Mask IoU vs fp16 engine, mean (5th pct) | Logit rel. error vs PyTorch | Score abs. diff | Best-score mask same as PyTorch |
|---|---|---|---|---|---|
| fp32 default | 0.9966 (0.986) | 0.9889 (0.954) | 0.0022 | 0.0007 | 99.8% |
| fp16 (floor) | 0.9891 (0.952) | - | 0.0089 | 0.0028 | 97.2% |
| FP8, all 4 | 0.9791 (0.924) | 0.9773 (0.923) | 0.0152 | 0.0044 | 95.6% |
| FP8, no attn proj | 0.9802 (0.934) | 0.9787 (0.924) | 0.0149 | 0.0044 | 96.7% |
| FP8, MLP only | 0.9800 (0.918) | 0.9785 (0.919) | 0.0141 | 0.0041 | 97.1% |

The fp16 floor is already about 0.01 below fp32 in mask IoU, mostly from the
fp16 mask decoder (0.9887 alone). FP8 adds another 0.02 (point) or 0.01 (box)
of mean IoU on top of it.

#### Parts checked separately

B200, TensorRT 11.4.

| Part | Check | fp32 | fp16 | FP8 |
|---|---|---|---|---|
| Image encoder | embedding relative L2 error vs PyTorch fp32 | 0.0014 | 0.0037 | 0.0327 (MLP only 0.0286) |
| Mask decoder alone (PyTorch fp32 embedding in) | mask IoU vs PyTorch, point / box | 0.9964 / 0.9966 | 0.9887 / 0.9891 | not quantized |
| Prompt encoder (CPU, numpy port) | max abs difference of the point embedding vs Hugging Face | - | - | 1.2e-7, not quantized |

Non-finite values: none in the encoder output on 350 real images, nor on noise
at std 1, 3 and 10 and on constant images (fp16 Dequantize).

#### Plan size and kernel evidence

| GPU | TensorRT | fp16 plan | FP8 plan | Change |
|---|---|---|---|---|
| B200 | 11.4 | 190.0 MiB | 108.6 MiB | -43% |
| B300 | 11.4 | 178.2 MiB | 96.8 MiB | -46% |

The plan inspector shows 120 e4m3 kernel mentions in the all-4 FP8 plan and none
in the fp16 plan, so the 48 projections really run as e4m3 GEMMs (the count
drops to 96 without the attention output projection and to 72 for MLP only).

#### Where the time goes

FP8 on all 48 linears saves 0.19 ms of the 4.2 ms fp16 encoder (B200, 11.4), so
the cap on the gain is set by the rest of the encoder. The encoder GEMMs have
4096 to 4900 rows, so the encoder is compute and attention bound, not
weight-bandwidth bound. A per-layer time split (attention, GEMMs, layer norm,
neck) was not measured for this family.

### 6. NVFP4

NVFP4 was not built. The encoder is compute and attention bound (GEMMs with
4096 to 4900 rows), and FP8 on all linears saves only 0.19 ms of 4.2 ms. NVFP4
can only change the same GEMM share, and it is lossier than FP8 (3x to 10x worse KL in earlier LLM families). With FP8 already at the quality limit, NVFP4 was
not expected to help.

### 7. What was not tried

| Item | Why it might matter | Expectation |
|---|---|---|
| Percentile calibration | Smaller activation scales for heavy-tailed inputs | Not measured. Max calibration was used. |
| Q/DQ on the layer-norm output (ModelOpt ViT recipe) | One shared Q/DQ before q, k, v and the MLP | Not measured. Our q, k, v already share one. |
| Fixed softmax-output scale (ModelOpt ViT recipe) | Quantizes the attention probabilities | Not measured. |
| FP8 mask decoder | Smaller decoder plan | Speed cannot change: the decoder is 0.3 ms with 7 row GEMMs. |
| Skipping the first or last blocks, per-layer sensitivity sweep | Could remove the worst layers | Not measured. |
| Mask IoU against COCO annotations | Ground-truth quality | Not measured. |
| H100 | Other GPU | Not measured. |
| `sam-vit-large`, `sam-vit-huge` | Larger encoders, more GEMM time | Not measured, no scales calibrated. |

We expect that these options may narrow the quality gap, but quality is
unlikely to return to the fp16 level: even MLP-only FP8 has the same embedding
error as all four projections. Speed would stay at about 1.05x, because FP8 already
changes all 48 GEMMs and saves 0.19 ms.

### 8. How to reproduce

The FP8 implementation was not merged, so the study cannot be rerun from this
repository. The numbers come from a local study with TensorRT 11.4.0.106 and
11.1.0.106, ModelOpt 0.46, on B200 and B300 (one GPU each, all variants built
in the same job).

Calibration:

1. Fetch 800 images of the COCO 2017 validation split (first shard of
   `detection-datasets/coco`) at a fixed stride, with their object boxes.
2. Load the original `facebook/sam-vit-base` in PyTorch fp32 with transformers.
3. Run ModelOpt `FP8_DEFAULT_CFG` (max calibration) over the image encoder with
   300 images (every second image), in batches of 4, with the checkpoint's own
   preprocessing (resize to 1024, pad, normalize).
4. Store `input_quantizer.amax / 448` of the 48 projections under the model
   config key. Fail if any projection is missing. Apply the 20x-median outlier
   rule (nothing was excluded).

Recipe knobs: the projection list, the activation scale source (static,
per tensor), the weight scale (amax / 448, per tensor), the Dequantize type
(fp16) and the outlier ratio (20). The ablations remove `attn.proj`, or all
attention projections, from the list.

Measurement: build all variants in one process on one GPU; time the median of
5 x 40 replays of a CUDA graph with batch 1 and a random 1024x1024 input;
inspect the plan for e4m3 kernels; evaluate masks as described in section 4
against a PyTorch fp32 `SamModel`.

Code needed: a Quantize/Dequantize insertion for the q, k, v, attention output,
`mlp.lin1` and `mlp.lin2` matmuls of every encoder block, a build-time weight
quantizer, an activation-scale table with a model-config key, a calibration
tool, and an evaluation harness for the metrics above.

### 9. When to revisit

- A recipe that holds mask quality: embedding error close to fp16 (0.0037) and
  mask IoU against the fp16 engine near the fp16-vs-fp32 level.
- Batch support above 1 in the family. It was not measured; the GEMMs already
  have 4096 or more rows at batch 1, so a larger gain from batch alone is not
  expected.
- A TensorRT release where FP8 gives a larger gain on the encoder (retest the
  speed table).
- A need for the 43% smaller encoder plan that outweighs the mask quality loss.
- Larger checkpoints (`sam-vit-large`, `sam-vit-huge`), after calibration.

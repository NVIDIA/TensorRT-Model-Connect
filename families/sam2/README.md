# SAM2

This family builds SAM2.1 Hiera-S (`facebook/sam2.1-hiera-small`) for
`video_segmentation` as six TensorRT plans: one image plan (Hiera backbone, FPN
neck, box head) and five tracker plans (the prompt plan for the first frame and
four recurrent plans for 1 to 4 history frames, each with memory attention,
memory encoder and mask decoder). The family precision is `mixed_bf16_fp32`,
batch size is 1 and the image plan input is 1024x1024. The family has no
quantized build.

## FP8 quantization study (not shipped)

FP8 self-quantization of the Hiera image encoder was built and measured on
TensorRT 11.4 and 11.1. It is not shipped: the image plan is only 1.02x faster
than the bf16 default on B200 (1.15x on B300), it is slower than or close to an
fp16 build of the same graph, a whole frame gains 1.01x to 1.06x, and the masks
get clearly worse. This section records the recipe, the method and the numbers,
so that the study does not have to be repeated. The FP8 code was not merged.
All numbers come from a local study; this repository has no FP8 build for SAM2.

### 1. Summary and decision

TensorRT 11.4, batch 1. "fp16 control" is the same graph with bf16 replaced by
fp16 (experiment only, not a family build).

| Item | bf16 (family default) | fp16 control | FP8 image encoder | FP8 vs default / vs fp16 control |
|---|---|---|---|---|
| Image plan, B200 | 1.976 ms | 1.877 ms | 1.935 ms | 1.02x / 0.97x |
| Image plan, B300 | 2.024 ms | 1.893 ms | 1.758 ms | 1.15x / 1.08x |
| Frame (image plan + recurrent H=4 plan), B200 | 4.558 ms | 4.477 ms | 4.517 ms | 1.01x / 0.99x |
| Frame (image plan + recurrent H=4 plan), B300 | 4.540 ms | 4.368 ms | 4.274 ms | 1.06x / 1.02x |
| Mask IoU vs PyTorch fp32, COCO box prompt, mean (min) | 0.9954 (0.861) | 0.9991 (0.988) | 0.9788 (0.399) | -0.017 mean |
| Image plan size, B300 | 85.6 MiB | 77.3 MiB | 55.7 MiB | -35% |
| Non-finite outputs | 0 | 0 | 0 | none |

Decision: do not ship FP8 for SAM2.

- Speed: no gain on B200 (1.02x over bf16, 0.97x of fp16), and a frame gains
  at most 1.06x (B300). On TensorRT 11.1 the image plan is 1.05x (B200) and
  1.07x (B300) faster than bf16 and 0.99x of fp16.
- Quality: mean mask IoU drops from 0.9954 to 0.9788 against PyTorch, the worst
  image drops to 0.399, and the FPN features have 4x to 6x the bf16 error. No
  subset of the projections that still gives a speedup keeps the quality
  (section 5).
- The tracker plans (memory attention, memory encoder, mask decoder) are not
  faster in FP8 (section 5), so the bundle only shrinks by about 8%.
- An fp16 build of the same graph is 5% to 8% faster than the bf16 default on
  the image plan and has a higher mask IoU (0.9991 vs 0.9954). It is a better
  first step than FP8 (not part of this study's code).

### 2. Model and plans

| Item | Value |
|---|---|
| Checkpoint | `facebook/sam2.1-hiera-small`, revision `6c381d9c16faed5e8a7c4a2cd99918bdca8316e4` (the public snapshot with the family's synthetic box head) |
| Plans | image plan; prompt plan; recurrent plans H=1, 2, 3, 4 |
| Family precision | `mixed_bf16_fp32` |
| Batch size | 1 (the family builds batch 1 only) |
| Image encoder model config | Hiera, embed dim 96, 16 blocks in 4 stages (1, 2, 11, 2), neck dim 256, image 1024 |
| Image plan outputs used by the tracker | FPN levels 0, 1, 2 |

### 3. The recipe and the design choices

No FP8 or NVFP4 checkpoint and no official quantization config exist for SAM2.
The recipe follows public ViT FP8 recipes (ModelOpt) and the FP8 builds of the
`siglip`, `dinov3` and `timm_vit` families: quantize the linear layers of the
encoder blocks, keep the rest in the family precision.

Every quantized projection is built as:

```
activation -> Quantize(FP8) -> Dequantize(bf16) -> MatMul <- Dequantize(bf16) <- FP8 weight constant
```

followed by the bias add.

| Choice | Decision | Reason |
|---|---|---|
| Quantized projections (51) | `attn.qkv`, `mlp.layers.0`, `mlp.layers.1` of the 16 Hiera blocks, and the shortcut `proj` of the three stage-transition blocks (1, 3, 14) | The GEMMs of the image encoder. |
| `attn.proj` | bf16 | TensorRT 11.4 cannot compile an FP8 Quantize after the fused attention of the image plan ("No matching rules found for input operand types" at the softmax, then no implementation for the encoder node). Not probed on 11.1. |
| Patch embedding, position embedding, layer norms, attention, FPN neck, box head | bf16 | Not linear projections of the blocks, or (box head) synthetic in the public snapshot. |
| Tracker plans | bf16 | No speedup in a timing experiment (section 5). |
| `attn.qkv` | one weight scale | The fused q, k, v module is one weight in the checkpoint. |
| Activation scale | static, per tensor | ModelOpt `FP8_DEFAULT_CFG` (max calibration), 400 COCO images, scale = amax / 448. |
| Weight scale | per tensor, amax / 448 at build time | The FP8 weights are quantized from the original checkpoint at build time. |
| Dequantize type and scale type | bf16 Dequantize, fp32 scale constants | The family runs bf16, whose range is the fp32 range, so the fp16 overflow case does not exist. bf16 scale constants were measured: no better quality, slower on B300, faster on B200 but still not faster than fp16 (section 5). |
| Outlier rule | a scale above 20x the median of its kind is stored as `null` and stays bf16 | Measured max / median: `qkv` 1.9, `layers.0` 2.3, `layers.1` 6.8, `proj` 1.0; nothing was excluded. |
| Missing scale, unknown model config, already-quantized checkpoint | build error | A partial table would build a half-quantized plan without any error. |

The activation-scale table was keyed by the model config plus two weight sums,
so a different checkpoint of the same architecture needs its own calibration.

### 4. How the results were measured

Speed: all variants of one table were built and timed in one job on the same
GPU (one job per GPU type), batch 1. Time is the median of 5 runs of 40
replays, plan only (no pre- or post-processing). "Frame" is the image plan
plus the recurrent H=4 plan.

Quality data and prompts:

| Item | Value |
|---|---|
| Source | COCO 2017 validation images from `detection-datasets/coco`, 800 images at a fixed stride |
| Calibration | 400 images |
| Evaluation | 300 held-out images, disjoint from the calibration images |
| Prompt | the largest object box of each image, on frame 0 |
| Video | 6 windows of 5 frames (stride 3) of the 200-frame `sam2` demo clip "bedroom", 3 objects per window (18), boxes from point prompts of the reference model |
| Reference | PyTorch fp32 `sam2` 1.1.0 video predictor, no autocast, same preprocessed pixels as the plans (not ground truth) |
| Mask | mask logits > 0 at 256x256 |

Metrics:

| Metric | Definition |
|---|---|
| Mask IoU vs PyTorch | IoU of the binary mask against the PyTorch fp32 mask. Mean and minimum over prompts. |
| Mask logit cosine | cosine of the 256x256 mask logits against PyTorch. |
| FPN relative L2 | norm(engine - PyTorch) / norm(PyTorch) of each FPN level of the image plan. |
| Non-finite scan | NaN and Inf count in mask logits, object pointer and memory features, all 300 images and 90 video frames. |

### 5. Results

#### Speed

| GPU | TensorRT | bf16 image plan (ms) | fp16 control (ms) | FP8 (ms) | FP8 vs bf16 / fp16, image plan | FP8 vs bf16 / fp16, frame |
|---|---|---|---|---|---|---|
| B200 | 11.4 | 1.976 | 1.877 | 1.935 | 1.02x / 0.97x | 1.01x / 0.99x |
| B300 | 11.4 | 2.024 | 1.893 | 1.758 | 1.15x / 1.08x | 1.06x / 1.02x |
| B200 | 11.1 | 2.167 | 2.044 | 2.069 | 1.05x / 0.99x | 1.02x / 0.99x |
| B300 | 11.1 | 2.121 | 1.960 | 1.984 | 1.07x / 0.99x | 1.03x / 0.99x |

Tracker plans, TensorRT 11.4 (the FP8 variant does not change them):

| GPU | Plan | bf16 (ms) | fp16 control (ms) |
|---|---|---|---|
| B200 | prompt | 0.648 | 0.627 |
| B200 | recurrent H=4 | 2.582 | 2.600 |
| B300 | prompt | 0.619 | 0.597 |
| B300 | recurrent H=4 | 2.516 | 2.475 |

#### Quality, COCO box prompt

B300, TensorRT 11.4, 300 held-out images. B200 gives the same FP8 numbers and
bf16 / fp16 numbers within 0.004.

| Image plan | Mask IoU vs PyTorch, mean (min) | Mask logit cosine vs PyTorch, mean (min) | FPN relative L2, level 0 / 1 / 2 |
|---|---|---|---|
| bf16 (default) | 0.9954 (0.861) | 1.0000 (0.992) | 0.015 / 0.034 / 0.017 |
| fp16 control (floor) | 0.9991 (0.988) | 1.0000 (1.000) | 0.003 / 0.007 / 0.004 |
| FP8 | 0.9788 (0.399) | 0.9968 (0.524) | 0.063 / 0.168 / 0.099 |
| FP8, bf16 scale constants | 0.9776 (0.394) | 0.9946 (0.254) | 0.064 / 0.169 / 0.099 |
| NVFP4 (experiment) | 0.9201 (0.019) | 0.9908 (-0.097) | 0.233 / 0.565 / 0.383 |

Mask IoU of FP8 against the fp16 control: 0.9789 mean, 0.399 min.

#### Quality, video propagation

B300, TensorRT 11.4, mean mask IoU vs PyTorch per frame (18 objects; frames 1
to 4 use the recurrent plans):

| Frame | bf16 (default) | fp16 control | FP8 | NVFP4 (experiment) |
|---|---|---|---|---|
| 0 | 0.990 | 0.999 | 0.978 | 0.890 |
| 1 | 0.991 | 0.997 | 0.975 | 0.909 |
| 2 | 0.993 | 0.997 | 0.973 | 0.907 |
| 3 | 0.976 | 0.982 | 0.962 | 0.917 |
| 4 | 0.979 | 0.996 | 0.960 | 0.910 |

The B200 run gives the same FP8 row, but lower bf16 and fp16 rows (0.959 to
0.974): on frame 0 their minimum IoU is 0.48 while the minimum logit cosine
stays 0.987, so a borderline mask flips in the B200 bf16 and fp16 plans. With 18 objects the video table
is a sanity check, not a quality metric; the COCO table is the main one.

#### Recipe variants and sensitivity

Fake quantization with ModelOpt (100 images, mask IoU against the unquantized
PyTorch model, mean (min)):

| Quantized | Mask IoU |
|---|---|
| all 51 projections (the recipe) | 0.9828 (0.720) |
| `qkv` only | 0.9927 (0.846) |
| `layers.0` only | 0.9892 (0.778) |
| `layers.1` only | 0.9917 (0.908) |
| shortcut `proj` only | 0.9934 (0.902) |
| stage 0 only (block 0) | 0.9878 (0.550) |
| stage 3 only (blocks 14, 15) | 0.9983 (0.980) |
| blocks 4 to 13 | 0.9904 (0.838) |
| all but one block (16 runs) | 0.9797 to 0.9859 |

The loss is spread over all blocks and all projection kinds; there is no single
outlier layer. Only the last stage keeps the mask IoU near 0.998, but it is 2 of
16 blocks at the lowest resolution, a small part of the GEMM work (not timed as
a plan).

Other variants (TensorRT 11.4, image plan):

| Variant | B200 (ms) | B300 (ms) | Plan size B300 | Note |
|---|---|---|---|---|
| FP8 recipe | 1.935 | 1.758 | 55.7 MiB | |
| FP8, bf16 scale constants | 1.896 | 1.785 | 55.7 MiB | quality no better (table above) |
| NVFP4, W4A4 block 16, same 51 projections | 2.208 | 2.284 | 43.4 MiB | slower than bf16 |
| FP8 on all tracker linears (dummy scales, timing only), B300 | - | prompt 0.703 (bf16 0.619), recurrent H=4 2.554 (bf16 2.515) | | slower |
| FP8 on memory attention only, B300 | - | recurrent H=4 2.551 (bf16 2.515) | | slower |

#### Where the time goes

Kernel profile of the 40 largest kernels, B300, TensorRT 11.4 (ms per run):

| Plan | GEMM and convolution | fused attention | other (layer norm, position embedding, elementwise) |
|---|---|---|---|
| image plan, bf16 | 0.874 | 0.352 | 0.536 |
| image plan, FP8 | 0.497 | 0.352 | 0.467 |
| recurrent H=4, bf16 | 0.355 | 1.594 | 0.371 |

- In the image plan about half of the kernel time is GEMM and convolution,
  one fifth is fused attention and the rest is layer norm, position embedding
  and elementwise work. FP8 can only shorten the GEMM share and adds Q/DQ work.
  The FP8 plan holds e4m3 GEMM kernels (`gemm_f8_f8_f32_bf16`).
- One kernel resizes and tiles the Hiera position embedding on every inference
  (0.13 ms of 2.0 ms on B300), although it depends only on constants.
- The recurrent plan is bound by fused attention over up to 16k memory tokens
  (1.6 ms of 2.3 ms), so FP8 on its linears cannot help.

#### Plan size

| GPU | bf16 image plan | FP8 image plan | Change |
|---|---|---|---|
| B200 | 83.7 MiB | 55.2 MiB | -34% |
| B300 | 85.6 MiB | 55.7 MiB | -35% |

The six-plan set on B300 shrinks from 381.3 MiB to 351.4 MiB (-8%), as the
tracker plans are unchanged.

### 6. NVFP4

NVFP4 (W4A4, block 16, on the same 51 projections) was built as an experiment
and dropped. The image encoder is compute and elementwise bound at batch 1, not
weight-bandwidth bound: the NVFP4 plan is slower than bf16 (2.208 ms vs 1.976
ms on B200, 2.284 ms vs 2.024 ms on B300), and its FPN error is about 4x the FP8
error (mask IoU 0.920, min 0.019). All outputs were finite.

### 7. What was not tried

| Item | Expectation |
|---|---|
| `attn.proj` in FP8 | Needs a TensorRT release that compiles an FP8 Quantize after the fused attention. The rest of the encoder stays limited by layer norm, attention and elementwise work. |
| Percentile or entropy calibration, Q/DQ on the layer-norm output | Not measured. The sensitivity table shows the loss is spread over all layers, so a large gain is not expected. |
| Batch above 1 | Not measured; the family builds batch 1 only. |
| The delivered package with the real box head | Not calibrated; its weights differ, so the scale key differs. Detection quality was not evaluated. |
| Point prompts, mask IoU against COCO annotations, H100 | Not measured. |
| Native C++ runtime with an FP8 bundle | Not run; the FP8 plans have the same inputs and outputs and were run through the Python TensorRT runtime. |

### 8. How to reproduce

The FP8 implementation was not merged, so the study cannot be rerun from this
repository. The numbers come from a local study with TensorRT 11.4.0.106 and
11.1.0.106, ModelOpt 0.46.1, torch 2.14, `sam2` 1.1.0, on B200 and B300 (one
GPU each, all variants built in the same job).

Calibration:

1. Fetch 800 COCO 2017 validation images of `detection-datasets/coco` with
   their boxes.
2. Load `facebook/sam2.1-hiera-small` in PyTorch fp32 with the `sam2` package.
3. Run ModelOpt `FP8_DEFAULT_CFG` (max calibration) over the image encoder with
   400 images at a fixed stride, with the predictor's own preprocessing.
4. Store `input_quantizer.amax / 448` of the 51 projections under the model
   config key. Fail if any projection is missing. Apply the 20x-median outlier
   rule (nothing was excluded).

Code needed: a Quantize/Dequantize insertion for the 51 Hiera projections in
the image plan builder, a build-time weight quantizer, an activation-scale
table with a model-config key, a calibration tool, and an evaluation harness
that feeds the same preprocessed pixels to the plans and to the PyTorch video
predictor.

### 9. When to revisit

- A recipe that holds mask quality: mean mask IoU against PyTorch near the bf16
  default (0.995) with no collapsed images.
- A TensorRT release that builds `attn.proj` in FP8 after the fused attention,
  or gives a larger FP8 gain on B200 (retest the speed table on both GPUs).
- A change of the family default to fp16 (measured 5% to 8% faster than bf16 on
  the image plan): FP8 must then beat fp16, which it does not on 11.4 B200 or
  11.1.
- Batch support above 1 in the family.

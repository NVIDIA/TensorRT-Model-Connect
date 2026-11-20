# SegFormer

This family builds SegFormer semantic segmentation models (SegFormer-B0 to B5,
for example `nvidia/segformer-b0-finetuned-ade-512-512`) for the
`segmentation` task as one TensorRT engine: input `pixel_values` (batch 1, the
checkpoint input size), output `logits` (class scores at 1/4 of the input
resolution). The encoder is a hierarchical transformer (overlap patch
embedding, efficient self-attention with sequence reduction, Mix-FFN); the
decode head is a light all-MLP head. The b0 premerge manifest uses fp16. The
family has no quantized build.

## FP8 quantization study (not shipped)

FP8 self-quantization of the encoder linears was built and measured on TensorRT
11.4 (and 11.1) for `nvidia/segformer-b0-finetuned-ade-512-512` and
`nvidia/segformer-b5-finetuned-ade-640-640`. It is not shipped: the plan is 19%
(b0) to 28% (b5) smaller, but latency is 0.97x to 1.00x of fp16 at batch 1 on
B200 and B300, and 1% to 2% of the pixels change class against the fp16
engine. This section records the recipe, the method and the numbers, so that
the study does not have to be repeated. The FP8 code was not merged; this
family has no FP8 build.

### Summary and decision

TensorRT 11.4, batch 1, B200 / B300 (x is fp16 time over FP8 time):

| Item | fp16 (manifest) | FP8 (fp16 engine) | FP8 vs fp16 |
|---|---|---|---|
| b0 512 latency | 0.456 / 0.433 ms | 0.471 / 0.437 ms | 0.97x / 0.99x |
| b5 640 latency | 3.749 / 3.484 ms | 3.874 / 3.498 ms | 0.97x / 1.00x |
| b0 / b5 plan size (B200) | 10.1 / 178.3 MiB | 8.1 / 129.2 MiB | -19% / -28% |
| b0 / b5 ADE20K mIoU (B200) | 0.3260 / 0.4482 | 0.3285 / 0.4486 | +0.003 / +0.000 |
| b0 / b5 pixels equal to the fp16 engine (B200) | 1.0000 | 0.9790 / 0.9915 | 2.1% / 0.9% changed |
| Non-finite outputs | 0 | 0 | none |

Decision: do not ship FP8 for SegFormer.

- Speed: at batch 1 every encoder matmul is small (hidden size 32 to 512) and
  takes about 9 to 13 us in fp16 and in FP8 alike (layer profiler). The FP8
  GEMM saves no time, and the extra Quantize and layout kernels make FP8
  slightly slower.
- Quality: mIoU against the ground truth does not drop, but 2.1% (b0) and 0.9%
  (b5) of the pixels change class against the fp16 engine (mIoU of the FP8 map
  against the fp16 map: 0.87 and 0.95). The fp32 engine changes 0.05%.
- The only gain is plan size (2 MiB for b0, 49 MiB for b5).

### Recipe that was measured

No FP8 or NVFP4 SegFormer checkpoint exists on the Hub (`nvidia/` and the model
repos), and no official quantization config exists for it. The recipe follows
the vision encoder guidance used for the `timm_swin` FP8 build: quantize the
linears of the transformer blocks, keep convolutions, norms, attention and the
head. Each quantized projection is built as
`input -> Quantize -> Dequantize(fp16) -> MatMul`, with the weight
`FP8 constant -> Dequantize(fp16)`.

| Choice | Decision |
|---|---|
| Quantized, per encoder block | query (`attn.q`), key and value (`attn.kv`, one shared input scale, own weight scales), Mix-FFN `mlp.fc1` and `mlp.fc2`: 40 matmuls in b0 (8 blocks), 260 in b5 (52 blocks) |
| fp16 | attention output projection, overlap patch embedding, sequence-reduction and depthwise convolutions, layer norms, attention; the decode head runs in fp32 as in the fp16 build |
| Attention output projection | stays fp16: TensorRT 11.4 cannot compile an FP8 Quantize after the fused attention (the Quantize moves into the attention and an fp8 operand reaches the softmax: "Could not infer output types for operation: softmax"; b0 and b5, B200 and B300) |
| Weight scale | per tensor, amax / 448 of the original weight, at build time; FP8 weights are not stored |
| Activation scale | static, per tensor; ModelOpt 0.46.1 `FP8_DEFAULT_CFG` (max calibration), 200 ADE20K validation images; stored per model config (depths, hidden sizes, heads, reduction ratios, input size and a weight fingerprint) |
| Outlier rule | a scale above 20x the median of the same projection in the other blocks stays fp16; measured: no outlier (largest ratio 3.3, `mlp.fc2` of b5; largest input amax 38) |
| Dequantize type | fp16; no non-finite value, fp32 Dequantize was not needed |

### Method

- GPUs: B200 and B300, one GPU per job; TensorRT 11.4.0.106 (target) and
  11.1.0.106. All variants of a model were built and timed in the same job.
- Variants: fp32 (the build default), fp16 (the manifest precision) and fp16
  with FP8, from the same graph code.
- Data: ADE20K validation (`zhoubolei/scene_parse_150`, 2000 images). Images
  0, 10, ..., 1990 (200) were used for calibration; quality uses 300 of the
  other 1800 images, resized to the model input size like the engine input.
- Quality: mIoU and pixel accuracy against the ground truth labels (150
  classes, logits upsampled to the label size); pixel agreement and mIoU of
  each engine's class map against the fp16 engine's class map.
- Non-finite scan: the logits of the 300 held-out images.
- Speed: batch 1 (the only batch the family builds), CUDA graph replay and
  eager enqueue, median of 5 interleaved rounds of 300 runs.
- Checkpoints: b0 at the manifest revision; b5 from the Hub (only a
  `pytorch_model.bin`, converted to safetensors for the study).

### Results

Speed and size (x is fp16 time over FP8 time, CUDA graph; eager in the last
latency column):

| Model | GPU | TensorRT | fp32 ms | fp16 ms | FP8 ms | x | x eager | plan MiB fp16 to FP8 |
|---|---|---|---|---|---|---|---|---|
| b0 512 | B200 | 11.4 | 0.607 | 0.456 | 0.471 | 0.97 | 0.99 | 10.1 to 8.1 |
| b0 512 | B300 | 11.4 | 0.574 | 0.433 | 0.437 | 0.99 | 0.97 | 10.0 to 8.2 |
| b5 640 | B200 | 11.4 | 5.276 | 3.749 | 3.874 | 0.97 | 0.97 | 178.3 to 129.2 |
| b5 640 | B300 | 11.4 | 5.007 | 3.484 | 3.498 | 1.00 | 0.99 | 174.4 to 125.3 |
| b0 512 | B200 | 11.1 | 0.619 | 0.488 | 0.499 | 0.98 | 1.00 | 10.2 to 8.3 |
| b0 512 | B300 | 11.1 | 0.578 | 0.453 | 0.461 | 0.98 | 0.99 | 10.2 to 8.3 |
| b5 640 | B200 | 11.1 | 5.408 | 3.950 | 4.017 | 0.98 | 0.98 | 178.6 to 129.6 |
| b5 640 | B300 | 11.1 | 5.063 | 3.534 | 3.586 | 0.99 | 0.98 | 175.4 to 126.4 |

An earlier run of the same code on TensorRT 11.4 gave the same picture (FP8
0.97x to 1.00x of fp16).

Quality, 300 held-out images, TensorRT 11.4, B200 (B300 in brackets where it
differs; TensorRT 11.1 is within 0.001):

| Model | Engine | mIoU | pixel accuracy | pixels equal to fp16 engine | mIoU vs fp16 engine map | non-finite |
|---|---|---|---|---|---|---|
| b0 | fp32 | 0.3260 | 0.7737 | 0.9995 | 0.997 (0.996) | 0 |
| b0 | fp16 | 0.3260 | 0.7737 | 1 | 1 | 0 |
| b0 | FP8 | 0.3285 (0.3279) | 0.7740 (0.7733) | 0.9790 (0.9794) | 0.872 (0.876) | 0 |
| b5 | fp32 | 0.4482 | 0.8210 | 0.9997 | 0.998 | 0 |
| b5 | fp16 | 0.4482 (0.4481) | 0.8210 | 1 | 1 | 0 |
| b5 | FP8 | 0.4486 (0.4491) | 0.8215 | 0.9915 (0.9918) | 0.946 (0.947) | 0 |

mIoU against the ground truth stays the same, but the FP8 class map moves away
from the fp16 engine. The b0 premerge case compares the engine class map with
the PyTorch reference (pixel agreement at least 0.99, mIoU at least 0.94). On
the held-out images the FP8 b0 map is at 0.979 and 0.87 against the fp16 map,
so an FP8 b0 bundle would likely fail that case (not run).

Kernel time per group, TensorRT 11.4, batch 1, B200 (layer profiler, ms
(count); the profiler adds a fixed cost per layer, so only the comparison
between the columns is meaningful):

| Kernel group | b0 fp16 | b0 FP8 | b5 fp16 | b5 FP8 |
|---|---|---|---|---|
| matmuls | 0.410 (42) | 0.415 (42) | 2.531 (261) | 2.523 (261) |
| attention | 0.085 (8) | 0.086 (8) | 0.765 (52) | 0.763 (52) |
| convolutions | 0.243 (20) | 0.247 (20) | 1.399 (107) | 1.495 (107) |
| GELU pointwise (with the next Quantize in FP8) | 0.062 (8) | 0.063 (8) | 0.557 (52) | 0.559 (52) |
| norms and other pointwise | 0.253 (29) | 0.242 (29) | 1.359 (160) | 1.358 (160) |
| layout and Quantize | 0.134 (46) | 0.166 (50) | 0.901 (97) | 0.639 (63) |

The matmul time does not change: in b5 the matmul kernels take 8.6 to 12.7 us
each in both engines. With hidden sizes of 32 to 512 these GEMMs are too small
to gain from FP8 tensor cores at batch 1. B300 shows the same picture (b0
matmuls 0.417 fp16, 0.421 FP8; b5 2.504 fp16, 2.500 FP8).

### NVFP4 (not tried)

The matmuls are the same speed in FP8 and fp16, so a lower weight precision
cannot make them faster at batch 1; NVFP4 would only reduce plan size further,
with more quality loss.

### What was not tried

| Item | Note |
|---|---|
| FP8 attention output projection | does not compile on TensorRT 11.4 (see the recipe table) |
| Batch above 1, tensor-parallel FP8 | the family builds batch 1 only |
| SegFormer-B1 to B4, H100, the native runtime with an FP8 bundle, the full 2000-image validation set | not measured |
| Percentile or entropy calibration | not measured |

### When to revisit

- A use case that needs larger batches or larger inputs, where the encoder
  matmuls become compute bound.
- A plan size limit that matters more than latency (the only measured gain).

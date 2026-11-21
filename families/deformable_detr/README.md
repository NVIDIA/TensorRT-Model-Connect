# Deformable DETR

This family builds `SenseTime/deformable-detr` (ResNet-50 backbone, 4 feature
levels, 6 encoder and 6 decoder layers, d_model 256, 300 queries, 91 classes)
for object detection as one TensorRT engine: batch 1, dynamic height and width
up to the resize longest edge (optimization point 800 x 800), fp16 or fp32 (the
manifest uses fp16 at 800 x 800). Multi-scale deformable attention is built from
explicit matmul, softmax and GridSample layers. The family has no quantized
build.

## FP8 quantization study (not shipped)

FP8 self-quantization of the encoder and decoder linears was built and measured
on TensorRT 11.4 (B200, B300) and 11.1 (B200). It is not shipped: latency stays
at the fp16 level (1.01x at 800 x 800, 0.97x at 800 x 1067), the plan is 8% to 9%
smaller and COCO AP drops by 0.2 point. All linears together take 1% to 3% of the
engine time, so FP8 has no room to pay. This section records the recipe, the
method and the numbers, so that the study does not have to be repeated. The FP8
code was not merged; this family has no FP8 build.

### Summary and decision

| Item | fp16 (manifest) | FP8, 72 linears (fp16 engine) | FP8 vs fp16 |
|---|---|---|---|
| Latency 800 x 800, B200 / B300 (TensorRT 11.4) | 5.889 / 5.816 ms | 5.850 / 5.764 ms | 1.01x / 1.01x |
| Latency 800 x 1067, B200 / B300 (TensorRT 11.4) | 14.747 / 14.487 ms | 15.230 / 14.861 ms | 0.97x / 0.97x |
| Plan size, B200 / B300 | 83.0 / 84.9 MiB | 75.8 / 78.0 MiB | -9% / -8% |
| COCO AP, B200 / B300 (PyTorch fp32 0.4653) | 0.4644 / 0.4641 | 0.4625 / 0.4619 | -0.2 point |
| Non-finite outputs | 0 | 0 | none |

Decision: do not ship FP8 for Deformable DETR.

- Speed: the GEMMs of all 100 linears take 0.19 to 0.21 ms of a 7.4 to 16.2 ms
  layer-profiled run. FP8 saves 0.04 to 0.11 ms of GEMM time, which is within
  the run-to-run noise.
- At 800 x 1067 FP8 is slower: the six largest fused kernels (the encoder
  deformable-attention weighted sums) gain a multiply and cast, which is the
  Quantize of the `output_proj` input, and each goes from 0.59 to 0.68 ms on B200
  (+0.5 ms in all).
- The time is in GridSample (2.1 ms at 800 x 800, 9.7 ms at 800 x 1067), the
  fused normalization and elementwise kernels (4.3 to 5.5 ms) and the backbone
  convolutions (0.8 ms). None of these is changed by FP8 linears.
- The only gain is plan size (7 MiB less), which is not a reason to ship.

### Recipe that was measured

No FP8 or NVFP4 checkpoint of Deformable DETR exists on the Hub, and no official
quantization config exists for it. The recipe follows the small-encoder and ViT
practice (FP8 for the projection and MLP linears, keep the sensitive parts in
fp16), adapted per part by measurement. Each quantized linear is
`input -> Quantize -> Dequantize(fp16) -> MatMul -> + bias`, with the weight as
an FP8 constant and `Dequantize(fp16)`.

| Part | Linears | Decision |
|---|---|---|
| Backbone (ResNet-50), input projections | convolutions | fp16 (no FP8 convolution path in this family) |
| Encoder | `value_proj`, `output_proj`, `fc1`, `fc2` (6 layers, 24 linears) | FP8 |
| Encoder | `sampling_offsets`, `attention_weights` | fp16 (each alone matches fp16; most sensitive, about 0.6 MiB to save) |
| Decoder self-attention | `q_proj`, `k_proj`, `v_proj`, `o_proj` | FP8 |
| Decoder cross-attention | `value_proj`, `output_proj` | FP8 |
| Decoder cross-attention | `sampling_offsets`, `attention_weights` | fp16 (non-finite output when both are FP8) |
| Decoder FFN | `fc1`, `fc2` | FP8 (decoder: 48 linears in all) |
| Heads | `class_embed`, `bbox_embed` (3 layers) | fp16 (box IoU vs fp16 0.947, AP -1.0 point) |

| Choice | Decision |
|---|---|
| Weight scale | per tensor, amax / 448, at build time |
| Activation scale | static, per tensor; ModelOpt 0.46.1 `FP8_DEFAULT_CFG` (max calibration) on the Hugging Face PyTorch model, 256 COCO 2017 validation images (batch 1, native preprocessing, no padding) |
| Outlier rule | an input scale above 20x the median of its kind stays fp16; measured: none triggered |
| Dequantize type | fp16; no non-finite value with the recipe, fp32 Dequantize was not needed |

### Method

- GPUs: B200 and B300, one GPU per job, all variants of a job built and timed on
  the same GPU; TensorRT 11.4.0.106 (target) and 11.1.0.106 (B200 only).
- Quality: 400 COCO 2017 validation images of `detection-datasets/coco`,
  disjoint from the 256 calibration images (800 images at a fixed stride).
  Reference: Hugging Face `DeformableDetrForObjectDetection` in fp32 on the same
  preprocessed input. AP is `pycocotools` COCO AP (IoU 0.50 to 0.95) of the top
  100 detections per image against a ground truth without the `iscrowd` flag, so
  it is comparable only inside these tables. Agreement: detections with score
  above 0.3 matched by label and IoU 0.5 (precision / recall), mean box IoU of
  those queries, and correlation of the class logits.
- Non-finite scan: logits and boxes of all 400 images.
- Speed: batch 1, `execute_async_v3`, CUDA events, median of 200 runs after 30
  warm-up runs, one variant after the other in the same process. 800 x 800 is
  the build size; 800 x 1067 is the usual 4:3 COCO size on the same engine.
- Kernel time: TensorRT layer profiler, ms per run (sum above the wall time).
- Code: the FP8 build was written on top of `main` at `94fcf143`; the family's
  graph code has not changed since. The B300 run (November 2026) used the same
  code and harness as the B200 runs.

### Results

Speed and size (x is fp16 time over FP8 time):

| GPU | TensorRT | fp32 ms 800x800 / 800x1067 | fp16 ms | FP8 ms | x 800x800 / 800x1067 | plan MiB fp16 to FP8 |
|---|---|---|---|---|---|---|
| B200 | 11.4 | 6.626 / 13.988 | 5.889 / 14.747 | 5.850 / 15.230 | 1.01 / 0.97 | 83.0 to 75.8 |
| B300 | 11.4 | 6.419 / 13.622 | 5.816 / 14.487 | 5.764 / 14.861 | 1.01 / 0.97 | 84.9 to 78.0 |
| B200 | 11.1 | - | 5.889 / 14.766 | 5.841 / 15.223 | 1.01 / 0.97 | 82.8 to 76.3 |

Quality of the recipe (vs PyTorch fp32 unless noted):

| GPU, TensorRT | Engine | AP | AP50 | logit corr | precision / recall | box IoU | corr / prec / rec vs fp16 engine |
|---|---|---|---|---|---|---|---|
| - | PyTorch fp32 | 0.4653 | 0.6411 | - | - | - | - |
| B200, 11.4 | fp32 engine | 0.4652 | 0.6417 | 1.0000 | 0.999 / 0.999 | 0.999 | - |
| B200, 11.4 | fp16 engine | 0.4644 | 0.6409 | 0.9918 | 0.970 / 0.941 | 0.972 | - |
| B200, 11.4 | FP8 engine | 0.4625 | 0.6391 | 0.9914 | 0.969 / 0.931 | 0.968 | 0.9991 / 0.984 / 0.975 |
| B300, 11.4 | fp16 engine | 0.4641 | 0.6407 | 0.9918 | 0.970 / 0.941 | 0.972 | - |
| B300, 11.4 | FP8 engine | 0.4619 | 0.6392 | 0.9913 | 0.970 / 0.930 | 0.966 | 0.9991 / 0.986 / 0.974 |
| B200, 11.1 | fp16 engine | 0.4636 | 0.6405 | 0.9918 | 0.971 / 0.943 | 0.972 | - |
| B200, 11.1 | FP8 engine | 0.4620 | 0.6374 | 0.9913 | 0.971 / 0.932 | 0.967 | 0.9991 / 0.985 / 0.974 |

Per part (only that part in FP8; TensorRT 11.4, B200, one separate job with fp16
AP 0.4639):

| Variant | Linears | AP | corr vs fp16 | precision / recall vs fp16 | box IoU vs fp16 |
|---|---|---|---|---|---|
| fp16 engine | 0 | 0.4639 | 1 | 1 / 1 | 1 |
| encoder `value_proj` + `output_proj` | 12 | 0.4632 | 0.9995 | 0.991 / 0.985 | 0.986 |
| encoder FFN | 12 | 0.4648 | 0.9998 | 0.991 / 0.990 | 0.991 |
| encoder `sampling_offsets` | 6 | 0.4639 | 1.0000 | 0.995 / 0.996 | 0.994 |
| encoder `attention_weights` | 6 | 0.4645 | 0.9999 | 0.994 / 0.995 | 0.994 |
| decoder self-attention | 24 | 0.4637 | 0.9999 | 0.992 / 0.994 | 0.996 |
| decoder cross `value_proj` + `output_proj` | 12 | 0.4647 | 0.9999 | 0.993 / 0.995 | 0.992 |
| decoder cross `sampling_offsets` + `attention_weights` | 12 | 0.0309 | non-finite | 0 / 0 | non-finite |
| decoder FFN | 12 | 0.4636 | 0.9999 | 0.991 / 0.992 | 0.996 |
| class and box heads | 4 | 0.4536 | 0.9995 | 0.991 / 0.982 | 0.947 |
| recipe | 72 | 0.4637 | 0.9991 | 0.986 / 0.976 | 0.983 |
| all 100 linears | 100 | 0.0309 | non-finite | 0 / 0 | non-finite |

The decoder cross `sampling_offsets` alone (AP 0.4644) and `attention_weights`
alone (AP 0.4639) are finite (B200 11.4); only the pair gives non-finite output.
The cause was not isolated.

Kernel time per group (layer profiler, ms per run, fp16 to FP8):

| GPU, TensorRT, size | GEMM | GridSample | other fused kernels | convolution | sum |
|---|---|---|---|---|---|
| B200, 11.4, 800 x 800 | 0.193 to 0.150 | 2.056 to 2.058 | 4.321 to 4.412 | 0.791 to 0.790 | 7.36 to 7.41 |
| B200, 11.4, 800 x 1067 | 0.209 to 0.149 | 9.684 to 9.686 | 5.449 to 6.053 | 0.839 to 0.843 | 16.18 to 16.73 |
| B300, 11.4, 800 x 800 | 0.189 to 0.090 | 2.036 to 2.040 | 4.340 to 4.450 | 0.784 to 0.784 | 7.35 to 7.36 |
| B300, 11.4, 800 x 1067 | 0.201 to 0.093 | 9.520 to 9.517 | 5.442 to 5.972 | 0.825 to 0.827 | 15.99 to 16.41 |
| B200, 11.1, 800 x 1067 | 0.409 to 0.073 | 9.708 to 9.705 | 5.228 to 6.084 | 0.843 to 0.837 | 16.19 to 16.70 |

The "other fused kernels" growth at 800 x 1067 is the six encoder
deformable-attention sum kernels (0.59 to 0.68 ms each on B200, 0.59 to 0.66 ms
on B300), which also compute the `output_proj` Quantize in the FP8 engine.

Side finding (not FP8): at 800 x 1067 the fp16 engine is slower than the fp32
engine (B200 14.75 vs 13.99 ms, B300 14.49 vs 13.62 ms). GridSample takes 9.7 ms
in fp16 against 8.0 ms (B200) and 7.8 ms (B300) in fp32, and 2.1 ms at the
800 x 800 build size, so a 4:3 input costs 4.7x the GridSample time for 1.33x
the pixels.

### NVFP4 (not tried)

The linears are 1% to 3% of the engine time and FP8 gives no speedup, so a 4-bit
weight cannot help. The convolutions have no NVFP4 path in TensorRT.

### What was not tried

| Item | Note |
|---|---|
| FP8 convolutions (backbone, input projections, about 11% of the time at 800 x 800) | the family has no FP8 convolution path |
| fp32 Dequantize for the decoder cross `sampling_offsets` + `attention_weights` pair | not measured |
| Percentile or entropy calibration, batch above 1 (the family builds batch 1 only) | not measured |
| Full 5000-image COCO validation, H100, GB300, native runtime with an FP8 bundle | not measured |

### When to revisit

- A faster GridSample (or a fused deformable-attention kernel) in TensorRT that
  makes the linears a larger share of the time.
- A TensorRT release that keeps the Quantize out of the deformable-attention sum
  kernel or makes it free there, so that 800 x 1067 is not slower.
- A family change that adds FP8 convolutions to the backbone.

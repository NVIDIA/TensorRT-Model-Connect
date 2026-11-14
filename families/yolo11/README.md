# YOLO11

This family builds the YOLO11 detector `yolo11n` (`yolo11n.pt` of
`Ultralytics/YOLO11`) for `object_detection` as one TensorRT engine: 640 x 640
input, batch 1, fp16 or fp32 (the manifest uses fp16). The head is not end to
end: the engine reports every anchor with its box and strongest class, and the
runtime drops weak scores and runs class-aware non-max suppression. The family
has no quantized build.

## FP8 quantization study (not shipped)

FP8 self-quantization of the convolutions was built and measured on TensorRT
11.4 (and 11.1) for `yolo11n`. It is not shipped: FP8 is slower than fp16
(0.90x to 0.93x at batch 1, 0.88x to 0.89x at batch 8), the plan saves only
about 2 MiB, and COCO AP drops by 1.2 points. This section records the recipe,
the method and the numbers, so that the study does not have to be repeated.
The FP8 code was not merged; this family has no FP8 build.

### Summary and decision

| Item | fp16 (manifest) | FP8 (fp16 engine) | FP8 vs fp16 |
|---|---|---|---|
| Latency, batch 1, B200 / B300 (TensorRT 11.4) | 0.433 / 0.400 ms | 0.467 / 0.443 ms | 0.93x / 0.90x |
| Latency, batch 8, B200 / B300 (TensorRT 11.4) | 0.802 / 0.754 ms | 0.901 / 0.854 ms | 0.89x / 0.88x |
| Plan size, B200 | 6.56 MiB | 4.65 MiB | -29% |
| COCO AP, B200 (PyTorch fp32 0.4161) | 0.4158 | 0.4034 | -1.2 points |
| Detections that match PyTorch (prec / rec) | 0.998 / 0.998 | 0.904 / 0.905 | about 10% changed |
| Non-finite outputs | 0 | 0 | none |

Decision: do not ship FP8 for YOLO11.

- Speed: the FP8 convolutions run on FP8 kernels, but bias, SiLU and the
  Quantize of the next convolution are not fused into the FP8 convolution. They
  run as extra pointwise kernels (45 in the fp16 plan, 64 in the FP8 plan; 137
  to 159 plan layers) and take the gain back. `yolo11n` is small, so it is
  launch and pointwise bound, and FP8 ends up slower than fp16.
- Depthwise convolutions have no FP8 kernel on TensorRT 11.4 (the build fails
  with "no tactics"). `yolo11n` has 7 of them (the position convolution of the
  attention block and 6 in the class branch of the head); they stay fp16.
- The only gain is plan size (about 2 MiB less).
- The sibling families `yolov8` and `yolov10` were measured with the same
  recipe; see their READMEs.

### Recipe that was measured

No FP8 or NVFP4 checkpoint of any YOLO detector exists on the Hub, and no
official quantization config exists for one. The recipe is the one of the
`yolov10` study: quantize the dense convolutions, keep the input convolution.
Each quantized convolution is built as
`input -> Quantize -> Dequantize(fp16) -> Convolution(+bias)`, with the kernel
`FP8 constant -> Dequantize(fp16, axis 0)`.

| Choice | Decision |
|---|---|
| Quantized (77 of 87 convolutions) | every dense convolution whose input and output channels are multiples of 16: Conv blocks, C3k2, C2PSA attention (qkv, projection, feed-forward), SPPF, head box and class convolutions |
| fp16 | stem `model.0` (3 input channels), the 2 convolutions with 8 channels in `model.2`, the 7 depthwise convolutions, DFL decode, top-1 class |
| Weight scale | per output channel, amax / 448 of the batch-norm-folded kernel, at build time |
| Activation scale | static, per tensor; max calibration (the statistic ModelOpt `FP8_DEFAULT_CFG` keeps) on the family's PyTorch reference, 256 letterboxed COCO 2017 validation images |
| Outlier check | 88 convolution inputs, median amax 11.4, largest 147 (13x): no outlier |
| Dequantize type | fp16; no non-finite value, fp32 Dequantize was not needed |

### Method

- GPUs: B200 and B300, one GPU per job; TensorRT 11.4.0.106 (target) and
  11.1.0.106 (B200).
- Quality: COCO 2017 validation images 300 to 799 of `detection-datasets/coco`
  (500 held out; 0 to 255 are the calibration set), letterboxed like the
  runtime. AP is `pycocotools` COCO AP (IoU 0.50 to 0.95), comparable only
  inside these tables. Agreement: detections at score 0.25 matched to the
  PyTorch fp32 detections (same class, IoU at least 0.5, NMS IoU 0.7 on both
  sides).
- Non-finite scan: the 500 images and 3 stress inputs (noise, black, white).
- Speed: CUDA graph replay, median of 9 interleaved rounds, all variants of a
  row in one job. Batch 8 is a scratch build (the family builds batch 1 only).

### Results

Speed and size (x is fp16 time over FP8 time):

| GPU | TensorRT | fp32 ms | fp16 ms | FP8 ms | x batch 1 | batch 8 fp16 / FP8 ms | x batch 8 | plan MiB fp16 to FP8 |
|---|---|---|---|---|---|---|---|---|
| B200 | 11.4 | 0.592 | 0.433 | 0.467 | 0.93 | 0.802 / 0.901 | 0.89 | 6.56 to 4.65 |
| B300 | 11.4 | 0.553 | 0.400 | 0.443 | 0.90 | 0.754 / 0.854 | 0.88 | 6.68 to 4.66 |
| B200 | 11.1 | - | 0.434 | 0.469 | 0.92 | 0.795 / 0.908 | 0.88 | 6.61 to 4.62 |

Quality, B200, TensorRT 11.4 (B300 and TensorRT 11.1 within 0.001 AP):

| Engine | AP | AP50 | prec / rec vs PyTorch | box IoU | score error |
|---|---|---|---|---|---|
| PyTorch fp32 | 0.4161 | 0.5572 | - | - | - |
| fp32 engine | 0.4163 | 0.5573 | 0.999 / 0.999 | 0.9994 | 0.0005 |
| fp16 engine | 0.4158 | 0.5551 | 0.998 / 0.998 | 0.995 | 0.0010 |
| FP8 engine | 0.4034 | 0.5464 | 0.904 / 0.905 | 0.951 | 0.047 |
| FP8 without the six last head convolutions | 0.4075 | 0.5501 | 0.917 / 0.900 | 0.960 | 0.045 |

The variant without the six last head convolutions (71 quantized) runs at 0.92x
(B200) and 0.90x (B300) of fp16 at batch 1, so keeping the head in fp16 does
not help either.

### NVFP4 (not tried)

TensorRT does not support NVFP4 on convolutions (4-D tensors), and the model
has no large matmul (the C2PSA attention block is small). The engine is launch
bound, not weight-bandwidth bound, so 4-bit weights would not help.

### What was not tried

| Item | Note |
|---|---|
| Percentile or entropy calibration, per-tensor weight scales | not measured |
| A SiLU form that TensorRT might fuse into the FP8 convolution epilogue | not measured |
| Larger widths (s, m, l, x), full 5000-image COCO validation, H100, native runtime with an FP8 bundle | not measured; the family builds only `yolo11n` |

### When to revisit

- A TensorRT release that fuses bias, SiLU and the next Quantize into the FP8
  convolution epilogue (as it does for most fp16 convolutions), or that adds
  FP8 depthwise convolution kernels.
- The family adds a larger width: on `yolov10x` the same recipe reached fp16
  speed with a 43% smaller plan, so a large YOLO11 may be worth a new
  measurement.

Batch 8 gave a lower ratio than batch 1, so larger batches alone do not change
the result.

# YOLOv5

This family builds the YOLOv5 detector `yolov5n` (`yolov5n.pt` of
`Ultralytics/YOLOv5`) for `image_to_boxes` as one TensorRT engine: 640 x 640
input, batch 1, fp16 or fp32 (the manifest uses fp16). The head is the anchor
head: three levels, three anchors per cell, objectness times class score. The
engine reports every anchor with its box and strongest class, and the runtime
drops weak scores and runs greedy per-class non-max suppression. The family has
no quantized build.

## FP8 quantization study (not shipped)

FP8 self-quantization of the convolutions was built and measured on TensorRT
11.4 (and 11.1) for `yolov5n`. It is not shipped: FP8 is slower than fp16
(0.89x to 0.95x at batch 1, 0.90x to 0.93x at batch 8), the plan saves only
1.4 MiB, and COCO AP drops by 0.8 point. This section records the recipe, the
method and the numbers, so that the study does not have to be repeated. The
FP8 code was not merged; this family has no FP8 build.

### Summary and decision

| Item | fp16 (manifest) | FP8 (fp16 engine) | FP8 vs fp16 |
|---|---|---|---|
| Latency, batch 1, B200 / B300 (TensorRT 11.4) | 0.337 / 0.307 ms | 0.355 / 0.337 ms | 0.95x / 0.91x |
| Latency, batch 8, B200 / B300 (TensorRT 11.4) | 0.952 / 0.921 ms | 1.055 / 0.993 ms | 0.90x / 0.93x |
| Plan size, B200 | 7.92 MiB | 6.51 MiB | -18% |
| COCO AP, B200 (PyTorch fp32 0.3075) | 0.3075 | 0.2991 | -0.8 point |
| Detections that match PyTorch (prec / rec) | 0.997 / 0.995 | 0.917 / 0.837 | 8% to 16% changed |
| Non-finite outputs | 0 | 0 | none |

Decision: do not ship FP8 for YOLOv5.

- Speed: the FP8 convolutions run on FP8 kernels, but bias, SiLU and the
  Quantize of the next convolution are not fused into the FP8 convolution. They
  run as extra pointwise kernels (18 in the fp16 plan, 27 in the FP8 plan; 75
  to 92 plan layers) and take the gain back. `yolov5n` is small, so it is
  launch and pointwise bound, and FP8 ends up slower than fp16.
- `yolov5n` has no depthwise convolution, so the missing FP8 depthwise kernel
  on TensorRT 11.4 does not limit it.
- The only gain is plan size (1.4 MiB less).
- The same recipe gave the same result on `yolov8n` (see the `yolov8` family
  README) and on the small `yolov10` widths.

### Recipe that was measured

No FP8 or NVFP4 checkpoint of any YOLO detector exists on the Hub, and no
official quantization config exists for one. The recipe is the one of the
`yolov10` study: quantize the dense convolutions, keep the input convolution.
Each quantized convolution is built as
`input -> Quantize -> Dequantize(fp16) -> Convolution(+bias)`, with the kernel
`FP8 constant -> Dequantize(fp16, axis 0)`.

| Choice | Decision |
|---|---|
| Quantized (56 of 60 convolutions) | every dense convolution whose input and output channels are multiples of 16: Conv blocks, C3, bottlenecks, SPPF |
| fp16 | stem `model.0` (3 input channels), the three head convolutions (255 output channels, not a multiple of 16), anchor decode, top-1 class |
| Weight scale | per output channel, amax / 448 of the batch-norm-folded kernel, at build time |
| Activation scale | static, per tensor; max calibration (the statistic ModelOpt `FP8_DEFAULT_CFG` keeps) on the family's PyTorch reference, 256 letterboxed COCO 2017 validation images |
| Outlier check | 60 convolution inputs, median amax 9.8, largest 141 (14x): no outlier |
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
  row in one job. Batch 8 is a scratch build (the same graph with an 8-image
  input and the raw head maps as outputs); the family builds batch 1 only.

### Results

Speed and size (x is fp16 time over FP8 time):

| GPU | TensorRT | fp32 ms | fp16 ms | FP8 ms | x batch 1 | batch 8 fp16 / FP8 ms | x batch 8 | plan MiB fp16 to FP8 |
|---|---|---|---|---|---|---|---|---|
| B200 | 11.4 | 0.404 | 0.337 | 0.355 | 0.95 | 0.952 / 1.055 | 0.90 | 7.92 to 6.51 |
| B300 | 11.4 | 0.385 | 0.307 | 0.337 | 0.91 | 0.921 / 0.993 | 0.93 | 7.95 to 6.60 |
| B200 | 11.1 | - | 0.319 | 0.357 | 0.89 | 0.945 / 1.033 | 0.91 | 7.94 to 6.52 |

Quality, B200, TensorRT 11.4 (B300 and TensorRT 11.1 within 0.0002 AP):

| Engine | AP | AP50 | prec / rec vs PyTorch | box IoU | score error |
|---|---|---|---|---|---|
| PyTorch fp32 | 0.3075 | 0.4695 | - | - | - |
| fp32 engine | 0.3075 | 0.4695 | 1.000 / 0.999 | 0.9993 | 0.0004 |
| fp16 engine | 0.3075 | 0.4697 | 0.997 / 0.995 | 0.993 | 0.0008 |
| FP8 engine | 0.2991 | 0.4580 | 0.917 / 0.837 | 0.931 | 0.043 |

### NVFP4 (not tried)

TensorRT does not support NVFP4 on convolutions (4-D tensors), and the model
has no matmul. The engine is launch bound, not weight-bandwidth bound, so
4-bit weights would not help.

### What was not tried

| Item | Note |
|---|---|
| Percentile or entropy calibration, per-tensor weight scales | not measured |
| A SiLU form that TensorRT might fuse into the FP8 convolution epilogue | not measured |
| Larger widths (s, m, l, x), full 5000-image COCO validation, H100, native runtime with an FP8 bundle | not measured; the family builds only `yolov5n` |

### When to revisit

- A TensorRT release that fuses bias, SiLU and the next Quantize into the FP8
  convolution epilogue (as it does for most fp16 convolutions).
- The family adds a larger width: on `yolov10x` the same recipe reached fp16
  speed with a 43% smaller plan, so a large YOLOv5 may be worth a new
  measurement.

Batch 8 gave about the same ratio as batch 1, so larger batches alone do not
change the result.

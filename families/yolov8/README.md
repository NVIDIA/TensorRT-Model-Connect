# YOLOv8

This family builds the YOLOv8 detector `yolov8n` (`yolov8n.pt` of
`Ultralytics/YOLOv8`) for `image_to_boxes` as one TensorRT engine: 640 x 640
input, batch 1, fp16 or fp32 (the manifest uses fp16). The head is not end to
end: the engine reports every anchor with its box and strongest class, and the
runtime drops weak scores and runs greedy per-class non-max suppression. The
family has no quantized build.

## FP8 quantization study (not shipped)

FP8 self-quantization of the convolutions was built and measured on TensorRT
11.4 (and 11.1) for `yolov8n`. It is not shipped: FP8 is slower than fp16
(0.92x to 0.97x at batch 1, 0.89x to 0.91x at batch 8), the plan saves only
2.7 MiB, and COCO AP drops by 0.5 point. This section records the recipe, the
method and the numbers, so that the study does not have to be repeated. The
FP8 code was not merged; this family has no FP8 build.

### Summary and decision

| Item | fp16 (manifest) | FP8 (fp16 engine) | FP8 vs fp16 |
|---|---|---|---|
| Latency, batch 1, B200 / B300 (TensorRT 11.4) | 0.317 / 0.287 ms | 0.335 / 0.312 ms | 0.95x / 0.92x |
| Latency, batch 8, B200 / B300 (TensorRT 11.4) | 0.612 / 0.578 ms | 0.673 / 0.643 ms | 0.91x / 0.90x |
| Plan size, B200 | 7.36 MiB | 4.60 MiB | -37% |
| COCO AP, B200 (PyTorch fp32 0.3971) | 0.3969 | 0.3922 | -0.5 point |
| Detections that match PyTorch (prec / rec) | 0.998 / 1.000 | 0.926 / 0.888 | 7% to 11% changed |
| Non-finite outputs | 0 | 0 | none |

Decision: do not ship FP8 for YOLOv8.

- Speed: the FP8 convolutions run on FP8 kernels, but bias, SiLU and the
  Quantize of the next convolution are not fused into the FP8 convolution. They
  run as extra pointwise kernels (23 in the fp16 plan, 30 in the FP8 plan; 89
  to 99 plan layers) and take the gain back. `yolov8n` is small, so it is
  launch and pointwise bound, and FP8 ends up slower than fp16.
- Depthwise convolutions have no FP8 kernel on TensorRT 11.4 (the build fails
  with "no tactics"). `yolov8n` has no depthwise convolution, so this does not
  limit it.
- The only gain is plan size (2.7 MiB less).
- The sibling families `yolov5` (`yolov5n`) and `yolo11` (`yolo11n`) were
  measured with the same recipe and gave the same result: FP8 at 0.89x to 0.95x
  and 0.90x to 0.93x of fp16 at batch 1, AP 0.3075 to 0.2991 and 0.4158 to
  0.4034.

### Recipe that was measured

No FP8 or NVFP4 checkpoint of any YOLO detector exists on the Hub, and no
official quantization config exists for one. The recipe is the one of the
`yolov10` study: quantize the dense convolutions, keep the input convolution.
Each quantized convolution is built as
`input -> Quantize -> Dequantize(fp16) -> Convolution(+bias)`, with the kernel
`FP8 constant -> Dequantize(fp16, axis 0)`.

| Choice | Decision |
|---|---|
| Quantized (62 of 63 convolutions) | every dense convolution whose input and output channels are multiples of 16: Conv blocks, C2f, bottlenecks, SPPF, head box and class convolutions |
| fp16 | stem `model.0` (3 input channels), DFL decode, top-1 class |
| Weight scale | per output channel, amax / 448 of the batch-norm-folded kernel, at build time |
| Activation scale | static, per tensor; max calibration (the statistic ModelOpt `FP8_DEFAULT_CFG` keeps) on the family's PyTorch reference, 256 letterboxed COCO 2017 validation images |
| Outlier check | 64 convolution inputs, median amax 11.0, largest 178 (16x): no outlier |
| Dequantize type | fp16; no non-finite value, fp32 Dequantize was not needed |

### Method

- GPUs: B200 and B300, one GPU per job; TensorRT 11.4.0.106 (target) and
  11.1.0.106.
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
| B200 | 11.4 | 0.477 | 0.317 | 0.335 | 0.95 | 0.612 / 0.673 | 0.91 | 7.36 to 4.60 |
| B300 | 11.4 | 0.455 | 0.287 | 0.312 | 0.92 | 0.578 / 0.643 | 0.90 | 7.38 to 4.72 |
| B200 | 11.1 | - | 0.321 | 0.332 | 0.97 | 0.597 / 0.672 | 0.89 | 7.33 to 4.62 |

Quality, B200, TensorRT 11.4 (B300 and TensorRT 11.1 within 0.0003 AP):

| Engine | AP | AP50 | prec / rec vs PyTorch | box IoU | score error |
|---|---|---|---|---|---|
| PyTorch fp32 | 0.3971 | 0.5389 | - | - | - |
| fp32 engine | 0.3967 | 0.5391 | 0.999 / 1.000 | 0.9996 | 0.0005 |
| fp16 engine | 0.3969 | 0.5387 | 0.998 / 1.000 | 0.995 | 0.0008 |
| FP8 engine | 0.3922 | 0.5336 | 0.926 / 0.888 | 0.955 | 0.042 |
| FP8 without the six last head convolutions | 0.3931 | 0.5338 | 0.931 / 0.892 | 0.962 | 0.040 |

The variant without the six last head convolutions (56 quantized) runs at 0.95x
(B200) and 0.91x (B300) of fp16 at batch 1, so keeping the head in fp16 does
not help either.

### NVFP4 (not tried)

TensorRT does not support NVFP4 on convolutions (4-D tensors), and the model
has no large matmul. The engine is launch bound, not weight-bandwidth bound,
so 4-bit weights would not help.

### What was not tried

| Item | Note |
|---|---|
| Percentile or entropy calibration, per-tensor weight scales | not measured |
| A SiLU form that TensorRT might fuse into the FP8 convolution epilogue | not measured |
| Larger widths (s, m, l, x), full 5000-image COCO validation, H100, native runtime with an FP8 bundle | not measured; the family builds only `yolov8n` |

### When to revisit

- A TensorRT release that fuses bias, SiLU and the next Quantize into the FP8
  convolution epilogue (as it does for most fp16 convolutions).
- The family adds a larger width: on `yolov10x` the same recipe reached fp16
  speed with a 43% smaller plan, so a large YOLOv8 may be worth a new
  measurement.

Batch 8 gave a lower ratio than batch 1, so larger batches alone do not change
the result.

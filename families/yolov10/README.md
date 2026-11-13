# YOLOv10

This family builds YOLOv10 detectors (`jameslahm/yolov10n`, `yolov10s`,
`yolov10x`) for `image_to_boxes` as one end-to-end TensorRT engine: 640 x 640
input, batch 1, fp16 or fp32 (the manifests use fp16). The one-to-one head emits
one box per object, so decode and top-k run in the engine and no non-max
suppression is needed. The family has no quantized build.

## FP8 quantization study (not shipped)

FP8 self-quantization of the convolutions was built and measured on TensorRT
11.4 (and 11.1) for `jameslahm/yolov10x`. It is not shipped: the plan is 43%
smaller, but latency stays at the fp16 level (0.98x to 1.02x at batch 1) and COCO
AP drops by 0.3 point. On yolov10n and yolov10s FP8 is slower than fp16. This
section records the recipe, the method and the numbers, so that the study does
not have to be repeated. The FP8 code was not merged; this family has no FP8
build.

### Summary and decision

| Item | fp16 (manifest) | FP8 (fp16 engine) | FP8 vs fp16 |
|---|---|---|---|
| yolov10x latency, batch 1, B200 / B300 (TensorRT 11.4) | 1.406 / 1.323 ms | 1.377 / 1.319 ms | 1.02x / 1.00x |
| yolov10x latency, batch 8, B200 / B300 (TensorRT 11.4) | 4.059 / 3.815 ms | 4.077 / 3.861 ms | 1.00x / 0.99x |
| yolov10x plan size | 61.9 MiB | 35.5 MiB | -43% |
| yolov10x COCO AP, B200 / B300 (PyTorch fp32 0.5731) | 0.5730 / 0.5734 | 0.5697 / 0.5701 | -0.3 point |
| yolov10n / yolov10s latency, batch 1, B200 | - | - | 0.93x / 0.97x |
| Non-finite outputs | 0 | 0 | none |

Decision: do not ship FP8 for YOLOv10.

- Speed: the FP8 convolutions are about 11% faster than the fp16 ones, but the
  bias, SiLU and the Quantize of the next convolution are not fused into the FP8
  convolution. They run as separate pointwise kernels (62 in the fp16 plan, 104
  in the FP8 plan) and take the gain back.
- The depthwise convolutions (about a sixth of the time) have no FP8 kernel and
  stay fp16.
- Smaller models: yolov10n and yolov10s are slower in FP8 and lose 0.35 to 0.9
  AP point.
- The only gain is plan size (27 MiB less for yolov10x).

### Recipe that was measured

No FP8 or NVFP4 checkpoint of any YOLO detector exists on the Hub, and no
official quantization config exists for one. The recipe follows the convolution
guidance used for the `stable_diffusion` FP8 build: quantize the convolutions,
keep the input convolution. Each quantized convolution is built as
`input -> Quantize -> Dequantize(fp16) -> Convolution(+bias)`, with the kernel
`FP8 constant -> Dequantize(fp16, axis 0)`.

| Choice | Decision |
|---|---|
| Quantized (108 of 173 convolutions in yolov10x) | every dense convolution whose input and output channels are multiples of 16: Conv blocks, C2f, bottleneck 3x3, CIB 1x1, SCDown `cv1`, SPPF, PSA `cv1`, `cv2`, `attn.qkv`, `attn.proj`, `ffn`, head dense steps and the final box and class convolutions |
| fp16 | stem `model.0` (3 input channels), all 64 depthwise convolutions (CIB, SCDown `cv2`, PSA `attn.pe`, class branch), attention matmuls and softmax, DFL decode and top-k |
| Weight scale | per output channel, amax / 448 of the batch-norm-folded kernel, at build time |
| Activation scale | static, per tensor; ModelOpt 0.46.1 `FP8_DEFAULT_CFG` (max calibration) on the family's PyTorch reference, 256 letterboxed COCO 2017 validation images |
| Outlier rule | a scale above 20x the median of its kind (1x1 or spatial) stays fp16; measured: no outlier (largest input amax 164, median 14.5) |
| Dequantize type | fp16; no non-finite value, fp32 Dequantize was not needed |

### Method

- GPUs: B200 and B300, one GPU per job; TensorRT 11.4.0.106 (target) and
  11.1.0.106.
- Quality: COCO 2017 validation images 300 to 799 of `detection-datasets/coco`
  (500 held out; 0 to 255 are the calibration set), letterboxed like the
  runtime. AP is `pycocotools` COCO AP (IoU 0.50 to 0.95) of the 300 detections
  per image, comparable only inside these tables. Agreement: detections at score
  0.25 matched to the PyTorch fp32 detections (same class, IoU at least 0.5).
- Non-finite scan: the 500 images and 20 stress inputs (noise, constants).
- Speed: CUDA graph replay, median of 9 interleaved rounds. Batch 8 is a scratch
  build (the family builds batch 1 only).

### Results

Speed and size, yolov10x (x is fp16 time over FP8 time):

| GPU | TensorRT | fp32 ms | fp16 ms | FP8 ms | x batch 1 | x batch 8 | plan MiB fp16 to FP8 |
|---|---|---|---|---|---|---|---|
| B200 | 11.4 | 2.021 | 1.406 | 1.377 | 1.02 | 1.00 | 61.9 to 35.5 |
| B300 | 11.4 | 1.927 | 1.323 | 1.319 | 1.00 | 0.99 | 62.0 to 35.5 |
| B200 | 11.1 | 2.015 | 1.412 | 1.385 | 1.02 | 1.03 | 61.6 to 35.1 |
| B300 | 11.1 | 1.919 | 1.305 | 1.329 | 0.98 | 1.02 | 61.7 to 35.2 |

Quality, yolov10x, B200 (B300 in brackets where it differs; TensorRT 11.1 gives
the same numbers as 11.4):

| Engine | AP | AP50 | prec / rec vs PyTorch | box IoU | score error |
|---|---|---|---|---|---|
| PyTorch fp32 | 0.5731 | 0.7334 | - | - | - |
| fp16 engine | 0.5730 (0.5734) | 0.7337 | 0.999 / 0.999 (0.998 / 0.998) | 0.995 | 0.0010 |
| FP8 engine | 0.5697 (0.5701) | 0.7302 (0.7294) | 0.977 / 0.959 (0.977 / 0.958) | 0.983 | 0.024 |

Kernel time per group, yolov10x, batch 1, B200, TensorRT 11.4 (layer profiler,
ms (count)):

| Kernel group | fp16 | FP8 |
|---|---|---|
| dense convolutions | 1.415 (111) | 1.212 FP8 (108) + 0.049 fp16 (3) |
| depthwise convolutions | 0.531 (64) | 0.529 (64) |
| bias + SiLU (+ Quantize) pointwise | 0.484 (62) | 0.800 (104) |
| layout transposes | 0.343 (47) | 0.386 (54) |
| other | 0.326 | 0.175 |
| sum | 3.09 | 3.15 |

B300 shows the same picture (sum 3.02 fp16, 3.12 FP8).

Variants (TensorRT 11.4, B200 speed vs fp16; one separate job, so the AP of the
first row differs slightly from the table above):

| Variant | x batch 1 / batch 8 | plan MiB | AP B200 / B300 |
|---|---|---|---|
| recipe above | 1.02 / 1.00 | 35.5 | 0.5707 / 0.5720 |
| per-tensor weight scale (ModelOpt default) | 1.01 / - | 35.4 | 0.5681 / 0.5681 |
| without the six last head convolutions | 1.00 / - | 35.4 | 0.5699 / 0.5699 |
| 3x3 convolutions only (33) | 1.01 / 1.02 | 55.4 | 0.6014 (100 images, fp16 0.5953) |
| 1x1 convolutions only (75) | 0.96 / 0.95 | 42.4 | 0.5956 (100 images, fp16 0.5953) |
| builder optimization level 3 / 5 | 1.00 / 1.01 | 35.2 | - |
| depthwise convolutions in FP8 | build fails on 11.4 ("no tactics") | | |
| yolov10n | 0.93 (B300 0.92) / 0.88 | 6.2 to 4.5 | 0.4128 / 0.4102 (fp16 0.4190) |
| yolov10s | 0.97 (B300 0.95) / 0.95 (0.94) | 15.7 to 9.5 | 0.4865 / 0.4882 (fp16 0.4917) |

### NVFP4 (not tried)

TensorRT does not support NVFP4 on convolutions (4-D tensors), and the only
matmuls of the model (the PSA attention) are a negligible share of the time.

### What was not tried

| Item | Note |
|---|---|
| FP8 attention matmuls, percentile or entropy calibration | not measured |
| A SiLU form that TensorRT might fuse into the FP8 convolution epilogue | not measured |
| Full 5000-image COCO validation, H100, native runtime with an FP8 bundle | not measured |

### When to revisit

- A TensorRT release that fuses bias, SiLU and the next Quantize into the FP8
  convolution epilogue (as it does for most fp16 convolutions): the saved 0.15
  to 0.2 ms of convolution time would then show up as a speedup.
- A TensorRT FP8 kernel for depthwise convolutions.

Batch 8 gave the same ratio as batch 1, so larger batches alone do not change
the result.

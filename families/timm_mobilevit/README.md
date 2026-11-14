# timm MobileViT

This family builds the timm MobileViT image classifiers (`timm/mobilevit_xxs`,
`mobilevit_xs` and `mobilevit_s`, `cvnets_in1k`) for `image_to_class_scores` as
one TensorRT engine: 256 x 256 input, batch 1, fp16 or fp32 (the manifests use
fp16). Most of the network is MobileNetV2 inverted residual blocks; three stages
end in a MobileViT block (convolutions plus a small transformer over 2 x 2
patches). The family has no quantized build.

## FP8 quantization study (not shipped)

FP8 self-quantization of the transformer linears was built and measured on
TensorRT 11.4 (and 11.1) for all three checkpoints. It is not shipped: FP8 is
slower than fp16 on B200 and B300 (0.89x to 0.96x at builder optimization
level 1, 0.94x to 0.97x at level 3), and top-1 agreement with the fp16 engine
drops to 95.7% to 98.7%. The plan is 4% to 19% smaller. This section records the
recipe, the method and the numbers, so that the study does not have to be
repeated. The FP8 code was not merged; this family has no FP8 build.

### Summary and decision

TensorRT 11.4, batch 1, builder optimization level 1 (the level of this family);
x is fp16 time over FP8 time.

| Item | fp16 (manifest) | FP8 (fp16 engine) | FP8 vs fp16 |
|---|---|---|---|
| mobilevit_s latency, B200 / B300 | 0.431 / 0.420 ms | 0.462 / 0.442 ms | 0.93x / 0.95x |
| mobilevit_xs latency, B200 / B300 | 0.408 / 0.406 ms | 0.449 / 0.425 ms | 0.91x / 0.96x |
| mobilevit_xxs latency, B200 / B300 | 0.357 / 0.363 ms | 0.403 / 0.385 ms | 0.89x / 0.94x |
| Same, builder optimization level 3 | - | - | 0.94x to 0.97x |
| Plan size s / xs / xxs | 12.6 / 6.3 / 4.3 MiB | 10.2 / 5.6 / 4.0 MiB | -19% / -12% / -6% |
| Top-1 agreement with the fp16 engine, s / xs / xxs (B200) | 100% | 98.5% / 97.7% / 95.8% | - |
| Non-finite outputs on ImageNet images | 0 | 0 | none |

Decision: do not ship FP8 for timm MobileViT.

- Speed: the quantized GEMMs are tiny (token width 64 to 240, 16 to 256
  tokens per sequence, 9 transformer layers per model), and the engine
  is dominated by the convolutions and by per-layer launch time (about 0.4 ms
  for 150 to 165 layers). The FP8 engine adds 9 to 12 layers (Quantize and
  Dequantize that do not fuse), which cost more than the FP8 GEMMs save.
- Builder optimization level 3 makes both engines faster, but FP8 stays slower
  than fp16 at that level too.
- Quality: per-tensor FP8 activations change the top-1 class of 1.5% (s) to
  4.2% (xxs) of the images compared with the fp16 engine; the smaller the
  model, the larger the loss.
- The only gain is plan size (2.4 MiB less for mobilevit_s).

### Recipe that was measured

No FP8 or NVFP4 checkpoint of these models exists on the Hub, and no official
quantization config exists for them. The recipe follows the ModelOpt ViT recipe
(attention and MLP linears quantized, convolutions and patch embedding kept).
Each quantized linear is built as
`input -> Quantize -> Dequantize(fp16) -> MatMul`, with the weight
`FP8 constant -> Dequantize(fp16)`.

| Choice | Decision |
|---|---|
| Quantized | `attn.qkv` (one shared input scale), `mlp.fc1`, `mlp.fc2` of every transformer layer (9 layers in each model: 2, 4 and 3 per MobileViT stage) |
| fp16 | stem, inverted residual convolutions, MobileViT 1x1 and fusion convolutions, attention matmuls and softmax, layer norms, classifier, `attn.proj` |
| `attn.proj` | a Quantize on its input makes the TensorRT 11.4 attention fusion try an FP8 softmax and the build fails ("Could not infer output types for operation softmax"; xxs, xs, s); kept fp16 |
| Weight scale | per tensor, amax / 448, at build time (no FP8 weights stored) |
| Activation scale | static, per tensor; ModelOpt 0.46.1 `FP8_DEFAULT_CFG` (max calibration) on the timm PyTorch model, 500 ImageNet validation images |
| Outlier rule | a scale above 20x the median of its kind stays fp16; measured: no outlier |
| Dequantize type | fp16; no non-finite value, fp32 Dequantize was not needed |

### Method

- GPUs: B200 and B300, one GPU per job, all variants of one model built and
  timed in the same job; TensorRT 11.4.0.106 (target) and 11.1.0.106.
- Builds: the family's own engine build at builder optimization level 1 (this
  family) and level 3 (the TensorRT default), fp32, fp16 and FP8.
- Data: 2000 ImageNet validation images from `clip-benchmark/wds_imagenet1k`
  (every 25th image of the class-sorted set): 500 spread over the classes for
  calibration, 1500 of the others held out. Checkpoint preprocessing (timm).
- Quality: top-1 against the labels, top-1 agreement and logit cosine against
  the fp16 engine and against the timm PyTorch fp32 model, batch 1.
- Non-finite scan: the 1500 held-out images and 8 stress inputs (constants 0,
  +-1, +-10 and Gaussian noise with standard deviation 1, 3 and 10).
- Speed: CUDA graph replay, median of 15 interleaved rounds of 200 calls,
  spread at most 0.1%.

### Results

Speed and size, batch 1 (ms per engine call; x is fp16 time over FP8 time;
L1 and L3 are builder optimization levels 1 and 3):

| Model | GPU | TensorRT | fp32 L1 | fp16 L1 | FP8 L1 | x L1 | fp16 L3 | FP8 L3 | x L3 | plan MiB fp16 to FP8 (L1) |
|---|---|---|---|---|---|---|---|---|---|---|
| s | B200 | 11.4 | 0.929 | 0.431 | 0.462 | 0.93 | 0.398 | 0.414 | 0.96 | 12.6 to 10.2 |
| s | B300 | 11.4 | 0.897 | 0.420 | 0.442 | 0.95 | 0.371 | 0.388 | 0.96 | 12.7 to 10.3 |
| xs | B200 | 11.4 | 0.749 | 0.408 | 0.449 | 0.91 | 0.385 | 0.401 | 0.96 | 6.3 to 5.6 |
| xs | B300 | 11.4 | 0.714 | 0.406 | 0.425 | 0.96 | 0.355 | 0.367 | 0.97 | 6.4 to 5.6 |
| xxs | B200 | 11.4 | 0.614 | 0.357 | 0.403 | 0.89 | 0.338 | 0.358 | 0.94 | 4.3 to 4.0 |
| xxs | B300 | 11.4 | 0.583 | 0.363 | 0.385 | 0.94 | 0.313 | 0.331 | 0.95 | 4.3 to 4.0 |
| s | B200 | 11.1 | 0.915 | 0.447 | 0.462 | 0.97 | 0.412 | 0.410 | 1.01 | 12.6 to 10.3 |

Layers in the plan (TensorRT 11.4, B200, fp16 to FP8): s 154 to 163, xs 165 to
176, xxs 153 to 164 at level 1; 152 to 161, 162 to 173, 149 to 161 at level 3.

Quality, TensorRT 11.4, level 1, 1500 held-out images (B300 in brackets where
it differs by more than 0.1 point):

| Model | PyTorch fp32 top-1 | fp16 engine top-1 | FP8 engine top-1 | FP8 agreement with fp16 top-1 | FP8 logit cosine to fp16, mean / min |
|---|---|---|---|---|---|
| s | 78.00% | 78.00% | 78.07% | 98.5% | 0.9971 / 0.896 (0.854) |
| xs | 74.33% | 74.47% | 74.20% (74.33%) | 97.7% | 0.9961 / 0.931 |
| xxs | 68.87% | 68.73% | 68.40% | 95.8% | 0.9951 / 0.951 |

The fp16 engine agrees with the PyTorch top-1 on 99.7% to 99.8% of the images
(logit cosine 0.99995 or higher on average). Level 3 gives the same quality as
level 1 (FP8 agreement with fp16 95.7% to 98.7%).

Non-finite values: none on the 1500 images for any engine. One of the 8
stress inputs gives non-finite logits on mobilevit_xxs in fp16 and in FP8 alike
(level 1 and 3, B200 and B300; PyTorch fp32 is finite); mobilevit_xs and
mobilevit_s are finite. This is fp16 behavior of the existing build, not
caused by FP8.

### NVFP4 (not tried)

The only GEMMs are the small transformer linears (K = 64 to 480); FP8 is
already slower than fp16 there, and NVFP4 would add block-scale handling on the
same tiny GEMMs and lose more accuracy. TensorRT does not support NVFP4 on the
convolutions that dominate the time.

### What was not tried

| Item | Note |
|---|---|
| FP8 convolutions, FP8 attention matmuls, percentile or entropy calibration | not measured |
| Batch size above 1 | the family builds batch 1 only |
| H100, native runtime with an FP8 bundle | not measured |

### When to revisit

- A TensorRT release that fuses the Quantize into the producing layer (layer
  norm, GELU) for small GEMMs, so that FP8 adds no layers.
- FP8 for `attn.proj` once the TensorRT attention fusion accepts a Quantize on
  its output.
- A batched or larger-resolution variant of this family, where the transformer
  GEMMs would be large enough to be limited by math.

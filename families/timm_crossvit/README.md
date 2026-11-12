# timm CrossViT

This family builds the timm CrossViT classifier (`timm/crossvit_9_240.in1k`) for
`image_to_class_scores` as one TensorRT engine: 240 x 240 input, batch 1, fp16 or
fp32 (the manifest uses fp16). The two branches (patch 12, width 128, 401 tokens;
patch 16, width 256, 197 tokens) and the cross-attention between their class
tokens run in the same engine. The family has no quantized build.

## FP8 quantization study (not shipped)

FP8 self-quantization of the encoder and cross-attention projections was built
and measured on TensorRT 11.4 (and 11.1) for `timm/crossvit_9_240.in1k`. It is
not shipped: at the family's builder optimization level (1) FP8 is 3% slower
than fp16 on B200 and B300, and the FP8 engine changes the top-1 class of about
3% of the images. The plan is 31% smaller. This section records the recipe, the
method and the numbers, so that the study does not have to be repeated. The FP8
code was not merged; this family has no FP8 build.

### Summary and decision

Batch 1, TensorRT 11.4, 1500 held-out ImageNet validation images (x is fp16 time
over FP8 time):

| Item | fp16 (manifest) | FP8 (fp16 engine) | FP8 vs fp16 |
|---|---|---|---|
| Latency, builder level 1 (family), B200 / B300 | 0.484 / 0.475 ms | 0.499 / 0.491 ms | 0.97x / 0.97x |
| Latency, builder level 3, B200 / B300 | 0.452 / 0.436 ms | 0.453 / 0.425 ms | 1.00x / 1.03x |
| Plan size, level 1 / level 3 (B200) | 18.2 / 20.2 MiB | 12.5 / 14.7 MiB | -31% / -27% |
| Top-1 agreement with PyTorch fp32, B200 / B300 | 99.7% / 99.7% | 97.3% / 96.9% | -2.5 to -2.9 points |
| ImageNet top-1, B200 / B300 (PyTorch fp32 73.47%) | 73.53% / 73.53% | 73.53% / 73.40% | within noise |
| Non-finite outputs | 0 | 0 | none |

Decision: do not ship FP8 for CrossViT.

- Speed: at the family's level 1 FP8 is slower than fp16 on both GPUs. At
  level 3 it is 1.00x on B200 and 1.03x on B300, so it is not faster on both.
- Quality: the logit cosine to PyTorch drops from 0.99999 to 0.9959 (mean) and
  from 0.999 to 0.976 (worst image); the top-1 class changes on about 3% of the
  images. The fp16 engine changes it on 0.3%.
- The only clear gain is plan size (5.7 MiB less).

### Recipe that was measured

No FP8 or NVFP4 checkpoint of a CrossViT model exists on the Hub, and no
official quantization config exists for it. The recipe follows the ViT FP8
recipe used for the `timm_vit`, `siglip` and `ast` FP8 builds: quantize the
linears that read all tokens. Each quantized projection is built as
`input -> Quantize -> Dequantize(fp16) -> MatMul`, with the weight
`FP8 constant -> Dequantize(fp16)`.

| Choice | Decision |
|---|---|
| Quantized | `attn.qkv`, `mlp.fc1`, `mlp.fc2` of the 12 encoder blocks (36), `attn.wk` and `attn.wv` of the 6 cross-attention blocks (12; one shared input scale) |
| fp16 | encoder `attn.proj` (TensorRT 11.4 fails to build an FP8 Quantize right after the fused attention, also on a tiny engine), cross-attention `wq` and `proj` and the class-token projections (one row), patch embedding, norms, attention matmuls and softmax, heads |
| Weight scale | per tensor, amax / 448 of the original fp32 weight, at build time |
| Activation scale | static, per tensor; ModelOpt 0.46.1 `FP8_DEFAULT_CFG` (max calibration) on the timm model, 500 ImageNet validation images |
| Outlier rule | an input scale above 20x the median of its kind stays fp16 (none found); a scale below the smallest normal fp16 value stays fp16 (one: `blocks.2.fusion.1.attn.wk`, weight scale 2.3e-5) |
| Dequantize type | fp16; no non-finite value, fp32 Dequantize was not needed |

### Method

- GPUs: B200 and B300, one GPU per job; TensorRT 11.4.0.106 (target) and
  11.1.0.106. In each job all engines were built and timed together.
- Engines: fp16 from the family code at builder optimization level 1 (the
  family setting) and at level 3 (only the level value replaced); FP8 from the
  study code at the same two levels. The fp16 engine of the study code has the
  same layer count as the family engine (188 on 11.4, 200 on 11.1).
- Quality: ImageNet validation images from `clip-benchmark/wds_imagenet1k`. The
  calibration set is 500 images at a stride of 100 over the class-sorted 50000.
  The held-out set is 1500 images spread over the classes from a stride-12 set,
  with every image of the calibration stride removed (500 on TensorRT 11.1).
  Preprocessing is the timm transform; the reference is the timm model in fp32;
  the fp16 engine is the floor.
- Non-finite scan: the held-out images and 19 stress inputs (noise at std 1, 3,
  10 and 30, constant images).
- Speed: CUDA graph replay of 100 runs, median of 15 interleaved rounds (spread
  under 0.2%).

### Results

Speed and size, batch 1 (x is fp16 time over FP8 time at the same level):

| GPU | TensorRT | fp16 L1 ms | FP8 L1 ms | x L1 | fp16 L3 ms | FP8 L3 ms | x L3 | plan MiB fp16 to FP8 (L1 / L3) |
|---|---|---|---|---|---|---|---|---|
| B200 | 11.4 | 0.484 | 0.499 | 0.97 | 0.452 | 0.453 | 1.00 | 18.2 to 12.5 / 20.2 to 14.7 |
| B300 | 11.4 | 0.475 | 0.491 | 0.97 | 0.436 | 0.425 | 1.03 | 18.2 to 12.5 / 21.2 to 14.6 |
| B200 | 11.1 | 0.496 | 0.508 | 0.98 | 0.462 | 0.459 | 1.01 | 18.3 to 12.5 / 20.2 to 14.6 |
| B300 | 11.1 | 0.486 | 0.502 | 0.97 | 0.445 | 0.424 | 1.05 | 18.2 to 12.6 / 21.2 to 14.6 |

Quality, TensorRT 11.4, 1500 held-out images (PyTorch fp32 top-1 73.47%):

| Engine | GPU | Top-1 | Agreement with PyTorch | Agreement with fp16 L1 | Logit cosine mean / min | KL(PyTorch, engine) |
|---|---|---|---|---|---|---|
| fp16 L1 | B200 | 73.53% | 99.73% | - | 0.99999 / 0.99904 | 0.00001 |
| FP8 L1 | B200 | 73.53% | 97.27% | 97.40% | 0.99590 / 0.97777 | 0.0060 |
| fp16 L3 | B200 | 73.47% | 99.73% | 99.87% | 0.99999 / 0.99911 | 0.00001 |
| FP8 L3 | B200 | 73.60% | 97.00% | 97.13% | 0.99590 / 0.97582 | 0.0058 |
| fp16 L1 | B300 | 73.53% | 99.73% | - | 0.99999 / 0.99911 | 0.00001 |
| FP8 L1 | B300 | 73.40% | 96.87% | 96.93% | 0.99592 / 0.97766 | 0.0059 |
| fp16 L3 | B300 | 73.47% | 99.87% | 99.87% | 0.99999 / 0.99900 | 0.00001 |
| FP8 L3 | B300 | 73.53% | 97.13% | 97.27% | 0.99593 / 0.97913 | 0.0059 |

TensorRT 11.1 (500 held-out images, PyTorch fp32 70.80%): FP8 agrees with
PyTorch on 96.6% to 97.0% of the images, fp16 on 99.6% to 99.8%; the mean
logit cosine is the same as on 11.4 (0.9959). No engine produced a non-finite
output.

Earlier run (TensorRT 11.4, level 1, B200; 1500 images that include 375 of the
calibration images, so only the speed columns are comparable):

| Variant | x batch 1 / 8 / 32 / 128 | plan MB | Agreement with PyTorch |
|---|---|---|---|
| recipe above | 1.00 / 1.02 / 1.04 / 1.05 | 13.1 | 96.2% |
| MLP only (`fc1`, `fc2`) | 1.02 / - / - / - | 15.4 | 96.5% |

Batches 8 to 128 are scratch builds (the family builds batch 1 only); B300 gave
1.03x to 1.06x at batch 8 to 128.

### Why there is no gain

- At batch 1 the GEMMs are small (197 or 401 rows, widths 128 to 768). The
  plan runs about 190 kernels of a few microseconds each, so the engine is
  bound by kernel count, not by matmul math; FP8 GEMMs of this size are not
  faster than fp16 ones, and each FP8 GEMM adds a Quantize of its input.
- Not every projection is FP8: the 12 encoder `attn.proj` cannot be quantized
  after the fused attention on TensorRT 11.4, and the class-token projections
  have one row.
- The gain grows only with batch (1.05x at batch 128), which the family does
  not build.
- The model is sensitive to FP8 activations: even the MLP-only variant changes
  the top-1 class of 3.5% of the images.

### NVFP4 (not tried)

NVFP4 was not built. FP8 already gives no speedup at these GEMM sizes, and a
4-bit format would lose more quality on a model that changes about 3% of its
top-1 classes in FP8.

### What was not tried

| Item | Note |
|---|---|
| FP8 encoder `attn.proj` | build fails on TensorRT 11.4; retest on a newer release |
| Q/DQ on the layer-norm output, fixed softmax-output scale, percentile or entropy calibration | not measured |
| Native runtime with an FP8 bundle, H100 | not measured |

### When to revisit

- The family supports larger batches: FP8 then gives about 1.05x at batch 128.
- A TensorRT release that builds the FP8 `attn.proj` after the fused attention
  and fuses the input Quantize into the previous kernel.
- A calibration or recipe that keeps the top-1 agreement within about 1 point
  of the fp16 engine.

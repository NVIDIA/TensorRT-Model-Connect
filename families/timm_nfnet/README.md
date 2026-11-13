# timm NFNet

This family builds the timm normalizer-free NFNet classifiers (`dm_nfnet_f*`;
`timm/dm_nfnet_f0.dm_in1k` in the premerge manifest, `timm/dm_nfnet_f3.dm_in1k`
in the nightly manifest) as one TensorRT engine: batch 1, the input size of the
checkpoint's `pretrained_cfg`, fp16 in the manifests (fp32 is also supported),
builder optimization level 3. The layout (stage depths, group counts, second 3x3
convolution, squeeze gate) is read from the checkpoint; see `model.py`.

## FP8 (`quantization=fp8`)

No FP8 or NVFP4 checkpoint of these models exists on the Hub, so `quantization=fp8`
quantizes the original checkpoint at build time (fp16 precision only):

| Part | Precision |
|---|---|
| Dense (one-group) convolutions on a spatial map: stem convolutions after the first, `conv1`, `conv3`, `downsample.conv`, the first-stage `conv2` / `conv2b`, `final_conv` | FP8 e4m3: weights per output channel (amax / 448 of the standardised weight), activations per tensor (static scale) |
| First stem convolution (3 input channels), grouped 3x3 convolutions (group width 128), squeeze gate, classifier | fp16 |

The activation scales were calibrated once with ModelOpt `FP8_DEFAULT_CFG` (max
calibration, 500 ImageNet validation images) and are stored in
`fp8_activation_scales.json`, keyed by model config and a weight fingerprint
(`dm_nfnet_f0` 34 scales, `dm_nfnet_f3` 112 scales, one of them `null`: an input that
is almost zero stays fp16). Another checkpoint or input size raises until
`calibrate_fp8_scales.py` is run for it. An already quantized checkpoint is refused.

The grouped 3x3 convolutions stay fp16: in FP8 they make the whole engine slower at
batch 1 (0.78x to 0.84x of fp16 on B200 and B300, TensorRT 11.4).

Measured at batch 1, TensorRT 11.4, CUDA graph, median of 9 interleaved rounds
(x is fp16 time over FP8 time); quality on 1500 held-out ImageNet validation images
against the timm fp32 model:

| Model | GPU | fp16 ms | FP8 ms | x | Top-1 fp16 / FP8 | FP8 top-1 agreement with timm | FP8 KL mean |
|---|---|---|---|---|---|---|---|
| dm_nfnet_f0 | B200 | 0.605 | 0.572 | 1.059 | 81.67% / 81.33% | 98.2% | 0.0054 |
| dm_nfnet_f0 | B300 | 0.579 | 0.561 | 1.033 | 81.67% / 81.33% | 98.2% | 0.0056 |
| dm_nfnet_f3 | B200 | 2.438 | 2.356 | 1.035 | 86.33% / 86.33% | 98.8% | 0.0054 |
| dm_nfnet_f3 | B300 | 2.315 | 2.236 | 1.035 | 86.33% / 86.27% | 98.7% | 0.0052 |

TensorRT 11.1 (B200): 1.044x (dm_nfnet_f0) and 1.036x (dm_nfnet_f3), the same quality. No non-finite output
on 2000 images and 29 stress inputs. The plan is about 20% smaller (dm_nfnet_f0 138
to 111 MiB, dm_nfnet_f3 494 to 403 MiB).

NVFP4 is not supported: TensorRT has no NVFP4 convolution.

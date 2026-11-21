# Qwen-Image build geometry

`image_generation` uses `BuildRequest.image_height` and `image_width` as the
static output size. For the currently supported `image_edit` build, those two
required fields are the raw condition-image size; the output stays at the
checkpoint's default 1024 x 1024 size. The direct family E2E reads the real
condition fixture dimensions before calling `build()`.

# FP8 quantization

`--quantization fp8` quantizes the MMDiT denoiser of an `image_generation` build to FP8
(`quantization.py`). The original BF16 checkpoint is used as-is: weight scales are computed at
build time, and the activation scales are committed in `fp8_activation_scales.json`, keyed by
model config (`qwen_image-h3072-l60-i12288` covers Qwen-Image and Qwen-Image-2512). They were
calibrated once with `calibrate_fp8_scales.py` (ModelOpt `FP8_DEFAULT_CFG` over 32 prompts, 20
denoising steps with guidance). The text encoder, VAE, modulation linears, input and output
projections and the image attention output projection stay in BF16. `image_edit` is not supported.

## NVFP4

NVFP4 was measured and is not shipped: it is faster than FP8, but the quality
loss is more than 2x the FP8 loss, so NVFP4 is not shipped. No NVFP4 code is in
the family.

Setup: the published ModelOpt NVFP4 checkpoint `nvidia/Qwen-Image-NVFP4`
(784 NVFP4 projections in transformer blocks 2 to 57; the other blocks stay
BF16) loaded into the family's MMDiT denoiser, `image_generation`, one job per
GPU and size. A DiT call is the median of 120 calls. LPIPS is the mean over 12
images against the BF16 engine (same prompts and seeds). The non-finite check
covers 480 DiT calls per engine.

| GPU, TensorRT, size | FP8 DiT call ms | NVFP4 DiT call ms (vs FP8) | LPIPS vs BF16 engine, FP8 / NVFP4 (x FP8) | PSNR vs BF16 engine, FP8 / NVFP4 | Non-finite calls |
|---|---|---|---|---|---|
| B200, 11.4, 1024 x 1024 | 95.35 | 78.17 (1.22x) | 0.1005 / 0.2617 (2.60x) | 26.13 / 18.76 dB | 0 of 480 |
| B300, 11.4, 1024 x 1024 | 89.53 | 77.41 (1.16x) | 0.1005 / 0.2617 (2.60x) | 26.13 / 18.76 dB | 0 of 480 |
| B200, 11.4, 512 x 512 | 36.05 | 29.71 (1.21x) | 0.0641 / 0.1494 (2.33x) | 26.33 / 20.63 dB | 0 of 480 |
| B300, 11.4, 512 x 512 | 34.95 | 28.82 (1.21x) | 0.0526 / 0.1651 (3.14x) | 27.22 / 19.91 dB | 0 of 480 |
| B300, 11.1, 1024 x 1024 | 89.86 | 78.45 (1.15x) | 0.1005 / 0.2429 (2.42x) | 26.13 / 18.67 dB | 0 of 480 |

The DiT plan is 37.84 GiB in BF16, 25.80 GiB with FP8 and 12.38 GiB with
NVFP4. The BF16 engine against PyTorch has an LPIPS of 0.034 to 0.045
(1024 x 1024).

Reasons:

- Speed: a DiT call is 1.15x to 1.22x faster than FP8 and the plan is 52%
  smaller.
- Quality: LPIPS is 2.3x to 3.1x the FP8 LPIPS and PSNR drops by 5.7 to
  7.5 dB.
  Weight-only FP4 (BF16 activations, B300, 1024 x 1024) still gives LPIPS
  0.217, so most of the loss comes from the 4-bit weights of the checkpoint.
- No non-finite values.

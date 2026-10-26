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
model shape (`qwen_image-h3072-l60-i12288` covers Qwen-Image and Qwen-Image-2512). They were
calibrated once with `calibrate_fp8_scales.py` (ModelOpt `FP8_DEFAULT_CFG` over 32 prompts, 20
denoising steps with guidance). The text encoder, VAE, modulation linears, input and output
projections and the image attention output projection stay in BF16. `image_edit` is not supported.

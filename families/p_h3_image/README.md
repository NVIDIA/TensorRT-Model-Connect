# P-H3-Image

This family owns literal-prompt `text_to_image` with the released mixed
BF16/FP32 checkpoint. The optional prompt upsampler is disabled. The native
pipeline includes the Qwen3-VL text backbone, text connector, two text refiner
blocks, all 50 denoiser blocks, the four internally denoised audio rows, the
eight-step checkpoint schedule and the complete tiled ViT image decoder.

Build one image with sides divisible by 256 and an area of 2–9 megapixels.
The default profile is 1024×2048, with up to 256 prompt tokens. The direct
test builds a 128-token profile. Parallelism, quantization, editing and prompt
upsampling are outside this profile.

The public Task accepts `seed` as an integer from 0 to 2^32−1. For exact input
replay, `initial_latents` contains float32 image noise in `[24,H/16,W/16]`
channel-major order followed by the `[4,32]` channel-major audio noise. Empty
input selects Gaussian initialization. The result is one float32 HWC RGB image
in `[0,1]`. The image and audio schedules use the source's deterministic blend
update and provided sigma grid without shifting it again.

Tests compare two full native images against the pinned official Diffusers
pipeline, using the public Task API and identical initial noise. The reference
retains its default 256px VAE tiles and 64px minimum overlap.

Checkpoint weights and generated engines are not distributed in this source
tree. Obtain the checkpoint under its published licenses, including the
MiniMax H3 Community License terms.

Native tokenization requires the ICU `uc` and `i18n` development libraries.
Selecting `-DTRTMC_BUILD_FAMILIES=p_h3_image` checks these at configuration time.
A default build may skip this optional family when ICU is absent; explicitly
building `trtmc_model_p_h3_image` then reports the missing prerequisite.

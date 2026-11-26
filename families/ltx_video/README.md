# LTX-Video

This family builds Lightricks LTX-Video (`Lightricks/LTX-Video` in the manifests,
a 2B diffusion transformer with 28 blocks and hidden size 2048) for
`image_generation` of short videos, fp16. A bundle holds three engines: the T5
text encoder, the DiT denoiser and the VAE decoder. With classifier-free
guidance the denoiser runs twice per step.

## Quantization

`quantization=fp8` (fp16 precision) quantizes the original checkpoint at build
time. The two feed-forward linears of every block (`ff.net.0.proj` and
`ff.net.2`, 56 projections) run as FP8 GEMMs: per-tensor weight scale
amax / 448, static activation scales from `fp8_activation_scales.json`, fp16
Dequantize. Attention, modulation, embeddings, the T5 encoder and the VAE stay
fp16. On B300 (TensorRT 11.1 and 11.4) one DiT call is 1.14x to 1.17x faster
than fp16 at 480x704 with 25 to 161 frames (1.05x at 256x256x9), and the
denoiser plan is 0.87 GiB smaller.

### NVFP4

NVFP4 for the feed-forward was measured in the investigation before the FP8
change and is not shipped. Only the quality result of that investigation was
recorded. No NVFP4 code is in the family; `quantization=nvfp4` is refused.

| Item | NVFP4 feed-forward | FP8 feed-forward (shipped) |
|---|---|---|
| Speed vs FP8 and fp16 | not recorded | 1.14x to 1.17x vs fp16 per DiT call (480x704) |
| Quality: LPIPS against the fp16 engine (lower is better) | 0.16 to 0.27 | 0.031 to 0.059 (mean per prompt set and size) |
| Non-finite outputs | not recorded | 0 in 3,360 DiT calls |
| GPU, TensorRT | not recorded | B300, TensorRT 11.1 and 11.4 |

The NVFP4 and FP8 values come from different runs (prompts and calibration
differ), so compare them only roughly.

Reason: the quality loss. With NVFP4 the videos move too far from the fp16
engine (LPIPS 0.16 to 0.27), while the shipped FP8 build is already at 0.031
to 0.059 with a tail of single prompts at 0.11 to 0.22.

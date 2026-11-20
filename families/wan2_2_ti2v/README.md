# Wan2.2 TI2V

`families/wan2_2_ti2v` builds Wan2.2 TI2V-5B (`Wan-AI/Wan2.2-TI2V-5B`) for
text-to-video and image-to-video: a UMT5 text encoder, the 30-block DiT denoiser
and the Wan2.2 VAE as TensorRT plans, run by the family runtime. The DiT is
built in BF16 for fixed generation profiles (1280x704 or 704x1280 with 121
frames and 50 steps, and a 672x384, 5-frame profile for tests).

## Quantization

`--quantization fp8` quantizes the DiT at the 1280x704, 121-frame profile only.
In each of the 30 blocks, `ffn.net.0.proj`, `ffn.net.2` and `attn2.to_q` (90
projections) run as FP8 GEMMs: per-tensor FP8 e4m3 weights computed at build
time (`amax / 448`) and static per-tensor activation scales from the packaged
scale file `data/wan22-ti2v-5b-921dbaf3-fp8-scales.json` (`fp8_profile.py`).
`attn2.to_out.0` stays BF16 because TensorRT 11.4 cannot compile an FP8
Quantize on the output of the fused cross-attention. The text encoder, the VAE,
self-attention, the cross-attention key and value, the modulation and the
input and output projections stay as they are. The FP8 DiT call is about
1.18x faster than BF16 and the plan is 29% smaller.

### NVFP4

`--quantization nvfp4` builds the same 90 DiT projections (`ffn.net.0.proj`,
`ffn.net.2` and `attn2.to_q` in each of the 30 blocks) as NVFP4 W4A4 GEMMs
instead of FP8; everything else is built as in the FP8 build. Like FP8, it
supports only the 1280x704, 121-frame profile, needs `--precision bf16` and
needs a Blackwell GPU:

```bash
python -m tensorrt_model_connect build /path/to/Wan2.2-TI2V-5B \
  --family wan2_2_ti2v --task image_generation --precision bf16 \
  --image-width 1280 --image-height 704 --video-num-frames 121 \
  --quantization nvfp4 -o wan22-ti2v-5b-nvfp4.bundle
```

No ModelOpt NVFP4 checkpoint from NVIDIA exists for Wan2.2 TI2V-5B, so the
original checkpoint is quantized at build time (`quantization.py`). Weights:
packed E2M1 values, one e4m3 scale per 16 inputs and one
`weight_scale_2 = amax / (6 x 448)` per tensor. Activations: dynamic block-16
FP4 quantization (`add_dynamic_quantize`) with a static global scale from
`data/wan22-ti2v-5b-921dbaf3-nvfp4-scales.json`; each value is the FP8
activation scale of the same projection divided by 6, so both recipes use
one calibration. The plan inspector shows a block-scaled FP4 GEMM for each
of the 90 projections.

Speed: DiT plan at 1280x704, 121 frames, 512 text tokens, batch 1. The three
plans were built and timed in the same job per GPU, 12 interleaved rounds of
5 calls, median.

| GPU | TensorRT | Build | Plan | DiT call | vs FP8 |
|---|---|---|---|---|---|
| B200 | 11.4.0.106 | BF16 (`none`) | 9.46 GiB | 494.3 ms | 0.85x |
| B200 | 11.4.0.106 | FP8 | 6.74 GiB | 419.5 ms | 1.00x |
| B200 | 11.4.0.106 | NVFP4 | 5.55 GiB | 410.4 ms | 1.02x |
| B300 | 11.4.0.106 | BF16 (`none`) | 9.46 GiB | 422.9 ms | 0.85x |
| B300 | 11.4.0.106 | FP8 | 6.74 GiB | 359.5 ms | 1.00x |
| B300 | 11.4.0.106 | NVFP4 | 5.55 GiB | 354.7 ms | 1.01x |

TensorRT 11.1.0.106 (B200, same job layout): BF16 500.1 ms, FP8 428.0 ms,
NVFP4 412.1 ms (1.04x of FP8), same plan sizes, no non-finite output.

A video calls the DiT 100 times (50 steps with guidance), so the request
gain is about the per-call gain. The gain is small because NVFP4 only changes
projections that are already FP8; attention, the other projections and the
norms stay the same. The main gain is the plan size (18% smaller than FP8).

Quality: 4 prompts, seed 42, 50 steps, guidance 5, 1280x704x121. The text
encoder and the VAE run in PyTorch; only the DiT plan differs. LPIPS
(AlexNet, mean over the 121 frames) and PSNR of the final video against the
BF16 plan. Per-call error: relative L2 of the noise prediction against the
BF16 plan, open loop, on a random latent at timesteps 999, 500 and 1. CLIP:
text-image score, 4 frames per video, mean over the prompts. B200 and B300
gave the same videos and numbers.

| Build | LPIPS (x FP8) | PSNR | CLIP | Per-call error t=999 / 500 / 1 | Non-finite outputs |
|---|---|---|---|---|---|
| BF16 (`none`) | - | - | 30.8 | - | 0 of 400 calls |
| FP8 | 0.219 | 17.9 dB | 31.7 | 1.4% / 1.6% / 5.1% | 0 of 400 calls |
| NVFP4 | 0.273 (1.25x) | 16.5 dB | 30.2 | 4.9% / 5.4% / 16.9% | 0 of 400 calls |

Video diffusion over 50 steps turns small numeric differences into different
motion and details: the BF16 plan against the PyTorch transformer already has
LPIPS 0.168 (measured in #1945 with the same prompts).

NaN scan: no non-finite output on B200 and B300 in the 400 calls per build of
the video runs and on 12 stress inputs (timesteps 0, 1, 500 and 999, a zero
latent, latents x4, x10 and x30, text embeddings x0, x4 and x8, prompts of 1
and 8 tokens with zero padding, and the image-to-video timestep pattern with
a clean first frame). `ffn.net.2` follows a GELU with bias and its input has
27280 rows, which leaves a partial last 128-row tile; NVFP4 on that input
gave no NaN.

E2E: `tests/manifests/wan22-ti2v-5b-nvfp4.json` builds the NVFP4 bundle, runs
it with the native runtime and compares it with the official reference, with
the same checks as the FP8 case.

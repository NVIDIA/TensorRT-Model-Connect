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

NVFP4 is not shipped: the best variant is 1.04x faster than FP8 per DiT call
on B200 and B300, below the 1.05x needed to ship. It also adds image error on
top of FP8 (LPIPS 1.25x to 1.34x the FP8 LPIPS, per-call error 2.5x to 3.4x
FP8).

No ModelOpt NVFP4 checkpoint from NVIDIA exists for Wan2.2 TI2V-5B, so NVFP4
W4A4 was quantized from the original checkpoint at build time. Weights: packed
E2M1 values, per-16 e4m3 block scales and one `weight_scale_2 = amax / (6 x
448)` per tensor. Activations: dynamic block-16 FP4 quantization with a static
global scale, the packaged FP8 activation scale divided by 6. The plan
inspector shows a block-scaled FP4 GEMM for every NVFP4 projection. Three
variants were built, each with the rest of the FP8 recipe unchanged:

| Variant | NVFP4 | FP8 |
|---|---|---|
| NVFP4 all | `ffn.net.0.proj`, `ffn.net.2`, `attn2.to_q` | - |
| NVFP4 FFN | `ffn.net.0.proj`, `ffn.net.2` | `attn2.to_q` |
| NVFP4 FFN up | `ffn.net.0.proj` | `ffn.net.2`, `attn2.to_q` |

Speed: DiT plan at 1280x704, 121 frames, 512 text tokens, batch 1, TensorRT
11.4.0.106. All plans were built and timed in the same job per GPU, 12
interleaved rounds of 5 calls, median.

| GPU | Variant | Plan | DiT call | vs FP8 |
|---|---|---|---|---|
| B200 | BF16 | 9.46 GiB | 496.6 ms | 0.85x |
| B200 | FP8 (main) | 6.74 GiB | 422.3 ms | 1.00x |
| B200 | NVFP4 all | 5.55 GiB | 412.8 ms | 1.02x |
| B200 | NVFP4 FFN | 5.66 GiB | 404.7 ms | 1.04x |
| B200 | NVFP4 FFN up | 6.20 GiB | 433.1 ms | 0.98x |
| B300 | BF16 | 9.46 GiB | 422.5 ms | 0.85x |
| B300 | FP8 (main) | 6.74 GiB | 360.0 ms | 1.00x |
| B300 | NVFP4 all | 5.55 GiB | 353.9 ms | 1.02x |
| B300 | NVFP4 FFN | 5.66 GiB | 347.5 ms | 1.04x |
| B300 | NVFP4 FFN up | 6.20 GiB | 363.8 ms | 0.99x |

Inside the video runs below, the median DiT call of NVFP4 FFN was 1.06x (B200)
and 1.05x (B300) faster than FP8. These calls ran one plan after the other,
not interleaved. A video calls the DiT 100 times (50 steps with guidance), so
the request gain is about the same as the per-call gain.

Quality: 4 prompts, seed 42, 50 steps, guidance 5, 1280x704x121. The text
encoder and the VAE run in PyTorch; only the DiT plan differs. LPIPS (AlexNet,
mean over the 121 frames) and PSNR against the BF16 plan. Per-call error:
relative L2 of the noise prediction against the BF16 plan, open loop, on a
random latent with timestep 500. B200 and B300 gave the same videos and
numbers.

| Variant | LPIPS (x FP8) | PSNR | Per-call error (x FP8) | Non-finite outputs |
|---|---|---|---|---|
| FP8 (main) | 0.219 | 17.9 dB | 1.6% | 0 of 400 calls |
| NVFP4 all | 0.273 (1.25x) | 16.5 dB | 5.4% (3.4x) | 0 of 400 calls |
| NVFP4 FFN | 0.294 (1.34x) | 16.0 dB | 5.1% (3.1x) | 0 of 400 calls |
| NVFP4 FFN up | 0.286 (1.30x) | 16.2 dB | 4.1% (2.5x) | 0 of 400 calls |

Video diffusion over 50 steps turns small numeric differences into different
motion and details: the BF16 plan against the PyTorch transformer already has
LPIPS 0.168 (measured in #1945 with the same prompts). Measured above that floor, NVFP4 adds about 2x to 2.5x the FP8
loss. The CLIP text-image score (4 frames per video, mean over the prompts)
is 30.8 for BF16, 31.7 for FP8 and 30.1 to 30.7 for NVFP4.

NaN scan: no non-finite output for any variant on B200 and B300, in the 400
calls of the video runs and on six stress inputs (timesteps 1, 500 and 999,
latents x4 and x10, text embeddings x4). `ffn.net.2` follows a GELU with bias,
and its input has 27280 rows, which leaves a partial last 128-row tile. NVFP4
on that input gave no NaN here.

The gain is small because NVFP4 only changes projections that are
already FP8 (FP8 saves about 75 ms per call on B200 against BF16, NVFP4 FFN
about 18 ms more); attention, the other projections and the norms stay the same.
Revisit NVFP4 if a later TensorRT gives a larger FP4 GEMM gain on this model,
or with a recipe that keeps the quality closer to FP8.

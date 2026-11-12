# Wan2.1 text-to-video

This family builds Wan2.1 text-to-video (`Wan-AI/Wan2.1-T2V-1.3B-Diffusers` in
the manifests, a 1.3B diffusion transformer with 30 blocks and hidden size 1536)
for `image_generation` of short videos. A bundle holds the T5 text encoder, the
DiT and the VAE. The manifests build fp16 bundles at 384x672 with 5 frames and
480x832 with 17 frames, with tensor or context parallel size 1 or 4 (the tensor
parallel case uses fp32).

## Quantization

`quantization=fp8` (fp16 precision, no tensor or context parallelism) quantizes
the original checkpoint at build time. `fc1` and `fc2` of all 30 blocks (60
projections) run as FP8 GEMMs: per-tensor weight scale amax / 448, static
activation scales from `fp8_activation_scales.json`, fp16 Dequantize.
Attention, modulation, embeddings, T5 and the VAE stay fp16. On B300 (TensorRT
11.1 and 11.4) one DiT call is 1.10x (81 frames) to 1.19x (17 frames) faster
than fp16 and the DiT plan goes from 2.60 to 1.84 GiB. LPIPS against the fp16
engine is 0.105 to 0.150: scene, colors and motion stay, fine detail (faces,
hair, object layout) changes.

### NVFP4

NVFP4 was measured in the investigation before the FP8 change (B300) and is not
shipped. Only the quality result was recorded; there are no NVFP4 speed or NaN
numbers. No NVFP4 code is in the family; `quantization=nvfp4` is refused.

| Variant | LPIPS vs fp16 engine, 17 frames | LPIPS vs fp16 engine, 81 frames | Visual result |
|---|---|---|---|
| FP8 feed-forward (shipped, 8 prompts) | 0.105 | 0.150 | same scene, fine detail changes |
| NVFP4 feed-forward | 0.255 | 0.118 | faces change, haze |
| NVFP4 q/k/v + feed-forward | 0.316 | 0.153 | faces change, haze |

The NVFP4 values at 81 frames come from a second set of only 2 prompts, so they
are not directly comparable with the 8-prompt FP8 value.

Reason: the quality loss. At 17 frames NVFP4 has 2.4x to 3.0x the LPIPS of the
shipped FP8 build, and the frame sheets show changed faces and haze; the
result was judged not usable.

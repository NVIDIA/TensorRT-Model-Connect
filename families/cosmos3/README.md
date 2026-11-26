# Cosmos3

This family builds the native Cosmos3-Nano text-to-video bundle
(`nvidia/Cosmos3-Nano` in the manifests, bf16, with and without context
parallelism). The denoiser is a dual-stream transformer (an understanding
stream for the text tokens and a generation stream for the video latents); the
bundle also holds the VAE step.

## Quantization

`quantization=fp8` builds the denoiser from the official ModelOpt FP8
checkpoint on the `fp8` revision of `nvidia/Cosmos3-Nano`, using its FP8
weights and static per-tensor scales as stored (no calibration). 12 of the 14
linear layers per transformer layer run as FP8 GEMMs (q/k/v and the three MLP
projections of each stream); Dequantize and the matmul run in fp32 and the
result is cast to bf16. The attention output projections, VAE, embeddings,
norms, time embedder and input and output projections stay bf16. Context
parallelism is not supported with FP8. A denoiser call is 1.14x faster than
bf16.

### NVFP4

NVFP4 was measured and is not shipped: it is faster than FP8, but the quality
loss is more than 2x the FP8 loss, so NVFP4 is not shipped. No NVFP4 code is in
the family.

Setup: no NVFP4 checkpoint exists, so NVFP4 was self-quantized from the BF16
checkpoint (block size 16, e4m3 block scales, activations quantized at run
time, input scale = FP8 checkpoint input scale / 6). "NVFP4 12 projections"
puts the 12 FP8 projections per layer in NVFP4; "NVFP4 MLP" puts only the 6
MLP projections per layer in NVFP4 and keeps q/k/v in bf16. One B200,
TensorRT 11.4, full profile (720p, 189 frames, 4096 text tokens); a denoiser
call is the median of 6 interleaved rounds of 3 calls on a real input.
Quality: 3 prompts, 35 steps. Open-loop error is the mean relative L2 error of
the noise prediction against the bf16 plan along the PyTorch trajectory (22
calls per prompt); LPIPS compares 8 decoded frames with the bf16 plan.

| Variant | Plans GiB | Denoiser call ms (vs FP8) | Open-loop error vs bf16 plan (x FP8) | LPIPS vs bf16 plan | Non-finite outputs |
|---|---|---|---|---|---|
| bf16 | 26.77 | 2445.2 | - | - | 0 of 276 |
| FP8 (shipped) | 15.13 | 2159.8 (1.00x) | 0.0335 (1.0x) | 0.252 | 0 of 276 |
| NVFP4 12 projections | 10.03 | 2006.6 (1.08x) | 0.0990 (2.95x) | 0.404 | 0 of 276 |
| NVFP4 MLP | 12.43 | 2047.9 (1.05x) | 0.0803 (2.40x) | 0.338 | 0 of 276 |

The bf16 plan against PyTorch has an open-loop error of 0.019 to 0.034 and an
LPIPS of 0.178. On B300 (TensorRT 11.4) NVFP4 gives the same quality (open-loop
error 0.0993 and 0.0804) and is 1.16x (12 projections) and 1.15x (MLP) faster
than bf16. Stress inputs (scaled and random latents, random text tokens, t = 0
and t = 1) are finite.

Reasons:

- Speed: a denoiser call is 1.05x to 1.08x faster than FP8 and the plans are
  18% to 34% smaller; attention, norms and the attention output projections
  are not changed by NVFP4.
- Quality: the open-loop error is 2.40x to 2.95x the FP8 error, and LPIPS
  against the bf16 plan rises from 0.252 to 0.338 to 0.404.
- No non-finite values.

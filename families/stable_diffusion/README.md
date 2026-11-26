# Stable Diffusion

This family builds text-to-image bundles for Stable Diffusion UNet models
(`stable-diffusion-v1-5/stable-diffusion-v1-5` and `stabilityai/sdxl-turbo` in
the manifests): the CLIP text encoder (two for SDXL), the UNet denoiser and the
VAE decoder, one square image size per bundle. The UNet is built in fp16 (or
fp32); the SDXL VAE decoder is always fp32.

## Quantization

`--precision fp16 --quantization fp8` quantizes the SDXL-Turbo UNet at build
time (#1949); SD 1.5 and the other model types refuse it. Every Conv2d and
Linear layer is FP8 except `conv_in`, `conv_out`, `conv_shortcut`, the resnet
`conv2` layers and the time and added-condition embeddings (743 layers). The
weight scale is `amax / 448` per tensor; the activation scales are static per
tensor from `fp8_activation_scales.json` (calibrated at 512 and 1024). The
convolutions are what make the plan faster: FP8 Linear layers alone gave no
speedup. Per UNet call FP8 is 1.06x to 1.32x faster than fp16 (depends on GPU
and size) and the plan is 54% smaller; LPIPS against the fp16 engine is 0.07
to 0.12.

### NVFP4

`--precision fp16 --quantization nvfp4` builds the SDXL-Turbo UNet with the
FP8 recipe above and one change: the GEGLU projection (`ff.net.0.proj`, the
first feed-forward projection of each of the 70 UNet transformer blocks) is
NVFP4 W4A4 instead of FP8. The other 673 layers stay FP8, everything else
stays as in the FP8 build. It needs a Blackwell GPU and supports the same
model, sizes (512 and 1024) and precision (fp16 only) as FP8; SD 1.5 and the
other model types refuse it.

```bash
trtmc build stabilityai/sdxl-turbo -o sdxl-turbo-nvfp4.bundle \
  --precision fp16 --quantization nvfp4 --image-height 512 --image-width 512
```

No ModelOpt NVFP4 checkpoint of SDXL-Turbo exists, so the GEGLU weights are
self-quantized from the original checkpoint at build time: E2M1 values with
one e4m3 scale per 16 inputs and one `weight_scale_2 = amax / (6 x 448)` per
weight. The layer input is block-quantized to FP4 at run time
(`add_dynamic_quantize`, per-16 e4m3 block scales) with a static global scale:
the FP8 activation scale of the same layer from `fp8_activation_scales.json`
divided by 6. No other calibration file is needed.

Only the GEGLU projection is NVFP4 because it is the only choice that made the
plan faster than FP8 with an image error close to 2x the FP8 error (study in
#2289, TensorRT 11.4): all 722 Linear layers in NVFP4 were 0.93x to 1.01x of
FP8 with 2.0x to 3.1x the FP8 LPIPS; the GEGLU projection and the
feed-forward output together were 0.97x to 1.00x with 1.7x to 2.3x. Each FP4
GEMM needs a dynamic quantize step in front of it, which cancels most of the
faster GEMM for the smaller layers. TensorRT has no FP4 convolution, so the
convolutions stay FP8. The feed-forward output (`ff.net.2`, input after the
GEGLU bias and GELU) stays FP8.

Speed: UNet call, batch 1, all plans built by the family builder and timed in
the same job per GPU (TensorRT 11.4.0.106), median of 4 interleaved rounds of
150 calls, stream launch (CUDA graph gives the same ratios within 0.01).

| GPU | Size | fp16 | FP8 | NVFP4 | NVFP4 vs FP8 | NVFP4 vs fp16 |
|---|---|---|---|---|---|---|
| B200 | 512 | 7.68 ms | 7.17 ms | 6.99 ms | 1.03x | 1.10x |
| B200 | 1024 | 12.60 ms | 11.37 ms | 10.90 ms | 1.04x | 1.16x |
| B300 | 512 | 7.38 ms | 6.38 ms | 6.24 ms | 1.02x | 1.18x |
| B300 | 1024 | 13.54 ms | 10.24 ms | 10.23 ms | 1.00x | 1.32x |

Plan size: fp16 4.80 GiB, FP8 2.58 GiB, NVFP4 2.25 GiB (13% smaller than FP8).

Quality: LPIPS (AlexNet) against the fp16 engine, as for FP8: held-out Parti
prompts (not calibration prompts, 16 at 512 and 12 at 1024), fixed seeds, the
Turbo schedule at 1 and 4 steps, fp32 PyTorch text encoders and VAE so only
the UNet differs. B200 first, B300 in brackets.

| Size, steps | FP8 | NVFP4 | NVFP4 / FP8 |
|---|---|---|---|
| 512, 1 | 0.067 (0.065) | 0.130 (0.133) | 1.9x (2.1x) |
| 512, 4 | 0.113 (0.107) | 0.181 (0.183) | 1.6x (1.7x) |
| 1024, 1 | 0.089 (0.090) | 0.154 (0.148) | 1.7x (1.6x) |
| 1024, 4 | 0.116 (0.117) | 0.179 (0.171) | 1.5x (1.5x) |

Per step (B200, 4 steps, mean over the prompts, each engine on its own
trajectory): the UNet output cosine against the fp16 engine is 0.9999 for
both at step 1 and 0.957 (FP8) and 0.934 (NVFP4) at step 4 (512); the latent
error relative to the fp16 engine after step 4 is 0.28 (FP8) and 0.39 (NVFP4)
at 512, 0.29 and 0.39 at 1024. Against the diffusers fp32 pipeline (E2E case,
512, 1 step) the NVFP4 image has PSNR 20 to 23 dB and SSIM 0.90 to 0.94
over 12 builds on B200 and B300 (FP8: 23.9 to 25.1 dB, 0.95 to 0.96); the
image changes a little from build to build because TensorRT picks the kernels
per build, while the mean LPIPS above changes by at most 0.011 across three
builds. The premerge case `sdxl-turbo-nvfp4` requires PSNR 19 dB and SSIM
0.85.

NaN: no non-finite UNet output on either GPU: every call of the quality runs
(80 at 512, 60 at 1024) and 5 stress inputs per size (random latents and text
embeddings at 1x and 4x scale, timesteps 999, 500 and 1). TensorRT 11.1.0.106
(B200, 512) also builds and runs the NVFP4 plan: 1.08x of FP8, LPIPS 2.0x of
FP8 on 4 prompts, no non-finite output.

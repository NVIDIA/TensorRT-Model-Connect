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

NVFP4 is not shipped: no variant is 1.03x faster than FP8 on both B200 and
B300, and the image error is 1.5x to 3.1x the FP8 error. No ModelOpt NVFP4
checkpoint of SDXL-Turbo exists, so NVFP4 W4A4 was self-quantized from the
original checkpoint: per-16 e4m3 block scales and one `weight_scale_2 = amax /
(6 x 448)` per weight, dynamic block-16 FP4 quantization of the activations
with a static global scale (the FP8 calibration divided by 6). TensorRT has no
FP4 convolution, so the convolutions of the FP8 recipe stay FP8 in every
variant; only Linear layers can be NVFP4:

| Variant | NVFP4 | FP8 |
|---|---|---|
| NVFP4 linear | all 722 Linear layers of the FP8 recipe (attention `proj_in`, `proj_out`, `to_q`, `to_k`, `to_v`, `to_out`, GEGLU projection, feed-forward output) | 21 convolutions |
| NVFP4 feed-forward | GEGLU projection and feed-forward output (140 layers) | the other 603 layers |
| NVFP4 GEGLU projection | GEGLU projection (70 layers) | the other 673 layers |

Speed: UNet call, batch 1, all plans built by the family builder and timed
in the same job per GPU (TensorRT 11.4.0.106), median of 4 interleaved rounds
of 150 calls, stream launch (CUDA graph gives the same ratios within 0.02).

| GPU | Size | fp16 | FP8 (main) | NVFP4 linear | NVFP4 feed-forward | NVFP4 GEGLU projection |
|---|---|---|---|---|---|---|
| B200 | 512 | 7.67 ms | 7.14 ms | 7.27 ms (0.98x FP8) | 7.29 ms (0.98x) | 6.99 ms (1.02x) |
| B200 | 1024 | 12.56 ms | 11.26 ms | 11.18 ms (1.01x) | 11.22 ms (1.00x) | 10.93 ms (1.03x) |
| B300 | 512 | 7.59 ms | 6.51 ms | 7.01 ms (0.93x) | 6.72 ms (0.97x) | 6.42 ms (1.02x) |
| B300 | 1024 | 13.48 ms | 10.40 ms | 10.73 ms (0.97x) | 10.43 ms (1.00x) | 10.17 ms (1.02x) |

Plan size: fp16 4.80 GiB, FP8 2.58 GiB, NVFP4 linear 1.67 GiB, feed-forward
2.07 GiB, GEGLU projection 2.25 GiB. The linear layers are not a small part
of the UNet (in the FP8 plan, GEMM kernels take 63% to 66% of the layer time
and convolution kernels 3% to 5%, B300 layer profile), but NVFP4 adds a
dynamic quantize step in front of each FP4 GEMM (878 non-GEMM layers in the
NVFP4 linear plan against 571 in the FP8 plan at 512), which cancels the
faster FP4 GEMMs.

Quality: LPIPS (AlexNet) against the fp16 engine, as in #1949: held-out Parti
prompts (not calibration prompts, 16 at 512 and 12 at 1024), fixed seeds, the
Turbo schedule at 1 and 4 steps, fp32 PyTorch text encoders and VAE so only
the UNet differs. B200 first, B300 in brackets; ratio to the FP8 LPIPS.

| Size, steps | FP8 (main) | NVFP4 linear | NVFP4 feed-forward | NVFP4 GEGLU projection |
|---|---|---|---|---|
| 512, 1 | 0.069 (0.065) | 0.205 (0.201), 3.0x (3.1x) | 0.145 (0.149), 2.1x (2.3x) | 0.127 (0.126), 1.8x (1.9x) |
| 512, 4 | 0.115 (0.112) | 0.235 (0.253), 2.0x (2.3x) | 0.194 (0.186), 1.7x (1.7x) | 0.173 (0.173), 1.5x (1.5x) |
| 1024, 1 | 0.083 (0.087) | 0.220 (0.230), 2.6x (2.6x) | 0.169 (0.174), 2.0x (2.0x) | 0.148 (0.148), 1.8x (1.7x) |
| 1024, 4 | 0.109 (0.112) | 0.227 (0.234), 2.1x (2.1x) | 0.192 (0.195), 1.8x (1.7x) | 0.170 (0.172), 1.6x (1.5x) |

NaN: no non-finite UNet output for any variant on either GPU: every call of
the quality runs (80 at 512, 60 at 1024) and 5 stress inputs per size (random
latents and text embeddings at 1x and 4x scale, timesteps 999, 500 and 1).
The feed-forward output (input after the GEGLU bias and GELU) runs NVFP4 in
two variants without NaN. No new TensorRT problem was found.

Revisit NVFP4 when TensorRT fuses the dynamic FP4 quantize into the producing
kernel for these layers, or when it has FP4 convolutions.

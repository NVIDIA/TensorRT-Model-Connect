# Nemotron Labs Diffusion

This family builds Nemotron Labs Diffusion, a diffusion language model with a
dense Ministral-style decoder, for `text_generation`
(`nvidia/Nemotron-Labs-Diffusion-8B` in the manifests, bf16). The decoder runs
in two modes: autoregressive (AR) decode and block diffusion, where a block of
masked rows is denoised with bidirectional attention inside the block. A bundle
holds one dual-profile decoder engine (prefill and decode).

## Quantization

`quantization=fp8` (fp16 or bf16) quantizes the original checkpoint at build
time for the 3B and 8B configs. q, k, v, gate, up and down of every layer run
as FP8 GEMMs (per-tensor weight scale amax / 448, static activation scales from
`fp8_activation_scales.json`, ModelOpt `FP8_DEFAULT_CFG` in both modes);
Dequantize and the matmul run in fp32 and the result is cast back. `w_o`, the
diffusion head, embedding and norms stay 16 bit. On the 3B model (B200,
TensorRT 11.1) AR decode is 1.28x faster than bf16.

### NVFP4

NVFP4 was measured and is not shipped: it is faster than FP8, but the quality
loss is more than 2x the FP8 loss, so NVFP4 is not shipped. No NVFP4 code is in
the family.

Setup: Nemotron-Labs-Diffusion-8B, bf16, `max_len` 1024, TensorRT 11.4, one
job per GPU (B200 and B300). NVFP4 self-quantized from the BF16 checkpoint
(block size 16, activations quantized at run time, input scale from the FP8
calibration). "NVFP4 MLP" has gate/up/down in NVFP4 and q/k/v in FP8; "NVFP4
all" has q/k/v/gate/up/down in NVFP4. Decode is one token with a CUDA graph;
the diffusion step is one 32-row block. KL is against the bf16 engine:
WikiText-2 (4 windows of 512 tokens) and 8 masked diffusion blocks. The
non-finite scan covers the quality runs and 3 x 932 stress tokens.

| GPU | Variant | Plan GiB | Decode ms (vs FP8) | Diffusion block ms (vs FP8) | Prefill 512 ms (vs FP8) | WikiText KL (x FP8) / top-1 | Block KL (x FP8) | Non-finite |
|---|---|---|---|---|---|---|---|---|
| B200 | bf16 engine | 15.83 | 3.980 | 5.653 | 15.46 | - | - | 0 |
| B200 | FP8 (shipped) | 9.46 | 3.245 (1.00x) | 4.661 (1.00x) | 10.64 (1.00x) | 0.0084 (1.0x) / 95.5% | 0.0149 (1.0x) | 0 |
| B200 | NVFP4 MLP | 7.02 | 2.870 (1.13x) | 4.292 (1.09x) | 8.57 (1.24x) | 0.0547 (6.5x) / 88.7% | 0.0865 (5.8x) | 0 |
| B200 | NVFP4 all | 6.67 | 3.050 (1.06x) | 4.475 (1.04x) | 8.87 (1.20x) | 0.0806 (9.6x) / 85.8% | 0.1310 (8.8x) | 0 |
| B300 | bf16 engine | 15.83 | 4.260 | 5.558 | 11.71 | - | - | 0 |
| B300 | FP8 (shipped) | 9.46 | 3.133 (1.00x) | 4.542 (1.00x) | 8.92 (1.00x) | 0.0085 (1.0x) / 95.5% | 0.0146 (1.0x) | 0 |
| B300 | NVFP4 MLP | 7.02 | 2.917 (1.07x) | 4.360 (1.04x) | 8.48 (1.05x) | 0.0540 (6.4x) / 88.3% | 0.0816 (5.6x) | 0 |
| B300 | NVFP4 all | 6.67 | 3.127 (1.00x) | 4.558 (1.00x) | 8.70 (1.03x) | 0.0796 (9.4x) / 87.1% | 0.1136 (7.8x) | 0 |

AR greedy decode on 8 held-out prompts (238 tokens), tokens equal to the
Hugging Face bf16 model: bf16 engine 225 / 209 (B200 / B300), FP8 141 / 151,
NVFP4 MLP 52 / 78, NVFP4 all 21 / 34.

Reasons:

- Speed: NVFP4 MLP is 1.07x to 1.13x faster than FP8 at decode and 1.04x to
  1.09x per diffusion block, with a 26% smaller plan. NVFP4 all is not faster
  on B300.
- Quality: KL is 5.6x to 6.5x the FP8 KL for NVFP4 MLP (7.8x to 9.6x for NVFP4
  all), and the AR greedy match with Hugging Face drops from 141 to 151 tokens
  (FP8) to 52 to 78 tokens.
- No non-finite values.

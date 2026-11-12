# LFM2

This family builds the dense LFM2 and LFM2.5 decoders (a hybrid of
short-convolution blocks and grouped-query attention blocks) for
`text_generation` (`LiquidAI/LFM2-350M`, `LFM2-700M`, `LFM2-1.2B` and
`LFM2-2.6B` in the manifests, fp16 or bf16). A bundle holds one static
single-token engine; prompt tokens run through the same engine, so every step
is a GEMV.

## Quantization

`quantization=fp8` quantizes the original checkpoint at build time for
LFM2-350M, 700M, 1.2B, 2.6B and LFM2.5-1.2B-Instruct. Attention q/k/v, MLP
`w1`/`w2`/`w3` and the conv-block `in_proj`/`out_proj` run as FP8 GEMMs
(per-tensor weight scale amax / 448, static activation scales from
`fp8_activation_scales.json`, ModelOpt `FP8_DEFAULT_CFG`); Dequantize and the
matmul run in fp32. The depthwise convolution, attention `out_proj`, embedding,
norms and `lm_head` stay 16 bit. On TensorRT 11.4, B200, decode is 1.20x to
1.38x faster than the unquantized engine.

### NVFP4

NVFP4 was measured and is not shipped: it is faster than FP8, but the quality
loss is more than 2x the FP8 loss, so NVFP4 is not shipped. No NVFP4 code is in
the family.

Setup: LFM2-1.2B (bf16) and LFM2-2.6B (fp16), one job per GPU and TensorRT
version, 4096-token cache. NVFP4 self-quantized from the BF16 checkpoint (block
size 16, activations quantized at run time, input scale from the FP8
calibration). "NVFP4 all" puts every FP8 projection of the shipped recipe in
NVFP4; "NVFP4 MLP" puts only `w1`/`w2`/`w3` in NVFP4 and keeps the rest in FP8.
Decode is one token at position 100 with a CUDA graph. KL is against the
Hugging Face model on wikitext-2 (2056 tokens; 1028 on TensorRT 11.1).

| Model | GPU, TensorRT | FP8 decode ms | NVFP4 all ms (vs FP8) | NVFP4 MLP ms (vs FP8) | KL FP8 | KL NVFP4 all (x FP8) | KL NVFP4 MLP (x FP8) | Plan GiB FP8 / all / MLP |
|---|---|---|---|---|---|---|---|---|
| LFM2-1.2B | B200, 11.4 | 0.939 | 0.825 (1.14x) | 0.850 (1.10x) | 0.0407 | 0.3193 (7.9x) | 0.2036 (5.0x) | 1.25 / 0.83 / 0.92 |
| LFM2-1.2B | B300, 11.4 | 0.920 | 0.862 (1.07x) | 0.883 (1.04x) | 0.0407 | 0.3165 (7.8x) | 0.1991 (4.9x) | 1.50 / 1.08 / 1.17 |
| LFM2-1.2B | B200, 11.1 | 0.922 | 0.858 (1.07x) | 0.867 (1.06x) | 0.0538 | 0.3382 (6.3x) | 0.2259 (4.2x) | 1.25 / 0.83 / 0.92 |
| LFM2-2.6B | B200, 11.4 | 1.760 | 1.536 (1.15x) | 1.554 (1.13x) | 0.0393 | 0.2942 (7.5x) | 0.1713 (4.4x) | 2.56 / 1.58 / 1.75 |
| LFM2-2.6B | B300, 11.4 | 1.735 | 1.596 (1.09x) | 1.630 (1.06x) | 0.0390 | 0.2866 (7.3x) | 0.1634 (4.2x) | 2.81 / 1.83 / 2.00 |
| LFM2-2.6B | B200, 11.1 | 1.749 | 1.581 (1.11x) | 1.582 (1.11x) | 0.0369 | 0.2780 (7.5x) | 0.1581 (4.3x) | 2.56 / 1.58 / 1.75 |

The unquantized engine has KL 0.0012 to 0.0018. Top-1 agreement with Hugging
Face: FP8 0.86 to 0.90, NVFP4 all 0.65 to 0.72, NVFP4 MLP 0.71 to 0.79. No
non-finite logits or cache values in any run (quality runs and 3 x 3032 stress
tokens per engine).

Reasons:

- Speed: NVFP4 all is 1.07x to 1.15x faster than FP8 at decode and the plan is
  28% to 38% smaller.
- Quality: KL is 6.3x to 7.9x the FP8 KL (4.2x to 5.0x for NVFP4 MLP), and
  top-1 agreement falls from about 0.89 to about 0.70.

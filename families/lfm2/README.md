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

### INT4-AWQ

`quantization=int4_awq` (W4A16) quantizes the original checkpoint at build time
for `LiquidAI/LFM2-1.2B` @ `40f3da0d`, following the llama family recipe: the
per-projection AWQ smoothing scales (ModelOpt `INT4_AWQ_CFG`, `awq_lite`) are
committed in `awq_scales/` and selected by float64 sums of checkpoint values;
the build recomputes group-128 block scales (`amax / 7`) and runs
`INT4 constant -> Dequantize -> MatMul` at builder optimization level 3, where
Ampere and Ada fuse the dequantization into the GEMM. The quantized projections
are the FP8 set (conv-block `in_proj`/`out_proj`, attention q/k/v, `w1`/`w2`/`w3`).
Another checkpoint needs its own scale file:
`python -m families.lfm2.calibrate_awq_scales`.

RTX 6000 Ada, TensorRT 11.4, fp16 engine, 512-token cache. Latency is plain
`execute_async_v3`, prefill 128 tokens (two 64-token launches) and decode one
token after them; 5 warmup + 50 timed runs, median of 3 measurements. KL and
top-1 agreement are against Hugging Face fp32 on wikitext-2 (5,120 tokens).

| Engine | Plan GB | Prefill ms | Decode ms | Fused INT4 GEMMs prefill / decode | KL | Top-1 | ARC-Easy (of 300) |
|---|---|---|---|---|---|---|---|
| fp16 | 2.34 | 6.65 | 3.20 | - | 0.0001 | 99.2% | 267 |
| FP8 | 1.34 | 4.49 | 2.16 | - | 0.0146 | 92.3% | 269 |
| INT4-AWQ | 1.00 | 4.93 | 1.86 | 70 / 74 of 86 | 0.126 | 77.4% | 262 |

INT4-AWQ decodes 1.72x faster than fp16 and 1.16x faster than FP8 with a 25%
smaller plan than FP8, and loses more quality than FP8. At optimization level 1
no INT4 GEMM fuses and decode takes 6.29 ms.

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

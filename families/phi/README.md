# Phi

This family builds the dense Phi-3 decoders for `text_generation`
(`microsoft/Phi-3-mini-4k-instruct` in the manifests, fp16, tensor parallel
size 1 and 4). The model uses RMSNorm, RoPE and SwiGLU; the fused `qkv_proj`
and `gate_up_proj` of the checkpoint are split at load. A single-GPU bundle
holds a one-token decode engine and a prefill engine.

## Quantization

`quantization=fp8` (fp16 or bf16, tensor parallel size 1) quantizes the
original checkpoint at build time for Phi-3-mini-4k and Phi-3-medium-4k. q, k,
v, gate, up and down of every layer run as FP8 GEMMs (per-tensor weight scale
amax / 448, static activation scales from `fp8_activation_scales.json`,
ModelOpt `FP8_DEFAULT_CFG`), except a few `down_proj` layers with a massive
input activation. `w_o`, embedding, norms and `lm_head` stay 16 bit. On
TensorRT 11.4, B300, Phi-3-mini decode is 1.25x and Phi-3-medium decode 1.41x
faster than fp16.

### NVFP4

NVFP4 was measured and is not shipped: it is faster than FP8, but the quality
loss is more than 2x the FP8 loss, so NVFP4 is not shipped. No NVFP4 code is in
the family. (On Phi-3-mini only the gate/up layout is faster than FP8.)

Setup: one B300, TensorRT 11.4, fp16, 256-token build. NVFP4 on the MLP
(block size 16, activations quantized at run time, input scale from the FP8
calibration), q/k/v in FP8; "gate/up only" keeps `down_proj` in FP8. KL is
against the Hugging Face bf16 model on 7620 wikitext-2 tokens; the non-finite
scan feeds held-out prompts through prefill and 16 decode steps.

| Model | Variant | Decode plan GiB | Decode ms (vs FP8) | Prefill 256 tokens vs fp16 | WikiText KL (x FP8) | WikiText ppl | ARC-Easy (of 300) | Non-finite |
|---|---|---|---|---|---|---|---|---|
| Phi-3-mini | FP8 (shipped) | 4.05 | 1.920 (1.00x) | 1.42x | 0.0220 (1.0x) | 14.370 | 209 | 0 of 220 prompts |
| Phi-3-mini | NVFP4 MLP | 3.08 | 2.008 (0.96x) | 1.37x | 0.1725 (7.8x) | 16.255 | 208 | 0 of 220 prompts |
| Phi-3-mini | NVFP4 gate/up only | 3.39 | 1.856 (1.03x) | 1.40x | 0.0930 (4.2x) | 15.454 | 208 | not scanned |
| Phi-3-medium | FP8 (shipped) | 14.55 | 4.435 (1.00x) | 1.65x | 0.0358 (1.0x) | 8.192 | 221 | 0 of 220 prompts |
| Phi-3-medium | NVFP4 MLP | 10.17 | 4.057 (1.09x) | 1.67x | 0.3224 (9.0x) | 10.055 | 213 | 0 of 175 prompts |

Reasons:

- Speed: Phi-3-medium decode is 1.09x faster than FP8 with a 30% smaller
  plan. On Phi-3-mini the full MLP layout is slower than FP8, and gate/up only
  is 1.03x faster.
- Quality: KL is 4.2x to 9.0x the FP8 KL; Phi-3-medium WikiText perplexity rises
  from 8.04 (fp16) to 10.06.
- No non-finite values in the scans.

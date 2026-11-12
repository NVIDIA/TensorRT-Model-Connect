# DeepSeek-V2

This family builds DeepSeek-V2 decoders for `text_generation`
(`deepseek-ai/DeepSeek-V2-Lite` and a tiny test checkpoint in the manifests,
fp16, tensor parallel size 1, 2 and 4). The model uses Multi-head Latent
Attention (K/V are decompressed and cached in full) and Mixture of Experts:
one dense MLP layer, then MoE layers with 64 routed experts (top-6) and 2
shared experts.

## Quantization

`quantization=fp8` (fp16, tensor parallel size 1) quantizes only the routed
experts of the original checkpoint at build time, as weight-only FP8 (W8A16:
FP8 weights with one scale per layer and projection, amax / 448; activations
stay fp16; no calibration). The routed experts hold about 93% of the weights,
so the bundle drops from 29.27 to 15.86 GiB. Attention, the dense MLP, the
shared experts, the router, embedding, norms and `lm_head` stay fp16. On
TensorRT 11.4 decode is 1.04x of fp16 on B200 and 0.92x on B300.

### NVFP4

NVFP4 was measured and is not shipped: it is faster than FP8, but the quality
loss is more than 2x the FP8 loss, so NVFP4 is not shipped. No NVFP4 code is in
the family.

Setup: DeepSeek-V2-Lite, fp16, `max_len` 256, TensorRT 11.4, one job per GPU
(B200 and B300). The routed experts are NVFP4 W4A4 (block size 16, one second
scale per layer and projection, activations quantized at run time, input scale
from a max calibration on 64 prompts); everything else as in the FP8 recipe.
TensorRT cannot gather FP4 weights, so the decode engine runs one batched FP4
GEMM over all 64 experts and keeps the 6 selected rows. The NVFP4 engines are
built without the B300 build option `-quantize_ppg=off` that the FP8 build
uses (with it, NVFP4 decode takes 140.5 ms on B300). Decode is one token with a
CUDA graph. KL and top-1 are against the fp16 engine on 508 wikitext-2
positions; the non-finite scan covers these steps and 1080 stress decode
steps.

| GPU | Variant | Plan GiB | Decode ms (vs FP8) | KL vs fp16 engine (x FP8) | Top-1 vs fp16 engine | WikiText ppl | Non-finite |
|---|---|---|---|---|---|---|---|
| B200 | fp16 | 29.27 | 4.577 | - | - | 10.65 | 0 |
| B200 | FP8 (shipped) | 15.86 | 4.414 (1.00x) | 0.00170 (1.0x) | 97.6% | 10.66 | 0 |
| B200 | NVFP4 experts, batched | 9.99 | 4.015 (1.10x) | 0.0343 (20.2x) | 93.7% | 10.83 | 0 |
| B200 | NVFP4 experts, weight-only with gather | 9.99 | 7.329 (0.60x) | 0.0176 (10.4x) | 94.7% | 10.84 | 0 |
| B300 | fp16 | 29.26 | 4.025 | - | - | 10.65 | 0 |
| B300 | FP8 (shipped) | 15.86 | 4.381 (1.00x) | 0.00182 (1.0x) | 97.8% | 10.66 | 0 |
| B300 | NVFP4 experts, batched | 9.99 | 3.877 (1.13x) | 0.0349 (19.1x) | 92.9% | 11.10 | 0 |
| B300 | NVFP4 experts, weight-only with gather | 9.99 | 6.717 (0.65x) | 0.0178 (9.8x) | 95.3% | 10.84 | 0 |

Reasons:

- Speed: the batched NVFP4 experts are 1.10x (B200) and 1.13x (B300) faster
  than FP8 at decode, and the plan is 37% smaller. The weight-only gather
  layout is slower than FP8.
- Quality: KL is 19x to 20x the FP8 KL for the batched layout; even weight-only
  FP4 (no activation quantization) has about 10x the FP8 KL.
- No non-finite values.

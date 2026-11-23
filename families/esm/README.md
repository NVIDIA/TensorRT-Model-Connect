# ESM-2

This family builds ESM-2 protein encoders for the
`masked_text_to_token_scores` task (`facebook/esm2_t33_650M_UR50D` in the
manifests, fp32 and fp16, 256-token window). One engine returns the vocabulary
logits for every row of the window at batch 1. Positions use rotary embeddings
inside each layer, and the token-dropout scale is computed in the graph from
the input ids.

## Quantization

`quantization=fp8` (with `precision=fp16`) quantizes the original checkpoint at
build time. q, k, v, the attention output projection and both MLP projections
of every layer run as FP8 GEMMs (per-tensor weight scale amax / 448, static
activation scales from `fp8_activation_scales.json`, ModelOpt
`FP8_DEFAULT_CFG`); the embedding, norms, attention matmuls and `lm_head` stay
fp16. On TensorRT 11.4, batch 1, 256-token window, esm2_t33_650M_UR50D is 1.14x
(B200) to 1.35x (B300) faster than fp16, with masked-token perplexity +0.9%.

### NVFP4

NVFP4 was measured and is not shipped. The study built real TensorRT engines
(no NVFP4 code is in the family): MLP (2/3 of the weights) in NVFP4, q/k/v and
attention output in FP8, fp16 work type, batch 1, TensorRT 11.4. The 3B rows
used a checkpoint merged into one file, because the family loader read only
one checkpoint file at the time of the study.

| Model | GPU | Window 256 vs FP8 | Window 1024 vs FP8 | Plan vs FP8 | KL at 256 (x FP8) | Perplexity at 256: NVFP4 (FP8, HF) | Non-finite |
|---|---|---|---|---|---|---|---|
| 650M | B200 | 0.97x | 1.02x | -29% | 5.7x | 8.414 (8.080, 8.009) | 0 |
| 650M | B300 | 0.89x | 0.91x | -29% | 5.7x | 8.431 (8.080, 8.009) | 0 |
| 3B | B200 | 0.98x | 1.05x | -29% | 5.3x | 7.361 (7.168, 7.126) | 0 |
| 3B | B300 | 0.90x | 0.94x | -29% | 5.3x | 7.337 (7.168, 7.126) | 0 |
| 650M, all six NVFP4 | B200 | 0.96x | 0.98x | -43% | 8.7x | 8.658 (8.080, 8.009) | 0 |

Reasons:

- Speed: 0.89x to 1.05x of FP8 (faster in 2 of 8 cells, at most 5%). The
  dynamic-quantize kernels (0.2 to 0.6 ms per run at window 256) cost about
  what the FP4 GEMMs save.
- Quality: KL 5.3x to 5.7x that of FP8 (the rule allows about 2x); perplexity
  +5.1% to +5.3% (650M) and +3.0% to +3.3% (3B) against HF, where FP8 is +0.9%
  and +0.6%.
- Size: -29% against FP8, not large enough for the quality cost.
- TensorRT problem: NVFP4 with the fp32 work type does not build on TensorRT
  11.4 (no implementation for the fused node that holds the dynamic quantize).

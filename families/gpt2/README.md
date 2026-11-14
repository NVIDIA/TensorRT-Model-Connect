# GPT-2

This family builds GPT-2 and DistilGPT-2 decoders for `text_generation`
(`distilbert/distilgpt2` fp16 and `openai-community/gpt2` fp32 in the
manifests, tensor parallel size 1 and 4). The model uses LayerNorm, learned
positions, `gelu_new` and a fused `c_attn` QKV projection stored as Conv1D
weights. A bundle holds a one-token decode engine and a dynamic-length prefill
engine.

## Quantization

`quantization=fp8` (with `precision=fp16`, tensor parallel size 1) quantizes the
original checkpoint at build time for DistilGPT-2 and GPT-2. q, k, v, `c_fc`
and the MLP `c_proj` of every layer run as FP8 GEMMs (per-tensor weight scale
amax / 448, static activation scales from `fp8_activation_scales.json`,
ModelOpt `FP8_DEFAULT_CFG`). The attention output projection, `lm_head`,
embeddings and norms stay fp16. On TensorRT 11.4, B200, decode is 1.03x and
prefill (128 tokens) 1.07x to 1.10x faster than fp16.

### NVFP4

NVFP4 was measured and is not shipped. The study built real TensorRT engines
(no NVFP4 code is in the family): NVFP4 on `c_fc` (fc1) and the MLP `c_proj`
(fc2) of every layer, q/k/v in FP8. One job on one B200, TensorRT 11.4,
256-token build; decode is a CUDA graph with one token; the NaN scan feeds 96
real prompts to the prefill engine.

| Model | Variant | Decode plan GiB | Decode vs fp16 | Prefill 128 vs fp16 | KL vs HF (x FP8) | Prompts with non-finite prefill |
|---|---|---|---|---|---|---|
| DistilGPT-2 | FP8 (shipped) | 0.189 | 1.03x | 1.10x | 0.0171 (1.0x) | 0 of 96 |
| DistilGPT-2 | NVFP4 MLP | 0.177 | 0.99x | 1.02x | 0.0870 (5.1x) | 96 of 96 |
| DistilGPT-2 | NVFP4 MLP, fp32 Dequantize | 0.178 | 0.85x | 0.95x | 0.0866 (5.1x) | 0 of 96 |
| DistilGPT-2 | NVFP4 fc1 only | 0.183 | 1.03x | 1.02x | 0.0527 (3.1x) | 0 of 96 |
| GPT-2 | FP8 (shipped) | 0.232 | 1.03x | 1.07x | 0.0333 (1.0x) | 0 of 96 |
| GPT-2 | NVFP4 MLP | 0.209 | 0.98x | 1.00x | 0.0974 (2.9x) | 96 of 96 |
| GPT-2 | NVFP4 MLP, fp32 Dequantize | 0.210 | 0.85x | 0.94x | 0.0966 (2.9x) | 0 of 96 |
| GPT-2 | NVFP4 fc1 only | 0.221 | 1.00x | 1.00x | 0.0715 (2.1x) | 0 of 96 |

Reasons:

- Speed: every finite NVFP4 layout is either slower than fp16 at decode
  (fp32 Dequantize) or no faster than FP8 (fc1 only). Decode at these sizes is
  bound by per-layer launch cost; the weights are only 13% to 16% of decode
  time.
- Quality: NVFP4 MLP has 2.9x to 5.1x the FP8 KL (fc1 only: 2.1x to 3.1x).
- Size: the plan is only 6% (DistilGPT-2) and 10% (GPT-2) smaller than FP8.
- TensorRT problem: with fc1 and fc2 in NVFP4, TensorRT fuses bias, GELU and
  the FP4 quantize of the fc2 input into the fc1 GEMM, and that fused kernel
  returns non-finite prefill rows (TensorRT 11.1 and 11.4). FP8 fc1 with NVFP4
  fc2 does not build.

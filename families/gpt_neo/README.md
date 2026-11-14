# GPT-Neo

This family builds GPT-Neo decoders (`EleutherAI/gpt-neo-125m` in the
manifests) for `text_generation`: learned position embeddings, LayerNorm,
separate q, k and v projections, GELU MLP, alternating global and local
(window 256) attention and a tied `lm_head`. It builds a decode engine and a
prefill engine, fp16, bf16 or fp32, with optional tensor parallelism. The family
has no quantized build.

## FP8 quantization study (not shipped)

FP8 self-quantization of the decoder projections was built and measured for
`gpt-neo-125m` on TensorRT 11.4 (B200 and B300) and 11.1 (B300). It is not
shipped: the FP8 engine is 1.10x faster in decode and 24% smaller, but its KL
against the original model is about 37 times the fp16 engine's and wikitext-2
perplexity rises by 9% to 10%. No subset of projections keeps the quality near
the fp16 engine and also gives a speedup. This section records the recipe, the
method and the numbers, so that the study does not have to be repeated. The
FP8 code was not merged; this family has no FP8 build.

### Summary and decision

| Item | fp16 engine (manifest) | FP8, all 5 projection kinds | FP8 vs fp16 |
|---|---|---|---|
| Decode tok/s, cache 256, B200 / B300 (TensorRT 11.4) | 2456 / 2485 | 2706 / 2729 | 1.10x / 1.10x |
| Prefill 256 tokens, B200 / B300 (TensorRT 11.4) | 0.945 / 1.002 ms | 0.831 / 0.898 ms | 1.14x / 1.12x |
| Plan size, decode engine | 313.6 MiB | 239.5 MiB | -24% |
| KL(PyTorch bf16 \|\| engine) per token, B200 / B300 | 0.0022 / 0.0023 | 0.0832 / 0.0844 | about 37x |
| Next-token top-1 agreement with PyTorch, B200 / B300 | 96.3% / 96.3% | 81.5% / 80.6% | -15 points |
| Wikitext-2 perplexity (PyTorch bf16 58.55), B200 / B300 | 58.52 / 58.57 | 64.06 / 64.44 | +9% to +10% |
| Non-finite logits or K/V | 0 | 0 | none |

Decision: do not ship FP8 for GPT-Neo.

- Quality: every projection kind adds a large error on its own (8 to 22 times
  the fp16 KL each, see Variants). Keeping the first and last layers, the
  attention projections or the MLP in fp16 still leaves the KL 16 to 30 times
  the floor.
- Speed: the only variant with a clear speedup is the full recipe (1.10x
  decode). The variant with the lowest loss (`c_fc` only, KL 8 times the floor)
  gives no speedup (0.99x to 1.01x decode, 0.98x to 0.99x prefill).
- Little to gain on a 125M model: the full recipe removes 74 MiB of weights
  from the decode plan but makes decode only 10% faster; the MLP-only variant
  (54 MiB less) gives 1.00x to 1.02x decode.

### Recipe that was measured

No ModelOpt FP8 or NVFP4 checkpoint of GPT-Neo exists on the Hub, and no
official quantization config exists for it. The recipe follows the dense text
LLM FP8 recipe of the public NVIDIA FP8 checkpoints. Each quantized projection
is built as `input -> Quantize -> Dequantize(fp16) -> MatMul(+bias)`, with the
weight `FP8 constant -> Dequantize(fp16)`.

| Choice | Decision |
|---|---|
| Quantized | `attn.attention.q_proj`, `k_proj`, `v_proj`, `mlp.c_fc`, `mlp.c_proj` of all 12 layers (60 projections) |
| 16 bit | `attn.attention.out_proj` (see below), embeddings, position table, LayerNorms, attention matmuls, tied `lm_head` |
| Weight scale | per tensor, amax / 448, computed from the loaded weight at build time (FP8 weights are never stored) |
| Activation scale | static, per tensor; ModelOpt 0.46.1 `FP8_DEFAULT_CFG` (max calibration) on the PyTorch model in bf16, 192 prompts of 512 tokens (64 code, 64 math, 64 stem from `nvidia/Nemotron-Post-Training-Dataset-v1`, as plain text) |
| Outlier rule | a scale above 20x the median of its kind stays 16 bit; measured: none (largest `c_proj` scale 11x its median) |
| Dequantize type | fp16; no non-finite value (largest projection output of the bf16 model on 200 texts: 5056 for `c_proj`, 2240 for `out_proj`) |

`out_proj` stays fp16 because TensorRT 11.4 cannot build an FP8 Quantize after
the fused attention of this family: a small engine of one attention block fails
with `Input 0's element type (fp8_e4m3) differs from input 1's element type
(half)` in decode and prefill form, with fp16 and with fp32 Dequantize.
TensorRT 11.1 builds it.

### Method

- GPUs: B200 and B300, one GPU per job, all variants of a table built and timed
  in the same job; TensorRT 11.4.0.106 (target) and 11.1.0.106 (B300 only).
- Engines: the family's decode and prefill engines, fp16, `max_sequence_length`
  256 (2048 where noted), batch 1. Decode: one token per step with CUDA graph
  replay; prefill: n tokens; median of 9 interleaved rounds.
- Quality, against `GPTNeoForCausalLM` in bf16 (PyTorch): 1536 teacher-forced
  wikitext-2 test tokens in 12 chunks (KL(PyTorch || engine) per token,
  next-token top-1 agreement, perplexity), 300 ARC-Easy test questions scored by
  the answer-letter logit, 8 short prompts with 24 greedy tokens. None of this
  data is in the calibration set. The fp16 engine is the floor.
- Non-finite scan: prefill of 200 real texts (120 wikitext lines, 80
  calibration prompts, up to 255 tokens) and the teacher-forced decode above;
  logits and K/V outputs.

### Results

Speed and size, full recipe, fp16 Dequantize (x is fp16 time over FP8 time):

| GPU | TensorRT | Decode tok/s fp16 / FP8 | x decode | x prefill 32 / 128 / 256 | Plan MiB decode fp16 to FP8 |
|---|---|---|---|---|---|
| B200 | 11.4 | 2456 / 2706 | 1.10 | 1.09 / 1.11 / 1.14 | 313.6 to 239.5 |
| B300 | 11.4 | 2485 / 2729 | 1.10 | 1.08 / 1.09 / 1.12 | 313.6 to 239.5 |
| B300 | 11.1 | 2359 / 2682 | 1.14 | 1.09 / 1.09 / 1.12 | 313.6 to 239.5 |

B300, TensorRT 11.4, cache 2048: decode 1816 / 1894 tok/s (1.04x), prefill 512
and 2048 tokens 1.20x and 1.13x. The bundle through `build()` (B200) is 631.7
MiB fp16 and 483.4 MiB FP8.

Quality (fp16 Dequantize):

| Engine | GPU, TensorRT | KL | Top-1 agree | Wikitext-2 ppl | ARC-Easy (PyTorch 74/300) | Greedy tokens equal to PyTorch |
|---|---|---|---|---|---|---|
| fp16 engine | B200, 11.4 | 0.0022 | 96.3% | 58.52 | 73/300 | 148/192 |
| FP8 | B200, 11.4 | 0.0832 | 81.5% | 64.06 | 75/300 | 95/192 |
| fp16 engine | B300, 11.4 | 0.0023 | 96.3% | 58.57 | 73/300 | 150/192 |
| FP8 | B300, 11.4 | 0.0844 | 80.6% | 64.44 | 75/300 | 72/192 |
| fp16 engine | B300, 11.1 | 0.0024 | 96.2% | 58.57 | 73/300 | 148/192 |
| FP8 | B300, 11.1 | 0.0858 | 80.8% | 63.65 | 73/300 | 101/192 |

ARC-Easy is near chance for a 125M model and shows only that the answers did
not collapse. The activation scales are not the cause: 8 of 72 projection inputs
exceed the calibrated range on the held-out texts, by at most 1.27x, and
multiplying every activation scale by 1.5 does not help (KL 0.0913).

Variants, TensorRT 11.4. B300 rows come from one job, except the MLP-only row,
which comes from the job of the tables above (the full-recipe KL is 0.0871 here
and 0.0844 there, so the resolution is about 0.003 KL). B200 rows come from the
B200 job of the tables above.

| Variant | FP8 projections | KL | Top-1 agree | Wikitext-2 ppl | x decode | x prefill 256 | Plan MiB |
|---|---|---|---|---|---|---|---|
| fp16 engine | 0 | 0.0023 | 96.3% | 58.57 | 1.00 | 1.00 | 313.6 |
| full recipe | 60 | 0.0871 | 81.6% | 64.03 | 1.09 | 1.13 | 239.5 |
| q, k, v only | 36 | 0.0502 | 86.4% | 62.42 | 1.07 | 1.00 | 293.4 |
| `c_fc` and `c_proj` only (MLP) | 24 | 0.0366 | 87.5% | 60.68 | 1.02 | 1.12 | 259.7 |
| MLP only, B200 | 24 | 0.0366 | 87.7% | 60.52 | 1.00 | 1.14 | 259.7 |
| `c_fc` only | 12 | 0.0183 | 91.3% | 59.58 | 0.99 | 0.99 | 286.6 |
| `c_fc` only, B200 | 12 | 0.0180 | 91.7% | 59.95 | 1.01 | 0.98 | 286.6 |
| `c_proj` only | 12 | 0.0253 | 89.5% | 60.45 | 0.99 | 1.06 | 286.7 |
| all but `c_proj` | 48 | 0.0685 | 83.6% | 63.28 | 1.10 | 0.99 | 266.4 |
| all but q and k | 36 | 0.0490 | 85.8% | 61.48 | 0.99 | 1.13 | 253.0 |
| all, first and last 2 layers fp16 | 40 | 0.0455 | 85.6% | 61.29 | 1.05 | 1.08 | 264.3 |
| all, activation scales x1.5 | 60 | 0.0913 | 81.4% | 63.96 | 1.10 | 1.12 | 239.5 |

Dequantize type, full recipe, TensorRT 11.4 (fp32 Dequantize plus a cast back
to fp16):

| Dequantize | GPU | KL | x decode | x prefill 32 / 128 / 256 |
|---|---|---|---|---|
| fp16 | B200 | 0.0832 | 1.10 | 1.09 / 1.11 / 1.14 |
| fp32 | B200 | 0.0839 | 1.01 | 1.19 / 1.19 / 1.20 |
| fp16 | B300 | 0.0844 | 1.10 | 1.08 / 1.09 / 1.12 |
| fp32 | B300 | 0.0831 | 0.99 | 1.44 / 1.41 / 1.39 |

fp32 Dequantize does not change the quality; it moves the gain from decode to
short prefill.

### NVFP4 (not tried)

NVFP4 is lossier than FP8, and FP8 already costs about 37 times the fp16 KL
(16 times with the MLP only). It pays only when decode is bound by weight
reads; here the FP8 MLP, which halves those weights, gives 1.00x to 1.02x
decode, so halving them again cannot give a measurable gain.

### What was not tried

| Item | Note |
|---|---|
| FP8 `out_proj` | does not build on TensorRT 11.4 (see Recipe) |
| Per-channel weight scales, percentile or SmoothQuant calibration | not measured |
| `gpt-neo-1.3B` and `gpt-neo-2.7B` (more weight traffic per token) | not calibrated or built |
| Batch above 1, native runtime with an FP8 bundle | not measured (the family builds batch 1) |

### When to revisit

- A larger GPT-Neo checkpoint in the manifests: decode would be more bound by
  weight reads, so FP8 might pay; it still needs a quality check, because the
  loss here comes from every projection kind.
- A calibration method that brings the KL of the full recipe near the fp16
  engine (for example per-channel weight scales or SmoothQuant).

# Qwen3-Omni

This family builds the Thinker of `Qwen/Qwen3-Omni-30B-A3B-Instruct` for
`text_generation` as one TensorRT engine: the Qwen3 MoE text decoder (48 layers,
hidden size 2048, 32 query and 4 key-value heads, 128 routed experts with top-8 softmax routing, expert
intermediate size 768), bf16, batch 1. The audio and vision encoders, the talker
and the code-to-wave decoder are not part of the engine. The family has no
quantized build: `quantization` other than `none` is rejected.

## FP8 quantization study (not shipped)

FP8 self-quantization of the Thinker was built and measured on TensorRT 11.1 on
B200 (the expert path was also checked on a tiny model on TensorRT 11.4, B200 and
B300). It is not shipped: FP8 routed experts give wrong output (logit cosine
about -0.09 against the bf16 engine), and even before that the expert path is
not faster (decode 0.93x, 64-token prefill 0.66x of bf16). FP8 attention q/k/v
alone is correct but covers under 2% of the weights: decode 1.05x, plan -0.8%,
perplexity +1.8%. This section records the recipe, the method and the numbers,
so that the study does not have to be repeated. The FP8 code was not merged;
this family has no FP8 build.

### Summary and decision

B200, TensorRT 11.1, one job, max sequence length 256 (the manifest value).
Speed ratio is bf16 time over FP8 time.

| Item | bf16 (manifest) | FP8 attention only | FP8 experts only | FP8 both |
|---|---|---|---|---|
| Plan size | 61.1 GB | 60.6 GB (-0.8%) | 32.1 GB (-47%) | 31.6 GB (-48%) |
| Decode step, ms (ratio) | 7.81 | 7.41 (1.05x) | 8.39 (0.93x) | 7.99 (0.98x) |
| Prefill of 64 tokens, ms (ratio) | 157.2 | 157.5 (1.00x) | 236.7 (0.66x) | 236.5 (0.66x) |
| Perplexity, 12 x 128 wikitext tokens (Hugging Face bf16: 14.945) | 14.889 | 15.150 | 545,097 | 653,873 |
| Logit cosine vs bf16 engine, mean | 1 | 0.9964 | -0.084 | -0.097 |
| Top-1 agreement vs bf16 engine | 1 | 92.6% | 0% | 0% |
| Non-finite logits | 0 | 0 | 0 | 0 |

Decision: do not ship FP8 for Qwen3-Omni.

- Correctness: the FP8 expert path (FP8 stack, Dequantize, then Gather of the
  selected experts) gives wrong output at full size. The same graph on a tiny
  random model already loses accuracy on TensorRT 11.1 and gives NaN on 11.4
  (see "Tiny-model check"), while the same FP8-rounded weights stored as a bf16
  constant are exact. The problem is the
  run-time Dequantize + Gather of an FP8 constant, not the scale values.
- Speed: even ignoring correctness, the FP8 expert engine is not faster. Decode
  is flat or slower and prefill is 1.5x slower. A second execution context of
  the FP8 engine asked for 19.3 GB of scratch memory (about 403 MB per layer),
  so TensorRT seems to write the dequantized expert stacks to memory instead of
  reading FP8 weights in the matmul.
- FP8 attention only: correct, but q, k and v hold 0.5B of the 30.5B Thinker
  parameters; the gain (decode 1.05x, 0.5 GB) does not pay for the quality
  loss (perplexity 14.889 to 15.150, greedy 24-token match 69% vs the bf16
  engine).

### Recipe that was measured

No ModelOpt (`nvidia/*`) FP8 or NVFP4 checkpoint of Qwen3-Omni exists on the
Hub; the quantized repositories are community exports (128 x 128 block FP8 or
compressed-tensors FP8 with dynamic activation scales, third-party NVFP4, AWQ,
GPTQ, GGUF, MLX). So the original bf16 checkpoint was quantized at build time.

Parameter split of the Thinker (30.5B): routed experts 29.0B (95%), attention
0.9B (q, k, v 0.5B), embedding and `lm_head` 0.6B.

| Part | Decision |
|---|---|
| Attention q, k, v (W8A8) | FP8 weight, per-tensor scale amax / 448; static per-tensor activation scale, max calibration on 192 prompts of `nvidia/Nemotron-Post-Training-Dataset-v1` run through the bf16 engine (144 scales, keyed by model config); `Quantize -> Dequantize(fp32) -> MatMul(fp32) -> Cast` |
| Routed experts gate, up, down (W8A16) | one FP8 constant per projection and layer, `[experts, in, out]`, one scale per expert (amax / 448); `Dequantize(axis 0)` on the whole stack, then the existing Gather of the top-8 experts and batched matmuls in bf16 |
| Kept bf16 | attention output projection, attention, router, embedding, `lm_head`, norms |

TensorRT does not accept a Gather directly on an FP8 constant, and it folds a
Dequantize of a constant into a bf16 constant at build time (no size gain). To
keep the stack in FP8, the expert scale was multiplied by a value that is 1 at
run time but is computed from the router indices, so it cannot be folded.

### Method

- GPU: B200, one GPU per job; TensorRT 11.1.0.106. The engines were built from
  the family code at `4f9ab925` with the FP8 change applied; the bf16 engine is
  the unchanged family build.
- Quality: 12 wikitext-2 test texts (longer than 600 characters, no headings),
  128 tokens each, teacher-forced one token per step: perplexity, logit cosine,
  top-1 and KL against the bf16 engine and against the Hugging Face
  `Qwen3OmniMoeForConditionalGeneration` Thinker in bf16. Free running: 24
  greedy tokens after a 32-token prefix of each text. Non-finite scan of all
  logits.
- Speed: decode step at a cache length of a few tokens, mean of 60 steps after
  5 warm-up steps (the step includes the Python input copy; engine-only
  re-execution gives about the same ratios: 6.91, 6.52, 7.48, 7.09 ms); prefill of 64
  tokens, best of 5 runs. All variants in the same job, built and timed one
  after the other (not interleaved).

### Results

Quality against the Hugging Face bf16 Thinker (same 12 x 128 tokens):

| Engine | Perplexity | Cosine mean / min | Top-1 | KL mean | Greedy match (24 tokens) |
|---|---|---|---|---|---|
| Hugging Face bf16 | 14.945 | - | - | - | - |
| bf16 engine | 14.889 | 0.9983 / 0.939 | 94.8% | 0.0104 | 79.9% |
| FP8 attention only | 15.150 | 0.9964 / 0.874 | 92.6% | 0.0199 | 55.6% |
| FP8 experts only | 545,097 | -0.084 / -0.515 | 0% | 11.05 | 0% |
| FP8 both | 653,873 | -0.096 / -0.542 | 0% | 11.24 | 0% |

Build, B200, TensorRT 11.1:

| Engine | Weight load s | Build s | Plan bytes |
|---|---|---|---|
| bf16 | 78 | 310 | 61,117,691,308 |
| FP8 attention only | 76 | 292 | 60,619,197,004 |
| FP8 experts only | 167 | 292 | 32,125,842,996 |
| FP8 both | 160 | 282 | 31,627,355,468 |

### Tiny-model check of the expert path

Random Thinker with 2 layers, hidden size 64, 8 experts (top 2), expert size
64; FP8 experts only; logit cosine against the bf16 engine of the same weights.

| Expert graph | TensorRT 11.1, B200 and B300 | TensorRT 11.4, B200 and B300 |
|---|---|---|
| FP8 stack, Dequantize with per-expert scale x run-time 1, Gather (the recipe) | 0.9921 | NaN |
| Same, run-time value added as `scale + 0` or multiplied as ones | 0.9921 | NaN |
| FP8 stack, single scalar scale, Gather (checks the plumbing; wrong scale values) | 0.9977 | 0.9976 |
| FP8 stack, Dequantize of a plain constant scale (TensorRT folds it to bf16) | 0.99997 | 0.99997 |
| FP8-rounded weights dequantized on the host, bf16 constant | 0.99997 | 0.99997 |

Storing the FP8-rounded weights as a bf16 constant keeps the result exact, so
the FP8 rounding is not the cause. Every graph that keeps the stack in FP8 and
dequantizes it at run time before the Gather loses accuracy on TensorRT 11.1,
with any form of the scale; at full size (128 experts, 48 layers) the error is
total. On TensorRT 11.4 the same graph gives NaN logits on the tiny model
(B200 and B300), so 11.4 does not fix it. The full model was not rebuilt on
11.4.

### NVFP4 (not tried)

The expert path is the limit for any low-precision weight format here, and the
NVFP4 exports on the Hub are third-party, not `nvidia/*`, with a different
expert layout.

### What was not tried

| Item | Note |
|---|---|
| B300 and TensorRT 11.4 on the full model | not measured (only the tiny-model check above ran there) |
| Interleaved timing rounds, longer sequences, batch above 1 | not measured |
| FP8 attention output projection, FP8 `lm_head` | not measured |
| An FP8 grouped GEMM (MoE) path for the experts | not available in this family |

### When to revisit

- A TensorRT release where Dequantize + Gather of an FP8 constant gives
  correct results and reads FP8 weights inside the expert matmul (no
  dequantized copy in scratch memory).
- An FP8 grouped-GEMM expert path in this family.

The bf16 baseline itself builds on main, runs and matches Hugging Face (cosine
0.9983, perplexity 14.889 vs 14.945); no problem on main was found.

# K2-Horizon-7B-Uno

This family builds `IFM/K2-Horizon-7B-Uno` for `text_generation`: the
K2-Horizon-7B base (bf16) plus the Uno conditional LoRA adapter, as one dynamic
TensorRT engine that takes 1 to 8 rows per call and a per-row `lora_mask`. Rows
with mask 1 apply the adapter (draft rows); rows with mask 0 run the base model
(prompt, verify and autoregressive rows). Decode is greedy linear speculative
decoding: per cycle one draft call (`[seed, 7 noise tokens]`, mask
`[0, 1, ..., 1]`) and one verify call (8 rows, mask 0). The committed output
equals greedy autoregressive decoding of the base model. The manifest builds the
model in bf16 with `max_sequence_length` 256. Batch size is 1.

## Engine profiles and CUDA graph

The engine has two optimization profiles over the same weights. Profile 0
takes one `lora_mask` value per row and runs the LoRA branch; profile 1 takes
an empty `lora_mask` and gives the LoRA GEMMs no rows. The native runtime
deserializes the engine once, keeps one execution context per profile on the
same KV cache, and sends a call to profile 0 only when its mask has a nonzero
value (draft calls). Prompt, verify and autoregressive calls (mask all 0) use
profile 1; their logits and K/V are bit-identical to the same call with an
all-0 mask on profile 0 and on the single-profile engine of earlier versions.
Draft calls on profile 0 run the LoRA rows through other GEMM kernels than
before, so draft logits differ in the last bits (max abs 0.4 to 1.8 on one
test call). This changes only which draft tokens are proposed at near ties;
the committed tokens are the verify argmax and did not change on any measured
prompt (48 of 48 runs; 3150 to 3162 forwards against 3156). Bundles built
before this change have one profile and must be rebuilt.

The draft context replays a CUDA graph (8 rows in every cycle except the
last). The base context runs without a graph: its row count changes in every
request (prompt chunks of 8, k and 1 rows, then verify blocks), and a recapture
(2 to 13 ms) costs more than a replay saves (about 0.2 ms per call).

One call, 8 rows, `max_sequence_length` 256, TensorRT 11.4, ms (in brackets:
1024):

| Call | B200 before | B200 now | B300 before | B300 now |
|---|---|---|---|---|
| Draft (mask `[0, 1, ..., 1]`) | 5.51 (5.64) | 4.73 (4.88), graph | 5.06 (5.12) | 4.91 (5.00), graph |
| Verify or prompt (mask all 0) | 5.51 (5.63) | 4.53 (4.59) | 5.04 (5.12) | 4.16 (4.18) |
| 1 row (AR, last prompt row) | 5.61 | 4.56 | 5.13 | 4.13 |

Native runtime, 16 chat prompts x 3 rounds, 128 new tokens, greedy, same
tokens as before on every run: speculative decode 141.6 -> 151.6 tokens/s
(B200, 1.07x) and 148.9 -> 151.6 (B300, 1.02x); prompt time per request 26.6
-> 22.2 ms and 24.2 -> 20.6 ms; autoregressive mode 1.16x on both GPUs. The
end-to-end gain is smaller than the per-call gain because the host side of a
call (copy of the float32 logits and the argmax over 8 x 250624 values on the
CPU, about 7 ms per 8-row call) is longer than the engine time.

## FP8 quantization study (not shipped)

FP8 self-quantization of the base projections was built and measured on
TensorRT 11.4 (B200, B300) and 11.1 (B200), with the recipe of the merged
k2_horizon FP8 build (#2021). It is not shipped: in the measured setup without
a CUDA graph every FP8 layout is slower per call than bf16 (0.69x to 0.96x),
FP8 lowers the draft acceptance, and end-to-end speculative decode is 0.84x to
0.88x of bf16. This section records the recipe, the method and the numbers, so
that the study does not have to be repeated. The FP8 code was not merged; this
repository has no FP8 build for K2-Horizon-7B-Uno.

### 1. Summary and decision

TensorRT 11.4, `max_sequence_length` 1024, one 8-row call at cache position
504, ms (ratio = bf16 time / FP8 time).

| Item | bf16 | FP8 (recipe) | Change |
|---|---|---|---|
| 8-row call, CUDA graph, B200 / B300 | 5.32 to 5.69 / 4.96 to 5.00 | 5.38 to 5.91 / 5.52 to 5.58 | 0.96x to 0.99x / 0.89x |
| 8-row call, no CUDA graph (enqueue + sync), B200 / B300 | 5.61 / 5.13 | 5.83 / 5.84 | 0.96x / 0.88x |
| Tokens per forward (40 chats x 128 tokens), B200 / B300 | 1.552 / 1.552 | 1.438 / 1.488 | -7% / -4% |
| Speculative decode, host loop, tokens/s, B200 / B300 | 250 / 280 | 220 / 236 | 0.88x / 0.84x |
| KL vs Transformers bf16, mean / p99 (B200) | 0.0052 / 0.092 | 0.0180 / 0.159 | 3.5x |
| Top-1 agreement (B200) | 97.4% | 92.9% | -4.5 points |
| ARC-Easy, 300 questions (Transformers bf16 291) | 291 | 291 | same |
| Prompts with NaN or Inf (200 prompts) | 0 | 0 | none |
| Bundle size | 17.44 GiB | 11.38 GiB | -35% |

Decision: do not ship FP8 for K2-Horizon-7B-Uno.

- The bf16 LoRA branch (rank 128 on q, k, v, o, gate, up and down, plus the
  scale, mask and add per projection) is about 20% of a call, and FP8 does not
  touch it.
- The FP8 gain needs TensorRT to run the LoRA branches on auxiliary streams.
  On TensorRT 11.4 the builder does this only for some FP8 graphs; for the
  recipe at 1024 and 4096 it logs `[MS] Multi stream is disabled because the
  stream assignment failed`. Layouts that get auxiliary streams (FP8 in every
  layer) are 1.09x to 1.35x faster in a CUDA graph but lose quality (KL 1.8x to
  2.4x the recipe).
- The native runtime does not use a CUDA graph. Without a graph the
  multi-stream plans are much slower (0.68x to 0.78x), and the single-stream
  FP8 plan is still slower than bf16 (kernels per call 830 to 1134, more
  Quantize kernels).
- FP8 lowers draft acceptance: tokens per forward 1.55 to 1.35 to 1.49 over
  all FP8 layouts (4% to 13% fewer).
- Only the 35% smaller bundle remains.

### 2. Model and engine

| Item | Value |
|---|---|
| Adapter | `IFM/K2-Horizon-7B-Uno` at `ec92bbd768f4a404319625204544782e3377bcd7` (manifest revision) |
| Base | `IFM/K2-Horizon-7B` at `586b03f0fd1fbbf2f13eeafc33749e95ae34dd10` (manifest revision) |
| Model config | hidden 4096, 36 layers, MLP 12288, vocabulary 250624 |
| Adapter config | LoRA rank 128, scale 64, on q, k, v, o, gate, up, down |
| Engine | one dynamic bf16 engine, 1 to 8 rows per call, per-row `lora_mask` |
| `max_sequence_length` | 1024 for the main numbers (manifest: 256); also 256, 2048 and 4096 |

Checkpoints on the Hub:

| Repo | Format | Used |
|---|---|---|
| `IFM/K2-Horizon-7B-FP8` (publisher, base only) | compressed-tensors FP8, 128 x 128 block weight scales, dynamic per-group activations | no: the block-scale format does not build in this family; used only as the recipe source (every Linear except `lm_head` in FP8) |
| `ProCreations/K2-Horizon-7B-Uno-NVFP4` (community) | ModelOpt NVFP4 W4A4 base (every Linear except `lm_head`) plus a bf16 adapter | no: not an official checkpoint, and every layer is NVFP4 |
| `nvidia/*` | none for the base or for Uno | - |

So the weights were self-quantized from the bf16 base.

### 3. The recipe

Same recipe as the merged k2_horizon FP8 build (#2021). Only the base matmul
of each conditional projection is quantized; the LoRA branch stays bf16:

```
activation -> Quantize(FP8) -> Dequantize(bf16) -> MatMul <- Dequantize(bf16) <- FP8 weight constant
activation -> LoRA A (bf16) -> LoRA B (bf16) -> x scale x lora_mask -> add
```

| Choice | Decision | Reason |
|---|---|---|
| Quantized projections | base q, k, v, o, gate, up, down of layers 2 to 35 (237 projections) | As k2_horizon. |
| Kept bf16 | layers 0 and 1, `layer.35.w_down` (input amax 73x the median), LoRA A and B, embedding, norms, `lm_head` | As k2_horizon; outlier rule. |
| Weight scale | per tensor, amax / 448, at build time | From the bf16 base. |
| Activation scale | static, per tensor | ModelOpt `FP8_DEFAULT_CFG` on the base plus the conditional LoRA wrapper (LoRA quantizers off). 512 samples (Nemotron chat, code, math, STEM and cnn_dailymail), each fed as `ids + 7 noise rows` and `ids[:half] + 7 noise rows`, so both verify and draft rows are seen. |
| Dequantize type | bf16 | No non-finite value in the scans (section 5). |
| Dual-GEMM fusion | off | As k2_horizon. |

Draft rows (mask 1) have smaller input amax than verify rows: median ratio
0.80 (MLP 0.43); 7 of 252 inputs are larger on draft rows, at most 2.55x
(`w_o`). The resulting scales equal the k2_horizon scales within 0.5% to 2%.
Re-running the calibration reproduces all 252 entries exactly.

### 4. How the results were measured

| Item | Value |
|---|---|
| GPUs | B200 and B300, one GPU per job; bf16 and FP8 of one GPU type in one job |
| TensorRT | 11.4.0.106 (target); 11.1.0.106 on B200 |
| Harness | Python, the engine of the bundle called directly (not the native runtime) |
| Call speed | one 8-row (and 1-row) call at cache position 504, median time; replayed CUDA graph, and enqueue plus sync without a graph |
| Speculative decode | 40 held-out chats x 128 new tokens, greedy, draft and verify as the native runtime; host loop without a CUDA graph, prefill included |
| Reference | Transformers bf16 (`trust_remote_code`), base model; the committed output of speculative decode is base greedy output |

| Metric | Definition |
|---|---|
| Tokens per forward | committed tokens / engine calls in speculative decode |
| KL, top-1 | wikitext-2, 5080 positions, KL(Transformers bf16, engine) per token; top-1 = engine argmax equals the Transformers argmax |
| Perplexity | wikitext-2, 5120 tokens (Transformers bf16 28.29) |
| ARC-Easy | 300 questions, correct answers |
| Non-finite scan | 200 held-out prompts, prefill and speculative decode up to 64 tokens (about 15700 calls, draft and verify rows), NaN and Inf in logits and KV cache |
| Manifest case | "Reply with the word OK.", 26 tokens, 18 forwards |

### 5. Results

#### Speed of one call

TensorRT 11.4, `max_sequence_length` 1024, 8 rows, ms (ratio vs bf16).

| Variant | B200 CUDA graph | B200 no graph | B300 CUDA graph | B300 no graph | Auxiliary streams |
|---|---|---|---|---|---|
| bf16 | 5.32 to 5.69 | 5.61 | 4.96 to 5.00 | 5.13 | 0 |
| FP8 recipe (layers 0, 1 bf16) | 5.38 to 5.91 (0.96x to 0.99x) | 5.83 (0.96x) | 5.52 to 5.58 (0.89x) | 5.84 (0.88x) | 0 (stream assignment failed) |
| FP8 every layer | 3.96 to 4.52 (1.26x to 1.35x) | 7.16 (0.78x) | 4.41 to 4.56 (1.09x to 1.13x) | 7.41 (0.69x) | 3 |
| FP8, layers 0, 1 bf16 except `w_o` | 4.04 to 4.52 (1.25x to 1.32x) | 7.30 (0.77x) | 4.42 to 4.55 (1.10x to 1.12x) | 7.57 (0.68x) | 3 |
| bf16 without the LoRA branch (control) | 4.61 (1.24x) | - | 4.07 (1.23x) | - | - |
| FP8 recipe without the LoRA branch (control) | 3.64 (1.57x) | - | 3.27 (1.53x) | - | - |

Other lengths, CUDA graph, 8 rows: the FP8 recipe is 1.26x (B200) and 1.10x
(B300) at 256, and 0.97x and 0.90x at 4096. The `w_o` variant is 0.97x and
0.90x at 4096 and 0.89x at 2048 (B300): it loses the auxiliary streams there
too. TensorRT 11.1 (B200, 1024, CUDA graph): FP8 recipe 1.11x (8 rows) and
1.16x (1 row).

#### Speculative decode and quality

B200, TensorRT 11.4, `max_sequence_length` 1024 unless noted.

| Engine | Tokens per forward | Host-loop tokens/s | KL mean / p99 | Top-1 | Perplexity | ARC-Easy | Non-finite prompts |
|---|---|---|---|---|---|---|---|
| Transformers bf16 | - | - | - | - | 28.29 | 291 | - |
| bf16 | 1.552 | 250 to 258 (B300 280 to 281) | 0.0052 / 0.092 | 97.4% | 28.25 | 291 | 0 |
| FP8 recipe | 1.438 (B300 1.488) | 220 (B300 236) | 0.0180 / 0.159 | 92.9% | 28.39 | 291 | 0 |
| FP8 every layer | 1.384 | 175 (B300 174) | 0.0438 / 0.698 | 91.4% | 28.28 | 290 | not run |
| FP8, layers 0, 1 bf16 except `w_o` | 1.399 | 186 (B300 179) | 0.0333 / 0.497 | 91.8% | 28.24 | 289 | 0 |
| FP8 recipe, TensorRT 11.1 | 1.346 | 147 (bf16 217) | 0.0181 / 0.153 | 92.8% | 28.45 | 288 | 0 |
| FP8 with the k2_horizon scales | 1.542 | 236 | 0.0185 | 93.3% | 28.54 | 289 | not run |

End to end, speculative decode in the host loop is 0.88x (B200) and 0.84x
(B300) of bf16 on TensorRT 11.4 (same job), and 0.68x on 11.1. The manifest
case passes for the FP8 recipe on 11.4 and 11.1; it fails with the k2_horizon
scales (the text departs at a near tie).

### 6. NVFP4 (measured, not shipped)

NVFP4 for the MLP (gate, up, down of layers 2 to 35), FP8 elsewhere as in the
recipe, dual-GEMM fusion on.

| Item | bf16 | NVFP4 MLP | Change |
|---|---|---|---|
| Plan size | 17837 MiB | 9515 MiB | -47% |
| 8-row call, CUDA graph, `max_sequence_length` 256 to 4096, B200 / B300 | - | - | 1.27x to 1.35x / 1.11x to 1.14x |
| Tokens per forward, B200 / B300 | 1.552 | 1.41 / 1.44 | -9% / -7% |
| Host-loop speculative decode, B200 / B300 | 250 to 258 / 280 to 281 tok/s | 179 / 181 tok/s | 0.69x to 0.72x / 0.64x |
| KL mean (B200) | 0.0052 | 0.092 | 5x FP8 |
| Top-1 / perplexity (B200) | 97.4% / 28.25 | 84.0% / 30.05 | worse |
| Manifest case | pass | fails on B300 | - |

Not shipped: the quality loss is large and speculative decode is slower.

### 7. Found on the way (main, independent of FP8; not fixed here)

1. The Uno native runtime never enables CUDA graphs. A bf16 8-row call runs
   about 830 kernels; the host loop spends about 6 ms per 8-row call against
   5.3 to 5.7 ms in a graph.
2. The LoRA branch is computed for every row of every call, also for prompt and
   verify calls whose mask is all 0. It is 19% to 23% of a call: bf16 without
   the branch is 1.21x to 1.24x faster. Skipping the branch when the mask is
   all 0 (or a second engine profile without LoRA for prompt and verify) would
   speed up bf16 decode more than FP8 does.
   Addressed: see "Engine profiles and CUDA graph" (item 1 for draft calls).
3. TensorRT 11.4 stream assignment fails for some FP8 graphs of this family
   (`[MS] Multi stream is disabled because the stream assignment failed`).
   Without a CUDA graph the multi-stream plans are 1.3x to 1.5x slower than
   bf16 (0.68x to 0.78x), although they are the fastest plans in a graph.
4. Speculative decode equals the engine's own greedy autoregressive output on
   13 of 14 prompts only (bf16): 8-row and 1-row calls differ in the last bits,
   which changes the argmax at near ties.

### 8. What was not tried

| Item | Why it might matter |
|---|---|
| FP8 or NVFP4 for the LoRA branch | The branch is about 20% of a call and stays bf16. |
| The publisher block-scale FP8 checkpoint | The format does not build in this family. |
| The native C++ runtime with an FP8 bundle | Not run; the Python harness called the engine. |
| H100, TensorRT 11.1 on B300 | Not measured. |

### 9. How to reproduce

1. Calibrate per-tensor activation scales with ModelOpt `FP8_DEFAULT_CFG` on
   the base plus the conditional LoRA wrapper, feeding each sample as verify
   rows and as draft rows (section 3).
2. Build the bf16 and FP8 bundles of the checkpoints above with the recipe of
   section 3.
3. Time one 8-row call with and without a CUDA graph, then run speculative
   decode and the quality metrics of section 4 against Transformers bf16.

### 10. When to revisit

- The native runtime replays calls as CUDA graphs, and TensorRT assigns
  auxiliary streams to the FP8 graph reliably.
- The LoRA branch becomes cheaper (skipped for all-0 masks, or a separate
  profile without LoRA), so the base GEMMs dominate a call.
- A need for the 35% smaller bundle that outweighs the lower draft acceptance.

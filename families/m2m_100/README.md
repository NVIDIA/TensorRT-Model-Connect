# M2M-100 / NLLB

This family builds M2M-100 and NLLB-200 translation checkpoints (`model_type`
`m2m_100`) for `text_generation` as two TensorRT engines: the encoder and a
one-token decoder step with a static KV cache. The manifest builds
`facebook/nllb-200-distilled-600M` in fp16 with `max_sequence_length` 256.
Batch size is 1.

## FP8 quantization study (not shipped)

FP8 self-quantization of the encoder and decoder projections was built and
measured on TensorRT 11.4 for `facebook/nllb-200-distilled-600M` and
`facebook/nllb-200-distilled-1.3B`. It is not shipped: end-to-end translation
is slower than fp16 (0.94x to 0.96x) and the decode step is not faster (1.00x
to 1.02x). Quality is close to fp16. This section records the recipe, the
method and the numbers, so that the study does not have to be repeated. The FP8
code was not merged; this repository has no FP8 build for M2M-100 or NLLB.

### 1. Summary and decision

| Item | fp16 | FP8 | Change |
|---|---|---|---|
| End to end (encoder + 64 decode steps), 600M, B200 / B300 | 83.7 / 74.9 ms | 89.0 / 78.6 ms | 0.94x / 0.95x |
| End to end, 1.3B, B200 / B300 | 160.5 / 145.7 ms | 169.2 / 151.0 ms | 0.95x / 0.96x |
| Decode step (CUDA graph), 600M, B200 / B300 | 0.934 / 0.850 ms | 0.923 / 0.846 ms | 1.01x / 1.00x |
| Decode step, 1.3B, B200 / B300 | 1.840 / 1.692 ms | 1.798 / 1.668 ms | 1.02x / 1.01x |
| Encoder (128 tokens), 600M, B200 / B300 | 0.488 / 0.404 ms | 0.429 / 0.402 ms | 1.14x / 1.00x |
| Encoder, 1.3B, B200 / B300 | 1.117 / 1.076 ms | 0.914 / 0.860 ms | 1.22x / 1.25x |
| chrF2 / BLEU vs human reference, 600M (B200; Transformers fp32 58.59 / 33.49) | 58.53 / 33.42 | 58.46 / 33.43 | -0.07 / +0.01 |
| chrF2 / BLEU, 1.3B (B200; Transformers fp32 60.78 / 35.86) | 60.74 / 35.81 | 60.79 / 35.95 | +0.05 / +0.14 |
| KL vs Transformers fp32, 600M / 1.3B (B200) | 3e-6 / 3e-6 | 0.0037 / 0.0028 | small |
| Top-1 agreement, 600M / 1.3B (B200) | 99.9% / 99.9% | 98.1% / 98.3% | -1.8 / -1.6 points |
| Non-finite sentences (300 per model and GPU) | 0 | 0 | none |
| Bundle size, 600M / 1.3B | 2201 / 3644 MiB | 1901 / 2661 MiB | -14% / -27% |

Decision: do not ship FP8 for M2M-100 / NLLB.

- Speed: the decode step at batch 1 is bound by the number of small kernels and
  by the host enqueue, not by weight reads. FP8 halves the projection weights
  but adds 48 (600M) and 96 (1.3B) Quantize layers (decoder plan 355 to 403 and
  703 to 799 layers). The step is 1.00x to 1.02x in a CUDA graph and 0.93x with
  a host sync per step (B300: 1.21 to 1.30 ms; TensorRT enqueue 627 to 666 us).
  The native runtime enqueues each step without a CUDA graph, so end to end FP8
  is 4% to 6% slower. Only the encoder of the 1.3B model gains (1.22x to
  1.25x), and the encoder runs once per sentence.
- Quality: small loss, similar to the merged t5 FP8 and smaller than in the
  bart study. Top-1 agreement is 98.1% to 98.4%; 61% to 65% of greedy
  translations equal the reference output (fp16: 97% to 98%).
- The only clear gain is the 14% to 27% smaller bundle. The bundle is dominated
  by the 256206 x 1024 shared embedding, stored three times (section 7).

### 2. Models and engines

| Item | `facebook/nllb-200-distilled-600M` | `facebook/nllb-200-distilled-1.3B` |
|---|---|---|
| Revision | `f8d333a098d19b4fd9a8b18f94170487ad3f821d` (manifest revision) | `7be3e24664b38ce1cac29b8aeed6911aa0cf0576` |
| Model config | hidden 1024, 12 + 12 layers, MLP 4096, 16 heads, vocab 256206 | hidden 1024, 24 + 24 layers, MLP 8192, 16 heads, vocab 256206 |
| Quantized projections | 156 | 312 |
| `max_sequence_length` (decoder cache) | 256 (manifest) | 256 |
| Engines | encoder (128 source tokens, fixed), decoder step (fp16) | same |

No official FP8 or NVFP4 checkpoint exists for NLLB-200 or M2M-100 on the Hub
(only CTranslate2 int8 and GGUF conversions), so the weights were
self-quantized.

### 3. The recipe

No official FP8 config exists for NLLB. The recipe follows the ModelOpt
guidelines for encoder-decoder text models (as the merged t5 FP8 and the bart
study): quantize the linear projections of the encoder and decoder blocks, keep
everything else in fp16. Every quantized projection is built as:

```
activation -> Quantize(FP8) -> Dequantize(fp16) -> MatMul <- Dequantize(fp16) <- FP8 weight constant
```

| Choice | Decision | Reason |
|---|---|---|
| Encoder projections | q, k, v, `fc1`, `fc2` of every layer | The GEMMs of the encoder block. |
| Decoder projections | self-attention q, k, v; cross-attention q, k, v; `fc1`, `fc2` of every layer | Cross-attention k and v project the encoder output on every decode step. |
| Attention output projections (`out_proj` of self- and cross-attention) | fp16 | TensorRT 11.4 does not build an FP8 Quantize right after the fused attention: on a tiny model the encoder and decoder self-attention output projections fail ("Input 0's element type (fp8_e4m3) differs from input 1's element type (half)") and the cross-attention output projection aborts the process. TensorRT 11.1 builds all three. |
| Embeddings, tied `lm_head`, sinusoidal positions, LayerNorms | fp16 | Not quantized. |
| Activation scale | static, per tensor, one per projection input | ModelOpt `FP8_DEFAULT_CFG` (max calibration) on PyTorch fp32 while the model greedily translates 576 WMT16 validation sentences (cs, de, fi, ro, ru, tr with en, both directions, 48 each, forced target-language token), so the decoder sees decode-time activations. Scale = amax / 448. |
| Weight scale | per tensor, amax / 448, at build time | FP8 weights are quantized from the original checkpoint at build time. A checkpoint that is already quantized is rejected. |
| Dequantize type | engine type (fp16 for the measured builds) | The FP8 GEMM writes fp16 and the bias stays in the epilogue. No non-finite value in 1200 evaluated sentences. The largest input scale is 0.255; the inputs are LayerNorm or ReLU outputs. fp32 Dequantize was not needed. |
| Outlier rule | a scale above 20x the median of the same projection in the other layers is stored as `null` and stays fp16 | Measured: no outlier; 156 of 156 (600M) and 312 of 312 (1.3B) projections quantized. |
| Scale table key | model config plus a weight fingerprint | The two models need their own scales. A missing scale is a build error. |

### 4. How the results were measured

| Item | Value |
|---|---|
| GPUs | B200 and B300, one GPU per job, all builds of one GPU type in one job |
| TensorRT | 11.4.0.106 (target); 11.1.0.106 for a build and a 60-sentence quality check on B200 (section 5) |
| Calibration data | WMT16 validation splits (576 sentences) |
| Evaluation data | WMT16 test splits (newstest2016), 6 pairs, both directions, 25 sentences per direction = 300 sentences per model |
| Reference | Transformers fp32 greedy output, forced target-language token, up to 128 new tokens, Transformers tokenizer framing |
| Harness | Python, the engines of the bundle called directly (not the native runtime); encoder and decoder masks as in the native runtime |

| Metric | Definition |
|---|---|
| KL | mean KL(reference, engine) per token, with the reference greedy output teacher-forced through the engine (forced language token skipped) |
| Top-1 | fraction of those tokens where the engine argmax equals the reference token |
| Greedy equal | fraction of sentences where the free-running engine greedy output equals the reference output |
| chrF2 / BLEU | sacrebleu 2.6.0 corpus scores of the free-running output against the human reference |
| Non-finite scan | NaN and Inf in every encoder output and every logit (teacher-forced and free-running) |

Speed: median of 7 interleaved rounds per build. Encoder: one engine call (128
tokens). Decode step: one replayed CUDA graph of the decoder engine (200
replays per round). End to end: encoder plus 64 decode steps with a host argmax
per step, as the native runtime does. Host enqueue: host wall time of
`execute_async_v3`.

### 5. Results

#### Speed

TensorRT 11.4, batch 1, ms, fp16 / FP8 (ratio = fp16 time / FP8 time).

| Model | GPU | Encoder | Decode step (graph) | End to end (64 steps) |
|---|---|---|---|---|
| 600M | B200 | 0.488 / 0.429 (1.14x) | 0.934 / 0.923 (1.01x) | 83.7 / 89.0 (0.94x) |
| 600M | B300 | 0.404 / 0.402 (1.00x) | 0.850 / 0.846 (1.00x) | 74.9 / 78.6 (0.95x) |
| 1.3B | B200 | 1.117 / 0.914 (1.22x) | 1.840 / 1.798 (1.02x) | 160.5 / 169.2 (0.95x) |
| 1.3B | B300 | 1.076 / 0.860 (1.25x) | 1.692 / 1.668 (1.01x) | 145.7 / 151.0 (0.96x) |

Host cost per step, 600M, B300, fp16 / FP8: enqueue 627 / 666 us; step with a
host sync 1.21 / 1.30 ms (0.93x); end to end at 16 / 32 / 64 / 128 steps
20.5 / 21.9, 40.3 / 43.1, 79.8 / 85.4 and 159.0 / 170.2 ms (0.93x to 0.94x at
every length).

The FP8 decoder plans have 132 (600M) and 264 (1.3B) layers with e4m3 inputs.
The fp16 `lm_head` (256206 x 1024, 0.088 ms in the per-layer profile) is the
largest single kernel of the decode step and stays fp16.

#### Quality

TensorRT 11.4, 300 held-out sentences per row.

| Model | GPU | Build | chrF2 | BLEU | KL | KL p99 | Top-1 | Greedy equal | Non-finite |
|---|---|---|---|---|---|---|---|---|---|
| 600M | - | Transformers fp32 | 58.59 | 33.49 | - | - | - | - | - |
| 600M | B200 | fp16 | 58.53 | 33.42 | 3e-6 | 2e-5 | 99.89% | 97.7% | 0 |
| 600M | B200 | FP8 | 58.46 | 33.43 | 0.0037 | 0.028 | 98.11% | 61.0% | 0 |
| 600M | B300 | fp16 | 58.54 | 33.45 | 6e-6 | 4e-5 | 99.88% | 97.0% | 0 |
| 600M | B300 | FP8 | 58.43 | 33.32 | 0.0036 | 0.026 | 98.26% | 61.0% | 0 |
| 1.3B | - | Transformers fp32 | 60.78 | 35.86 | - | - | - | - | - |
| 1.3B | B200 | fp16 | 60.74 | 35.81 | 3e-6 | 2e-5 | 99.94% | 98.3% | 0 |
| 1.3B | B200 | FP8 | 60.79 | 35.95 | 0.0028 | 0.020 | 98.32% | 63.3% | 0 |
| 1.3B | B300 | fp16 | 60.70 | 35.77 | 4e-6 | 3e-5 | 99.89% | 97.0% | 0 |
| 1.3B | B300 | FP8 | 60.45 | 35.60 | 0.0027 | 0.020 | 98.35% | 65.0% | 0 |

Per direction (600M, B200) chrF2 changes from fp16 to FP8 by -1.32 (fi-en) to
+0.81 (en-tr); no direction breaks. Encoder output cosine against Transformers:
min 0.993, mean 0.999.

TensorRT 11.1 (B200, 600M, 60 sentences): fp16 equals Transformers on 100% of
tokens; FP8 KL 0.0037, top-1 98.2%, chrF2 57.02 (fp16 56.63).

The FP8 code did not change the unquantized builds: the fp16 plans of main and
of the study code have the same layer count (decoder 355, encoder 100), the
same size within 96 bytes, and bit-identical outputs on the same inputs (B200
and B300).

### 6. NVFP4

NVFP4 was not built. NVFP4 pays only when decode is bound by weight reads. Here
the decode step does not get faster when FP8 halves the projection weights
(1.00x to 1.02x), so 4-bit weights would not help either: the step is set by
the kernel count, the host enqueue and the fp16 `lm_head`. The encoder (128
tokens) is compute bound and FP8 already gives at most 1.25x there.

### 7. Found on the way (main, independent of FP8; not fixed here)

1. The native runtime frames the source as `text </s> src_lang` (the legacy
   NLLB framing, see `runtime/request_tokens.h`). The Transformers NLLB
   tokenizer frames it as `src_lang text </s>` (its default since
   `legacy_behaviour=False`). Transformers fp32 600M on the 300 test sentences
   with the runtime framing: chrF2 58.59 to 58.10 (-0.5), BLEU 33.49 to 32.76
   (-0.7); 84 of 300 outputs differ; worst direction en-tr, BLEU 20.12 to
   18.49.
2. The shared 256206 x 1024 embedding is stored three times (encoder gather
   table, decoder gather table, decoder `lm_head`): about 1.5 GiB of the
   2.2 GiB 600M bundle (encoder plan 790 MiB, decoder plan 1389 MiB).
3. Cross-attention K and V are projected from the encoder output on every
   decode step (24 GEMMs of 128 x 1024 x 1024 per step for 600M, see
   `_add_m2m100_decoder_layer` in `model.py`) instead of once per sentence.
   The cost was not timed separately.
4. Each decode step costs about 0.63 ms of host enqueue (B300), close to its
   GPU time (0.85 ms in a CUDA graph); the native runtime does not use a CUDA
   graph.

### 8. What was not tried

| Item | Why it might matter |
|---|---|
| FP8 attention output projections | Do not build on TensorRT 11.4 (section 3); retest on newer releases. |
| fp32 Dequantize, percentile calibration, per-channel weight scales | Could lower the drift. |
| Batch above 1 | The family supports batch 1 only; larger GEMMs would favor FP8. |
| M2M-100 checkpoints, NLLB 3.3B | No scales calibrated. |
| H100, TensorRT 11.1 speed | Not measured. |
| Native runtime with an FP8 bundle | Not run; the Python harness called the engines. |

### 9. How to reproduce

1. Calibrate per-tensor activation scales with ModelOpt `FP8_DEFAULT_CFG` on
   PyTorch fp32 during greedy translation of the calibration sentences.
2. Build the fp16 and FP8 bundles of the checkpoints above with the recipe of
   section 3.
3. Time the engines and run the quality metrics of section 4 against
   Transformers fp32 on the evaluation sentences.

### 10. When to revisit

- A TensorRT release that builds FP8 after the fused attention and gives a
  decode-step gain at batch 1.
- Batch support above 1 in the family, or a native runtime that replays the
  decode step as a CUDA graph.
- A need for the 14% to 27% smaller bundle that outweighs the small drift.

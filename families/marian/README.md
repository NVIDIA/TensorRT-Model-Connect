# Marian

This family builds Marian (OPUS-MT) translation checkpoints (`model_type`
`marian`) for `text_generation` as two TensorRT engines: the encoder and a
one-token decoder step with a static KV cache. The CLI default precision is
fp32; the manifest builds `Helsinki-NLP/opus-mt-en-ru` in fp16 with
`max_sequence_length` 256 (tensor parallel size 1 and 4). Batch size is 1.

## FP8 quantization study (not shipped)

FP8 self-quantization of the encoder and decoder projections was built and
measured on TensorRT 11.4 for `Helsinki-NLP/opus-mt-en-ru`. It is not shipped:
end-to-end translation is slower than fp16 (0.91x on B200, 0.96x on B300), the
decode step is 0.98x to 1.04x and the encoder 0.97x to 0.99x. Quality is close
to fp16 but measurably lower. This section records the recipe, the method and
the numbers, so that the study does not have to be repeated. The FP8 code was
not merged; this repository has no FP8 build for Marian.

### 1. Summary and decision

| Item | fp32 (CLI default) | fp16 (manifest) | FP8 (fp16 engine) | FP8 vs fp16 |
|---|---|---|---|---|
| End to end, 40 held-out sentences, mean per sentence, B200 / B300 | 21.37 / 20.58 ms | 18.53 / 17.77 ms | 20.45 / 18.60 ms | 0.91x / 0.96x |
| End to end, encoder + 64 decode steps, B200 / B300 | 43.0 / 41.3 ms | 38.0 / 36.4 ms | 42.2 / 38.3 ms | 0.90x / 0.95x |
| Decode step (CUDA graph), B200 / B300 | 0.426 / 0.379 ms | 0.377 / 0.376 ms | 0.383 / 0.362 ms | 0.98x / 1.04x |
| Encoder (512 positions), B200 / B300 | 1.134 / 1.090 ms | 0.281 / 0.252 ms | 0.288 / 0.255 ms | 0.97x / 0.99x |
| chrF2 / BLEU vs human reference, B200 (Transformers fp32 54.49 / 28.23) | 54.49 / 28.23 | 54.50 / 28.22 | 54.45 / 28.06 | -0.05 / -0.16 |
| chrF2 / BLEU, B300 | 54.49 / 28.23 | 54.55 / 28.34 | 54.59 / 28.21 | +0.04 / -0.13 |
| KL vs Transformers fp32, mean / p99 (B200) | 0 / 2e-7 | 4e-6 / 2e-5 | 0.0057 / 0.038 | about 1500x |
| Top-1 agreement (B200) | 99.99% | 99.89% | 97.79% | -2.1 points |
| Greedy output equal to the reference (B200) | 99.7% | 96.7% | 55.7% | -41 points |
| Non-finite sentences (300 per build and GPU) | 0 | 0 | 0 | none |
| Bundle size | 424 MiB | 278 MiB | 240 MiB | -13% |

Decision: do not ship FP8 for Marian.

- Speed: the model is small (hidden 512, 6 + 6 layers, about 76M parameters).
  The decode step at batch 1 is bound by the number of small kernels and the
  host, not by weight reads: FP8 adds 26 Quantize layers to the decoder (187 to
  211 layers in the plan) and the step stays at 0.98x to 1.04x in a CUDA graph.
  With a host argmax per step, as the native runtime does, end to end is 4% to
  9% slower. The encoder GEMMs (512 x 512 x 512 and 512 x 512 x 2048) are too
  small for FP8 to win (0.97x to 0.99x).
- Against the fp32 CLI default, FP8 is 1.05x to 1.11x faster end to end, but
  fp16 (already supported, used by the manifest) is 1.15x to 1.16x faster with
  KL 4e-6. FP8 with an fp32 engine (312 MiB) is 1.07x to 1.14x over fp32, still
  not better than fp16.
- Quality: small but clear loss (top-1 97.8%, KL 0.0057, BLEU -0.13 to -0.16).
  No NaN or Inf with the fp16 Dequantize.
- The only clear gain is the 13% smaller bundle; the bundle is dominated by the
  shared embedding stored three times (section 7).

### 2. Model and engines

| Item | `Helsinki-NLP/opus-mt-en-ru` |
|---|---|
| Revision | `bb09c99d180016eac6819df3dae68edb1690fdee` (manifest revision) |
| Model config | hidden 512, 6 + 6 layers, MLP 2048 (SiLU), 8 heads, vocab 62518, post-LayerNorm |
| Quantized projections | 78 (encoder 30, decoder 48) |
| `max_sequence_length` (decoder cache) | 256 (manifest) |
| Engines | encoder (512 source positions, fixed), decoder step |

No official FP8 or NVFP4 checkpoint exists for the OPUS-MT models on the Hub
(only CTranslate2 int8 and ONNX int8 conversions), so the weights were
self-quantized.

### 3. The recipe

No official FP8 config exists for Marian. The recipe follows the ModelOpt
guidelines for encoder-decoder text models (as the merged t5 FP8 and the bart
study): quantize the linear projections of the encoder and decoder blocks, keep
everything else in 16 bit. Every quantized projection is built as:

```
activation -> Quantize(FP8) -> Dequantize(engine type) -> MatMul <- Dequantize(engine type) <- FP8 weight constant
```

| Choice | Decision | Reason |
|---|---|---|
| Encoder projections | q, k, v, `fc1`, `fc2` of every layer | The GEMMs of the encoder block. |
| Decoder projections | self-attention q, k, v; cross-attention q, k, v; `fc1`, `fc2` of every layer | Cross-attention k and v project the 512-row encoder output on every decode step. |
| Attention output projections (`out_proj` of self- and cross-attention) | 16 bit | TensorRT 11.4 does not build an FP8 Quantize right after the fused attention: on a tiny model the encoder and decoder self-attention output projections fail ("Input 0's element type (fp8_e4m3) differs from input 1's element type (half)") and the cross-attention output projection aborts the process. TensorRT 11.1 builds all three. |
| Embeddings, tied `lm_head`, sinusoidal positions, LayerNorms | engine type | Not quantized. |
| Activation scale | static, per tensor, one per projection input | ModelOpt `FP8_DEFAULT_CFG` (max calibration) on PyTorch fp32 while the model greedily translates 512 English sentences of the WMT16 ru-en validation split (newstest2015), up to 128 new tokens, so the decoder sees decode-time activations. Scale = amax / 448, range 0.012 to 0.164. |
| Weight scale | per tensor, amax / 448, at build time | FP8 weights are quantized from the original checkpoint at build time. A checkpoint that is already quantized is rejected. |
| Dequantize type | engine type (fp16 or fp32) | fp16 Dequantize gives no non-finite value in 600 evaluated sentences (B200 and B300). fp32 Dequantize was not needed. |
| Outlier rule | a scale above 20x the median of the same projection is stored as `null` and stays 16 bit | Measured: no outlier, 78 of 78 projections quantized. |
| Scale table key | model config plus a weight fingerprint | A missing scale is a build error. |
| Tensor parallelism | refused with FP8 | The tensor-parallel decoder builder is fp32 only and has no Q/DQ path. |

### 4. How the results were measured

| Item | Value |
|---|---|
| GPUs | B200 and B300, one GPU per job, all builds of one GPU type in one job |
| TensorRT | 11.4.0.106 (target); 11.1.0.106 for a build and a 60-sentence quality check on B200 (section 5) |
| Calibration data | WMT16 ru-en validation, English side, first 512 sentences |
| Evaluation data | WMT16 ru-en test (newstest2016), English to Russian, first 300 sentences (8795 reference tokens) |
| Reference | Transformers fp32 greedy output, up to 128 new tokens |
| Harness | Python, the engines of the bundle called directly (not the native runtime); encoder and decoder masks as in the native runtime |

| Metric | Definition |
|---|---|
| KL | mean KL(reference, engine) per token, with the reference greedy output teacher-forced through the engine |
| Top-1 | fraction of those tokens where the engine argmax equals the reference token |
| Greedy equal | fraction of sentences where the free-running engine greedy output equals the reference output |
| chrF2 / BLEU | sacrebleu 2.6.0 corpus scores of the free-running output against the human reference |
| Non-finite scan | NaN and Inf in every encoder output and every logit (teacher-forced and free-running) |

Speed: median of 7 interleaved rounds. Encoder: one engine call. Decode step:
one replayed CUDA graph of the decoder engine (200 replays per round). End to
end (fixed): encoder plus 64 decode steps with a host argmax per step. End to
end (real): greedy translation until EOS of the first 40 held-out sentences
(about 30.6 output tokens each), sentences interleaved across builds, 3 rounds,
mean per sentence.

### 5. Results

#### Speed

TensorRT 11.4, batch 1, ms. FP8 (fp32) is FP8 with an fp32 engine.

| GPU | Build | Encoder | Decode step (graph) | End to end, 64 steps | End to end, real sentence | ms per output token |
|---|---|---|---|---|---|---|
| B200 | fp32 | 1.134 | 0.426 | 43.0 | 21.37 | 0.700 |
| B200 | fp16 | 0.281 | 0.377 | 38.0 | 18.53 | 0.605 |
| B200 | FP8 (fp16) | 0.288 | 0.383 | 42.2 | 20.45 | 0.670 |
| B200 | FP8 (fp32) | 0.504 | 0.418 | 40.3 | 19.93 | 0.646 |
| B300 | fp32 | 1.090 | 0.379 | 41.3 | 20.58 | 0.674 |
| B300 | fp16 | 0.252 | 0.376 | 36.4 | 17.77 | 0.581 |
| B300 | FP8 (fp16) | 0.255 | 0.362 | 38.3 | 18.60 | 0.608 |
| B300 | FP8 (fp32) | 0.477 | 0.377 | 36.4 | 17.99 | 0.583 |

Per-layer profile of one decode step (B200, sum of layer times): fp16 1.07 ms
in 187 layers (attention 13%); FP8 1.15 ms in 211 layers (FP8 GEMMs and Q/DQ
49%, attention 13%, 16-bit GEMMs 10%). The fp16 `lm_head` (62518 x 512,
0.025 ms) is the largest single layer and stays 16 bit. The plan inspector
finds 72 layers with e4m3 inputs in the FP8 decoder and 36 in the encoder.

#### Quality

TensorRT 11.4, 300 held-out sentences, English to Russian.

| GPU | Build | chrF2 | BLEU | KL | KL p99 | Top-1 | Greedy equal | Encoder cosine min | Non-finite |
|---|---|---|---|---|---|---|---|---|---|
| - | Transformers fp32 | 54.49 | 28.23 | - | - | - | - | - | - |
| B200 | fp32 | 54.49 | 28.23 | 0 | 2e-7 | 99.99% | 99.7% | 1.0000 | 0 |
| B200 | fp16 | 54.50 | 28.22 | 4e-6 | 2e-5 | 99.89% | 96.7% | 1.0000 | 0 |
| B200 | FP8 (fp16) | 54.45 | 28.06 | 0.0057 | 0.038 | 97.79% | 55.7% | 0.9986 | 0 |
| B200 | FP8 (fp32) | 54.18 | 27.78 | 0.0055 | 0.038 | 97.66% | 55.3% | 0.9986 | 0 |
| B300 | fp16 | 54.55 | 28.34 | 6e-6 | 4e-5 | 99.87% | 96.3% | 1.0000 | 0 |
| B300 | FP8 (fp16) | 54.59 | 28.21 | 0.0057 | 0.038 | 97.79% | 54.3% | 0.9986 | 0 |
| B300 | FP8 (fp32) | 54.10 | 27.70 | 0.0056 | 0.039 | 97.66% | 54.7% | 0.9986 | 0 |

TensorRT 11.1 (B200, first 60 sentences): fp16 equals Transformers on 100% of
tokens; FP8 KL 0.0058, top-1 97.6%, chrF2 52.82 (fp16 53.87). No NaN or Inf.

The FP8 code did not change the unquantized builds: the fp16 plans of main and
of the study code have the same layer count (decoder 187, encoder 58) and the
same size within 112 bytes.

### 6. NVFP4

NVFP4 was not built. NVFP4 pays only when decode is bound by weight reads. Here
the decode step does not get faster when FP8 halves the projection weights
(0.98x to 1.04x): with hidden 512 every projection weight is 0.25 to 1 MiB and
the step is set by the kernel count, the host and the 16-bit `lm_head`. The
encoder is too small for FP8 to gain, so 4-bit weights would not help either.

### 7. Found on the way (main, independent of FP8; not fixed here)

1. The encoder always runs at `max_position_embeddings` (512) positions,
   whatever the source length, and the decoder projects cross-attention K and V
   from the full 512-row encoder output on every decode step (12 GEMMs of
   512 x 512 x 512 plus 12 fp32 to fp16 casts of 512 x 512 per step), instead
   of once per sentence for the real source length. Held-out sources are about
   30 tokens. The cost was not timed separately. See `_build_marian_encoder`
   and `_add_marian_decoder_layer` in `model.py` and `run_encoder` in
   `runtime/plugin.cpp`.
2. The shared 62518 x 512 embedding is stored three times (encoder gather
   table, decoder gather table, decoder `lm_head`): about 183 MiB of the
   278 MiB fp16 bundle (encoder plan 99 MiB, decoder plan 172 MiB).
3. OPUS-MT "tc-big" checkpoints (`model_type` `marian`) cannot be built. The
   loader requires `model.encoder.embed_positions.weight`, which these
   checkpoints do not store (Transformers computes the sinusoidal table):
   `Helsinki-NLP/opus-mt-tc-big-en-fr` (revision `6e062862`) stops with
   `KeyError: 'Tensor not found: model.encoder.embed_positions.weight'`. The
   family also hard-codes SiLU, while these checkpoints use
   `activation_function: relu`, so adding only the position table would give
   wrong output.

### 8. What was not tried

| Item | Why it might matter |
|---|---|
| FP8 attention output projections | Do not build on TensorRT 11.4 (section 3); retest on newer releases. |
| fp32 Dequantize, percentile calibration, per-channel weight scales | Could lower the drift. |
| Batch above 1 | The family supports batch 1 only; larger GEMMs would favor FP8. |
| Other OPUS-MT checkpoints | No scales calibrated; tc-big checkpoints do not load (section 7). |
| H100, TensorRT 11.1 speed | Not measured. |
| Native runtime with an FP8 bundle | Not run; the Python harness called the engines. |

### 9. How to reproduce

1. Calibrate per-tensor activation scales with ModelOpt `FP8_DEFAULT_CFG` on
   PyTorch fp32 during greedy translation of the calibration sentences.
2. Build the fp16 and FP8 bundles of the checkpoint above with the recipe of
   section 3.
3. Time the engines and run the quality metrics of section 4 against
   Transformers fp32 on the evaluation sentences.

### 10. When to revisit

- A TensorRT release that builds FP8 after the fused attention and gives a
  decode-step gain at batch 1.
- Batch support above 1 in the family.
- After the encoder runs at the real source length and cross-attention K and V
  are computed once per sentence (section 7), retest: the decode step then has
  less fixed work.

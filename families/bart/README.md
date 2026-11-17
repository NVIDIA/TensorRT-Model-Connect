# BART

This family builds BART encoder-decoder checkpoints (`model_type` `bart` and
`mbart`) for `text_generation` as two TensorRT engines: the encoder and a
one-token decoder step with a static KV cache. Precision is fp32 or fp16, batch
size is 1, and `max_sequence_length` defaults to 256 (at most the checkpoint's
`max_position_embeddings`). The manifest builds `facebook/bart-base` in fp16
with `max_sequence_length` 256.

## FP8 quantization study (not shipped)

FP8 self-quantization of the encoder and decoder projections was built and
measured on TensorRT 11.4 for `facebook/bart-base` and
`facebook/bart-large-cnn`. It is not shipped: end-to-end generation is not
faster than fp16 (0.93x to 1.04x), and `bart-large-cnn` drifts far from the
reference model. This section records the recipe, the method and the numbers,
so that the study does not have to be repeated. The FP8 code was not merged;
this repository has no FP8 build for BART.

### 1. Summary and decision

| Item | fp16 | FP8 | Change |
|---|---|---|---|
| End to end, `bart-large-cnn`, `max_sequence_length` 256, B200 / B300 | 157.9 / 174.6 ms | 169.0 / 181.5 ms | 0.93x / 0.96x |
| End to end, `bart-base`, 256, B200 / B300 | 80.8 / 76.0 ms | 86.8 / 80.4 ms | 0.93x / 0.94x |
| End to end, `bart-large-cnn`, 1024, B200 / B300 | 176.8 / 172.0 ms | 170.6 / 174.1 ms | 1.04x / 0.99x |
| KL vs Transformers fp32, `bart-large-cnn` 256 (B200) | 0.0005 | 0.20 | 370x larger |
| Top-1 agreement, `bart-large-cnn` 256 (B200) | 98.2% | 77.6% | -20.6 points |
| KL vs Transformers fp32, `bart-base` 256 (B200) | 6e-8 | 0.0002 | small |
| Bundle size, `bart-large-cnn` (B200) | 879.5 MiB | 579.9 MiB | -34% |
| Bundle size, `bart-base` | 417.5 MiB | 333.3 MiB | -20% |

Decision: do not ship FP8 for BART.

- Speed: the encoder is 1.03x to 1.10x faster and the decode step 0.98x to
  1.03x. End to end (encoder plus 128 decode steps) FP8 is 0.93x to 0.96x at
  256 and 0.99x to 1.04x at 1024. Batch 1 is the only supported batch size.
- Quality: `bart-base` stays close to fp16 (top-1 99.98%), but
  `bart-large-cnn` agrees with the reference on only 77.6% of tokens and only
  33% of the greedy summaries are equal to the reference. No subset of the
  projections reaches the fp16 floor (section 5).
- The only clear gain is the 20% to 34% smaller bundle.

### 2. Models and engines

| Item | `facebook/bart-base` | `facebook/bart-large-cnn` |
|---|---|---|
| Revision | `aadd2ab0ae0c8268c7c9693540e9904811f36177` | `37f520fa929c961707657b28798b30c003dd100b` |
| Model config | hidden 768, 6 + 6 layers, MLP 3072 | hidden 1024, 12 + 12 layers, MLP 4096 |
| Quantized projections | 78 | 156 |
| `max_sequence_length` measured | 256 | 256 and 1024 |
| Engines | encoder, decoder step (fp16) | encoder, decoder step (fp16) |

`bart-large` was not measured.

### 3. The recipe

No official FP8 checkpoint or config exists for BART. The recipe follows the
ModelOpt guidelines: quantize the linear projections of the encoder and decoder
blocks, keep everything else in fp16. Every quantized projection is built as:

```
activation -> Quantize(FP8) -> Dequantize(fp16) -> MatMul <- Dequantize(fp16) <- FP8 weight constant
```

| Choice | Decision | Reason |
|---|---|---|
| Encoder projections | q, k, v, `fc1`, `fc2` of every layer | The GEMMs of the encoder block. |
| Decoder projections | self-attention q, k, v; cross-attention q, k, v; `fc1`, `fc2` of every layer | Cross-attention k and v project the encoder output on every decode step. |
| Attention output projections (`out_proj` of self- and cross-attention) | fp16 | On TensorRT 11.4 an FP8 Quantize right after the fused attention does not build (encoder and decoder build error). The same graph builds on TensorRT 11.1. |
| Embeddings, tied `lm_head`, layer norms | fp16 | Not quantized. |
| Activation scale | static, per tensor, one per projection input | ModelOpt `FP8_DEFAULT_CFG` (max calibration) on PyTorch fp32 while the model greedily generates up to 128 tokens for 256 CNN/DailyMail training articles, so the decoder sees decode-time activations. Scale = amax / 448. |
| Weight scale | per tensor, amax / 448, at build time | FP8 weights are quantized from the original checkpoint at build time. A checkpoint that is already quantized is rejected. |
| Dequantize type | fp16 | The FP8 GEMM writes fp16 and the bias stays in the GEMM epilogue. No non-finite value was found (section 4). fp32 Dequantize was not measured. |
| Outlier rule | a scale above 20x the median of the same projection in the other layers is stored as `null` and stays fp16 | Measured: no outlier; 78 of 78 (`bart-base`) and 156 of 156 (`bart-large-cnn`) projections quantized. |
| Scale table key | model config plus a weight fingerprint | `bart-large` and `bart-large-cnn` share the model config and need their own scales. A missing scale is a build error. |
| Precision | FP8 needs an fp16 engine | fp32 builds have no FP8 variant. |
| Tensor parallelism | not supported with FP8 | Not built. |

### 4. How the results were measured

| Item | Value |
|---|---|
| GPUs | B200 and B300, one GPU per job, all builds of one GPU type in one job |
| TensorRT | 11.4.0.106 (target); the unit tests of the FP8 code also passed on 11.1.0.106 |
| Calibration data | first 256 articles of the CNN/DailyMail 3.0.0 training split |
| Evaluation data | CNN/DailyMail 3.0.0 test split, first 300 articles (`bart-large-cnn`) and 200 (`bart-base`), truncated to `max_sequence_length` |
| Reference | Transformers fp32 greedy output, up to 128 new tokens |
| Harness | Python, the engines of the bundle called directly (not the native runtime) |
| Forced BOS | `bart-large-cnn` runs with `forced_bos_token_id=0` (its generation config) in the reference and in the harness |

Metrics:

| Metric | Definition |
|---|---|
| KL | mean KL(reference, engine) per token, with the reference greedy output teacher-forced through the engine |
| Top-1 | fraction of those tokens where the engine argmax equals the reference token |
| Greedy equal | fraction of articles where the free-running engine greedy output equals the reference output |
| ROUGE-1 / 2 / L | free-running output against the reference highlights (`bart-large-cnn` only) |
| Non-finite scan | NaN and Inf in every encoder output and every logit of the evaluation |

Speed: median of 7 interleaved rounds per variant. Encoder: one engine call at
`max_sequence_length`. Decode step: one replayed CUDA graph of the decoder
engine. End to end: encoder plus 128 decode steps with a host argmax per step,
as the native runtime does.

### 5. Results

#### Speed

TensorRT 11.4, batch 1, times in ms, ratio = fp16 time / FP8 time.

| Model, `max_sequence_length` | GPU | Encoder fp16 / FP8 | Decode step fp16 / FP8 | End to end fp16 / FP8 |
|---|---|---|---|---|
| `bart-large-cnn`, 1024 | B200 | 0.888 / 0.810 (1.10x) | 1.163 / 1.131 (1.03x) | 176.8 / 170.6 (1.04x) |
| `bart-large-cnn`, 1024 | B300 | 0.821 / 0.776 (1.06x) | 1.111 / 1.075 (1.03x) | 172.0 / 174.1 (0.99x) |
| `bart-large-cnn`, 256 | B200 | 0.502 / 0.487 (1.03x) | 0.882 / 0.871 (1.01x) | 157.9 / 169.0 (0.93x) |
| `bart-large-cnn`, 256 | B300 | 0.465 / 0.448 (1.04x) | 0.817 / 0.815 (1.00x) | 174.6 / 181.5 (0.96x) |
| `bart-base`, 256 | B200 | 0.240 / 0.230 (1.04x) | 0.416 / 0.426 (0.98x) | 80.8 / 86.8 (0.93x) |
| `bart-base`, 256 | B300 | 0.224 / 0.207 (1.08x) | 0.411 / 0.410 (1.00x) | 76.0 / 80.4 (0.94x) |

The fp16 output at 1024 is wrong on main (section 8); its timings are still
valid for the same graph.

The FP8 plans have more layers than fp16 (`bart-large-cnn` decoder 403 vs 355
at 256, 283 vs 259 at 1024; encoder 112 vs 100) because of the activation
Quantize layers. At batch 1 the decode-step GEMMs have one row, and the
smaller FP8 weights do not shorten the step.

#### Quality

TensorRT 11.4, `max_sequence_length` 256. No NaN or Inf in any run.

| Model | GPU | Build | KL | Top-1 | Greedy equal | ROUGE-1 / 2 / L |
|---|---|---|---|---|---|---|
| `bart-base` (200 articles) | B200 | fp16 | 6e-8 | 100% | 100% | - |
| `bart-base` | B200 | FP8 | 0.0002 | 99.98% | 98.5% | - |
| `bart-base` | B300 | fp16 | 2e-7 | 100% | 100% | - |
| `bart-base` | B300 | FP8 | 0.0003 | 99.97% | 97.0% | - |
| `bart-large-cnn` (300 articles) | - | Transformers fp32 | - | - | - | 30.5 / 13.3 / 23.5 |
| `bart-large-cnn` | B200 | fp16 | 0.0005 | 98.2% | 95.7% | 30.2 / 13.2 / 23.3 |
| `bart-large-cnn` | B200 | FP8 | 0.20 | 77.6% | 32.7% | 32.9 / 13.9 / 25.0 |
| `bart-large-cnn` | B300 | fp16 | 0.0010 | 98.1% | 92.0% | 30.3 / 13.2 / 23.4 |
| `bart-large-cnn` | B300 | FP8 | 0.22 | 77.8% | 34.7% | 33.2 / 14.2 / 25.4 |

ROUGE against the reference highlights does not drop with FP8, but the FP8
summaries are different text: only 33% (B200) and 34% (B300) of them equal the
fp16 summaries. The minimum encoder output cosine against Transformers drops
from 0.99998 (fp16) to 0.991 (FP8) on B200.

#### Sensitivity (which projections cause the drift)

`bart-large-cnn`, `max_sequence_length` 256, B200, TensorRT 11.4, 60 articles.
Speed of the subsets was not measured.

| Variant | Quantized projections | KL | Top-1 | Greedy equal |
|---|---|---|---|---|
| fp16 | none | 2e-5 | 98.2% | 98.3% |
| all (the recipe) | encoder and decoder, all of section 3 | 0.19 | 79.2% | 30.0% |
| attention only | q, k, v (encoder); q, k, v and cross q, k, v (decoder) | 0.016 | 95.5% | 60.0% |
| no `fc1` | all except `fc1` | 0.033 | 95.6% | 56.7% |
| no `fc2` | all except `fc2` | 0.082 | 83.2% | 40.0% |
| encoder only | encoder projections | 0.12 | 93.7% | 56.7% |
| decoder only | decoder projections | 0.13 | 77.2% | 48.3% |
| MLP only | `fc1`, `fc2` | 0.36 | 78.1% | 46.7% |
| no cross k, v | all except decoder cross k and v | 0.37 | 75.4% | 31.7% |

The MLP projections, `fc1` most of all, cause most of the drift. Even the best
subset (attention only) has a KL 750x the fp16 floor and leaves the MLP GEMMs,
the largest ones, in fp16.

#### Size and kernel evidence

| Model | GPU | fp16 bundle | FP8 bundle | Change |
|---|---|---|---|---|
| `bart-large-cnn`, 1024 | B200 | 879.5 MiB | 579.9 MiB | -34% |
| `bart-large-cnn`, 1024 | B300 | 977.7 MiB | 678.1 MiB | -31% |
| `bart-base`, 256 | B200 and B300 | 417.5 MiB | 333.3 MiB | -20% |

The plan inspector finds FP8 (e4m3) layers in every FP8 plan (`bart-large-cnn`:
60 in the encoder, 132 in the decoder; `bart-base`: 30 and 66) and none in the
fp16 plans.

### 6. NVFP4

NVFP4 was not built. FP8 gives no end-to-end gain at batch 1 for these model
configs (decode step 0.98x to 1.03x), and NVFP4 is lossier than FP8, while FP8
already drifts on `bart-large-cnn`.

### 7. What was not tried

| Item | Why it might matter |
|---|---|
| FP8 attention output projections | Do not build on TensorRT 11.4 (section 3); retest on newer releases. |
| fp32 Dequantize, percentile calibration, per-channel weight scales | Could lower the drift of `fc1` and `fc2`. |
| Batch above 1 | The family supports batch 1 only; larger GEMMs would favor FP8. |
| `bart-large`, mBART checkpoints | No scales calibrated. |
| H100, TensorRT 11.1 speed | Not measured. |
| Native runtime with an FP8 bundle | Not run; the Python harness called the engines. |

### 8. Found on the way (main, independent of FP8; not fixed here)

Items 2 and 3 have since been fixed on main; item 1 is open.

1. The fp16 engine at `max_sequence_length` 1024 gives wrong logits. The
   encoder output matches Transformers (cosine at least 0.9998), the decoder
   output does not. 256 is correct (section 5). The unquantized `bart-base`
   plans built from main and from the study branch have the same size at 256
   and at 1024.

   | Model, 1024, fp16 | Articles | KL | Top-1 | Greedy equal | ROUGE-1 (Transformers fp32) |
   |---|---|---|---|---|---|
   | `bart-large-cnn`, forced BOS, B200 | 300 | 4.78 | 49.4% | 11.0% | 10.7 (27.5) |
   | `bart-large-cnn`, forced BOS, B300 | 300 | 4.79 | 49.4% | 11.0% | 10.8 (27.5) |
   | `bart-large-cnn`, no forced BOS in the engine, B200 | 20 | 4.33 | 27.8% | 5.0% | - |
   | `bart-base`, B200 | 20 | 3.02 | 42.3% | 0% | - |

   512 and an fp32 build at 1024 were not measured. The manifest uses 256, so
   CI does not cover it.
2. The BART runtime does not apply `forced_bos_token_id`. `bart-large-cnn` sets
   it to 0 in its generation config; the bundle runtime config takes only
   `eos_token_id` from the generation config. The study forced the token in
   the harness. With it, the 1024 fp16 result is still wrong (item 1), so the
   two problems are separate.
3. Main cannot load `facebook/bart-large-cnn` at the revision above: its
   safetensors store the shared token table only as
   `model.decoder.embed_tokens.weight`, and the loader stops with
   `BART: cannot find shared embedding weight`. The study added this fallback
   (not in this change).

### 9. When to revisit

- A TensorRT release that builds FP8 after the fused attention and gives a
  larger decode-step gain at batch 1.
- Batch support above 1 in the family.
- A recipe that keeps `bart-large-cnn` near the fp16 floor (top-1 above 98%).
- A need for the 20% to 34% smaller bundle that outweighs the drift.

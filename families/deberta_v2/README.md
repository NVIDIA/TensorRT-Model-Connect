# DeBERTa-v2 / v3

This family builds DeBERTa-v2 and DeBERTa-v3 encoders (`DebertaV2Model`, for
example `microsoft/deberta-v3-base` and `microsoft/deberta-v3-large`) as one
TensorRT engine for the `text_to_token_features` task. The engine takes token ids
and an attention mask over a fixed window (`max_sequence_length`, 256 in the
manifests) and returns the final hidden rows. Batch size is 1. The build accepts
precision fp32 (default) and fp16; the manifests build both. The family has no
quantized build.

## FP8 quantization study (not shipped)

FP8 self-quantization of the encoder projections, on top of the fp16 engine, was
built and measured on TensorRT 11.4 (and 11.1) for `microsoft/deberta-v3-base`
and `microsoft/deberta-v3-large`. It is not shipped: at the manifest window (256
tokens, batch 1) FP8 is 0.94x to 0.99x of fp16 speed on B200 and 1.08x to 1.11x on
B300, and the feature error grows from about 0.2% to 6% to 7%. FP8 is only
clearly faster at a 512-token window (1.18x to 1.37x). This section records the
recipe, the method and the numbers, so that the study does not have to be
repeated. The FP8 code was not merged; this family has no FP8 build. All
numbers come from a local study.

### Summary and decision

Batch 1, window 256 (the manifest window), TensorRT 11.4; x is fp16 time over
FP8 time.

| Item | fp16 (manifest) | FP8 (fp16 engine) | FP8 vs fp16 |
|---|---|---|---|
| base latency, B200 / B300 | 0.636 / 0.682 ms | 0.674 / 0.633 ms | 0.94x / 1.08x |
| large latency, B200 / B300 | 1.397 / 1.468 ms | 1.407 / 1.325 ms | 0.99x / 1.11x |
| base / large plan size (B200) | 386.1 / 935.8 MiB | 306.0 / 650.8 MiB | -21% / -30% |
| token feature relative error vs PyTorch fp32, base / large | 0.0021 / 0.0023 | 0.070 / 0.060 | about 30x |
| MNLI head prediction agreement with PyTorch fp32, base / large (B200) | 100.00% / 100.00% | 99.25% / 99.40% | -0.6 to -0.75 point |
| Non-finite outputs | 0 | 0 | none |

Decision: do not ship FP8 for DeBERTa-v2/v3.

- Speed: at batch 1 and windows 128 and 256 FP8 is slower than fp16 on B200
  and at most 1.11x faster on B300. In the FP8 engine TensorRT no longer fuses
  the attention of most layers into one kernel (see
  [Why there is no speedup](#why-there-is-no-speedup)), which takes the GEMM
  gain back.
- Quality: the token features move by 6% to 7% (relative error) against 0.2%
  for fp16; MNLI predictions change for 0.45% to 0.75% of pairs; on one
  short-input build of the large model the features lose most of their
  direction (token cosine 0.63, see [Quality](#quality)).
- Size: the plan is 21% (base) and 30% (large) smaller, because the fp16 word
  table (128100 rows) stays. A size-only gain is not enough to ship.
- Against the fp32 default engine FP8 is 3.7x to 6.1x faster, but most of that
  is the fp16 gain, which comes with no quality loss, so the comparison that
  counts is against fp16.

### Recipe that was measured

No FP8 or NVFP4 checkpoint of a DeBERTa-v2/v3 model exists on the Hub, and no
official quantization config exists for one. The recipe is the usual one for
small text encoders: quantize the attention and feed-forward linears, keep the
embeddings, position tables, norms and the attention core in high precision.
Each quantized projection is built as
`input -> Quantize -> Dequantize(fp16) -> MatMul`, with the weight
`FP8 constant -> Dequantize(fp16)`.

| Choice | Decision |
|---|---|
| Quantized (6 per layer; 72 in base, 144 in large) | `attention.self.query_proj`, `key_proj`, `value_proj` (applied to the hidden rows), `attention.output.dense`, `intermediate.dense`, `output.dense` |
| fp16 | word embeddings, relative position table, the positional products of the disentangled attention (`query_proj` and `key_proj` applied to the relative table; they do not depend on the input), layer norms (statistics in float32), GELU, attention matmuls and softmax, c2p / p2c gathers |
| Weight scale | per tensor, amax / 448, computed at build time one projection at a time; FP8 weights are never stored |
| Activation scale | static, per tensor; ModelOpt 0.46.1 `FP8_DEFAULT_CFG` (max calibration) on the PyTorch model, 800 calibration texts (400 WikiText-2 train paragraphs, 400 MNLI train pairs), framed like the runtime (`[CLS] a [SEP]`, pairs `[CLS] a [SEP] b [SEP]`, up to 512 tokens) |
| Scale table key | model config plus two weight sums, so a fine-tune needs its own calibration |
| Outlier rule | a scale above 20x the median of its kind stays fp16; measured: no outlier (largest ratio 4.3x base, 2.5x large) |
| Dequantize type | fp16; no non-finite value, fp32 Dequantize was not needed |
| Builder | the dual-GEMM fusion was turned off for FP8 builds |

With `share_att_key` the reference also feeds the relative table through
`query_proj` and `key_proj`, so the calibrated input range of these two also
covers the table. The stored scale is at most 1.12x the range of the hidden rows
(measured against `value_proj`, which has the same input); this costs a little
resolution and never clips.

### Method

- GPUs: B200 and B300, one GPU per job, all variants of a model built and timed
  in the same job; TensorRT 11.4.0.106 (target) and 11.1.0.106 (B200).
- Speed: replayed CUDA graph, median of 9 interleaved rounds of 100 calls;
  spread between rounds at most 1.5%. Batch 32 is an experiment-only graph
  (the family builds batch 1 only).
- Feature quality: 800 held-out texts (400 WikiText-2 test paragraphs, 400 MNLI
  validation-matched pairs; calibration used the train splits), each framed by
  itself, against Transformers `DebertaV2Model` in float32 (TF32 off). Token
  cosine is per real token; relative error is
  `||engine - reference|| / ||reference||` per text.
- Task metric: the MNLI heads of `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`
  and `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` (fine-tunes of
  the two checkpoints, calibrated the same way for this check only) applied to
  the engine features, first 2000 pairs of MNLI validation-matched, window 256,
  against the same head on the PyTorch features.
- Non-finite scan: every output above (all rows, padded rows included) and 11
  stress inputs at window 512 (repeated tokens, random ids, `[CLS] [SEP]` only,
  one token, the longest texts).
- Kernel time: TensorRT layer profiler, window 256, average of 20 calls. The
  per-layer sum is larger than the CUDA graph time (per-layer timing adds
  overhead), so only the split between groups is used.

### Results

Speed and size, batch 1 (ms; x is fp16 time over FP8 time; plan MiB fp16 to FP8):

| Model | Window | B200 11.4 fp16 / FP8 | x | B300 11.4 fp16 / FP8 | x | B200 11.1 x | plan MiB (B200 11.4) |
|---|---|---|---|---|---|---|---|
| base | 128 | 0.457 / 0.502 | 0.91 | 0.489 / 0.469 | 1.04 | 0.91 | 381.4 to 301.9 |
| base | 256 | 0.636 / 0.674 | 0.94 | 0.682 / 0.633 | 1.08 | 0.99 | 386.1 to 306.0 |
| base | 512 | 1.084 / 0.920 | 1.18 | 1.093 / 0.873 | 1.25 | 1.19 | 404.0 to 325.5 |
| large | 128 | 1.109 / 1.154 | 0.96 | 1.153 / 1.097 | 1.05 | 0.98 | 929.4 to 644.5 |
| large | 256 | 1.397 / 1.407 | 0.99 | 1.468 / 1.325 | 1.11 | 1.03 | 935.8 to 650.8 |
| large | 512 | 2.507 / 2.002 | 1.25 | 2.607 / 1.906 | 1.37 | 1.22 | 959.5 to 676.3 |

An earlier run of the same code (one job per GPU type, TensorRT 11.4) gave the
same ratios within 0.02 (for example base 256: 0.95x B200, 1.07x B300).
In that earlier run the fp32 default engine took 2.534 ms (base) and 7.239 ms
(large) at window 256 on B200.

Batch 32 (experiment-only graph, TensorRT 11.4, ms fp16 / FP8):

| Model | Window | B200 | x | B300 | x |
|---|---|---|---|---|---|
| base | 128 | 2.505 / 2.352 | 1.07 | 2.755 / 2.259 | 1.22 |
| base | 512 | 16.677 / 16.207 | 1.03 | 17.706 / 15.575 | 1.14 |
| large | 128 | 6.321 / 5.790 | 1.09 | 7.404 / 5.524 | 1.34 |
| large | 512 | 46.659 / 43.636 | 1.07 | 49.787 / 41.863 | 1.19 |

#### Quality

Feature quality against PyTorch fp32, 800 held-out texts, TensorRT 11.4:

| Model | Window | GPU | Engine | Token cosine mean (min) | Relative error mean (p99) | Max abs error | Non-finite |
|---|---|---|---|---|---|---|---|
| base | 256 | B200 | fp16 | 0.999998 (0.9999) | 0.0021 (0.0025) | 0.14 | 0 |
| base | 256 | B200 | FP8 | 0.997525 (0.9334) | 0.070 (0.085) | 5.12 | 0 |
| base | 256 | B300 | FP8 | 0.997520 (0.9517) | 0.070 (0.086) | 4.01 | 0 |
| base | 512 | B200, B300 | FP8 | 0.997516 (0.9337) | 0.070 (0.086) | 4.40 | 0 |
| large | 256 | B200 | fp16 | 0.999997 (0.9971) | 0.0023 (0.0058) | 0.66 | 0 |
| large | 256 | B200 | FP8 | 0.998138 (0.8519) | 0.060 (0.086) | 6.24 | 0 |
| large | 256 | B300 | FP8 | 0.998154 (0.8748) | 0.060 (0.080) | 5.31 | 0 |
| large | 512 | B200 | FP8 | 0.998121 (0.8248) | 0.060 (0.089) | 6.16 | 0 |
| large | 512 | B300 | FP8 | 0.998162 (0.8468) | 0.060 (0.079) | 5.84 | 0 |

TensorRT 11.1 (B200, 400 texts, window 256): FP8 relative error 0.074 (base)
and 0.062 (large); fp16 0.0022 and 0.0014.

MNLI heads, 2000 validation-matched pairs, window 256, TensorRT 11.4 (PyTorch
fp32 accuracy: base 90.65%, large 91.90%):

| Model | GPU | Engine | Accuracy | Agreement with PyTorch | KL mean (p99) | Max logit difference |
|---|---|---|---|---|---|---|
| base | B200 | fp16 | 90.65% | 100.00% | 1.4e-06 (3.5e-05) | 0.024 |
| base | B200, B300 | FP8 | 91.00% | 99.25% | 1.7e-03 (2.9e-02) | 1.03 |
| large | B200 | fp16 | 91.90% | 100.00% | 9.9e-07 (2.4e-05) | 0.036 |
| large | B200 | FP8 | 92.10% | 99.40% | 2.3e-03 (4.4e-02) | 1.30 |
| large | B300 | FP8 | 92.05% | 99.55% | 2.6e-03 (4.7e-02) | 1.63 |

TensorRT 11.1 (B200, 1000 pairs): FP8 agreement 99.70% (base) and 99.60%
(large). On 11.4 the FP8 accuracy moves by +0.15 to +0.35 point, inside the
noise of 2000 pairs; the KL is about 1000x that of fp16.

Short inputs: on stress inputs FP8 keeps token cosine 0.986 to 0.9995 (base)
and 0.994 to 0.9997 (large), except the one-token input (`[CLS] the [SEP]`) on
the large model, which depends on the build: token cosine 0.631 (worst token
0.078) in one B300 build at window 512, 0.944 (worst 0.885) in two more B300
builds, 0.978 at window 256 (B300), and 0.973 to 0.995 in five B200 builds at
windows 256 and 512. fp16 stays at 0.9999 or better. No value was non-finite.

#### Why there is no speedup

Kernel time per group, window 256, batch 1, TensorRT 11.4, B200 (layer
profiler, ms (kernel count)):

| Kernel group | base fp16 | base FP8 | large fp16 | large FP8 |
|---|---|---|---|---|
| GEMMs (projections, attention matmuls not fused) | 0.679 (73) | 0.746 (85) | 1.428 (145) | 1.553 (171) |
| fused attention | 0.120 (12) | 0.010 (1) | 0.245 (24) | 0.011 (1) |
| separate softmax | - | 0.083 (11) | - | 0.175 (23) |
| transposes, concat, add, other | 0.076 (14) | 0.352 (58) | 0.156 (26) | 0.720 (118) |
| c2p / p2c gathers | 0.221 (25) | 0.199 (25) | 0.490 (49) | 0.406 (49) |
| layer norms (with Quantize in FP8) | 0.169 (25) | 0.181 (25) | 0.338 (49) | 0.339 (49) |
| sum | 1.264 | 1.571 | 2.658 | 3.204 |

B300 and TensorRT 11.1 show the same split.

- In the fp16 engine TensorRT runs the attention of every layer as one fused
  kernel. In the FP8 engine it does so for one layer only; in the other 11 (base)
  or 23 (large) layers the attention runs as separate matmuls, a softmax and
  extra transposes, concats and adds (pointwise kernels grow from 14 to 58 in
  base and from 26 to 118 in large). The cause was not isolated.
- The Quantize of the activations is fused into neighboring kernels (the layer
  norm kernel count and time do not change), so it costs little.
- At 128 and 256 tokens the projection GEMMs are small (256 x 768 x 768 in base),
  so FP8 saves little GEMM time, and the lost attention fusion costs more. At
  512 tokens the GEMM saving is larger and FP8 wins (1.18x to 1.37x).

### NVFP4 (not tried)

DeBERTa-v3 is not bound by weight reads at batch 1: reading all FP8 encoder
weights (85 MB base, 302 MB large) at about 8 TB/s takes 0.01 to 0.04 ms of a
0.6 to 1.5 ms call, so 4-bit weights could save at most that. On other small
encoders (`timm_swin`) NVFP4 was slower than FP8 at batch 1 to 32 because of
the extra dynamic quantize kernels. No NVFP4 engine was built.

### What was not tried

| Item | Note |
|---|---|
| FP8 attention matmuls (BMM1 / BMM2), FP8 for the positional products | not measured |
| Keeping `attention.output.dense` (or q/k/v) in fp16 to keep the attention fusion | not measured |
| Per-channel weight scales, SmoothQuant, percentile or entropy calibration | not measured |
| H100, the native runtime with an FP8 bundle | not measured |

### When to revisit

- A TensorRT release that keeps the fused attention kernel when the projections
  around it are FP8: the GEMM saving would then show up at 128 and 256 tokens.
- The family supports batch greater than 1 or a 512-token window in practice:
  FP8 is 1.14x to 1.34x faster at batch 32 on B300 and 1.18x to 1.37x at 512
  tokens.
- A recipe that brings the feature error near fp16 (per-channel scales,
  SmoothQuant) and removes the short-input loss on the large model.

# Laya structured decisions

This family supports all three checkpoints bundled in
[convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya), pinned to
`7b928d828b7b0e022f929d9bd2e44165aa270148`:

| Variant | Checkpoint directory | Default sequence capacity |
| --- | --- | --- |
| `english` | repository root | 512 |
| `multilingual` | `multilingual/` | 1024 |
| `typed-decisions` | `typed-decisions/` | 1024 |
| `router` | all three above | each checkpoint's default |

The native C++ runtime tokenizes the state and questions, runs the complete
ModernBERT encoder and decision/action heads through TensorRT, and returns
calibrated choice probabilities, expected scores, and binary probabilities.
It preserves the released `laya==0.3.20` sequence construction, temperatures,
option ordering, confidence fields, action head, and input-token accounting.
Python and Torch are used for building and reference validation.

## Build and run

Use the repository's [source-build environment](../../website/docs/getting-started/source-build.md)
and install the family requirements. Build the native family targets:

```bash
cmake --build build --parallel 8 --target \
  trtmc trtmc_backend_trt trtmc_model_laya trtmc_cli_laya

build/trtmc laya build convaiinnovations/laya \
  --revision 7b928d828b7b0e022f929d9bd2e44165aa270148 \
  --variant english -o laya.bundle

build/trtmc laya decide laya.bundle --runtime-root build \
  --record families/laya/tests/fixtures/email.json
```

Select `multilingual` or `typed-decisions` to build one of the other variants.
`--max-sequence-length` overrides the checkpoint's sequence capacity;
the multilingual encoder permits up to 8192 tokens. `--max-batch-size`
controls how many question rows execute together, and `--max-options` limits
the options in one question. Inference retains each checkpoint's question-head
budget and truncates the remaining state. Conversation arrays retain their
newest turns. Empty question sets return empty answers and zero tokens.

The build uses BF16 linear operations with FP32 residual streams, normalization,
and rotary arithmetic. The multilingual mmBERT encoder also uses explicit FP32
attention to preserve long-conversation accuracy. The checkpoint's FP16 embedding and
normalization values are preserved. The family rejects unsupported precision,
task, and backend choices.

The English checkpoint uses reference-generated rotary constants and a BF16
GELU lookup expressed entirely with TensorRT operations. The lookup preserves
the original activation's small negative tail and is checked against every
BF16 input bit pattern. The other checkpoints retain their independently
qualified arithmetic paths.

## Route across all three variants

```bash
build/trtmc laya build convaiinnovations/laya \
  --revision 7b928d828b7b0e022f929d9bd2e44165aa270148 \
  --variant router -o laya-router.bundle

build/trtmc laya decide laya-router.bundle --runtime-root build \
  --record families/laya/tests/fixtures/hindi.json
```

A router bundle loads all three engines. Its JSON record accepts the same
`state` and `questions` as a single-variant bundle and returns `routing` metadata.
The language/script detector operates on string values in the state and uses
the released SDK's tables. An optional `model` or `task` selects a variant
explicitly; `lang` and `lang_guess` accept language codes. Set
`auto_task_detection` to `true` to route the four recognized question schemas
to typed-decisions. The default for unidentified text is `english`, configurable
through the record's `default` field. Automatic routing follows the released
heuristic; callers that know the language can supply it explicitly.

For C++ integration, load the bundle with `trtmc::load_task`, obtain
`trtmc::IStructuredDecision`, and pass a JSON document in
`StructuredDecisionRequest`. Both single-variant and router bundles use this
same Task API. Laya accepts text and JSON state; image/video tensors are
rejected. SDK integrations and Python callbacks are outside this native
inference interface.

## Validation

Enable `TRTMC_BUILD_TESTS` and run the selected checkpoint cases:

```bash
TRTMC_RUNTIME_ROOT="$PWD/build" \
PYTHONPATH=core/builder:apps/benchmark:. \
python -m pytest families/laya/tests/test_e2e.py --e2e-model laya -q
```

The tests compare the native public Task against the original CUDA SDK with
`fast=False, compile=False`, including every option's logit/probability,
selected answers, action/confidence fields, repeated calls, and usage.
Logits and per-option probabilities retain their original-reference tolerances;
selected choices, usage, and routing must match. Derived fields use the original
SDK formulas on the validated native probabilities and must match their exact
four-decimal formatting. Entropy confidence and expected scores can therefore
differ from the original inference result as those probability errors propagate.
The native process's loaded libraries are checked for Python/Torch dependencies.
The fixtures include the card's English email, Hindi, Spanish, and quickstart
inputs, plus empty, single-option, and long-conversation records.

`tests/compare_record.py` verifies exact token IDs, markers, normalization,
and truncation for all three variants. `tests/compare_routing.py` checks
routing decisions and metadata without loading model weights.

## FP8 quantization study (not shipped)

FP8 self-quantization of the encoder and head projections was built and
measured on TensorRT 11.4 for all three checkpoints. It is not shipped: FP8
changes the selected option on 2.3% to 3.4% of held-out questions (BF16 engine:
0.3%) and moves single option probabilities by up to 0.56 (B300) and 0.66 (B200), far above
the family E2E tolerance (`atol=0.015` on probabilities), for a speed gain of
1.00x to 1.14x. This section records the recipe, the method and the numbers, so
that the study does not have to be repeated. The FP8 code was not merged; this
family has no FP8 build.

### Summary and decision

| Item | english | typed-decisions | multilingual |
|---|---|---|---|
| FP8 vs BF16 speed, B200 (1 x 128 to 8 x 1024 tokens) | 1.07x to 1.13x | 1.01x to 1.14x | 1.00x to 1.04x |
| FP8 vs BF16 speed, B300 | 1.08x to 1.11x | 1.00x to 1.14x | 1.02x to 1.04x |
| Plan size, BF16 / FP8 | 855 / 528 MB (-38%) | 856 / 529 MB (-38%) | 658 / 539 MB (-18%) |
| Same option as the reference, BF16 / FP8 (of 3158, B300) | 3147 / 3084 | 3150 / 3061 | 3149 / 3052 |
| KL mean, BF16 / FP8 (B300) | 0.00008 / 0.0054 | 0.00003 / 0.0017 | 0.00009 / 0.0105 |
| Max option probability error, BF16 / FP8 (B300) | 0.065 / 0.56 | 0.031 / 0.27 | 0.068 / 0.55 |
| Non-finite outputs | 0 | 0 | 0 |

Decision: do not ship FP8 for Laya.

- Quality: the output of this family is calibrated option probabilities. FP8
  changes the selected option on 2.3% (english), 3.1% (typed-decisions) and
  3.4% (multilingual) of held-out questions, against 0.3% for the BF16 engine,
  and moves single option probabilities by up to 0.56 (p99 0.09 to 0.27). This
  is 3.5x to 7x the largest error of the shipped clef FP8 (0.077). No FP8 E2E
  case could pass the existing family tolerance.
- Speed: one request is one forward pass over a few question rows. The
  ModernBERT-large checkpoints gain 1.07x to 1.14x, multilingual 1.00x to
  1.04x. Small inputs are bound by the many elementwise kernels (fp32 residual
  path, BF16 GELU lookup); multilingual is bound by its explicit fp32
  attention.
- The main gain is the 38% smaller plan of the ModernBERT-large checkpoints.

### Models

| Variant | Encoder | Model config | Head | Default capacity |
|---|---|---|---|---|
| `english` | ModernBERT-large fine-tune | hidden 1024, 28 layers, MLP 2624 | 2 TransformerEncoder layers (d 1024, ffn 4096) | 512 |
| `typed-decisions` | ModernBERT-large fine-tune | same model config, other weights | same | 1024 |
| `multilingual` | mmBERT-base fine-tune | hidden 768, 22 layers, MLP 1152, vocab 256k | 2 layers (d 768, ffn 3072) | 1024 |

Checkpoint `convaiinnovations/laya` at the revision above. No FP8 or NVFP4
checkpoint exists on the Hub (only LiteRT, GGUF, MLX and CoreML community
exports), so the weights were self-quantized.

### Recipe that was measured

FP8 W8A8, per-tensor scales, static input scales; the same recipe as the merged
modernbert FP8. Every quantized projection is built as Quantize(FP8) ->
Dequantize(BF16) -> MatMul, with an FP8 weight constant.

| Choice | Decision |
|---|---|
| Quantized | encoder `attn.Wqkv`, `mlp.Wi`, `mlp.Wo` of every layer; head `self_attn.in_proj`, `linear1`, `linear2`. multilingual (explicit attention) also `attn.Wo` and `self_attn.out_proj` |
| BF16 | embeddings, norms, rotary, attention math, marker scorer, action head; the attention output projection of the fused-attention checkpoints (an FP8 Quantize after the fused attention does not build on TensorRT 11.4) |
| Outlier rule | a scale above 20x the median of its kind is stored as `null` and stays BF16: english and typed-decisions `encoder.layers.14` and `19.mlp.Wo`, multilingual `encoder.layers.11` and `12.mlp.Wo` |
| Weight scale | per tensor, amax / 448, at build time |
| Dequantize type | BF16 (no overflow risk) |
| Activation scale | ModelOpt 0.46.1 `FP8_DEFAULT_CFG` (max calibration), scale = amax / 448 |
| Calibration data | per checkpoint 640 records, 2327 question rows, 0.77 to 1.05 M tokens: 320 CNN/DailyMail validation articles (news questions), 128 Nemotron-Post-Training-v1 chat turns (triage questions), 192 WMT16 validation passages in cs, de, fi, ro, ru, tr; each question row run alone through the released SDK model under its BF16 autocast |

### Method

- GPUs: B200 and B300, one GPU per job; TensorRT 11.4.0.106 (target), and
  11.1.0.106 on B200.
- Held-out data: 1936 records, 3158 question rows, no state shared with
  calibration: 200 CNN/DailyMail test articles, 64 other Nemotron chat turns,
  192 WMT16 test passages, the 8 family fixtures, SST-2 validation (872) and
  AG News test (600).
- Reference: the released SDK (`laya.Agent`, `fast=False`, BF16 autocast).
  Option probabilities with the checkpoint's temperatures; KL(reference ||
  engine) per question over its options; option probability error = absolute
  difference per option.
- Non-finite scan: every held-out row plus 8 random-token batches at full
  capacity per engine.
- Speed: engine only, median of 9 interleaved rounds of 50 runs.

### Speed (TensorRT 11.4, ms, BF16 / FP8)

| Rows x tokens | english B200 | english B300 | typed B200 | typed B300 | multilingual B200 | multilingual B300 |
|---|---|---|---|---|---|---|
| 1 x 128 | 1.61 / 1.50 (1.07x) | 1.50 / 1.38 (1.09x) | 1.52 / 1.49 (1.01x) | 1.41 / 1.41 (1.00x) | 1.50 / 1.50 (1.00x) | 1.40 / 1.38 (1.02x) |
| 4 x 128 | 1.77 / 1.64 (1.08x) | 1.66 / 1.53 (1.08x) | 1.70 / 1.55 (1.10x) | 1.56 / 1.44 (1.08x) | 1.49 / 1.44 (1.04x) | 1.39 / 1.34 (1.04x) |
| 4 x 512 | 5.57 / 4.93 (1.13x) | 5.27 / 4.77 (1.11x) | 4.80 / 4.22 (1.14x) | 4.58 / 4.01 (1.14x) | 6.39 / 6.15 (1.04x) | 6.09 / 5.88 (1.04x) |
| 8 x 512 | 9.78 / 8.74 (1.12x) | 9.36 / 8.44 (1.11x) | 8.25 / 7.27 (1.14x) | 7.88 / 6.90 (1.14x) | 11.68 / 11.21 (1.04x) | 11.17 / 10.77 (1.04x) |
| 8 x 1024 | - (capacity 512) | - | 19.19 / 17.25 (1.11x) | 18.39 / 16.41 (1.12x) | 36.22 / 35.40 (1.02x) | 34.84 / 34.06 (1.02x) |

The plan inspector finds 142, 142 and 136 FP8 layers in the english,
typed-decisions and multilingual FP8 plans. Build time grows from about 55 s to 83 s.

### Quality (TensorRT 11.4, 3158 held-out questions)

| Checkpoint | GPU | Engine | Same option | KL mean / p99 / max | Option probability error mean / p99 / max | AG News accuracy (reference) | Non-finite |
|---|---|---|---|---|---|---|---|
| english | B300 | BF16 | 3147 | 0.00008 / 0.0014 / 0.022 | 0.0023 / 0.024 / 0.065 | 94.0% (94.0%) | 0 |
| english | B300 | FP8 | 3084 | 0.0054 / 0.095 / 0.70 | 0.019 / 0.18 / 0.56 | 93.7% | 0 |
| english | B200 | FP8 | 3079 | 0.0054 / 0.095 / 0.70 | 0.019 / 0.18 / 0.56 | 93.7% | 0 |
| typed-decisions | B300 | BF16 | 3150 | 0.00003 / 0.0004 / 0.002 | 0.0016 / 0.011 / 0.031 | 93.3% (93.2%) | 0 |
| typed-decisions | B300 | FP8 | 3061 | 0.0017 / 0.023 / 0.21 | 0.013 / 0.091 / 0.27 | 93.8% | 0 |
| typed-decisions | B200 | FP8 | 3067 | 0.0017 / 0.023 / 0.25 | 0.013 / 0.092 / 0.29 | 93.8% | 0 |
| multilingual | B300 | BF16 | 3149 | 0.00009 / 0.0013 / 0.014 | 0.0028 / 0.023 / 0.068 | 93.5% (93.5%) | 0 |
| multilingual | B300 | FP8 | 3052 | 0.0105 / 0.17 / 0.82 | 0.029 / 0.27 / 0.55 | 93.3% | 0 |
| multilingual | B200 | FP8 | 3048 | 0.0107 / 0.17 / 1.12 | 0.030 / 0.28 / 0.66 | 93.3% | 0 |

- SST-2 is not a usable task score for the ModernBERT-large checkpoints: the
  released model gives the same answer for every review (49.1%).
  multilingual SST-2: reference 67.7%, BF16 67.8%, FP8 66.2%.
- The action-head probability error is 0 for every engine.
- Recipe variants (B300): first-layer `mlp.Wi` in BF16: KL 0.0054 / 0.0017 /
  0.0091, same option 3061 / 3062 / 3043 (english / typed-decisions /
  multilingual); encoder only (head in BF16): same as the full recipe;
  multilingual without the attention output projections: KL 0.0086, same
  option 3068, same speed. None brings the error near the BF16 level.
- TensorRT 11.1 (B200): the FP8 plans build for english (527 MB, 142 FP8
  layers) and multilingual (539 MB, 136). On the first 400 held-out records
  (1457 questions): english BF16 1451 same option, KL 0.00013; FP8 1383, KL
  0.0091, max error 0.65. multilingual BF16 1451, KL 0.00015; FP8 1382, KL
  0.0155, max error 0.46. No non-finite value. Same picture as 11.4.
- The FP8 code did not change the unquantized builds (TensorRT 11.4, B200):
  same layer counts (493 / 492 / 469) and plan sizes within 0.012%.

### NVFP4 (not tried)

NVFP4 was not built. There is no decode step and no weight-bound phase: one
forward pass of 0.4B parameters takes 1.4 to 35 ms and is dominated by
elementwise and attention kernels. FP8 already gains only 0% to 14% and fails
the quality bar. On the same ModernBERT encoders, NVFP4 measured 5x to 10x the
FP8 error and was slower than FP8 at batch 1 (merged modernbert FP8 study).

### Found on the way

No problem on main was found. The constant SST-2 answer of the ModernBERT-large
checkpoints comes from the released model itself.

### What was not tried

| Item | Note |
|---|---|
| Per-kind sensitivity sweep | not measured |
| SmoothQuant-style scaling, percentile calibration, per-channel weight scales | not measured |
| FP8 only for the largest batch shapes | not measured |
| H100, TensorRT 11.1 speed and typed-decisions on 11.1 | not measured |

### How to reproduce

1. Calibrate per-tensor activation scales with ModelOpt `FP8_DEFAULT_CFG` on the
   released SDK model under BF16 autocast, one question row at a time.
2. Build the BF16 and FP8 bundles of each checkpoint with the recipe above.
3. Time the engines and compare option probabilities with the released SDK on
   the held-out records.

### When to revisit

- A recipe that keeps the option probability error within the family E2E
  tolerance.
- A TensorRT release that builds FP8 after the fused attention, or larger
  batches of question rows where the GEMMs dominate.

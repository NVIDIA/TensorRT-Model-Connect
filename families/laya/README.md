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
The native process's loaded libraries are checked for Python/Torch dependencies.
The fixtures include the card's English email, Hindi, Spanish, and quickstart
inputs, plus empty, single-option, and long-conversation records.

`tests/compare_record.py` verifies exact token IDs, markers, normalization,
and truncation for all three variants. `tests/compare_routing.py` checks
routing decisions and metadata without loading model weights.

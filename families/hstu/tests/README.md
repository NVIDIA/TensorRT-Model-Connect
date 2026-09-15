# HSTU inference qualification

The family tests generate deterministic, seeded checkpoints and compare the
native C++ runtime with the pinned NVIDIA HSTU reference. They qualify the
implemented computation and feature handling. They do not measure trained
recommendation quality, recall, AUC, or production checkpoint accuracy.

## Run

Compile `trtmc`, `trtmc_backend_trt`, `trtmc_model_hstu`, and `trtmc_hstu` in a
matching TensorRT environment. The family target builds its native command as
a dependency. Tests prepare the exact Git revision recorded in
`reference-source.json` in a local cache when no source override is supplied:

```bash
TRTMC_BINARY="$PWD/build-hstu/trtmc" \
TRTMC_RUNTIME_ROOT="$PWD/build-hstu" \
TRTMC_E2E_ARTIFACT_DIR="$PWD/hstu-artifacts" \
python -m pytest families/hstu/tests/test_e2e.py --e2e-model hstu -q
```

`TRTMC_HSTU_BINARY` can override the default `trtmc-hstu` sibling of
`TRTMC_BINARY`. For offline validation, set `TRTMC_HSTU_REFERENCE_ROOT` to an
existing exact checkout including its `.git` metadata. Explicitly selected
cases fail when native artifacts, a GPU, verified reference source, or their
threshold sidecar cannot be obtained. Without an E2E
selector these hardware cases are skipped. No checkpoint download is needed.

## Numerical contract

All elementwise comparisons apply `abs(actual - reference) <= atol +
rtol * abs(reference)`. The fixed gates were declared before the first native
parity execution:

| Engine and original-reference precision | rtol | atol |
| --- | ---: | ---: |
| FP32 | 0.0001 | 0.00001 |
| FP16 | 0.001 | 0.0001 |
| BF16 | 0.01 | 0.002 |

FP32 uses tight absolute and relative bounds across independently evaluated
TensorRT and PyTorch operations. The lower precision bounds reflect the
respective mantissa resolution: FP16 has 10 fraction bits and BF16 has 7.
The absolute term covers values close to zero; neither a cosine average nor a
whole-tensor average can hide an individual element outside these bounds.
Every output must also be finite, have the declared shape, and preserve raw
candidate IDs exactly. Candidate vectors must have unit L2 norm and retrieval
scores must remain within the normalized dot-product range.

The oracle runs original upstream HSTU layer, attention-mask, SiLU attention,
normalization/gating, L2 postprocessing, prediction-MLP, and dot-product
methods. Its test-only adapters handle jagged tensor storage, embedding lookup,
and position/timestamp indexing. See `reference-source.json` for the precise
source and adaptation boundaries. The PyTorch oracle is not a runtime
backend.

The fixtures use two HSTU blocks, nonzero affine normalization and projection
weights, large sparse INT64 feature keys, different user sequence lengths,
and ranking item/action/context tables. The matrix covers ranking, retrieval,
causal and noncausal attention, grouped targets, contextual masking, optional
position/time encodings, optional residuals and learned normalization,
default and fixed sequence scaling, and FP32/FP16/BF16. Lower precision cases
use hidden width 24 so multiplying by its square root exercises non-exact
position scaling.

Upstream contextual attention intentionally lets context tokens read the full
history. With multiple blocks, those tokens can carry later-history
information to earlier history tokens. Strict history-prefix causality is
therefore only a valid invariant when contextual propagation is absent.
Batch-padding invariance is only a valid comparison with a fixed positive
`scaling_seqlen`; the upstream default `-1` divides by the padded sequence
length.

The evidence recorder retains native commands, raw JSON requests/results,
checkpoint information, a lossless `.npz` snapshot of every canonical tensor,
source provenance, thresholds, and each comparison's
maximum absolute/relative error and relative L2 error. Performance and trained
checkpoint accuracy require separate qualification.

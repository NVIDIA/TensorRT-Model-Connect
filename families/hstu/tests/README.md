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

## Native attention provider qualification

The optional original CUDA provider has a separate reproducible check. Supply
the exact FBGEMM and CUTLASS revisions from `../native_attention_source.json`
and the reference checkout from `reference-source.json`:

```bash
PYTHONPATH=core/builder:. python -m families.hstu.tests.native_e2e \
  --source "$HSTU_NATIVE_SOURCE" \
  --reference-source "$TRTMC_HSTU_REFERENCE_ROOT" \
  --runtime-root "$TRTMC_NATIVE_BUILD_DIR" \
  --output "$HSTU_NATIVE_OUTPUT"
```

The output directory must be new. This command builds Dense and Paged BF16
bundles through the public builder, verifies embedded notices and path-free
provenance, and compares native C++ outputs with the original CPU reference.
It covers batch sizes one and eight, CUDA Graphs off and on, history-cache
hits/appends/invalidation/CPU-tier restoration, and request-local sessions and
branches. The source directories are explicit inputs; this command never
downloads them. Its local artifacts establish correctness and packaging, not
serving performance or trained recommendation quality.

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

## Native cache and session microbenchmark

With `TRTMC_BUILD_TESTS=ON`, build `hstu_cache_benchmark` and reuse the bundle
and `session-trace.json` from a completed session qualification. For example,
this case prepares 200 history items and 256 candidates with a BF16 engine:

```bash
cmake --build build-hstu --target trtmc_model_hstu hstu_cache_benchmark -j 4
TRTMC_BINARY="$PWD/build-hstu/trtmc" \
TRTMC_RUNTIME_ROOT="$PWD/build-hstu" \
TRTMC_E2E_ARTIFACT_DIR="$PWD/hstu-benchmark-evidence" \
python -m pytest families/hstu/tests/test_e2e.py \
  --e2e-model hstu-session-hundreds-bf16 \
  --basetemp="$PWD/hstu-benchmark-fixture" -q
```

After qualification, run the native executable without other GPU workloads:

```bash
HSTU_FIXTURE="$(python -c 'from pathlib import Path; print(next(Path("hstu-benchmark-fixture").rglob("session-trace.json")).parent)')"
build-hstu/hstu_cache_benchmark \
  --bundle "$HSTU_FIXTURE/hstu-session-hundreds-bf16.bundle" \
  --runtime-root "$PWD/build-hstu" \
  --input-json "$HSTU_FIXTURE/session-trace.json" \
  --output-json "$PWD/hstu-cache-session-benchmark.json"
```

The input accepts optional `warmups` (default 10), `iterations` (default 100),
and `storage_max_bytes` (default `cache_max_bytes`). At least 30 measured
iterations are needed for `performance_sample_count_met` to be true. Keep
the fixture's `rtol` and `atol` unchanged. The benchmark requires a fixed
sequence scale and an attention configuration allowing complete history
reuse; assertions reject runs that do not exercise the named cache paths.

By default, the comparison disables cache reuse in the same cache-enabled
bundle. To compare against the ordinary HSTU graph, build another bundle
from the same checkpoint weights and settings with only
`enable_history_cache` changed to `false`, then add its path as
`baseline_bundle` in a copy of the input JSON. The report identifies which
baseline was used. Differences between the two graph implementations can
change whether caching saves time.

The JSON report records raw samples and nearest-rank p50/p95/p99 for full
recomputation, a GPU history hit, native host-tier reload, a cold history
miss with publication, and an append with publication. Both cache tiers
use explicit budgets, and publication writes through to native host
storage. Cache invalidation and seed setup occur outside these timings.

For the 20-step greedy session, the report separates session creation and
initial scoring from the append/score loop. `setup_plus_twenty_step_loop`
uses each sample's paired sum, covering session setup and 21 scoring calls.
It does not add independently calculated percentiles. Loop timings include
greedy token selection, measurement bookkeeping, and CUDA synchronization
before and after every step. All timings are synchronous C++ API wall
times including host preparation, data transfers, GPU execution, and output
processing. Loading, JSON I/O, and numerical checks are excluded.

Every measured cache result is compared with full recomputation outside the
timer. Session runs check all greedy choices, their final outputs, and each
step's cache reuse. This is a seeded model microbenchmark, with the actual
GPU reported in JSON; it does not measure a server, a separate optimized
NVIDIA runtime, trained accuracy, or performance on another GPU.

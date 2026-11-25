# DLRMv3 HSTU ranking

This family serves the **official MLPerf Inference v6.0 DLRMv3 dense model**:
five HSTU blocks, width 512, four attention heads of width 128, learned
content/category/action preprocessing, user context, position/time encoding,
item MLP and binary `rating` prediction head. All 86 dense checkpoint tensors
are retained (16,693,594 elements, 63.68 MiB). Model and checkpoint provenance:
[MLCommons reference](https://github.com/mlcommons/inference/tree/3fbc329939999c13d0a7b5e67fb2092287e06047/recommendation/dlrm_v3),
[checkpoint](https://inference.mlcommons-storage.org/dlrmv3_trained_checkpoint/non_sparse.ckpt),
[publisher's Apache-2.0 license](https://inference.mlcommons-storage.org/dlrmv3_trained_checkpoint/LICENSE.txt).

Copyright (c) Meta Platforms, Inc. and affiliates. The TensorRT graph and NumPy
reference are NVIDIA adaptations; the cache runtime and serving adapters are
NVIDIA additions. See the repository [LICENSE](../../LICENSE) and
[asset attribution](../../ASSET_LICENSES.md#dlrmv3-dense-hstu-model). Generated
bundles retain this attribution in `checkpoint_provenance.json` and include
`LICENSE.txt` copied from the project's canonical license.

The caller performs sparse embedding lookup and supplies embeddings. The full
release's roughly 1 TB sparse table is unnecessary for this dense interface.
Scores represent candidate relevance probabilities; this model was trained on
the benchmark's synthetic streaming data. Numerical parity tests below do not
establish production recommendation quality or full MLPerf accuracy.

The tested serving interface is the supplied Triton prototype's structured-ranking
C++ API and this family's native transport. **Current main's C Task SDK is not
wired for this family**: it requires an `internal::IModel` implementation and a
shared Task contract for these mixed named tensors. That API migration remains
a prerequisite for general C Task SDK support; this contribution remains a draft.

## Build and change the serving profile

Keep downloaded weights, bundles, build products and test results outside the
checkout. Conversion requires PyTorch; building requires TensorRT and NumPy.

```bash
export PYTHONPATH="$PWD/core/builder:$PWD"
mkdir -p /tmp/dlrmv3
curl -fL -A 'Mozilla/5.0' \
  https://inference.mlcommons-storage.org/dlrmv3_trained_checkpoint/non_sparse.ckpt \
  -o /tmp/dlrmv3/non_sparse.ckpt
python -m families.hstu.checkpoint /tmp/dlrmv3/model \
  --source /tmp/dlrmv3/non_sparse.ckpt --history-length 512 --candidates 128
```

```python
from pathlib import Path
from tensorrt_model_connect import BuildRequest, build

build(
    BuildRequest(
        model_dir=Path("/tmp/dlrmv3/model"),
        output_path=Path("/tmp/dlrmv3/model.bundle"),
        family="hstu",
        task="structured_ranking",
        precision="bf16",
        max_batch_size=4,
    )
)
```

`config.json` owns the serving limits: `serving.history_length`, `candidates`,
`batching` (`dynamic` or `fixed`), `min_batch_size` and `opt_batch_size`.
`max_batch_size` comes from `BuildRequest`. The default is history 512,
128 candidates, dynamic batches 1–4. The second test profile uses history 32,
eight candidates and fixed batch 4. A larger profile, for example history 2048
and 256 candidates, uses the same checkpoint and requires rebuilding engines.
HSTU divides attention weights by `T+C+1`, matching the upstream call with
those maximum lengths; changing capacities can change scores. Validate each
new profile against the reference at the same limits. The architecture and
weights stay fixed. Capacity limits are history 16384,
candidates 2048, batch 64; these are bounds, not qualification of every shape.
`cuda_graphs` enables capture for the scoring stage; prefix and request
preparation execute through the native TensorRT runtime.

## Tensor and Task interface

`trtmc::IStructuredRanking::score`, `cache_stats`, the native Python client and
Triton tensor transport preserve the shared serving APIs. The interface
header lives in this family at `include/task.h`. Tensor schemas and scores follow DLRMv3.

| Input | Type | Shape |
|---|---|---|
| `history_item_embeddings`, `history_category_embeddings` | FP32 | `[B,T,512]` |
| `history_lengths` | INT32 | `[B,1]`, 0 through T |
| `history_timestamps` | INT64 | `[B,T]`, seconds since epoch |
| `history_actions` | INT32 | `[B,T]`, bitmask 0–31 |
| `user_embeddings` | FP32 | `[B,1,512]` |
| `query_timestamp` | INT64 | `[B,1]`, shared candidate query time |
| `num_candidates` | INT32 | `[B,1]`, 0 through C |
| `candidate_item_embeddings`, `candidate_category_embeddings` | FP32 | `[B,C,512]` |
| **Output:** `scores` | FP32 | `[B,C]`, padded candidates zeroed |

T and C are engine profile capacities. Callers pad tensors to these capacities;
lengths select valid rows. Attention is causal for history. Each candidate sees
the history and itself, and never another candidate. All candidates in one
request share `query_timestamp`. Embeddings must be finite; timestamps are
nonnegative and bounded by 2^40. Runtime metadata contains the exact schema.

FP32 is the reference mode. The default `bf16_policy: "expanded"` runs block
3–4 projections and attention, all five candidate AV products, the item MLP's
two matrix multiplies, the timestamp projection and the hidden prediction
projection in BF16. History blocks 0–2 and their candidate projections/QK/SiLU
remain FP32. Normalization, residuals, input preprocessing, biases and the
final prediction projection/sigmoid remain FP32. Each block runs once.

`bf16_policy: "conservative"` restores the previous policy: blocks 3–4 use
BF16, with the other operations in FP32. `fp32_block_indices` selects the
protected history/projection/QK blocks for both policies; expanded candidate
AV products are explicit exceptions. `runtime.json` records the effective
policy and extra BF16 operations. Changes require rebuilding and qualification.
`tests/thresholds/` declares score and ranking gates before qualification.
The original MLCommons model independently validates the full-sequence oracle;
its production positional kernel is used because its PyTorch fallback has a
bucket-count bug. TensorRT controls fusion and tactics; BF16 alone does not
guarantee a fused HSTU kernel or a speedup.

## Qualification results

The expanded policy passed all six precision/profile cases and 20 cache lifecycle
cases on GB300/SM103 with TensorRT 11.4.0.106. In addition to eight qualification
seeds, 32 held-out seeds per serving case passed the unchanged score/ranking
gates. On those held-out inputs, maximum absolute score error was 0.0162,
aggregate top-1 agreement was at least 96.875% and top-10 overlap at least 99.14%.
These are numerical checks on synthetic dense inputs, not dataset AUC/NE or
MLPerf compliance. Moving whole early blocks to BF16, the FP16 fallback,
additional early BF16 output projections and enabling TF32 failed score checks
and were rejected. TF32 remains disabled in both FP32 and BF16 builds.

GB300 diagnostic latency, milliseconds: median of three rounds' P50 values,
20 warmups and 100 measured calls per round, seed 42, rotated measurement order.
All columns use the same updated runtime and scoring-stage CUDA Graphs.

| History / candidates / batch | FP32 | Conservative BF16 | Expanded BF16 | Expanded reduction vs FP32 |
|---|---:|---:|---:|---:|
| 512 / 128 / 1 | 7.19 | 6.86 | 6.90 | 4.1% |
| 512 / 128 / 4 | 10.00 | 9.40 | 9.26 | 7.3% |
| 32 / 8 / 4, fixed | 4.95 | 4.79 | 4.76 | 3.8% |

Wider BF16 coverage adds only 1.4% at default batch 4 and 0.7% on the short
profile; batch 1 has no gain. Differences are small relative to run variation.
Nsight confirmed 20 BF16 GEMM launches per default-profile request, up from 13.
The runtime also moves JSON bounds lookup out of the per-float validation loop,
preserving all finite/range checks. A matched before/after BF16 comparison at
default batch 4 improved from 24.07 to 9.26 ms; most of this 61.5% reduction is
CPU work, rather than the extra BF16 operations. FP32 benefits from the same fix.

The measurement boundary is `NativeClient.score`, including host copies,
validation and synchronization, excluding Triton transport. Reproduce with
`python -m families.hstu.tests.benchmark --root <prepared>/output --runtime-root <sdk> --warmup 20 --iterations 100 --output <temporary>/latency.json`.
The earlier conservative policy passed 26 cases on A100; its expanded policy
and the updated runtime have not been requalified on A100 in this tuning round.

### HSTU attention and fusion

Attention is `SiLU(QKᵀ / sqrt(128)) / (T+C+1)`, multiplied by the mask and V.
It is expressed as decomposed TensorRT operations; TensorRT selects fusion.
Engine inspection and Nsight Systems execution traces on GB300 distinguish:

| BF16 serving profile | Fused attention kernels per request |
|---|---:|
| 512 / 128, dynamic batch 1–4, executed B4 | 0 |
| 512 / 128, fixed batch 4 | 2 |
| 32 / 8, fixed batch 4 | 2 |

The fixed-profile kernels are `_gemm_mha_v2` HSTU prefix-attention cores in
candidate blocks 3–4; the self contribution remains separate. The previous
conservative short profile also fuses, so wider BF16 coverage did not introduce
that fusion. No compiler debug flags were used. The fixed 512/128 profile
passed the same 40-seed checks and reduced BF16 P50 from 9.26 to 8.95 ms, but
P95 increased from 9.70 to 10.04 ms. This compares complete static/dynamic builds,
rather than isolating fusion alone. To use it, set `serving.batching` to `fixed`,
build with `max_batch_size=4`, and send exactly four rows per request.

## KV budget, metrics and native Triton

Build `trtmc_model_hstu` and `hstu_triton` with the SDK. For the supplied
**amd64 Triton 26.10 / TRT-MC 0.2.0 prototype**, run its ABI adapter inside the
image, then build the bundles there with TensorRT 11.4.0.106:

```bash
python3 -m families.hstu.triton.build_image_plugin \
  --output /tmp/dlrmv3/deploy --json-include /path/to/nlohmann/include
python3 -m families.hstu.triton.native_repository \
  --bundle /tmp/dlrmv3/model.bundle --repository /tmp/dlrmv3/repository \
  --name dlrmv3 --runtime-root /tmp/dlrmv3/deploy --kv-cache-memory-ratio 0.05
LD_LIBRARY_PATH=/tmp/dlrmv3/deploy:$LD_LIBRARY_PATH \
  tritonserver --model-repository=/tmp/dlrmv3/repository
```

The adapter recompiles only the family plugin and bridge and copies the image's
SDK DSOs unchanged. It accounts for that prototype's INT64 ordinal and
four-field factory/six-argument loader ABI. Do not mix it with a main-built
SDK or assume it applies to another image release. Rebuild serialized engines
when the TensorRT version or target platform changes.

The native serving path loads `libtrtmc_model_hstu.so` directly through the
image's backend. `native_repository.py` generates its configuration;
`libhstu_triton.so` reads bundle metadata for that generator and supports the
direct validation client. `model.py` and `generate_repository.py` implement the
separate Python-backend endpoint used for snapshot hydration; they are optional
for native scoring and its seven backend-published metrics.

Building in this image requires matching TensorRT Python bindings. A converted
checkpoint directory can be passed to `tests.prepare` to avoid installing
PyTorch in the serving image.

Generated configuration uses `backend: "tensorrt"`, `platform: "model_connect"`.
Specify **one** parameter: `kv_cache_size_bytes` (positive bytes) or
`kv_cache_memory_ratio` (finite fraction in (0,1]). Ratio budgets use free GPU
memory after engine initialization, adding back the already allocated KV
scratch. Both limits must cover the entire engine profile's scratch allocation.
They bound K/V storage, not weights, workspace or all GPU memory. The family
client configures these limits through its cache capability; the generic main
SDK decoder `--kv-cache-size` flag is unsupported for this family.

The native backend publishes seven `hstu_kv_cache_*` metrics at
`http://localhost:8002/metrics`: `supported`, `budget_bytes`, `reserved_bytes`,
`used_bytes`, `peak_bytes`, `active_sequences`, `allocation_failures`.
Reservation includes profile scratch and any restored entries; used/active
track in-flight scoring and return to zero at idle. Peak is a high-water mark.
FP32 K/V storage is `2 × 5 × 4 × (T+1) × 128 × 4` bytes per sequence, including
the user-context token. Model instances own their caches and metrics separately. In the A100 test,
a 128 MiB cap reported `budget_bytes=134217728`,
`reserved_bytes=peak_bytes=42024960`, and zero used bytes/active sequences
at idle. Changing the ratio from 1% to 5% increased the reported budget fivefold.
Native and hydration endpoints also passed simultaneous-load cache isolation;
positive allocation failures were exercised through snapshot restore pressure.

## Restore precomputed history

The family retains `export_history`, `restore_history`, `score_with_history`,
`history_info` and `release_history`. An opaque v2 snapshot contains all five
layers' K/V plus auxiliary state. Restoration copies this state to GPU without
running history preparation. Query time, candidate count, user context, history
features/revision, precision and serving profile belong to snapshot identity.
Only candidate embeddings may change during cached scoring. In particular,
the original model's context gate depends on history length plus candidate count.

Snapshots are trusted binary inputs with bounded framing, tensor validation
and corruption checks. A per-build bundle UUID rejects incompatible bundles;
rebuilding produces a new identity. Handles belong to one model instance.
Restore admission enforces byte/entry limits, TTL, early-hydration leases and
LRU eviction. The maximum snapshot size is 1 GiB.

```bash
python -m families.hstu.examples.history_cache \
  --bundle /tmp/dlrmv3/model.bundle --checkpoint /tmp/dlrmv3/model \
  --runtime-root /path/to/sdk/build
```

For remote restore, `triton.generate_repository --history-cache` creates the
Python transport exposing `CACHE_CONTROL`, `CACHE_SNAPSHOT` and `CACHE_STATUS`.
It runs the same native family plugin. This endpoint owns a separate cache
from the native backend endpoint; a handle cannot cross between them.
See `tests/triton/hydration_serving.py` for HTTP/gRPC restore and score requests.

## Reproduce validation

```bash
python -m families.hstu.tests.prepare --root /tmp/dlrmv3/qualification \
  --checkpoint /tmp/dlrmv3/non_sparse.ckpt --runtime-root /path/to/sdk/build
HSTU_E2E_ROOT=/tmp/dlrmv3/qualification/output TRTMC_RUNTIME_ROOT=/path/to/sdk/build \
  pytest families/hstu/tests/test_e2e.py families/hstu/tests/test_history_cache_gpu.py
```

In the supplied image, serve `qualification/triton-models` with explicit model
control, the matching deployment directory on `LD_LIBRARY_PATH`, and ports
8000/8001/8002. Set `HSTU_TEST_ROOT` to the qualification directory and
`TRTMC_RUNTIME_ROOT` to the deployment directory; run
`families.hstu.tests.triton.native_serving`, `native_cache` and
`hydration_serving` with `python -m`. Results and Prometheus scrapes go there.
The supplied image still accepts four malformed configurations: ratio
`0.05junk`, or byte limits `-1`, `134217728.5` and `134217728junk`. The native
cache campaign reports these as failures; the family generators and native
client reject them. Fixing this parser requires a Triton backend change.

CI stages `external_files` from the manifests before offline execution. Set
`TRTMC_HSTU_MODEL_DIR` to that raw-checkpoint directory (or explicitly set
`DLRMV3_CHECKPOINT`); selected tests never download weights. CI's
`TRTMC_NATIVE_BUILD_DIR` locates the transport library while the Task loader
uses the isolated `TRTMC_RUNTIME_ROOT`. An independent upstream check is
available through `tests/upstream_check.py` with `DLRMV3_UPSTREAM`,
`DLRMV3_CHECKPOINT` and `DLRMV3_CONVERTED` set.

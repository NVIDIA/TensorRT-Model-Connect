# Trained public HSTU recommender

This is the recommended trained HSTU example. It imports a complete publicly
available checkpoint and preserves its relative position/time biases, all 16
blocks and cosine retrieval head. The family owns its builder, runtime,
reference and tests independently. The earlier `hstu` family remains a
one-block target-aware greedy-slate example; its weights and snapshots are
different.

## Checkpoint and task

The default is [tzchen07's run26](https://huggingface.co/tzchen07/hstu-checkpoints-2026-06-10),
`run26_ndcg1849.tar.gz`, revision
`fb6530eba870e842cfb236737586a4188ec3faef`. Its archive SHA-256 is
`9c4edcecb5e15c513ec7baa343f1a1e2ee0b8a200242e811612f55205a74c9ac`.
The publisher reports MovieLens-20M NDCG@10 of 0.1849. It is an existing trained
checkpoint, not a random initialization or a new training run. It was selected
over higher-scoring PRISM/FiLM variants because it is plain HSTU and can be
imported completely without dropping trained components. Config-only Hub
repositories are not pretrained alternatives.

Architecture: width 256, 16 blocks, eight heads, Q/K and V dimensions 32 per
head, 131,263 embedding rows, 211 learned positions, trained history limit 200,
128 time buckets. All 83 model tensors are imported; unsupported components
or masks fail conversion. Each block implements
`LN -> SiLU(U,V,Q,K) -> SiLU(QKᵀ + relative biases)/211 -> causal mask -> AV -> LN -> U gate -> projection + residual`.
There is no softmax. The final valid history embedding and candidate embeddings
are L2-normalized before cosine scoring. Training-loss parameters do not modify
inference scores.

The [Meta reference](https://github.com/meta-recsys/generative-recommenders/blob/474afe5cfdb3d16d603d04f943f4dbe7eb6866fd/generative_recommenders/research/modeling/sequential/hstu.py)
is Apache-2.0. **An explicit weight license was not found in the publisher's
repository.** Public download availability does not establish redistribution
rights. No weights or MovieLens data are included here; see `NOTICE`.

| Tensor | Type | Shape and meaning |
| --- | --- | --- |
| `history_ids` | INT64 | `[B,T]`, raw MovieLens movie IDs; right-pad with 0 |
| `history_lengths` | INT64 | `[B,1]`, valid lengths 1 through T |
| `history_timestamps` | INT64 | `[B,T+1]`, seconds, nonnegative and at most 2^40 |
| `candidate_ids` | INT64 | `[B,C]`, valid embedding IDs including padding |
| `num_candidates` | INT64 | `[B,1]`, valid counts 0 through C |
| `scores` | FP32 | `[B,C]`, cosine scores; unused positions are 0 |
| `picked_indices` | INT64 | `[B,1]`, stable top-1 candidate position; -1 for zero candidates |

The timestamp at index **history_lengths[b,0]** is the next-event/prediction
timestamp. The checkpoint's shifted time-bias calculation requires this value,
including for shorter histories. It is part of the snapshot's history identity.
Changing the prediction timestamp requires a new snapshot/revision. The caller
maps raw IDs, orders events, truncates, pads and supplies lengths/timestamps.
Sort the valid scores for top-K retrieval; this model does not generate a
50-round target-aware slate.

## Build and choose a serving profile

Use the repository TensorRT environment and keep generated files outside source:

```bash
export PYTHONPATH=core/builder:.
# Conversion additionally requires PyTorch >=2.6. It never executes Hub code.
python3 -m families.hstu_research.checkpoint /tmp/hstu-trained --download-default
cmake -S . -B /tmp/hstu-sdk -DTRTMC_BUILD_FAMILIES=hstu_research \
  -DTRTMC_ENABLE_BYOK=OFF -DTRTMC_BUILD_SERVER=OFF
cmake --build /tmp/hstu-sdk --target trtmc_model_hstu_research hstu_research_triton trtmc_backend_trt
python3 -m tensorrt_model_connect build /tmp/hstu-trained --family hstu_research \
  --task structured_ranking --precision bf16 --max-batch-size 4 \
  -o /tmp/hstu-trained/model.bundle
python3 -m families.hstu_research.examples.history_cache \
  --checkpoint /tmp/hstu-trained --bundle /tmp/hstu-trained/model.bundle \
  --runtime-root /tmp/hstu-sdk
```

`config.json` separates immutable `architecture` from `serving`. Default serving
is T=200, C=64, dynamic B=1..4 for the build above. Another qualified profile is
T=64, C=128, fixed B=4. Change only `serving` and rebuild:

```python
from dataclasses import replace
from families.hstu_research.config import Config, ServingProfile

directory = "/tmp/hstu-trained"
c = Config.load(directory)
replace(c, serving=ServingProfile(history_length=64, candidates=128, batching="fixed")).save(
    directory
)
```

Dynamic profiles accept `min_batch_size` and `opt_batch_size`, with
`1 <= min <= opt <= BuildRequest.max_batch_size`. Fixed profiles require the
exact built batch. T must be at most the trained limit (200 for run26); C can
be 1..4096. Larger shapes/batches need their own accuracy, memory and latency
qualification. Increasing T past 200 requires a compatible checkpoint; changing
padding never changes the learned positions or the attention divisor 211.
Architecture dimensions are loaded from checkpoint tensors and training config,
rather than changed to make a serving profile fit. The runtime allocates and
validates tensors using bundle metadata. It does not pad, split or schedule
application batches.

## BF16 behavior and measured result

The converted checkpoint selects `bf16_projections="all"`,
`fp32_block_indices=[0]`, `bf16_attention_epilogue=false`, `cuda_graphs=true`.
Block 0 stays in FP32. Blocks 1–15 use BF16 UVQK/output projections and QK/AV
matrix multiplication, with FP32 normalization, SiLU/bias/scaling/masking,
residuals and final cosine head. This is one encoder path with no second FP32
attention/head selector. FP32 builds execute all blocks in FP32.
`bf16_projections="none"` retains FP32 projections for a conservative
attention-only comparison. Precision options require rebuilding engines.

GB300, TensorRT 11.4.0.106, identical complete weights and inputs, T=200/C=64,
dynamic B=1..4, CUDA Graphs enabled for the scoring stage. History/candidate
preparation uses ordinary engine launches. Median synchronized `NativeClient` call time
in milliseconds, including host tensor serialization/copies, excluding load,
build and warmup. Thirty warmups per policy/mode and 300 alternating timed
calls per policy over three repeats; timings were collected without a profiler.

| Batch | History | FP32 | Selected BF16 | Reduction vs FP32 |
| --- | --- | --- | --- | --- |
| 1 | cold | 3.859 | 3.256 | 15.6% |
| 4 | cold | 4.091 | 3.318 | 18.9% |
| 1 | restored | 2.676 | 2.548 | 4.8% |
| 4 | restored | 2.846 | 2.666 | 6.3% |

AMD64 A100 40-GB PCIe, supplied 26.10/0.2.0 container, same TensorRT version,
profile, weights, graph policy and measurement procedure:

| Batch | History | FP32 | Selected BF16 | Reduction vs FP32 |
| --- | --- | --- | --- | --- |
| 1 | cold | 2.487 | 1.775 | 28.6% |
| 4 | cold | 2.939 | 2.059 | 29.9% |
| 1 | restored | 1.011 | 0.789 | 21.9% |
| 4 | restored | 1.193 | 0.984 | 17.6% |

These are within-host comparisons; host serialization costs differ between
machines. A separate GB300 trace contains ten scoring graph replays for ten
restored requests and BF16 QK/AV GEMM kernels with FP32 accumulation. It confirms
that reduced precision and CUDA Graphs execute; profiling was disabled for timing.

Twenty deterministic seeds at B1/B4 meet the unchanged BF16 pointwise gate
`abs(error) <= 0.02 + 0.02*abs(reference)` and have identical top-1 selections.
Largest score error was 0.01048. Quantizing every block, including block 0,
failed the same gate and is not the default. FP32 parity is checked against an
independent full-sequence NumPy oracle, itself checked against all 16 pinned
upstream Meta blocks.

On the first 512 MovieLens-20M users, holding out the final interaction with 63
sampled unseen negatives, selected BF16 and FP32 both measured sampled NDCG@10
0.907944 and hit-rate@10 0.994141. Mean top-10 overlap was 99.86%, mean absolute
score error 0.000157, maximum 0.001152. This is a relative precision check with
sampled candidates, **not** a reproduction of the publisher's full evaluation
or production-quality qualification. These numbers do not imply HSTU kernel
fusion; the dependency pin remains 11.4.

## Prefix restore, budget and Triton

The three engine stages remain history preparation, candidate preparation and
scoring. History preparation caches every layer's prefix K/V, excluding the last
valid token. Scoring replays that last query through the stack using the restored
prefix plus its self K/V, then evaluates candidates. Restored scoring makes zero
history-prepare calls. K/V storage remains FP32 in either precision mode.
All other features retain the same serving contract: SHA-256 snapshots,
deduplication, revisions, TTL, leases, release, eviction and metrics. Snapshots
bind to the exact bundle hash, including weights, engine build, profile and
precision; they cannot be transferred between rebuilt/different bundles.

For T=200, K/V consumes 6,553,600 bytes per history; max-B4 scratch reserves
26,214,400 bytes. T=64 consumes 2,097,152 per history and 8,388,608 scratch bytes.
Masks/query context are counted separately in `context_device_bytes`.
Hydration requires budget beyond scratch. The qualified 128-MiB budget supports
the example's scratch plus eight resident histories. Snapshots are bounded to
64 MiB; larger export batches are not qualified.

Native Triton serving uses `backend: "tensorrt"`, `platform: "model_connect"`:

```bash
python3 -m families.hstu_research.triton.native_repository \
  --bundle /tmp/hstu-trained/model.bundle --repository /tmp/hstu-models \
  --name hstu --kv-cache-memory-ratio 0.05
```

Alternatively use `--kv-cache-size-bytes 134217728`. The byte cap and ratio are
mutually exclusive. Ratio is applied at instance creation to available free GPU
memory with its already allocated scratch K/V added back. It is a budget ceiling,
not a promise to preallocate that fraction of the GPU. A budget smaller than the
complete profile's scratch allocation fails model loading.

The supplied 26.10/0.2.0 backend publishes these seven metrics at `/metrics`
(default port 8002), with model/version/device/instance labels:

- `hstu_kv_cache_supported`
- `hstu_kv_cache_budget_bytes`
- `hstu_kv_cache_reserved_bytes`
- `hstu_kv_cache_used_bytes`
- `hstu_kv_cache_peak_bytes`
- `hstu_kv_cache_active_sequences`
- `hstu_kv_cache_allocation_failures`

Native scoring does not expose snapshot-control operations in that image.
The separate `backend: "python"` endpoint transports the same native Task's
`CACHE_CONTROL`, `CACHE_STATUS` and `CACHE_SNAPSHOT` protocol. Generate it with
`triton.generate_repository --history-cache --runtime-root ...` and an explicit
budget. It additionally publishes resident/restore/hit/eviction/preparation
counters. `examples/history_cache.py` demonstrates the native restore API.

## Reproduce qualification

Install `tests/requirements.txt` in the test environment. Convert once; rebuild
plans on the target GPU/SDK. The supplied AMD64 26.10 image requires the
family-local `triton.build_image_plugin` helper because its prototype C++ dtype
ABI differs from current main. The helper preserves image SDK/backend binaries;
never use its plugin with a main-built SDK.

```bash
python3 -m families.hstu_research.tests.prepare --checkpoint /tmp/hstu-trained \
  --root /tmp/hstu-qa --runtime-root /tmp/hstu-sdk
HSTU_RESEARCH_E2E_ROOT=/tmp/hstu-qa/output TRTMC_RUNTIME_ROOT=/tmp/hstu-sdk \
  python3 -m pytest families/hstu_research/tests/test_e2e.py \
  --e2e-model hstu_research -q -p no:cacheprovider --basetemp=/tmp/hstu-pytest
HSTU_RESEARCH_E2E_ROOT=/tmp/hstu-qa/output TRTMC_RUNTIME_ROOT=/tmp/hstu-sdk \
  python3 -m pytest families/hstu_research/tests/test_history_cache_gpu.py \
  -q -p no:cacheprovider --basetemp=/tmp/hstu-history-tests
```

The 16 lifecycle cases cover both precisions at B1/B4, including revisions,
leases, LRU eviction, TTL expiry, reordered rows and changed candidates. CPU
tests can additionally execute the pinned Meta implementation by setting
`HSTU_RESEARCH_CHECKPOINT` and `HSTU_UPSTREAM_SOURCE` before running
`tests/test_reference.py`; the upstream comparison skips when those are absent.

Timing can be repeated with `tests.benchmark --fp32 <directory> --bf16 <directory>
--runtime-root <sdk> --output /tmp/timing.json`. The sampled quality check is
`tests.compare_precision` with the same arguments plus
`--ratings-archive /path/to/ml-20m.zip`. Run both as modules prefixed by
`families.hstu_research`; they require identical checkpoint weights.

Preparation writes the four bundle directories under `output` and the
`triton-models` directory under its root.
Set `HSTU_RESEARCH_TEST_ROOT` to their parent containing `output` and
`triton-models` when running the serving campaigns. Add `TRTMC_RUNTIME_ROOT` to
`LD_LIBRARY_PATH`, start a dedicated `tritonserver` with
`--model-repository="$HSTU_RESEARCH_TEST_ROOT/triton-models"`,
`--model-control-mode=explicit` and `--allow-gpu-metrics=false`, then run
`tests.triton.native_serving`, `tests.triton.hydration_serving`, and
`tests.triton.native_cache` as modules in this family. These write results and
scrapes under the supplied test root. The cache campaign deliberately retains
negative parser checks; the 26.10 backend accepts four malformed numeric
strings (`0.05junk`, `-1`, fractional bytes and trailing-junk bytes). Those are
upstream backend failures, not passing family validation.

On AMD64 A100 in that image, both profiles/precisions passed all six Task/cache
cases and all 12 native HTTP/gRPC comparisons. Twelve hydration transport cells
passed bitwise cold/restored parity, live metrics, corruption rejection, export
and model reload. The native budget/metric campaign passed 30 of 34 checks;
the four failures were the malformed strings listed above. It observed the
seven native metrics, effective 1%/5% budgets, insufficient-budget rejection,
mutual exclusion and model/instance isolation. Positive allocation-failure
counting was also verified through the snapshot endpoint under leased-entry
pressure; ordinary native scoring uses fixed scratch and does not exhaust a
loaded model's cache budget.
The 16 detailed lifecycle cases also passed on both GB300 and A100.

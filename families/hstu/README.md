# Public HSTU ranking

This family uses Meta's Apache-2.0 [HSTU implementation](https://github.com/meta-recsys/generative-recommenders/blob/474afe5cfdb3d16d603d04f943f4dbe7eb6866fd/generative_recommenders/research/modeling/sequential/hstu.py), revision `474afe5cfdb3d16d603d04f943f4dbe7eb6866fd`, as its mathematical reference. It contains one HSTU block and a public target-aware scoring head/greedy slate adapter. It is not Meta's complete pretrained recommender or its production DLRM model. No customer topology, checkpoint, feature names, or golden outputs are required.

The block computes `LN -> SiLU(U,V,Q,K) -> SiLU(QKᵀ)/N -> mask -> AV -> LN -> U gate -> projection + residual`. Attention has **no softmax**. Timestamp/relative bias, multiple HSTU blocks, and arbitrary upstream checkpoints are outside this profile. The tests execute the pinned upstream block itself, with only CPU dense/jagged transport shims.

Supported profiles: one block, width 256, eight heads, 128 padded history positions, 50 selection rounds, 50 or 64 candidates, FP32/BF16, variable batch size up to the built maximum. The application pads histories and supplies lengths. Candidate queries see valid history, previously selected items, and themselves. Duplicate item IDs use their first occurrence's score before stable, lowest-index tie breaking. Unselected scores remain zero; unused selection slots are `-1`.

| Input | Type | Shape |
| --- | --- | --- |
| `history_ids` | INT64 | `[B,128]` |
| `history_lengths` | INT64 | `[B,1]`, 0–128 |
| `candidate_ids` | INT64 | `[B,C]` |
| `num_candidates` | INT64 | `[B,1]`, 0–C |

IDs must be in the checkpoint vocabulary, including padded positions. Outputs are `scores: FP32[B,C]` and `picked_indices: INT64[B,50]`.

## Public checkpoints

Create a reproducible, **untrained validation fixture** without downloads:

```bash
export PYTHONPATH=core/builder:.
python3 -m families.hstu.checkpoint /tmp/hstu-public --candidates 50
python3 -m tensorrt_model_connect build /tmp/hstu-public --family hstu \
  --task structured_ranking --precision bf16 --max-batch-size 4 \
  -o /tmp/hstu-public/model.bundle
```

Alternatively, train example weights on a separately downloaded MovieLens 100K `u.data` file:

```bash
python3 -m families.hstu.train --ratings /path/to/ml-100k/u.data \
  --output /tmp/hstu-movielens --steps 100 --batch-size 8
```

The training command needs PyTorch. It trains the whole public block and head; it records the seed, data/weight hashes, loss, and sampled held-out evaluation in `checkpoint_provenance.json`. This short recipe is a training/integration example, not an official pretrained release or a recommendation-quality qualification. MovieLens has its own [dataset terms](https://files.grouplens.org/datasets/movielens/ml-100k-README.txt). Neither data nor checkpoint binaries are checked into this repository. Checkpoint arrays are validated by `checkpoint.py`; do not treat unrelated Hugging Face HSTU state dictionaries as compatible.

Hugging Face was checked as well. [Lingqu-Rec/HSTU-1B](https://huggingface.co/Lingqu-Rec/HSTU-1B) explicitly uses random weights for testing. [tzchen07/hstu-iter78-preservation](https://huggingface.co/tzchen07/hstu-iter78-preservation) contains trained MovieLens research artifacts, but its card distinguishes the code license from research-artifact weight terms and its architecture differs. Neither is silently substituted as a supported pretrained checkpoint here.

## Precision and runtime

FP32 is the baseline. Both BF16 policies run QK and attention-weight/V GEMMs with BF16 inputs/outputs, restoring FP32 for sensitive arithmetic. The tested A100 Tensor Core tactics accumulate in FP32; BF16 output rounding remains. Embeddings, UVQK projection, normalization, and the score head stay FP32. `config.json` selects:

- `"bf16_policy": "strict"` (default): FP32 output projection and a separate FP32 attention/head selector preserve greedy stability. The existing strict numerical gates apply.
- `"bf16_policy": "fast"`: one attention/head path produces scores and selections; the output projection also uses BF16. This trades full-slate/individual-score parity for lower latency. Ranking overlap, held-out quality, and mean score error have separate checks; passing them does not imply passing strict score parity.

These checkpoint fields are encoded in rebuilt bundles. New bundles default to `"cuda_graphs": true`: the native runtime captures/replays the reusable **step engine**, with preparation, the host round loop, and state copies outside capture. TensorRT invalidates its capture when shapes or bindings change. Set the checkpoint field to `false`, or pass `--no-cuda-graphs` below, to disable it. Older bundles without this metadata keep graphs disabled. This family checkpoint option is separate from the shared RTX-only loader graph flag.

Build all four serving variants with the fast BF16 policy and matched graph settings:

```bash
python3 -m families.hstu.tests.triton.prepare --root /tmp/hstu-fast \
  --runtime-root /tmp/hstu-sdk --checkpoint /tmp/hstu-movielens \
  --bf16-policy fast --cuda-graphs
HSTU_FAST_E2E_ROOT=/tmp/hstu-fast/output TRTMC_RUNTIME_ROOT=/tmp/hstu-sdk \
  python3 -m pytest families/hstu/tests/test_fast_precision.py -q
python3 -m families.hstu.tests.compare_precision --root /tmp/hstu-fast/output \
  --runtime-root /tmp/hstu-sdk --ratings /path/to/ml-100k/u.data \
  --examples 256 --output /tmp/hstu-fast/quality.json
```

The quality comparison requires identical FP32/BF16 weights. Its fixed limits are at most 1% relative NDCG@10 loss, one percentage point hit-rate loss, at least 98% mean top-10 overlap, and mean absolute score error at most 0.02. It holds out each sampled user's final interaction. The separate strict numerical gates are unchanged.

TensorRT owns fusion and kernel selection. The current A100 C50/B4 trace contains 150 BF16 Tensor Core GEMM launches in fast mode: 50 QK, 50 AV, and 50 output projections, with FP32 accumulation. FP32 has zero BF16 launches. Both modes replay 50 step CUDA Graphs per request; disabling the checkpoint graph switch produces zero graph launches. QK and AV remain separate; no dedicated fused HSTU kernel was observed. Softmax-MHA fusion results from another model do not transfer.

The staged runtime retains the existing three-engine orchestration, GPU slate state, KV byte accounting, SHA-256 snapshots, restore/deduplication, revisions, TTL, leases, and eviction. Cache storage is FP32 in both precision modes: **262,144 KV bytes per history**, plus a 512-byte history mask outside the KV budget. A max-batch-4 engine reserves 1,048,576 scratch KV bytes. Restored histories need budget beyond scratch. Snapshots are bound to the exact bundle hash and cannot migrate from the private model, a rebuilt engine, or a different precision bundle.

## Triton and KV budgets

Use the native backend in the supplied AMD64 `tritonserver:26.10-py3-trtmc-0.2.0` image:

```bash
python3 -m families.hstu.triton.native_repository \
  --bundle /tmp/hstu-public/model.bundle --repository /tmp/models \
  --name hstu --kv-cache-memory-ratio 0.05
```

It generates:

```protobuf
backend: "tensorrt"
platform: "model_connect"
default_model_filename: "model.bundle"
parameters { key: "kv_cache_memory_ratio" value { string_value: "0.05" } }
```

Alternatively specify `--kv-cache-size-bytes 8388608`. The two options are mutually exclusive. Ratio must be finite and in `(0,1]`; it caps KV bytes against free GPU memory after non-cache initialization, adding back already allocated scratch KV. An omitted cap defaults to scratch capacity. Each model instance resolves its own budget. Too little capacity fails model loading; the ratio is a cap, not an eager allocation of that fraction.

**Match the SDK ABI.** Current main and the prototype image use different release-coupled C++ tensor enums. Build ordinary native SDK libraries together with CMake (`TRTMC_BUILD_FAMILIES=hstu`). For this exact prototype image, run the following *inside the image* to create a separate SDK directory containing byte-identical image SDK libraries and the new public plugin/bridge compiled for its ABI:

```bash
python3 -m families.hstu.triton.build_image_plugin \
  --output /tmp/hstu-sdk --json-include /path/containing/nlohmann
LD_LIBRARY_PATH=/tmp/hstu-sdk:$LD_LIBRARY_PATH \
  tritonserver --model-repository=/tmp/models --model-control-mode=explicit
```

The helper pins the tested backend/SDK hashes. It changes neither the Triton backend nor TensorRT/CUDA libraries. Do not mix the image plugin with main-built SDK libraries. Rebuild engines inside the target image for the target GPU.

The native backend publishes these seven gauges at `http://localhost:8002/metrics`, labeled by model, version, device, and instance:

- `hstu_kv_cache_supported`
- `hstu_kv_cache_budget_bytes`
- `hstu_kv_cache_reserved_bytes`
- `hstu_kv_cache_used_bytes`
- `hstu_kv_cache_peak_bytes`
- `hstu_kv_cache_active_sequences`
- `hstu_kv_cache_allocation_failures`

Metrics publish asynchronously. Observe effective budgets and active/idle transitions, not just echoed configuration. For B4, active KV usage is 1,048,576 bytes; idle usage/active sequences return to zero while reserved storage and peak remain. The normal native scoring path has no additional cache allocation after loading, so its allocation-failure gauge normally stays zero.

## Restore example

```bash
python3 -m families.hstu.examples.history_cache \
  --checkpoint /tmp/hstu-public --bundle /tmp/hstu-public/model.bundle \
  --runtime-root /tmp/hstu-sdk --snapshot /tmp/history.kv
```

This exports a snapshot, destroys the producer, restores into another instance, scores using only candidate inputs, checks bit-identical cold/restored results, and verifies zero history preparation calls after restoration.

The prototype **native** Triton backend supports scoring and seven metrics, but has no snapshot-control input contract. This is the existing integration boundary. The family-owned Python transport exposes export/restore/score/release over Triton using the same C++ runtime:

```bash
python3 -m families.hstu.triton.generate_repository \
  --bundle /tmp/hstu-public/model.bundle --repository /tmp/models \
  --name hstu_history --runtime-root /tmp/hstu-sdk --history-cache \
  --kv-cache-size-bytes 9437184
```

That endpoint uses `backend: "python"` and optional `CACHE_CONTROL`/`CACHE_SNAPSHOT` inputs. It also publishes restoration/residency counters. No PyTorch is used for serving. The bridge is trusted in-process test/example transport, not a remote pointer API. The modern TRT-MC C SDK has no structured-ranking binding yet; use this C++ Task interface, the example bridge, or the prototype native Triton backend.

## Validation

Numerical gates are fixed in `tests/thresholds/correctness.json`. CPU tests run without a GPU. Selected E2E tests build deterministic fixtures in pytest temporary directories by default. Set `HSTU_E2E_ROOT` to use prepared `hstu50-fp32`, `hstu50-bf16`, `hstu64-fp32`, and `hstu64-bf16` directories instead:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest families/hstu/tests -q
HSTU_E2E_ROOT=/tmp/checkpoints TRTMC_RUNTIME_ROOT=/tmp/hstu-sdk \
  python3 -m pytest families/hstu/tests/test_e2e.py --e2e-model hstu -q
HSTU_UPSTREAM_SOURCE=/path/to/pinned/public/hstu.py \
  python3 -m pytest families/hstu/tests/test_upstream.py -q
```

Prepare all four variants and both endpoint configurations in a fresh test directory:

```bash
export HSTU_TEST_ROOT=/work/qa
export TRTMC_RUNTIME_ROOT=/work/hstu-sdk
python3 -m families.hstu.tests.triton.prepare --root "$HSTU_TEST_ROOT" \
  --runtime-root "$TRTMC_RUNTIME_ROOT" --checkpoint /path/to/public-checkpoint
# Omit --checkpoint to create untrained fixtures.
LD_LIBRARY_PATH="$TRTMC_RUNTIME_ROOT:$LD_LIBRARY_PATH" \
  tritonserver --model-repository="$HSTU_TEST_ROOT/triton-models" \
  --model-control-mode=explicit --allow-gpu-metrics=false
# In another shell inside the same container, with the same exports:
python3 -m families.hstu.tests.triton.native_serving
python3 -m families.hstu.tests.triton.hydration_serving
python3 -m families.hstu.tests.triton.native_cache
```

`tests/triton/{native_serving,native_cache,hydration_serving}.py` replay the container campaigns. Run them as Python modules against a dedicated test server and the directory prepared above. `HSTU_TEST_ROOT` defaults to `/work`; `TRTMC_RUNTIME_ROOT` defaults to its `deploy` subdirectory. Busy-gauge checks sustain scoring until an active sample appears, with a five-second observation deadline; short individual requests can fall between asynchronous metric publications. They preserve failure exit codes and write evidence outside the source directory. The history lifecycle test accepts `HSTU_HISTORY_BUNDLE`, `HSTU_HISTORY_REQUEST` (NPZ), and `HSTU_HISTORY_RUNTIME_ROOT`.

The 26.10 image has four pre-existing configuration-parser defects: it accepts ratio `"0.05junk"`, bytes `"-1"`, `"8388608.5"`, and `"8388608junk"`. The negative tests remain failures; the backend needs strict full-string numeric parsing. This family cannot recover the original strings after the backend has parsed them. Valid budget/metric, multi-model, and multi-instance cases pass.

## Measured result

AMD64 A100 PCIe 40 GB, supplied 26.10/0.2.0 image, TensorRT 11.4.0.106; public MovieLens 100-step example weights, seed 20261124. This is a short training/integration example, not production recommendation-quality qualification.

Matched `NativeClient.score` median milliseconds, including synchronized GPU work and host tensor transport, excluding model loading/compilation and HTTP. CUDA Graphs are **enabled for all three policies**. Each cell uses 50 warmups and 300 measured calls per policy, interleaved over three repeats:

| Candidates | Batch | FP32 ms | Strict BF16 ms | Fast BF16 ms | Fast latency reduction vs FP32 |
| --- | --- | --- | --- | --- | --- |
| 50 | 1 | 5.615 | 7.104 | 5.013 | 10.7% |
| 50 | 4 | 6.086 | 7.481 | 5.445 | 10.5% |
| 64 | 1 | 5.994 | 7.419 | 5.088 | 15.1% |
| 64 | 4 | 6.436 | 7.901 | 5.868 | 8.8% |

The earlier latency table used strict BF16 with graphs disabled. The fast policy removes the second full attention/head path and moves the output projection to BF16; it beats the matched FP32 baseline in these four staged cases. Unrolled/Loop variants and other GPUs are unqualified.

Held-out quality on all **943 MovieLens users per candidate count**, with matched FP32/BF16 weights and sampled negative candidates:

| Candidates | FP32 NDCG@10 | Fast BF16 NDCG@10 | Mean top-10 overlap | Top-1 agreement | Full-slate position agreement | Mean absolute score error |
| --- | --- | --- | --- | --- | --- | --- |
| 50 | 0.19304 | 0.19378 | 99.69% | 99.89% | 96.94% | 0.00597 |
| 64 | 0.15842 | 0.15842 | 99.73% | 99.89% | 95.71% | 0.00564 |

No NDCG@10 loss was measured in this smoke evaluation. **Fast mode does not preserve individual-score parity:** tail selections can diverge and alter later context, and C64 candidates can cross the selected/unselected boundary. Maximum individual score differences reached 2.486 (C50) and 1.885 (C64). Use strict mode for score calibration or full-slate parity requirements; its original numerical gates remain unchanged.

The optimization rerun includes eight strict numerical cells and eight fast-policy/graph-freshness tests per SDK (image and current main), 64 history lifecycle checks across both SDKs, four main-SDK C++ checks, 16 native HTTP/gRPC cells, and 16 snapshot transport cells with a positive allocation-failure metric check. The seven native gauges and effective KV budgets are checked by the native cache/config campaign; its four existing backend parser failures remain explicit. Source and evidence are recorded outside the repository. The generic modern C-SDK structured-ranking binding remains unavailable; these timings use the family-owned native Task bridge.

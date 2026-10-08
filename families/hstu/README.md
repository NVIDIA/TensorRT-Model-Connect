# HSTU recommendations

This family implements the HSTU ranking and retrieval mathematics from
[NVIDIA recsys-examples](https://github.com/NVIDIA/recsys-examples/tree/97062d97eef53115105063801e35184e36186df5/examples/hstu).
The deployed path is a Model Connect family DSO, TensorRT engines, and C++.
Python and PyTorch are used for checkpoint conversion, engine building, and
reference validation only.

## Model contract

The TensorRT graph contains embedding lookup, optional position and timestamp
embeddings, HSTU blocks, output normalization, and the ranking MLP. An HSTU block
computes input LayerNorm, a fused UVQK projection and SiLU, then
`SiLU(Q Kᵀ / sqrt(head_dim)) / scaling_seqlen` attention. It normalizes the
attention output across all heads, multiplies by U, projects, and optionally
adds the residual. There is no attention softmax.

Ranking consumes contextual features, history items and their optional actions,
and candidate items. C++ constructs the reference token order: contextual
features in checkpoint order, interleaved history items/actions, then candidate
items. Its mask implements causal/noncausal attention, contextual attention,
and candidate groups. It returns raw MLP logits and L2-normalized embeddings;
apply the task's own score interpretation outside the model.

Retrieval transduces contextual features and the item/action history and looks
up candidates separately. It returns dot products between the last history
item's HSTU embedding and normalized candidate item embeddings. The final
action token is not the retrieval query. Training softmax temperature is not
applied to these scores.

The bundle uses one GPU and resident embedding weights. Batch and sequence
lengths are dynamic within the build profile. FP32, FP16, and BF16 graphs use
explicit dtype boundaries. Optional native KV caching reuses a validated history
prefix and evaluates only its appended history and current candidates. The
deployed runtime uses Model Connect, TensorRT, CUDA, and C++; it does not load
PyTorch, TorchRec, DynamicEmb, NVE, AOTInductor, or Python. It does not implement
distributed training, tensor-parallel inference, dynamic table updates, ANN
indexing, or the optional input-projection MLP preprocessing variant.

## Convert an NVIDIA checkpoint

Prepare an explicit topology JSON matching the trained checkpoint. For example:

```json
{
  "model_type": "hstu",
  "schema_version": 1,
  "mode": "ranking",
  "hidden_size": 128,
  "num_heads": 4,
  "head_dim": 128,
  "num_layers": 1,
  "max_sequence_length": 512,
  "embedding_tables": [
    {"name": "item", "role": "item", "num_embeddings": 1000},
    {"name": "action", "role": "action", "num_embeddings": 10}
  ],
  "prediction_head": [512, 10],
  "prediction_activation": "relu",
  "prediction_bias": true,
  "position_buckets": 512,
  "time_buckets": 0,
  "scaling_seqlen": 512,
  "disable_contextual_mask": true,
  "enable_history_cache": true
}
```

Replace dimensions, table names, capacities, head, and semantic options with
the training configuration; the numbers above illustrate the format. The table
name must equal the source checkpoint table name. The roles identify the item
table, optional action table, and zero or more contextual tables. The complete
schema and defaults are in [config.py](config.py). Unknown topology fields and
unexpected tensors are rejected.

```bash
export PYTHONPATH=core/builder:.
python -m families.hstu.checkpoint \
  --checkpoint-dir /data/hstu/checkpoint \
  --config /data/hstu/topology.json \
  --source-layout fused \
  --output-dir /data/hstu/converted
```

The converter reads `torch_module/model.0.pth` (`model_state_dict`) and the
checkpoint's static or DynamicEmb tables. It accepts all shards of a DynamicEmb
table and retains arbitrary signed 64-bit keys. All dense tensor-parallel
weights must be consolidated first. Conversion reads distributed embedding
placeholders and TransformerEngine metadata without starting a source process
group, while retaining PyTorch's weights-only loading restriction. Select the
actual UVQK layout:

| Source layout | NVIDIA implementation |
| --- | --- |
| `fused` | Legacy FUSED layer, type-major transposed matrices |
| `native` | NATIVE/DEBUG layer, head-major `torch.nn.Linear` matrices |
| `paged` | Paged inference layer, type-major `torch.nn.Linear` matrices |

The output contains `config.json`, FP32 `model.safetensors`, and a conversion
receipt. BF16 source weights are promoted directly to FP32 without an FP16
intermediate. Runtime sparse-key lookups reject keys missing from the exported
table. There is no random fallback embedding.

## Build a bundle

Run inside the repository's TensorRT development environment:

```python
from pathlib import Path
from tensorrt_model_connect import BuildRequest, build

build(BuildRequest(
    model_dir=Path("/data/hstu/converted"),
    output_path=Path("/data/hstu/model.bundle"),
    family="hstu",
    task="recommendation",
    precision="fp32",
    max_batch_size=8,
    max_sequence_length=512,
))
```

Set `enable_history_cache: true` in the converted `config.json` before building
to expose history-cache and request-local session capabilities. Its default is
`false`. The enabled graph uses TensorRT's native KV-cache update layer and
explicit HSTU attention operations; no ONNX intermediate or PT2 artifact is
required. This switch applies to the supported HSTU family topology. It does
not transform an arbitrary exported PyTorch model into a cached model.

Cache-enabled bundles using the ordinary TensorRT attention path also contain
`prefill.plan`. C++ selects it when every sequence needs complete recomputation;
mixed batches and reusable prefixes use
`engine.plan`. Prefill writes native KV while attention consumes the sanitized
current K/V directly. Both paths preserve the same outputs and history-cache
contract. Older bundles without `prefill.plan` use their existing cached engine.
The additional engine increases bundle size and engine memory.

`scaling_seqlen=-1` uses the actual maximum sequence length in the request batch.
A positive value fixes the attention divisor. The reference's `-1` means the
caller-supplied `JaggedData.max_seqlen`, which can be a static dataset bound.
Set the positive divisor from that original data pipeline when importing such
a checkpoint; the training config's `-1` alone does not identify it. Consequently,
batching requests can change results under `-1`; use the value the checkpoint
was trained and evaluated with. Position-only encoding clamps all candidate
positions to the history boundary. Timestamp encoding uses reversed positions
and the reference's 2048 square-root time buckets.

## Native C++ execution

Build the family, native runner, and TensorRT backend:

```bash
cmake --build build --target trtmc_model_hstu trtmc_hstu trtmc_backend_trt -j8
build/trtmc-hstu \
  --bundle /data/hstu/model.bundle \
  --runtime-root build \
  --input-json /data/hstu/request.json \
  --output-json /data/hstu/result.json
```

Example request (IDs must exist in the exported tables):

```json
{
  "sequences": [{
    "history_item_ids": [12, 42],
    "history_action_ids": [1, 2],
    "candidate_item_ids": [57, 91]
  }]
}
```

Contextual features use `"contextual_features": [{"name": "user", "ids": [7]}]`.
When time encoding is enabled, supply `token_timestamps` in seconds for every
assembled token in the order described above. With retrieval, timestamps cover
history tokens only. A batch can contain different history and candidate counts.
Empty candidate lists produce empty candidate outputs.

C++ applications can avoid JSON and call the public task directly:

```cpp
#include <trtmc/runtime/family_loader.h>
#include <trtmc/task.h>
#include <stdexcept>

auto task = trtmc::load_task("model.bundle", "runtime_directory");
auto* recommender = dynamic_cast<trtmc::IRecommendation*>(task.get());
if (!recommender) throw std::runtime_error("Bundle has no recommendation task");
trtmc::RecommendationRequest request;
trtmc::RecommendationSequence user;
user.history_item_ids = {12, 42};
user.history_action_ids = {1, 2};
user.candidate_item_ids = {57, 91};
request.sequences.push_back(user);
auto result = recommender->recommend(request);
```

Ranking `logits` are row-major `[num_candidates, output_dim]`; `embeddings` are
row-major `[num_candidates, embedding_dim]`. Retrieval returns `scores` and
normalized candidate embeddings. `sequence_embeddings` contains the normalized
HSTU outputs in token order. Keep the task loaded to reuse its engine across
requests.

## Shared history cache

The serving application can attach one platform-owned cache to compatible
native task instances on the same CUDA device. Each request identifies its
subject, feature definition, and history lineage; the bundle contributes its
own artifact identity.

```cpp
#include <trtmc/history_cache.h>
#include <memory>

trtmc::HistoryCacheOptions options;
options.max_bytes = 64ULL * 1024 * 1024;
options.max_entries = 128;
auto cache = std::make_shared<trtmc::HistoryCache>(options);
auto* consumer = dynamic_cast<trtmc::IHistoryCacheConsumer*>(task.get());
if (!consumer) throw std::runtime_error("Rebuild with enable_history_cache");
consumer->set_history_cache(cache);

user.cache.subject_id = "user-42";
user.cache.feature_version = "item-action-v1";
user.cache.history_epoch = "history-v1";
auto result = recommender->recommend({{user}});
```

The canonical key is `(artifact_id, feature_version, subject_id, history_epoch)`.
A rebuild gets a new artifact identity. Keep `history_epoch` stable for an
append-only history lineage; change it for corrections, truncation, window
movement, or other application invalidation. Change `feature_version` when
feature definitions change. The runtime also compares actual encoded history
IDs, positional/time IDs, and attention scaling before using any cached KV.
An unchanged key and matching sequence length alone never prove a hit.

An empty `subject_id` disables shared history reuse for that sequence.
`cache.read_only=true` permits a valid hit but prevents publication. Passing
`nullptr` to `set_history_cache()` disables shared caching for the task.
The result's `cache` report exposes the source, reuse/recomputation reason,
history-token count, reused-token count, computed-token count, and publication
status. Counts refer to encoded tokens: 200 items with actions represent 400
history tokens, plus any context tokens.

Persistent snapshots contain each layer's history K/V and the normalized
history embeddings required by the public output contract. Candidate K/V is
request-local and is never published as reusable user history. Shared readers
hold immutable snapshots. Concurrent publications use generation checks, and
invalidated or replaced snapshots remain alive while readers hold them.
Pinned snapshots continue consuming the cache's byte budget.

### Native CPU storage and external adapters

GPU snapshots are the hot tier. An optional bounded native CPU tier stores
host-only snapshots:

```cpp
options.storage = std::make_shared<trtmc::InMemoryHistoryCacheStorage>(
    512ULL * 1024 * 1024, 1024);
options.write_through = true;
consumer->set_history_cache(std::make_shared<trtmc::HistoryCache>(options));
```

With write-through enabled, publication makes a synchronous device-to-host
snapshot outside the cache-manager lock. This adds transfer and storage cost
to the publishing call. CPU hits restore validated K/V to the GPU before
execution. Without write-through, cache publication does not make this CPU copy.

`IHistoryCacheStorage` defines `load`, `store`, and `erase` for a platform
adapter. The value contains versioned format metadata and named tensors with
shape, dtype, and host bytes. A cache miss or load failure triggers normal model
computation. Storage failures have counters; an erase failure disables further
lower-tier reads in that cache instance to prevent stale reuse. `invalidate(key)`
invalidates both tiers; `clear_memory()` releases resident entries while
preserving the lower tier. Use new key versions for model or feature rollouts.

The repository supplies the CPU implementation, not a FlexKV or RecSys
KVCache Manager adapter. SSD, remote, or FlexKV storage requires a native
adapter and its own deployment, compatibility, and latency validation. The
source interface version is `kHistoryCacheInterfaceVersion = 1`; this is not
a promise of a stable C++ compiler or standard-library ABI. Build the native
client, family library, and runtime together. The in-process cache coordinates
its own instances; distributed invalidation across separate managers is the
external adapter or platform's responsibility.

### When history must be recomputed

Exact-history reuse is available for causal ranking even with contextual
attention. Appended histories can reuse a prefix only when its representations
remain unchanged. The runtime recomputes when any of these conditions applies:

- History/context IDs, ordering, or the context boundary changed.
- Appended tokens affect context tokens that attend the entire history.
- Noncausal attention makes prior tokens depend on the new suffix. Noncausal
  ranking additionally depends on request candidates and does not persist KV.
- Timestamp buckets or reversed positions change prior token embeddings.
- The effective attention divisor changes, including a changed batch maximum
  with `scaling_seqlen=-1`.

These fallbacks preserve the checkpoint's semantics. Enable a fixed divisor
or disable contextual masking only when that matches the trained model;
neither setting should be changed just to obtain cache hits.

## Request-local decoding sessions

Cache-enabled HSTU tasks implement `IRecommendationSessionFactory`. A session
starts from a history prefix, allocates reusable GPU KV buffers once, and keeps
its subsequent appends separate from shared user history. `score()` returns the
same logits/scores and embeddings as the ordinary recommendation API.

```cpp
auto* sessions = dynamic_cast<trtmc::IRecommendationSessionFactory*>(task.get());
if (!sessions) throw std::runtime_error("Bundle has no recommendation sessions");
auto history = user;
history.candidate_item_ids.clear();
auto session = sessions->create_recommendation_session(history, 64ULL * 1024 * 1024);
auto first = session->score({57, 91});

// The application selects an item and supplies the action expected by its model.
trtmc::RecommendationHistoryAppend update;
update.item_ids = {57};
update.action_ids = {1};
session->append(update);
auto next = session->score({18, 29});
auto branch = session->branch();
```

The application owns the selection policy and loop. Every history item needs
an action when the model has an action table. Timestamp-enabled models require
one timestamp per appended encoded token; ranking `score()` also takes candidate
timestamps as its second argument. All tokens must fit the built sequence
capacity. The session budget must accommodate the preallocated KV buffers for
that capacity, rather than only the current history length.

`branch()` gives a branch its own GPU buffers and committed prefix; subsequent
appends do not change its parent or persistent user history. The native LINEAR
layout materializes the committed context into each session or branch once;
this is not a paged shared-context beam executor. Subsequent steps reuse those
buffers. Failed appends leave the session history unchanged. A session retains its native runtime and
can outlive the task object that created it. Calls sharing a task serialize
through its execution context. This supplies native state and branch operations;
token sampling, beam-search policy, continuous batching, and server scheduling
remain application responsibilities.

## Adoption boundaries

### Original NVIDIA attention in a native bundle

The family builder can compile the pinned original NVIDIA/FBGEMM CUDA attention body
into a native TensorRT plugin. Normalization, prediction and KV page updates
remain TensorRT layers. Biased UVQK projections with hidden width 256 use stock
cuBLASLt with BF16 inputs and FP32 accumulation; other projections use TensorRT.
No new GPU math kernel is maintained here.
Python, the CUDA toolkit, and the pinned original HSTU/CUTLASS headers are build
dependencies; Torch supplies compile-time headers, but its runtime is not linked.
The resulting bundle
contains the engine, native library, specialization manifest and the complete
`attention_native.NOTICE` attribution and license material. C++ loads that
library in the model's own TensorRT registry, without a process-wide preload.
Native-attention bundles contain executable code and must come from a trusted
build. Content identities detect changes; they do not authenticate a publisher.

`attention_implementation` in the checkpoint configuration accepts `auto`
(default), `tensorrt`, or `nvidia_hstu`. Automatic selection requires the original
source and a supported configuration. An explicit `nvidia_hstu` selection fails
if its dependencies, source hashes, target or semantics do not match; it never
silently builds another path. The current specialization is BF16, H4/D64,
causal ranking, target group one, fixed scale 1024, capacity at most 1024,
and no contextual or timestamp embeddings. Batch size and candidate count do
not select a different implementation. The same original M64/N128 attention
specialization, TensorRT graph, and cache layout are used across build targets.
GPU architecture determines compilation and artifact compatibility only.
Other configurations use the ordinary
TensorRT graph with their existing semantics.

At build time, set `native_kernel_source` in the checkpoint configuration to the
pinned FBGEMM checkout, or its `hstu_ampere` source directory, and initialize its
pinned `external/cutlass` dependency. Both identities are recorded in
[the source manifest](native_attention_source.json). Relative source paths are resolved
against the checkpoint directory; this build-only path is omitted from the
serving configuration. Packaged compiler provenance records content identities;
absolute dependency paths remain in local build receipts. Build on the serving
target GPU. Native artifacts record their compute capability and
the C++ loader rejects a mismatched target. Cross-target compilation alone does
not establish numerical or performance qualification on that GPU. The C++ runtime,
family DSOs and backend must be rebuilt together for the new native-library
loading interface.

The paged path accepts compact query rows internally and preserves the public
recommendation/session API. Only history K/V enters owned pages. Candidate
K/V stays in the current projection. Cache restoration initializes all private
tails, and ownership/generation checks protect append, invalidation and failure.
The CUDA provider requires zeroed unused entries in referenced partial pages;
native page updates clear dirty tails before attention. Unreferenced pages are
not read. Dense attention has no KV pages or page writes.
Sessions and branches still own independent GPU storage; this does not add
shared beam scheduling or a server.

Correctness validation covers recommendation outputs, cache restoration and
invalidation, session appends, and independent branches. Production qualification
still requires the deployment's checkpoint, workload, GPU, and runtime versions.
Attention microbenchmarks alone do not establish end-to-end latency or qualify
another deployment target.

| Requirement | Current boundary |
| --- | --- |
| Pure native HSTU ranking/retrieval | Model Connect + TensorRT + C++, with resident embedding tables |
| Persistent history and request-local decode KV | Shared cache service plus separate native sessions |
| SDPA models | Native SDPA/KV graph primitives exist; importing another model still requires its family-owned graph and cache semantics |
| Ordinary PyTorch export or PT2 input | No automatic graph rewrite or PT2 execution path |
| RecSys KVCache Manager / FlexKV | No linked integration; the native storage adapter boundary is available |
| Triton or routing infrastructure | Outside this implementation |
| Train and serve on different GPUs | Convert weights and build/qualify TensorRT engines for the serving GPU |
| End-to-end latency across autoregressive steps | Requires a representative checkpoint, workload, and measurements on the deployment hardware |

For a serving qualification, measure complete request latency and all decode
steps separately for GPU history hits, CPU hits, misses, and appends. Include
candidate count, model dimensions, batch size, transfers, cache publication, and
the required output tensors in the timing boundary. Server queueing and
transport need separate measurements when a server is added.

## Validation

The family-owned E2E suite creates deterministic, non-pretrained checkpoints
and compares real TensorRT/C++ execution with the pinned original NVIDIA
attention and layer methods. It exercises ranking, retrieval, both contextual
mask settings, candidate groups, noncausal attention, timestamp encoding,
learned and unlearned normalization, and all three precision choices.
Cache cases additionally cover mixed hits/misses, invalidation, CPU-tier
restoration, and prefix correctness. Session cases compare twenty append/score
steps and independent branches with complete recomputation and the pinned
reference, including 200 history items and 256 candidates.

Configure the native build with `-DTRTMC_BUILD_TESTS=ON` and build
`trtmc_model_hstu` to include the cache and session test drivers.

```bash
git clone https://github.com/NVIDIA/recsys-examples /data/recsys-examples
git -C /data/recsys-examples checkout 97062d97eef53115105063801e35184e36186df5
export TRTMC_HSTU_REFERENCE_ROOT=/data/recsys-examples
export TRTMC_HSTU_BINARY="$PWD/build/trtmc-hstu"
export TRTMC_RUNTIME_ROOT="$PWD/build"
export TRTMC_NATIVE_BUILD_DIR="$PWD/build"
PYTHONPATH=core/builder:. python -m pytest families/hstu/tests --e2e-model hstu -q
```

The explicit source checkout is optional: without `TRTMC_HSTU_REFERENCE_ROOT`,
the tests fetch the exact revision into a local cache and verify it on reuse.
Use the explicit checkout for offline validation. The family build target also
builds `trtmc-hstu`. Wheel packaging retains the executable in the package's
native runtime directory; it can also be invoked directly from the build tree
as shown above. Neither source preparation nor the reference oracle is part of
deployed inference.

Seeded reference parity validates implementation semantics. It does not measure
recommendation quality on MovieLens or KuaiRand, qualify a user-trained
checkpoint, or establish production-scale throughput. Those require the actual
checkpoint and workload.

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
explicit dtype boundaries. The implementation runs full sequences; it does not
use upstream PyTorch, TorchRec, DynamicEmb, NVE, AOTInductor, Triton kernels,
FlexKV, or paged KV-cache services at inference time. It does not implement
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
  "disable_contextual_mask": true
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

## Validation

The family-owned E2E suite creates deterministic, non-pretrained checkpoints
and compares real TensorRT/C++ execution with the pinned original NVIDIA
attention and layer methods. It exercises ranking, retrieval, both contextual
mask settings, candidate groups, noncausal attention, timestamp encoding,
learned and unlearned normalization, and all three precision choices.

```bash
git clone https://github.com/NVIDIA/recsys-examples /data/recsys-examples
git -C /data/recsys-examples checkout 97062d97eef53115105063801e35184e36186df5
export TRTMC_HSTU_REFERENCE_ROOT=/data/recsys-examples
export TRTMC_HSTU_BINARY="$PWD/build/trtmc-hstu"
export TRTMC_RUNTIME_ROOT="$PWD/build"
PYTHONPATH=core/builder:. python -m pytest families/hstu/tests --e2e-model hstu -q
```

The explicit source checkout is optional: without `TRTMC_HSTU_REFERENCE_ROOT`,
the tests fetch the exact revision into a local cache and verify it on reuse.
Use the explicit checkout for offline validation. The family build target also
builds `trtmc-hstu`, and wheel packaging installs the native command alongside
the generic `trtmc` command. Neither source preparation nor the reference oracle
is part of deployed inference.

Seeded reference parity validates implementation semantics. It does not measure
recommendation quality on MovieLens or KuaiRand, qualify a user-trained
checkpoint, or establish production-scale throughput. Those require the actual
checkpoint and workload.

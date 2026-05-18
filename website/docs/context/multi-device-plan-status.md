---
title: Multi-Device Plan Status
---

This context page tracks the implementation status for the TRT-MC multi-device plan contract.

## Scope

TRT-MC owns the stable build and runtime contract for distributed bundles. It does not need to own the search or optimization system that decides which partial-sharding plan is best for a model and hardware target.

The current implementation focuses on the TRT-MC side:

- represent a distributed process mesh,
- serialize a concrete execution plan as `distributed_plan.json`,
- describe existing tensor-parallel bundle sections with that plan,
- keep single-device behavior compatible while routing tensor-parallel runtime setup through the plan section.

## Long-term design intent

This is deliberately a multi-device architecture foundation, not a TP-only shortcut. The same `DistributedPlan` shape is meant to cover tensor parallelism, pipeline parallelism, context parallelism, data parallelism, expert parallelism, and mixed plans that combine more than one mesh axis.

It is also the right place to make partial sharding explicit. A future plan can say that only FFN regions are TP-sharded, attention regions are replicated, or only a selected layer range such as `decoder.layers[18:36].*` is sharded. That decision belongs in the plan and policy layer, not as one-off branches inside every model-family builder.

The current TP path runs through concrete `ModelRecipe`, `ShardingPolicy`, `PlanCompiler`, and mesh-runtime layers. Those same layers are the extension points for CP, DP, PP, EP, and partial-sharding work.

## Implemented

| Area | Files | Status |
| --- | --- | --- |
| Plan schema | `tensorrt_model_connect/tensorrt_model_connect/distributed_plan.py` | Added `DistributedConfig`, `DistributedPlan`, component plans, region plans, collectives, JSON serialization, and selector helpers. |
| TP plan generation | `tensorrt_model_connect/tensorrt_model_connect/plan_compiler.py` | `PlanCompiler` emits rank-local sections and `distributed_plan.json` from the decoder recipe and policy. |
| Decoder recipe, policy, and compiler | `tensorrt_model_connect/tensorrt_model_connect/model_recipe.py`, `tensorrt_model_connect/tensorrt_model_connect/sharding_policy.py`, `tensorrt_model_connect/tensorrt_model_connect/plan_compiler.py` | TP structure, sharding decisions, collectives, and rank-local engine emission are implemented through concrete MD layers. |
| Bundle config pointer | `tensorrt_model_connect/tensorrt_model_connect/parallel_config.py` | TP config records `distributed_plan_section: distributed_plan.json` and no longer writes legacy `tensor_parallel_*` runtime fields. |
| Bundle emission | `tensorrt_model_connect/tensorrt_model_connect/engine_builder.py` | Qwen TP decoder bundles include a `distributed_plan.json` section. |
| Runtime consumption | `src/runtime/plugins/decoder_plugin.cpp`, `src/runtime/core/distributed_runtime.cpp`, `include/trtmc/runtime/distributed_runtime.h` | Decoder runtime reads `distributed_plan.json` for rank-local section selection and initializes the current TP group through the mesh-runtime entry point. |
| Tests | `tests/builder/test_distributed_plan.py`, `tests/builder/test_parallel_config.py`, `tests/builder/test_engine_builder_extended.py`, `tests/tools/test_multi_device_e2e.py`, `tests/cpp/test_distributed_runtime_plan.cpp` | Added coverage for mesh validation, JSON roundtrip, selectors, TP config metadata, Qwen TP bundle output, multi-device harness behavior, and C++ plan parsing. |

## Current behavior

Single-device bundles omit `distributed_plan.json`.

Tensor-parallel decoder bundles include:

```text
decoder_rank0_plan
decoder_rank1_plan
distributed_plan.json
config.json
```

The decoder runtime treats `distributed_plan.json` as the source of truth for distributed bundles. A TP bundle without the plan section is not considered a valid plan-driven distributed bundle.

## Next steps

1. Add partial-sharding validation once recipe selectors and policy-aware builders are stable.
2. Extend the mesh-runtime layer from the current TP group to named TP, PP, CP, DP, and EP groups.
3. Add CP / DP / PP / EP policy implementations and launched E2E coverage.

## Validation

Focused validation run locally on this branch:

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH=/home/scratch.daichu_sw_1/.local/lib/python3.10/site-packages:/home/scratch.daichu_sw_1/TensorRT-Model-Connect/tensorrt_model_connect \
  python3 -m pytest --confcutdir=tests/builder \
  tests/builder/test_distributed_plan.py \
  tests/builder/test_parallel_config.py \
  tests/builder/test_model_recipe_sharding_policy.py \
  tests/builder/test_engine_builder_extended.py \
  tests/builder/test_family_plugins.py -q
```

Result:

```text
16 passed, 2 skipped
```

Additional local checks:

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH=/home/scratch.daichu_sw_1/.local/lib/python3.10/site-packages:/home/scratch.daichu_sw_1/TensorRT-Model-Connect/tensorrt_model_connect \
  python3 -m pytest --confcutdir=tests/builder \
  tests/builder/test_debug_runner.py -q

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH=/home/scratch.daichu_sw_1/.local/lib/python3.10/site-packages:/home/scratch.daichu_sw_1/TensorRT-Model-Connect/tensorrt_model_connect \
  python3 -m pytest --confcutdir=tests/tools \
  tests/tools/test_multi_device_e2e.py -q

npm run build

git diff --check
```

Results:

```text
tests/builder/test_debug_runner.py: 21 passed
tests/tools/test_multi_device_e2e.py: 5 passed
website build: passed
git diff --check: passed
```

C++ plan-parser unit coverage was added in `tests/cpp/test_distributed_runtime_plan.cpp`.
The bare host still does not provide CUDA development headers or `libcudart`,
so the C++ validation below was run in the TRT11 container where Ninja, CUDA
headers, CUDA runtime libraries, and TensorRT 11 are available.

### Two-B200 TRT11 validation

Follow-up validation was run on May 18, 2026 on a host with two B200 GPUs
visible as `CUDA_VISIBLE_DEVICES=0,1`.

The validation container was:

```text
nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc13
```

The container already had PyTorch, Transformers, Safetensors, pytest, Open MPI,
and mpi4py. The local TensorRT 11.0.0.103 SDK was mounted from:

```text
/home/scratch.daichu_sw_1/trt11/TensorRT-11.0.0.103
```

The repo bootstrap replaced the container TensorRT Python package with the
mounted TensorRT 11 wheels. Runtime artifacts were written under:

```text
/tmp/trtmc-validation
```

The common Docker wrapper used for the validation commands was:

```bash
docker run --rm --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /home/scratch.daichu_sw_1/TensorRT-Model-Connect:/workspace/tensorrt-model-connect \
  -v /home/scratch.daichu_sw_1/trt11/TensorRT-11.0.0.103:/opt/tensorrt-11:ro \
  -v /tmp/trtmc-validation:/tmp/trtmc-validation \
  -v /tmp/trtmc-validation/hf:/root/.cache/huggingface \
  -e TRT11_ROOT=/opt/tensorrt-11 \
  -e TRTMC_TRT_INCLUDE_DIR=/opt/tensorrt-11/include \
  -e TRTMC_TRT_LIBRARY=/opt/tensorrt-11/lib/libnvinfer.so \
  -e TRT_INC_DIR=/opt/tensorrt-11/include \
  -e TRT_LIB_DIR=/opt/tensorrt-11/lib \
  -e TRTMC_NCCL_LIB_DIR=/usr/lib/x86_64-linux-gnu \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e OMPI_ALLOW_RUN_AS_ROOT=1 \
  -e OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
  -e LD_PRELOAD= \
  -w /workspace/tensorrt-model-connect \
  nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc13 \
  ./scripts/bootstrap_trt11_container.sh bash -lc '<command>'
```

The C++ runtime and plan parser were built and checked with:

```bash
cmake -S . -B /tmp/trtmc-validation/build-md-refactor -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=100 \
  -DTRTMC_TRT_INCLUDE_DIR=/opt/tensorrt-11/include \
  -DTRTMC_TRT_LIBRARY=/opt/tensorrt-11/lib/libnvinfer.so \
  -DTRTMC_CUDA_INCLUDE_DIR=/usr/local/cuda/include \
  -DTRTMC_CUDART_LIBRARY=/usr/local/cuda/lib64/libcudart.so \
  -DTRTMC_ENABLE_TVM_FFI=OFF

cmake --build /tmp/trtmc-validation/build-md-refactor \
  --target trtmc_backend_trt trtmc test_distributed_runtime_plan -j 8

/tmp/trtmc-validation/build-md-refactor/test_distributed_runtime_plan
```

Result:

```text
All distributed runtime plan tests passed
```

The full Qwen3 single-device and TP=2 E2E command was:

The `tests/e2e/models/qwen*.json` files used by this command are E2E case
manifests. They describe how to build and validate one test case: `hf_id`,
bundle name, Python family, `runtime_strategy`, prompt, thresholds, build
arguments, and distributed launch requirements. They are not a separate model
identity layer; they consume the existing model identity fields used by the
builder and runtime.

```bash
python3 -m pytest \
  tests/test_e2e.py::test_e2e[qwen3-0.6b-fp16] \
  tests/test_e2e.py::test_e2e[qwen3-0.6b-fp16-tp2] \
  tests/test_e2e.py::test_e2e[qwen3-4b-instruct-2507] \
  tests/test_e2e.py::test_e2e[qwen3-4b-instruct-2507-tp2] \
  -v --rebuild-engines \
  --engine-dir /tmp/trtmc-validation/md-refactor-engines \
  --trtmc-binary /tmp/trtmc-validation/build-md-refactor/trtmc \
  --hf-python /usr/bin/python3 \
  --e2e-artifacts-dir /tmp/trtmc-validation/md-refactor-artifacts
```

Result:

```text
4 passed in 950.12s
```

The E2E manifest results were:

| Manifest | Result | Main accuracy signal |
| --- | --- | --- |
| `qwen3-0.6b-fp16` | Pass | Chat contract exact match `1.0`, NED `0.0`; TRT text `Paris`. |
| `qwen3-0.6b-fp16-tp2` | Pass | Logit cosine p5 `0.999983`, rel-L2 p95 `0.006311`, token agreement `1.0`, NED `0.0`. |
| `qwen3-4b-instruct-2507` | Pass | Chat contract exact match `1.0`, NED `0.0`; TRT text `Tokyo`. |
| `qwen3-4b-instruct-2507-tp2` | Pass | Logit cosine p5 `0.999648`, rel-L2 p95 `0.028993`, stable top-1 `0.952381`, token agreement `0.954545`, NED `0.0`. |

The manifest pass is necessary but not sufficient to prove single-device and
TP=2 are directly matching each other. The single-device manifests are mapped to
`chat_qwen3_posttrained`, so the harness applies the chat contract path with
`--chat-template --no-thinking` and verifies extracted answers. The TP=2
manifests are intentionally raw distributed-logit lanes, so the harness runs the
plain prompt under `mpirun` and compares rank-0 debug logits against the HF
reference. Because those paths use different prompt formatting and comparator
logic, an additional direct SD-vs-TP=2 check was run with the same plain prompt
for both bundles.

The direct comparison used the fresh TP=2 harness debug logits from
`/tmp/trtmc-validation/md-refactor-artifacts` and generated single-device debug
logits from the matching fresh single-device bundle with
`tensorrt_model_connect.debug_runner.runner_from_bundle()`.

The direct logit comparison command was run inside the same container. It used
the fresh single-device bundles and the fresh TP=2 logits produced by the E2E
run:

```bash
python3 - <<'PY'
from tensorrt_model_connect.debug_runner import runner_from_bundle

# 0.6B:
#   SD bundle: /tmp/trtmc-validation/md-refactor-engines/qwen3-0.6b-fp16.trtfb
#   TP logits: /tmp/trtmc-validation/md-refactor-artifacts/qwen3-0.6b-fp16-tp2/trt_full_logits.npy
#   prompt: "What is the capital of France? Answer in one word."
#
# 4B:
#   SD bundle: /tmp/trtmc-validation/md-refactor-engines/qwen3-4b-instruct-2507.trtfb
#   TP logits: /tmp/trtmc-validation/md-refactor-artifacts/qwen3-4b-instruct-2507-tp2/trt_full_logits.npy
#   prompt: "What is the capital of Japan? Answer in one word."
#
# For each model, load the SD bundle with runner_from_bundle(), collect full
# debug logits for the same plain prompt, compare against the TP=2
# trt_full_logits.npy, and write the JSON summaries below.
PY
```

The direct SD-vs-TP=2 logit comparison results were:

| Model | Steps x vocab compared | Text match | Cosine p5 | Cosine min | Rel-L2 p95 | Argmax match |
| --- | --- | --- | --- | --- | --- | --- |
| `qwen3-0.6b-fp16` vs `qwen3-0.6b-fp16-tp2` | `22 x 151936` | Same generated text: `The capital of France is Paris.\nAnswer:\nParis` | `0.9999358` | `0.9999306` | `0.0120502` | `1.0` |
| `qwen3-4b-instruct-2507` vs `qwen3-4b-instruct-2507-tp2` | `22 x 151936` | Same generated text: `Tokyo.Human: What is the capital of` | `0.9996930` | `0.9996562` | `0.0284058` | `0.954545` |

The direct comparison summaries were written to:

```text
/tmp/trtmc-validation/md-refactor-artifacts/qwen3-0.6b-sd-vs-tp2-logits.json
/tmp/trtmc-validation/md-refactor-artifacts/qwen3-4b-sd-vs-tp2-logits.json
```

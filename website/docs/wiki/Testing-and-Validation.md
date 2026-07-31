# Testing and Validation

The maintained command reference is
[Testing Reference](../reference/testing.md). This page explains the evidence
model and current E2E contract.

## Test layers

| Layer | Authority | What it proves |
| --- | --- | --- |
| Python builder | `tests/builder/` | Config, mapping, graph/build, bundle, and CLI units |
| C++ runtime | `tests/cpp/` and model-declared runtime tests | Core and family runtime behavior |
| Optimized implementation | Family adapter contract tests and exact profile data | Exact implementation/profile selection and embedded-DSO loading; target-hardware parity and performance require separate evidence |
| Repository tools | `tests/tools/` | CI selection, isolation, comparison, packaging, reports |
| E2E harness | `tests/e2e_harness/` | Manifest loading, orchestration, runner/comparator contracts |
| Model E2E | `tests/e2e/models/<family>/` | Exact checkpoint/task integration and artifacts |
| Nightly | Private Internal CI | Broad scheduled model and packaging evidence; Source receives no raw nightly logs or artifacts |

Counts change as models land. At this revision, all three ownership trees have
78 descriptors and the E2E tree declares 204 JSON manifests. Derive future
counts from descriptors; do not copy this snapshot into acceptance logic.

## Native manifest shape

The following is a valid structural example for the native path; values come
from the owning family:

```json
{
  "name": "distilgpt2",
  "hf_id": "distilbert/distilgpt2",
  "bundle": "distilgpt2.trtfb",
  "family": "gpt2",
  "runtime_strategy": "gpt2_decoder_kv_cache",
  "task_strategy": "text_generation_causal",
  "precision": "fp16",
  "max_cache_length": 256,
  "testcases": [
    {
      "name": "distilgpt2",
      "trace_id": "IT-E2E-GPT2-02",
      "reference_family": "causal_base_continuation",
      "user_contract": "model_card_generation_parity",
      "prompt": "Hello, I'm a language model",
      "max_new_tokens": 12
    }
  ]
}
```

The family `MODEL.toml` must list this file. A buildable manifest without a
non-empty `testcases` array is rejected. Task defaults may be inherited from
the model descriptor. On this path, `runtime_strategy` must exactly match the
family-owned model-plugin strategy and the run must load both its model DSO and
selected backend DSO.

## Optimized implementation shape

Optimized-runtime evidence is not a second native strategy. It consists of:

1. The family adapter's `IMPLEMENTATION.toml`, which declares the delegated
   runtime identity and private `libtrtmc_impl_*.so`.
2. An exact qualified `profiles/*.toml` entry binding the model revision,
   target, build options, and semantic hash.
3. Retained parity/performance results from a matching target-hardware
   producer. Source currently publishes no A100 producer or runner.
4. A bundle that is self-contained for the implementation DSO and
   provider-produced artifacts: `optimized_runtime.json`,
   `implementation.json`, the integrity-bound artifact tree, and the exact
   embedded implementation DSO. The host still supplies a matching NVIDIA
   driver (`libcuda.so.1`), versioned CUDA runtime
   (`libcudart.so.<major>`), TensorRT (`libnvinfer.so.<major>`), dynamic
   loader, and compatible system libraries.

The optimized descriptor claims the bundle before native strategy dispatch.
Its `runtime_strategy` may therefore be empty; the implementation and profile
IDs are the relevant selection identity. The host does not use the native
model-plugin index or select a `libtrtmc_backend_*.so` for this path.

## Commands

```bash
PYTHONPATH=python:. python3 -m pytest tests/builder -q
PYTHONPATH=python:. python3 -m pytest tests/tools -q
ctest --test-dir build --output-on-failure

PYTHONPATH=python:. python3 -m pytest \
  tests/e2e/models/<family> \
  --e2e-model <manifest-name> \
  --engine-dir /path/to/engines \
  --trtmc-binary ./build/trtmc \
  --model-plugin-dir ./build/models \
  -v
```

The final command requires literal family/manifest values and the declared
checkpoint, GPU, TensorRT environment, and binary. Native cases additionally
require the owning model and backend DSOs. Optimized cases require the
qualified implementation/profile evidence and a bundle carrying the exact
embedded implementation DSO.

## Evidence rules

- Compilation is not inference proof.
- A close metric is not exact parity when the acceptance contract requires
  exact output.
- A successful model run does not prove broad regression coverage.
- CI summaries are not sufficient if the tested SHA or artifact provenance is
  unknown.
- Thresholds must represent the intended user contract and must not be loosened
  merely to pass.
- Performance claims require the exact hardware, software revision, inputs,
  warmups, repetitions, and retained result artifacts.
- `TextResult` exposes setup, prefill, and decode timing fields, but the public
  header at this revision does not define zero as a universal availability
  sentinel. Verify which fields the selected pipeline populates before using
  them, and use synchronized public-call wall time for cross-provider latency
  or throughput comparisons.

# Testing and Validation

The maintained command reference is
[Testing Reference](../reference/testing.md). This page explains the evidence
model and current E2E contract.

## Test layers

| Layer | Authority | What it proves |
| --- | --- | --- |
| Python builder | `tests/builder/` | Config, mapping, graph/build, bundle, and CLI units |
| C++ runtime | `tests/cpp/` and model-declared runtime tests | Core and family runtime behavior |
| Repository tools | `tests/tools/` | CI selection, isolation, comparison, packaging, reports |
| E2E harness | `tests/e2e_harness/` | Manifest loading, orchestration, runner/comparator contracts |
| Model E2E | `tests/e2e/models/<family>/` | Exact checkpoint/task integration and artifacts |
| Nightly | `.github/workflows/nightly.yml` | Broad scheduled model and packaging evidence |

Counts change as models land. At this revision, all three ownership trees have
78 descriptors and the E2E tree declares 203 JSON manifests. Derive future
counts from descriptors; do not copy this snapshot into acceptance logic.

## Manifest shape

The following is a valid structural example; values come from the owning
family:

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
the model descriptor.

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
checkpoint, GPU, TensorRT environment, binary, and model DSO.

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

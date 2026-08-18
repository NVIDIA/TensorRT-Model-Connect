---
title: Validate a Model Contribution
---

Use this workflow after following
[Add a Model Family](add-model-family.md). It separates repository consistency,
focused tests, real inference, and qualification evidence so that a passing
lower-level check is not mistaken for model proof.

## Identify the ownership unit

A native model contribution connects three model-owned roots:

```text
python/tensorrt_model_connect/families/<builder-family>/MODEL.toml
src/runtime/models/<runtime-owner>/MODEL.toml
tests/e2e/models/<e2e-family>/MODEL.toml
```

Each descriptor `id` must match its own directory. The three physical names
normally match, but the Python family `model.py` and E2E manifest select the
runtime owner through the exact family-owned `runtime_strategy`. Do not
substitute a generic task name such as `text_generation_causal`;
`task_strategy` selects the reusable runner/comparator contract, while
`runtime_strategy` selects a concrete native model DSO.

Before testing, record:

- builder family, runtime owner, and E2E family;
- Hugging Face model ID and immutable revision;
- native runtime strategy and task strategy;
- literal manifest name and testcase;
- precision, quantization, tensor-parallel, and shape settings; and
- required checkpoint, runtime libraries, device count, and GPU capacity.

An exact delegated optimized-runtime implementation has an additional
family-owned `IMPLEMENTATION.toml`, exact profile, semantic-source digest,
embedded implementation DSO, and Source-side adapter/runtime-contract tests.
Its implementation/profile identity replaces native strategy dispatch for
that bundle. Target-hardware qualification is a separate external evidence
layer; the public Source tree does not publish the former qualification
descriptor or runner.

## 1. Validate repository ownership

Run the descriptor and impact-map checks:

```bash
PYTHONPATH=python:. python3 tools/model_ci.py validate
PYTHONPATH=python:. python3 tools/test_impact.py --validate
```

For a branch based on the repository's `github/main` remote, inspect the exact
model impact:

```bash
git fetch github main
PYTHONPATH=python:. python3 tools/model_ci.py impact \
  --base github/main \
  --head HEAD
PYTHONPATH=python:. python3 tools/test_impact.py \
  --base github/main \
  --head HEAD
```

If the canonical GitHub repository is named `origin` in your clone, use
`origin/main` consistently instead.

The runtime strategy matrix is a useful diagnostic:

```bash
PYTHONPATH=python:. python3 tools/check_runtime_strategy_matrix.py
```

At GitHub `main` commit
`e6b798cdb145c38caf1ede8eda7f5ce83f894138`, this diagnostic has known
repository-wide gaps for `diffusion_sana_wm` and five speech/omni runner
entries. Do not claim the command is green on that snapshot. A model change
must not add a new gap; report the pre-existing baseline separately from any
new output.

## 2. Run focused contract tests

These tests cover descriptor shape, runtime-strategy consistency behavior, and
model ownership:

```bash
PYTHONPATH=python:. python3 -m pytest \
  tests/builder/test_manifest_validation.py \
  tests/tools/test_runtime_strategy_matrix_checker.py \
  tests/tools/test_model_plugin_encapsulation_static.py -q
```

Then run the owning family's builder, C++, tool, and model-local tests affected
by the change. Prefer the exact tests selected by the impact report. Passing a
shared static test is not a substitute for testing the model-owned code.

## 3. Build and inspect one bundle

Build a representative checkpoint with the intended user options, then inspect
the result. This concrete example matches the current Qwen L0 manifest; adapt
the literal model, revision, and bundle name to the contribution under review:

```bash
trtmc build Qwen/Qwen3-0.6B \
  -o /tmp/qwen3-0.6b-native-l0.bundle
trtmc inspect /tmp/qwen3-0.6b-native-l0.bundle
```

For a native bundle, verify the exact `runtime_strategy`, precision, engine
sections, and required model/backend DSOs. For an optimized bundle, verify the
presence of `optimized_runtime.json`, implementation metadata, the
integrity-bound artifact tree, and the embedded implementation DSO.

For a qualified contribution, pin and record an immutable model revision even
when an older smoke manifest does not yet carry one. A successful compile or
inspection proves artifact construction, not inference parity.

## 4. Run the declared E2E case

Run the literal family and manifest declared by the E2E descriptor:

```bash
PYTHONPATH=python:. python3 -m pytest \
  tests/e2e/models/qwen \
  --e2e-model qwen3-0.6b-native-l0 \
  --engine-dir /path/to/engines \
  --trtmc-binary ./build/trtmc \
  --model-plugin-dir ./build/models \
  -v
```

Add `--hf-python /path/to/python` only when the selected runtime requires a
Python helper. This step needs the declared checkpoint, TensorRT/CUDA, suitable
GPU hardware, the compiled CLI, and all runtime libraries required by the
bundle path.

Confirm that:

1. the manifest contains a non-empty `testcases` array;
2. the testcase names its user contract, CI tier, request, oracle, and
   thresholds;
3. the runtime loads the intended implementation rather than a fallback;
4. comparison artifacts identify the exact model revision and bundle; and
5. failures remain failures rather than being hidden by a relaxed threshold.

## 5. Record evidence by level

Keep these evidence levels separate:

| Level | What it establishes |
| --- | --- |
| Implemented | The source and descriptors exist. |
| Repository-consistent | Ownership, manifest, and impact checks accept the tree. |
| Unit-tested | Focused builder, C++, or tool behavior passes. |
| Inference-tested | The exact bundle runs the declared user task on compatible hardware. |
| Parity-qualified | Retained comparison artifacts satisfy the intended reference contract. |
| Performance-qualified | Exact-hardware measurements retain inputs, warmups, repetitions, baseline, and raw results. |

A completion report should state the exact tested code revision, model
revision, commands, hardware, bundle, comparison artifacts, performance
artifacts when claimed, known baseline failures, and unverified paths.

For branch, pull-request, and one-shot `run-internal-ci` handling, follow
[Contributing](contributing.md). CI success does not widen the evidence boundary
beyond the jobs and models that actually ran.

{/* Collaborative review anchor: batch 2. */}

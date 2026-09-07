---
title: Validate a Model Contribution
---

Use this workflow after [Add a Model Family](add-model-family.md), or after
changing an existing family's build, runtime, dependency, or oracle.

## Identify the ownership unit

A family contribution is one vertical slice:

```text
families/<family>/
  support.py
  model.py
  requirements.txt              # optional
  runtime/CMakeLists.txt
  runtime/*.cpp
  tests/test_*.py
  tests/manifests/*.json
  tests/thresholds/*.json       # optional numeric overrides
```

Record the family, exact model ID and immutable revision, task, testcase,
precision, quantization, tensor/context parallel sizes, shape bounds, required
assets/dependencies, native targets, GPU count, and oracle before testing.

## 1. Validate ownership and impact

```bash
python3 -m tools.model_ci validate
python3 tools/test_impact.py --validate
python3 tools/test_impact.py --base github/main --head HEAD
```

The impact report should select the owning family. Production source must not
depend on a sibling family, and adding a normal family must not modify a
central registry or source list.

## 2. Run source and unit contracts

```bash
python3 -m pytest core/builder/tests tools/tests -q
python3 -m pytest families/<family>/tests -m 'not gpu and not trt and not e2e' -q
```

Build and run the exact family-owned native targets when present:

```bash
cmake -S . -B build -DTRTMC_BUILD_TESTS=ON
cmake --build build --target trtmc_model_<family>
ctest --test-dir build --output-on-failure
```

Passing shared architecture tests does not replace family-specific proof.

## 3. Build and inspect a representative bundle

```bash
python -m tensorrt_model_connect build Qwen/Qwen3-0.6B \
  --precision fp16 \
  --output /tmp/qwen3-0.6b.bundle
trtmc inspect /tmp/qwen3-0.6b.bundle
```

Verify the exact `family`, `task`, `backend`, and required family-owned
sections. Inspection proves container construction, not inference parity.

## 4. Run the declared family E2E

Family `test_e2e.py` files accept an explicit selection and require the native
binary/runtime root through the environment:

```bash
TRTMC_BINARY="$PWD/build/apps/cli/trtmc" \
TRTMC_RUNTIME_ROOT="$PWD/build/install/lib" \
python3 -m pytest families/qwen/tests/test_e2e.py \
  --e2e-testcase qwen3-0.6b-fp16 \
  -q -x
```

The selected test downloads or opens its declared checkpoint, builds through
the public Python API, loads exactly the owning family DSO, invokes the public
Task API, and applies its own oracle. Tensor-parallel cases additionally need
the declared GPU count and `mpirun`.

Confirm that the manifest has a meaningful task/testcase, exact inputs,
premerge selection where intended, and thresholds that reject adversarial or
known-wrong outputs. Never weaken a criterion to pass CI.

## 5. Report evidence by level

| Level | What it establishes |
| --- | --- |
| Repository-consistent | Ownership and impact validators accept the tree. |
| Unit-tested | Focused builder, runtime, tool, and family contracts pass. |
| Inference-tested | The exact bundle runs its declared Task on compatible hardware. |
| Parity-qualified | Retained comparison artifacts pass the intended official-reference contract. |
| Performance-qualified | Exact-hardware results retain inputs, warmups, repetitions, baseline, and raw measurements. |

State exactly which revision, checkpoint, hardware, commands, and cases ran,
plus unverified paths. For the full CI path and one-shot protected-CI label,
follow [Contributing](contributing.md).

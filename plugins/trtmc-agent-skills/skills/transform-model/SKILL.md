---
name: transform-model
description: >-
  Use when onboarding a Hugging Face model into TensorRT-Model-Connect or
  extending an existing family to produce a `.bundle` bundle. Drives
  ownership-first implementation across Python builder, native runtime, and
  model-owned E2E descriptors, then requires reference-consistency and runtime
  evidence before support is claimed.
---

# Transform Model

## Define The Support Claim

Record:

- exact Hugging Face model ID and immutable revision;
- task/modality and requested public operation;
- target hardware and precision/quantization;
- closest existing family and architectural differences;
- whether the expected bundle path is native or an exact optimized profile;
- requested evidence level: build, parity, E2E, performance, or qualification.

Do not begin from a generic manifest. Read the model config, reference
implementation, nearest family descriptors, and owned tests first.

## Ownership Map

A fully registered native model normally crosses three descriptors:

| Layer | Owner |
|---|---|
| Python build/family selection | `python/tensorrt_model_connect/families/<family>/MODEL.toml` |
| Native C++ strategy/plugin | `src/runtime/models/<family>/MODEL.toml` |
| E2E models, manifests, sidecars | `tests/e2e/models/<family>/MODEL.toml` |

The owning directories also contain family-local builders, graph helpers,
runtime sources, manifests, testcases, thresholds, and performance contracts.
Keep changes there unless multiple families demonstrably share the contract.

An optimized-runtime path instead requires an exact implementation, profile,
and qualification chain. Do not create a native strategy merely to mirror an
optimized implementation, and do not silently fall back after an optimized
profile has claimed the request.

## Choose Reuse Or A New Family

Extend an existing family when model type, checkpoint mapping, graph dataflow,
runtime strategy, and validation contract fit that owner. Create a new family
when those contracts materially differ.

For a standard decoder starting point:

```bash
python3 scripts/new_family.py \
  --model-type <model-type> \
  --hf-repo <org/model> \
  --family-name <family>
```

The scaffold is only a starting point. It fetches model configuration and emits
a decoder-oriented Python family. Review every generated match rule, import,
weight mapping, graph path, and descriptor; do not use it for non-decoder
architectures without redesigning the generated code.

## Implement The Smallest Owned Change

### Python Builder

- Parse model configuration without inventing defaults that change semantics.
- Map checkpoint tensors explicitly, including tied weights, fused/split
  projections, transposes, and expert layouts.
- Preserve strongly typed TensorRT network creation.
- Keep graph operations in the family directory; root graph helpers are
  intentionally absent.
- For GQA/MQA, keep K/V projections, bias, and cache at compact
  `num_key_value_heads * head_dim` unless a proven operation needs another
  layout.
- Use `$fp16-trt-network` for FP16/BF16 dtype and FP32-boundary work.
- Use the shared quantization plan and family hooks; do not build a parallel
  quantization path.
- Select TensorRT versus TensorRT-RTX before TensorRT imports and keep backend
  choice separate from model-architecture changes.

### Runtime FFI Graph Slots

Use `--recipe <recipe-id> <instance-id>` or `--graph-patch <json>` only when
the request is to replace an explicit TensorRT region with a reviewed,
family-owned TVM-FFI graph implementation. Read
`website/docs/features/tvm-ffi.md` and the owning recipe/instance contract.
Do not use a graph slot as a generic model-onboarding shortcut: current slots
reject TensorRT-RTX, optimized-runtime, quantized, and tensor-parallel builds.

### Runtime

- Reuse an existing runtime strategy only when its public request, cache/state,
  tokenizer/media, output, and lifecycle contracts match.
- Add a family-owned C++ model plugin for new native behavior.
- Keep common host/session behavior in shared runtime only when multiple model
  owners need the same contract.
- Ensure bundle metadata and runtime selection describe the actual path.

### Model-Owned Evidence

- Add the manifest under `tests/e2e/models/<family>/manifests/`.
- Register it in that family's E2E `MODEL.toml`.
- Use real task strategy, runtime strategy, testcase, inputs, and comparator.
- Put model-specific tolerances in the established threshold sidecar.
- Add `perf_validation.json` only when a reviewed performance contract exists.
- Bind a reference workload in `tests/validation/model_workloads.yaml` when the
  task has an aligned comparison implementation.
- Define a new workload in `tests/validation/workloads.yaml` only when the
  reference runner, TRTMC runner, comparator, gates, and reproduction contract
  are complete; implementation belongs in `tools/validation/`.

Never reduce a threshold, dataset, sample limit, or oracle merely to obtain a
pass.

## Build And Family Validation

Use the supported dev container when TensorRT/GPU dependencies are required:

```bash
./build/trtmc build <model-id-or-path> \
  --model-revision <immutable-revision> \
  -o <artifact-dir>/<model>.bundle

./scripts/validate_family.sh <model-id-or-path> \
  --binary ./build/trtmc \
  --bundle-dir <artifact-dir> \
  --isolate-model-plugin
```

The script builds the bundle, discovers the native runtime owner, runs
decoder-only diff tools only for a declared `decoder_debug` profile, and runs a
matching model-owned E2E case when present. A warning about a missing manifest
is not a support pass.

Also validate repository ownership and selection:

```bash
PYTHONPATH=python:. python3 tools/model_ci.py validate
PYTHONPATH=python:. python3 tools/test_impact.py --validate
```

## Reference Consistency

List and dry-run the owned workload:

```bash
PYTHONPATH=python:. python3 tools/trtmc_validate.py --list
PYTHONPATH=python:. python3 tools/trtmc_validate.py \
  <model> <workload> \
  --dry-run \
  --output <plan-dir>
```

Then execute it on the exact bundle and retain the artifact tree:

```bash
PYTHONPATH=python:. python3 tools/trtmc_validate.py \
  <model> <workload> \
  --bundle <bundle.bundle> \
  --output <comparison-dir>
```

Keep execution, reference, comparison, and validation status separate. Use
`$debug-trt-mismatch` to localize failures by modality; do not force decoder
diff tools onto audio, diffusion, or vision-language paths.

## Evidence Gates

| Claim | Minimum evidence |
|---|---|
| Builder implemented | focused family tests and successful bundle build |
| Native runtime connected | descriptor/selection checks and isolated plugin load |
| Output parity | retained model-first comparison artifact |
| Model supported | registered model-owned E2E plus target-hardware pass |
| Performance improved | comparable owned performance run |
| Optimized path qualified | exact implementation/profile/qualification artifacts |

Compilation, sample output, build success, and dry-run planning are not model
parity.

## Delivery

Before `$submit-github-pr`, run skill/static checks, relevant unit tests, the
model-owned E2E and comparison where hardware permits, and `git diff --check`.
Use `$write-git-messages`.

The PR must identify the exact model revision, all three ownership descriptors,
runtime/bundle path, build and validation commands, artifact hashes/paths,
comparison metrics, target hardware, performance evidence if claimed, and
unrun or blocked gates.

<!-- Collaborative review anchor: batch 2. -->

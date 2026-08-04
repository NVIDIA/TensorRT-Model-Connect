---
title: Feature Reference & Context
description: Current feature contracts and the design history that explains why they exist.
---

This section is the project's long-term feature record. It combines two kinds
of material that used to appear in separate menus:

- **Feature reference** describes behavior available in the current codebase.
- **Feature context** records the decision, migration, experiment, or retired
  workflow that explains how the current design was reached.

Most newcomers do not need this section to complete their first inference.
Start with [Getting Started](../getting-started/overview.md), then use the
[Learning Path](../learning-path.md). Return here when you need the exact
boundary or history of one feature.

## How to read a page

| Page kind | Use it for | Do not use it for |
| --- | --- | --- |
| Current reference | Supported options, ownership, runtime behavior, and validation expectations. | Proof that every model or hardware target passed. |
| Design record | Why a design was selected and which alternatives were rejected. | A step-by-step current runbook unless it explicitly says so. |
| Worklog | Debugging history, experiments, and lessons learned. | Current defaults or support claims. |
| Retired workflow | Understanding preserved tooling or an old process. | Operating the current project. |

Implementation, descriptors, tests, and exact-revision evidence remain the
source of truth when historical prose disagrees with current code.

## Categories

### Model and runtime integration

Use these pages to understand family ownership, native runtime strategies,
delegated optimized implementations, and the migrations that created those
boundaries.

- [Model Support](../getting-started/model-support.md)
- [Model Families](model-families.md)
- [Runtime Strategies](runtime-strategies.md)
- [Optimized-runtime family adapter record](../context/optimized-runtime-family-adapter-plan.md)
- [Model-plugin encapsulation record](../context/model-plugin-encapsulation-plan.md)

### Inference behavior and optimizations

Use these pages for multi-device execution, sampling, TriAttention, cache
behavior, and the evidence behind runtime optimizations.

- [Multi-Device Execution](multi-device.md)
- [TVM FFI Kernel Bridge](tvm-ffi.md)
- [Sampling](sampling.md)
- [TriAttention](triattention.md)
- [TriAttention native C++ worklog](../context/triattention-native-cpp-worklog.md)

### Build, quantization, and configuration

Use these pages for precision and quantization contracts, schema-driven
configuration, backend loading, and the design history of the config registry.

- [Quantization](quantization.md)
- [Configuration and Backends](config-and-backends.md)
- [Config-registry status](../context/config-registry-status.md)

### Validation, CI, and performance

Use these pages for test commands, evidence tiers, profiling, benchmarking,
PR-versus-nightly coverage, and the current traceability and safety status.

- [Testing Reference](../reference/testing.md)
- [Benchmarking](../reference/benchmarking.md)
- [Profiling](../reference/profiling.md)
- [E2E L0 Replacements](../reference/e2e-l0-replacements.md)
- [Traceability and Safety Status](../context/traceability-and-safety.md)

### Design and project history

Use these pages for ADRs, retired development workflows, and documentation
design research. They are intentionally last in the navigation.

- [Architecture Decision Records](../context/adr/)
- [Documentation research](../reference/documentation-research.md)

## Support claims require evidence

An API, parser option, source file, or manifest proves only that a surface is
declared. A support claim needs the evidence appropriate to its level:

1. implementation and ownership descriptors;
2. focused unit and contract tests;
3. exact-model task comparison;
4. target-hardware execution; and
5. performance or release qualification when those properties are claimed.

See [Model Support](../getting-started/model-support.md) for the live inventory
and [Testing Reference](../reference/testing.md) for the validation boundary.

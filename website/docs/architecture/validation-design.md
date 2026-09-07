---
title: Validation Design
description: Evidence layers for isolated family changes and shared contracts.
---

import Diagram from '@site/src/components/Diagram';

Different tests prove different things. A green source check does not prove
model accuracy, and one model E2E does not prove an unrelated family.

<Diagram
  src="/img/diagrams/architecture/validation-evidence-ladder.svg"
  alt="Evidence grows from ownership and source checks through focused units, bundle loading, exact model E2E, and reproducible benchmarks"
  caption="Run the smallest meaningful layer first, then add only the evidence required by the changed behavior."
/>

## Evidence layers

1. **Ownership and source checks** prove that a normal family change stays
   inside `families/<family>/`, declarations are valid, and no forbidden
   cross-family dependency was added.
2. **Focused unit tests** prove builder, bundle, Task, Engine, or family-local
   behavior without requiring a real checkpoint.
3. **Bundle/load integration** proves the header, backend, family DSO, factory,
   and abstract Task interface agree.
4. **Exact model E2E** builds and runs a declared checkpoint with the
   family-owned oracle and thresholds.
5. **Performance evidence** uses a recorded workload and reports build,
   setup, prefill, decode, or task latency without mixing timing scopes.

## Start small

For a family change, run its smallest real build/run path before expanding to
more manifests. For a shared core change, run the core contract tests first,
then representative families only where the changed contract is exercised.

```bash
PYTHONPATH=core/builder:apps/benchmark:. python3 -m tools.model_ci validate
PYTHONPATH=core/builder:apps/benchmark:. python3 -m pytest -q
git diff --check
```

The exact family manifest and native target determine additional commands.

## Family-owned evidence

A family keeps its manifests, reference code, test helpers, thresholds, and
source tests under `families/<family>/tests/`. Shared infrastructure may
schedule those tests and validate their shape; it must not encode model
semantics or thresholds.

Do not weaken a threshold, remove a case, or change pass criteria to make CI
green. When hardware is unavailable or known broken, report the unrun boundary
instead of claiming coverage.

## Benchmarks

Benchmark applications live under `apps/benchmark/` and call public Task APIs.
A report must identify the exact model input, revision, bundle settings,
hardware, software environment, warmup, repetitions, and timing scope. Bundle
or source hashes are not a substitute for those semantics.

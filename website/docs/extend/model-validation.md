---
title: Model Validation
description: Keep model semantics and evidence inside the owning family.
---

A family owns its validation under `families/<family>/tests/`.

```text
families/<family>/tests/
├── test_e2e.py
├── manifests/
├── thresholds/       # only when a numeric default is overridden
├── data/             # small redistributable fixtures only
└── cpp/              # optional family runtime contract tests
```

## Start with one minimal E2E

First build one small real bundle and call the public native CLI or C++ Task API.
Prove model resolution, TensorRT build, bundle publication, explicit
backend/family loading, and one meaningful output. Add more cases only after
that path works.

A manifest declares the exact family, task, bundle settings, model source,
parallel size, precision, and family-owned test cases. The family test invokes
the oracle and thresholds it actually owns; it does not fall through to a
generic semantic comparator.

## Validation order

1. Run family source/unit tests.
2. Run `python -m tools.model_ci validate`.
3. Build the affected family DSO and any family-local C++ tests.
4. Build and run the smallest real E2E manifest.
5. Add exact parity or performance evidence only when the change claims it.
6. Run `git diff --check`.

Shared test infrastructure may schedule family tests and validate manifest
shape. It must not own model inputs, reference semantics, thresholds, or
family-specific dependencies.

## Isolation checks

A normal family contribution should change only `families/<family>/**`.
Verify that:

- `support.py` imports only the model-support contract;
- `model.py` is a plain function module and does not inherit;
- no source imports or includes a sibling family;
- the runtime DSO links core interfaces, not the runtime loader or another
  family;
- optional packages are declared in that family's `requirements.txt`;
- examples and benchmarks remain consumers of public APIs.

Do not weaken pass criteria, remove a failing case, or broaden a waiver to make
CI green. Report unavailable hardware and unrun configurations explicitly.

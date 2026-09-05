---
name: doc-sync
description: >-
  Audit and update TensorRT-Model-Connect documentation when code, public APIs,
  family ownership, commands, or model support have changed.
---

# Synchronize Documentation

Treat implementation, tests, executable help, and family manifests as the
source of truth. Update only documentation affected by the requested change.

## Route the change

- Public build behavior: check `core/builder/` and the Python API docs.
- Native user behavior: check `core/runtime/include/trtmc/`, CLI parsing, and
  the C++ API docs.
- Model behavior: check only the owning `families/<family>/` implementation,
  manifests, thresholds, and model recipe.
- Developer workflow: check `README.md`, `CONTRIBUTING.md`, and the relevant
  page under `website/docs/`.
- Benchmark behavior: check `apps/benchmark/` and its performance README.
- Repo-local skills: check `plugins/trtmc-agent-skills/skills/`.

Verify every changed path, symbol, option, and command locally when practical.
If hardware or dependencies prevent execution, inspect the consuming code and
state that boundary explicitly. Do not turn source presence into a runtime,
parity, performance, or qualification claim.

Keep one canonical explanation and link to it instead of copying large policy
sections. Remove stale links and generated counts rather than replacing them
with unverified values.

## Validate

```bash
git diff --check
PYTHONPATH=core/builder:apps/benchmark:. python3 -m tools.model_ci validate
npm --prefix website run test:model-support
npm --prefix website run build
```

Run additional focused tests for any documented command or API that changed.
Summarize corrected claims, validation actually run, and remaining evidence
boundaries. Publish or open a pull request only when the user requested it.

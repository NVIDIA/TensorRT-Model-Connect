---
title: Validation Design
---

Validation follows the same ownership boundary as production code. Every
family keeps its Python tests, native tests, manifests, thresholds, fixtures,
and official-reference adapters below `families/<family>/tests/`.

| Layer | Location | Evidence |
| --- | --- | --- |
| Shared Python contracts | `core/builder/tests/` | Resolver, build request, writer, graph-transform, and BYOK behavior. |
| Shared native contracts | `core/runtime/tests/` | Bundle bounds, loader, Task API, and engine primitives. |
| Family source/unit tests | `families/<family>/tests/` | Family topology, preprocessing, runtime helpers, and failure semantics. |
| Family native tests | `families/<family>/tests/cpp/` | The owning DSO's C++ contracts. |
| Family E2E | `families/<family>/tests/test_e2e.py` | Checkpoint-to-bundle-to-Task behavior and model-owned oracle. |
| Application tests | `apps/*/tests/`, `examples/**/test_*.py` | Public API consumers without reverse dependencies. |

Each family manifest declares its exact checkpoint inputs, task, precision,
topology, premerge selection, and case-specific contract. Optional threshold
files contain numeric overrides read by that family only. Premerge runs each
family's owner-selected cases; nightly runs the complete declared inventory.

## Required properties

- Every family declares at least one premerge case.
- A family test never falls through to a generic comparator or sibling fixture.
- Thresholds remain meaningful and are never weakened to make CI pass.
- Build, reference, and native inputs are aligned before claiming an accuracy
  defect.
- Family changes select and run the owning family's Python, native, GPU, and
  checkpoint evidence without importing unrelated families.
- Package validation finds every family dependency file and DSO while avoiding
  unselected family imports.

## Local checks

```bash
python3 -m tools.model_ci validate
python3 tools/test_impact.py --validate
python3 -m pytest core/builder/tests tools/tests
python3 -m pytest families/qwen/tests
```

GPU and checkpoint cases require the exact environment variables and native
artifacts declared by the selected family. The repository CI pipeline remains
the authoritative source for full public and protected qualification.

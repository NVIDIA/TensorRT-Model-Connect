---
title: AI & Agent Guide
description: Machine-friendly entry points and repository safety rules for coding agents working with TensorRT-Model-Connect.
---

import useBaseUrl from '@docusaurus/useBaseUrl';

TensorRT-Model-Connect is structured so one model-family change can be
understood, tested, and reverted independently. The machine-readable site index
is <a href={useBaseUrl('/llms.txt')}>llms.txt</a>. Repository instructions in
[`AGENTS.md`](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/main/AGENTS.md)
remain authoritative; this page is an orientation, not a substitute for the
instructions that apply to the current checkout.

## Find the owner first

For normal model work, start in one directory:

```text
families/<family>/
├── support.py        # exact checkpoint identity and supported/default tasks
├── model.py          # plain build(request, writer) entrypoint
├── requirements.txt  # optional owner-only dependencies
├── runtime/          # native Task implementation and family DSO
└── tests/            # manifests, oracles, thresholds, fixtures, and units
```

Do not begin by adding a registry, base class, shared model helper, central
source list, or configuration layer. A family builder must not inherit from a
base builder. Duplication is acceptable when it keeps families independent.

Use shared paths only for shared contracts:

- `core/builder/` owns resolution, the build request, bundle writer, and BYOK
  graph-transform hook;
- `core/runtime/` owns bounded bundle reads, abstract Task and Engine APIs,
  loader mechanics, and TensorRT backend primitives;
- `apps/` and `examples/` consume public APIs in one direction; and
- `tools/` owns repository validation and CI orchestration.

## Required behavior

Before changing or running anything, an agent should:

1. read every applicable `AGENTS.md`;
2. inspect the branch, status, remotes, and existing user changes;
3. read the owning family before proposing shared code;
4. use exact family manifests instead of guessing checkpoint IDs, precision,
   task, or topology;
5. make the smallest current-contract change; and
6. report executed evidence separately from unrun GPU, parity, performance, or
   publication boundaries.

An agent must not weaken a threshold or test to make CI pass, silently swap a
checkpoint, treat compilation as inference proof, expose private inputs, or
perform destructive/external actions outside the user's authority.

## Minimal validation loop

Start with the narrowest real loop, then expand only after it passes:

```text
checkpoint -> support resolution -> family build() -> bundle
           -> exact family DSO -> abstract Task call -> checked output
```

Useful source-only checks from the repository root are:

```bash
PYTHONPATH=core/builder:apps/benchmark:. python3 -m tools.model_ci validate
PYTHONPATH=core/builder:apps/benchmark:. python3 tools/test_impact.py --validate
PYTHONPATH=core/builder:apps/benchmark:. python3 -m pytest -q
git diff --check
```

Documentation changes should also run the website inventory test and production
build described in [Testing](reference/testing.md).

## Suggested read-only onboarding prompt

```text
Use the current TensorRT-Model-Connect checkout. Read AGENTS.md, then follow
website/docs/getting-started/source-build.md and quick-start.md exactly. Do not
modify source, tests, git history, or remote state. Report the exact revision,
selected GPU, commands, bundle path, Task result, and every deviation or
validation step that was not run.
```

For implementation boundaries, continue with the
[Developer Guide](developer-guide/overview.md) and
[Architecture](architecture/ai-native-horizontal-scaling.md).

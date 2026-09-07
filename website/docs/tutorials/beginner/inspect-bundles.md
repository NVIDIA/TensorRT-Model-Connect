---
title: Beginner Tutorial - Inspect Bundles
---

# Inspect a bundle

A `.bundle` file is the handoff from a family-owned Python builder to that
family's native task implementation. Inspect it before debugging execution.

## Inspect the header and sections

```bash
trtmc inspect ./gpt2.bundle
```

`inspect` accepts exactly one bundle path. It prints the bundle header and
section inventory without loading the family task or running inference. The
current command has no `--list-engines` mode and does not interpret every
family-specific section.

Record:

1. the source model and family identity;
2. precision and TensorRT compatibility metadata;
3. engine-plan, tokenizer, configuration, and asset sections;
4. whether the expected sections for this family are present.

Section names are family-owned. Do not infer a removed runtime-strategy or
provider registry from them.

## Connect the artifact to source ownership

For a repository checkout, inspect the family that owns the bundle:

```bash
find families/gpt2 -maxdepth 2 -type f | sort
find families/gpt2/tests/manifests -type f | sort
```

The relevant boundaries are:

| Evidence | Owner |
| --- | --- |
| Model recognition and build graph | `families/gpt2/support.py` and `model.py` |
| Native orchestration | `families/gpt2/runtime/` |
| Supported model cases | `families/gpt2/tests/manifests/` |
| Shared artifact reader | `core/runtime/bundle/` |

## Debugging order

1. If inspection cannot parse the header, investigate the build or artifact.
2. If required sections are absent, investigate the family builder.
3. If inspection succeeds but loading fails, verify `--runtime-root` contains
   the installed core, backend, and family libraries from a compatible build.
4. If loading succeeds but a request fails, route the issue to the family task
   contract and model-owned tests.

Inspection proves artifact structure, not numerical correctness. Complete a
family-owned E2E or reference comparison before treating the bundle as
validated.

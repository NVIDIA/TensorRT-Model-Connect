---
title: AI & Agent Guide
description: Machine-friendly entry points and repository safety rules for coding agents working with TensorRT-Model-Connect.
---

import useBaseUrl from '@docusaurus/useBaseUrl';

TensorRT-Model-Connect is designed to be navigable by humans and agents. The
machine-readable index is <a href={useBaseUrl('/llms.txt')}>llms.txt</a>.
Repository instructions in
the [repository `AGENTS.md`](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/main/AGENTS.md)
remain authoritative; this page is an orientation, not a substitute for
reading the instructions that apply to the current checkout and path.

## Required agent behavior

Before changing or running anything, an agent must:

1. read the applicable `AGENTS.md` files and obey the closest scoped file;
2. inspect the current branch, status, remotes, and user-owned changes;
3. use current model descriptors and E2E manifests instead of guessing model
   IDs, paths, precision, topology, or support;
4. separate source/static proof, GPU execution, model parity, performance, and
   publication evidence in its report; and
5. state which meaningful validations were not run.

## Repository skills

The repository ships task-specific skills under
`plugins/trtmc-agent-skills/skills/`. They are registered through the Codex
plugin path, so another agent runtime will not list them: read
`plugins/trtmc-agent-skills/skills/<name>/SKILL.md` directly. An empty runtime
skill list is not evidence that no skill covers the task.

Start from the routing table in
[`AGENTS.md`](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/main/AGENTS.md).
The two that cover the most common contributions are `transform-model`, for
onboarding a Hugging Face model or extending a family, and
`debug-trt-mismatch`, for output that disagrees with the reference.

## Safety boundaries

An agent must not:

- delete, overwrite, reset, rebase, commit, push, open/merge a PR, or publish
  artifacts unless the user authorized that action and scope;
- weaken test criteria, tolerances, or gates for the purpose of making CI pass;
- silently replace an exact checkpoint with a same-family model or unpinned
  revision;
- treat a parser option, source file, manifest, or skipped test as target
  qualification;
- expose credentials, gated model assets, private URLs, or retained internal
  artifacts in public output; or
- mix native runtime support with a platform-specialized provider claim.

When requested work conflicts with one of these boundaries, stop, show the
concrete conflict, and ask for human direction.

## Agent workflow

```text
Read instructions
  → inspect exact repository state
  → identify model/task/runtime owner
  → make the smallest scoped change
  → run the smallest meaningful validation
  → report evidence and unrun boundaries
  → wait for authorization before external or destructive actions
```

Useful source-of-truth commands:

```bash
git status --short --branch
python3 tools/model_ci.py validate
python3 tools/check_doc_file_references.py --strict website/docs
```

## AI-native quick start prompt

Give a capable coding agent this goal from a clean terminal:

```text
/goal Use the current TensorRT-Model-Connect checkout, or clone
https://github.com/NVIDIA/TensorRT-Model-Connect.git if none is provided. Read
AGENTS.md, then follow website/docs/getting-started/source-build.md and
website/docs/getting-started/quick-start.md exactly. Do not modify source,
tests, Dockerfiles, git history, or remote state. Report the selected GPU,
exact commands, bundle path, inference output, and every deviation from the
documentation.
```

The agent should use [Build from Source](getting-started/source-build.md) and
[Quick Start](getting-started/quick-start.md), not invent a parallel setup.

{/* Collaborative review anchor: batch 2. */}

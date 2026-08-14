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
/goal Clone https://github.com/NVIDIA/TensorRT-Model-Connect.git into a new
directory, read the repository AGENTS.md, set up the documented development
environment, and follow the end-to-end Qwen quick start exactly. Build
Qwen/Qwen3-0.6B with --precision bf16 and --max-cache-length 16384, then run it
with --chat-template, --no-thinking, --temperature 0.7, --top-k 20, --top-p 0.8,
--seed 42, and --max-new-tokens 64. Do not change source, tests, git history,
or remote state. Report every command, the resulting bundle path, validation
output, and any deviation from the documented quick start.
```

The agent should follow [Get Started](getting-started/overview.md), not invent
paths from another machine.

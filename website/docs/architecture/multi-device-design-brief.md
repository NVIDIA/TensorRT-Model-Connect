---
title: Multi-Device Design Brief
---

This is the short presentation version of the multi-device design. It is meant
for readers who need the idea before they read the implementation map.

## The Simple Picture

Single-device TRT-MC builds one engine and runs it in one process.

Multi-device TRT-MC builds a bundle with a plan:

```text
.trtfb bundle
  config.json
  distributed_plan.json
  decoder_rank0_plan
  decoder_rank1_plan
```

`distributed_plan.json` tells runtime which rank-local engine section each
process should load and which distributed group it should join.

## What The User Specifies

The user asks for distributed behavior through the normal build config surface.
For TP=2, that looks like:

```bash
trtmc build Qwen/Qwen3-0.6B \
  --output qwen-tp2.trtfb \
  --set parallel.mode=tensor_parallel \
  --set parallel.tp_size=2
```

The user does not manually write `DistributedConfig` or rank mapping for the
current TP path. The builder creates those from the build request.

## What The Builder Does

The builder turns the user request into four things:

| Builder layer | Simple meaning |
| --- | --- |
| `DistributedConfig` | "There are two ranks, arranged as TP=2." |
| `ModelRecipe` | "These are the model regions: attention, MLP, LM head, and so on." |
| `ShardingPolicy` | "For this rank, slice these weights and add these collectives." |
| `PlanCompiler` | "Build one local engine section per rank and write the plan." |

The actual sharded weights are materialized inside the rank-local engine
sections. There is no separate sharding cache that the user manages.

## What Runtime Does

At run time, each launched process reads the bundle and the plan.

For TP=2:

| Rank | Runtime loads |
| --- | --- |
| Rank 0 | `decoder_rank0_plan` |
| Rank 1 | `decoder_rank1_plan` |

Runtime also checks that the launch matches the plan. For TP=2, the user should
launch two processes:

```bash
mpirun -np 2 ./build/trtmc run qwen-tp2.trtfb --prompt "..."
```

The runtime command does not need extra sharding flags because the bundle
already contains the plan.

## Why This Fits TRT-MC

The design keeps the existing TRT-MC split:

| Existing TRT-MC idea | Multi-device version |
| --- | --- |
| Python builds bundles | Python builds rank-local sections and writes `distributed_plan.json`. |
| `.trtfb` is the handoff | The distributed plan is another bundle section. |
| Runtime plugins own tasks | Decoder still owns text generation behavior. |
| Runtime core owns shared infrastructure | Mesh runtime owns ranks, devices, groups, and communicators. |
| Backend hides TensorRT details | The backend receives local engine bytes and communicator handles. |

Single-device stays isolated: no distributed plan means the old one-process
load path.

## Avoiding Branch Sprawl

The design avoids putting long chains of `if tp`, `if cp`, `if pp`, and `if dp`
inside every model family.

Instead:

| Question | Owner |
| --- | --- |
| What parts does the model have? | `ModelRecipe` |
| Which parts are sharded or replicated? | `DistributedPlan` |
| How do we build this rank's local tensors? | `ShardingPolicy` |
| Which local engine sections go into the bundle? | `PlanCompiler` |
| Which process groups exist at runtime? | Mesh runtime |

Future CP, DP, PP, EP, and partial-sharding work should extend those same
owners.

## Where To Extend

For a new model or new parallel mode:

| Need | Extend |
| --- | --- |
| New shardable model regions | `ModelRecipe` |
| New plan shape or selector | `DistributedPlan` schema |
| New local tensor or collective behavior | `ShardingPolicy` |
| New rank or stage artifact layout | `PlanCompiler` |
| New runtime process groups | Mesh runtime |
| New task scheduling behavior | The relevant runtime plugin |

For exact files and code snippets, use
[Multi-Device Implementation Map](multi-device-implementation-map.md).

## Current Validation Takeaway

Fresh two-B200 validation on May 18, 2026 passed for the current Qwen TP=2
path:

| Check | Result |
| --- | --- |
| Qwen3 0.6B SD and TP=2 E2E | Passed. |
| Qwen3 4B SD and TP=2 E2E | Passed. |
| 0.6B direct SD-vs-TP=2 debug logits | Cosine p5 `0.9999358`, argmax match `1.0`. |
| 4B direct SD-vs-TP=2 debug logits | Cosine p5 `0.9996930`, argmax match `0.954545`. |

The E2E pass proves the TP bundle matches its configured oracle. The direct
SD-vs-TP debug-logit comparison proves the distributed path tracks the
single-device path for the same plain prompt.

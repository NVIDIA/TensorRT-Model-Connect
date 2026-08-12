---
title: Multi-Device Execution
description: Build-time tensor and context parallelism, bundle layout, runtime launch, model ownership, and current limits.
---

Model Connect can build native TensorRT bundles whose model-owned graphs run
across multiple GPUs. The topology is fixed during the build. At runtime, one
process per visible GPU loads the bundle, maps its global rank to a CUDA device,
and joins the required NCCL communicators.

There are two public modes:

| Mode | Partitioned dimension | Typical bundle layout | Current model examples |
| --- | --- | --- | --- |
| Tensor parallelism (TP) | Weights, attention heads, or hidden/FFN projections. | One rank-specific plan per shard, such as `engine_plan_tp_rankN` or `denoiser_plan_tp_rankN`. | Text decoders, encoders, speech, vision-language, time-series, and selected diffusion families. |
| Context parallelism (CP) | The denoiser's sequence or media-token dimension, using Ulysses exchanges. | One shared `denoiser_plan_cp` containing distributed collectives. | FLUX.1-schnell and Wan2.1 in the current multi-device manifests. |

The CLI surface is generic; support is not. A family must accept the selected
`ParallelConfig`, construct a valid distributed graph, define its bundle
sections, and own a runtime that initializes the matching communicator. Use an
exact `ci_tier: multi_device` E2E manifest as the support contract for a model,
mode, size, task, and oracle.

## Build contract

Select exactly one mode:

```bash
# Tensor parallelism
trtmc build MODEL --tensor-parallel-size 4 -o model-tp4.bundle

# Context parallelism
trtmc build MODEL --context-parallel-size 4 -o model-cp4.bundle
```

`--tp-size` and `--cp-size` are short aliases. Both options accept `1`, `2`,
`4`, or `8`; values greater than one enable a distributed build. TP and CP are
mutually exclusive. Distributed builds require TensorRT 11.0 or newer.

The mode belongs to the bundle, not a request. The build stores these common
configuration fields:

```json
{
  "parallelism": {
    "mode": "tensor_parallel",
    "tp_size": 4,
    "cp_size": 1,
    "rank": -1,
    "require_mpirun": true
  },
  "parallel_mode": "tensor_parallel",
  "tensor_parallel_size": 4,
  "tensor_parallel_require_mpirun": 1
}
```

A CP bundle uses `context_parallel_size` and
`context_parallel_require_mpirun` instead. Model families can append
model-owned topology metadata, but they must not redefine the common mode or
requested world size.

### Tensor-parallel sections

For decoder-style TP, the builder compiles one engine per rank in a single
build process. The bundle records rank-specific sections such as:

```text
engine_plan_tp_rank0
engine_plan_tp_rank1
engine_plan_tp_rank2
engine_plan_tp_rank3
```

Diffusion TP families use rank-specific denoiser sections such as
`denoiser_plan_tp_rankN` while keeping unsharded text encoder and VAE sections
model-owned. Each runtime rank selects only its own distributed plan.

The model builder validates its mathematical constraints. Common projection
paths require hidden, attention-head, key/value-head, or FFN dimensions to be
divisible by TP size. Other families can impose narrower constraints.

### Context-parallel sections

Current CP diffusion families store one `denoiser_plan_cp`. Every rank loads
that plan and supplies rank identity through the distributed runtime or an
explicit graph input. Ulysses all-to-all operations exchange sequence and head
shards inside the graph.

CP currently applies to model-owned diffusion paths. The common builder rejects
CP for families that do not implement `parallel_config`; the FLUX family also
restricts Ulysses CP to FLUX.1.

## Runtime contract

The runtime is deliberately launcher-light:

1. `mpirun` or another compatible launcher starts one process per rank.
2. The runtime reads world size and rank from Open MPI, PMI, or generic
   `WORLD_SIZE`/`RANK` environment variables.
3. Global rank `N` binds visible CUDA device ordinal `N`.
4. Rank 0 writes an NCCL unique ID to a rendezvous file; other ranks read it.
5. The model runtime loads the plan for that rank or shared CP plan and passes
   the communicator to TensorRT.
6. Every rank participates until model-owned distributed work completes;
   current media pipelines reserve final VAE decode and user-facing artifacts
   for global rank 0.

The default rendezvous path is derived from the MPI job. Set an explicit path
when the launcher environment is unusual or multiple jobs share the same
host:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TRTMC_NCCL_RENDEZVOUS="$PWD/artifacts/model-tp4.nccl"
rm -f -- "$TRTMC_NCCL_RENDEZVOUS"

mpirun --tag-output -np 4 \
  -x LD_LIBRARY_PATH \
  -x CUDA_VISIBLE_DEVICES \
  -x TRTMC_NCCL_RENDEZVOUS \
  trtmc run model-tp4.bundle [task options...]
```

`libnccl.so.2` or `libnccl.so` must be discoverable at runtime. Model Connect
loads NCCL dynamically, so the core runtime does not need a compile-time NCCL
dependency. The communicator is owned for at least as long as its TensorRT
engines and execution contexts.

## Find supported model contracts

The repository intentionally avoids a hand-maintained global support table for
multi-device profiles. Query the executable manifests:

```bash
rg -l '"ci_tier"\s*:\s*"multi_device"' \
  tests/e2e/models --glob '*.json' | sort
```

To print model, topology, and world size together:

```bash
for manifest in $(
  rg -l '"ci_tier"\s*:\s*"multi_device"' \
    tests/e2e/models --glob '*.json' | sort
); do
  jq -r '[
    .name,
    .hf_id,
    .build_args.parallel.mode,
    (.build_args.parallel.tp_size // .build_args.parallel.cp_size),
    .distributed_runtime.world_size
  ] | @tsv' "$manifest"
done
```

The current inventory spans text decoders and encoders, MoE and recurrent
models, speech and audio, vision-language and vision, time-series models, and
diffusion. That breadth is not blanket support: each row names one exact model,
mode, world size, task, prerequisites, and evidence policy.

## Validation

Multi-device E2E cases are excluded from the ordinary single-device selection.
Run them explicitly on a compatible host:

```bash
pytest tests/test_e2e.py \
  --multi-device-only \
  --e2e-model MODEL_OR_FAMILY \
  --engine-dir /path/to/bundles \
  --trtmc-binary /path/to/trtmc \
  --model-plugin-dir /path/to/model/plugins
```

The harness reads `distributed_runtime.world_size`, wraps the task command with
the declared launcher, exports the rendezvous and library environment, and
applies the model-owned oracle. It can also gate on GPU count, launcher
availability, Python dependencies, and model-access credentials.

Keep validation tiers separate:

- static and contract tests prove configuration and graph-layout rules;
- a successful build proves TensorRT accepted the exact model graph;
- a full multi-rank task run proves runtime coordination and output shape;
- the model oracle determines parity or quality for the declared input;
- matched repeated measurements are required for a performance claim.

A skipped hardware preflight is not a pass. A plausible rank-0 artifact alone
does not prove that all ranks completed without a TensorRT or NCCL error.

## Current limits

- The implementation is single-node and binds rank to visible device ordinal.
- Distributed builds require native TensorRT 11.0 or newer and a discoverable
  NCCL runtime.
- Supported public sizes are 2, 4, and 8; the selected model can impose stricter
  shape or size constraints.
- TP and CP cannot be combined through the public build CLI.
- Topology is fixed at build time; runtime process count must match the bundle.
- Support is model-owned. A CLI option, parser path, or another family's
  manifest does not prove a new model works.
- Current graph-slot/TVM-FFI builds require tensor-parallel size 1 and cannot
  patch a collective layer.
- Current distributed media pipelines perform final decode and artifact output
  on global rank 0; upstream stages can still be replicated rather than
  distributed.
- The E2E harness uses `mpirun`; other launchers must provide compatible rank
  and world-size environment variables and a shared rendezvous path.

For complete commands and two worked models, see
[Run Inference on Multiple GPUs](../tutorials/advanced/multi-device-inference.md).

{/* Collaborative review anchor: batch 2. */}

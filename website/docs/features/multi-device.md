---
title: Multi-Device Execution
description: Family-owned tensor/context parallel builds and multi-rank runtime execution.
---

Tensor and context parallel sizes are direct build-request fields. The selected
family owns the distributed TensorRT graph, rank-specific sections,
communicator setup, runtime orchestration, and validation.

## Build

```bash
# Tensor parallel
python -m tensorrt_model_connect build MODEL \
  --tensor-parallel-size 4 \
  --output model-tp4.bundle

# Context parallel
python -m tensorrt_model_connect build MODEL \
  --context-parallel-size 4 \
  --output model-cp4.bundle
```

The core accepts positive sizes; the family must implement or reject the exact
values and combination. Topology is fixed into the family-owned bundle
sections. There are no `--tp-size`/`--cp-size` aliases or a shared
`ParallelConfig` contract in the public core.

Families commonly write one plan per TP rank, such as
`engine.rank0.plan`, or a family-specific shared CP plan. Section names and
metadata are private to the owner.

## Runtime

Families that emit distributed collectives own NCCL initialization and load it
dynamically. Replicated/rank-selected plans that contain no collectives do not
initialize NCCL. A typical multi-rank launch is:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TRTMC_NCCL_RENDEZVOUS="$PWD/model-tp4.nccl"

mpirun --tag-output -np 4 \
  -x LD_LIBRARY_PATH \
  -x CUDA_VISIBLE_DEVICES \
  -x TRTMC_NCCL_RENDEZVOUS \
  trtmc run model-tp4.bundle \
    --runtime-root /opt/trtmc/lib \
    --prompt "Hello"
```

The family maps launcher rank to a visible device and must keep its
communicator alive for the TensorRT engines that use it. Runtime process count,
visible devices, and bundle topology must agree.

## Find exact support

Do not infer support from the generic flags. Search family manifests for the
requested topology:

```bash
rg -n '"tensor_parallel_size"|"context_parallel_size"' \
  families/*/tests/manifests/*.json
```

Each result names an exact checkpoint, task, precision, topology, testcase,
and oracle. Run the owning `families/<family>/tests/test_e2e.py` with its
explicit selection and required GPU count.

## Evidence boundary

Static tests prove request and graph-layout rules; a build proves TensorRT
accepted the exact graph; a successful all-rank Task call proves runtime
coordination; and the family oracle determines output parity or quality. A
performance claim additionally needs matched repeated measurements.

Current execution is single-node and uses family-specific launcher/rank
handling. TP and CP combinations, supported world sizes, media rank-zero
behavior, and hardware requirements are family-owned rather than global
promises.

---
title: Multi-Device Execution
description: Family-owned tensor and context parallel builds and explicit multi-rank launch.
---

Tensor and context parallelism are build-time choices owned by the selected
family. The shared build request carries the requested sizes; a family either
builds that topology or rejects it. A parser accepting an option is not a
support claim.

| Mode | Build field | Typical family-owned sections |
| --- | --- | --- |
| Tensor parallelism (TP) | `tensor_parallel_size` | `engine.rankN.plan` or `denoiser.rankN.plan` |
| Context parallelism (CP) | `context_parallel_size` | A family-defined collective plan such as `denoiser.cp.plan` |

Use `families/<family>/tests/manifests/*.json` to find exact model, precision,
and world-size examples.

## Build contract

Select one topology when building:

```bash
# Tensor parallelism
python -m tensorrt_model_connect build MODEL \
  --output model-tp4.bundle \
  --tensor-parallel-size 4

# Context parallelism
python -m tensorrt_model_connect build MODEL \
  --output model-cp4.bundle \
  --context-parallel-size 4
```

The core does not restrict sizes beyond positive integers. Each family owns the
legal sizes, divisibility rules, engine sections, and whether TP and CP may be
combined. Current family builders generally reject a topology they do not
implement.

### Tensor-parallel sections

Qwen TP4, for example, writes exactly:

```text
engine.rank0.plan
engine.rank1.plan
engine.rank2.plan
engine.rank3.plan
```

Diffusion families can use rank-specific denoiser sections while keeping their
text encoder, VAE, tokenizer, and preprocessor sections local to the same
family. Section names are not a shared schema; read that family's `model.py`
and manifest.

### Context-parallel sections

Families that put collective communication inside an engine own its graph,
communicator setup, and NCCL dependency. A family that only selects a
rank-specific replicated plan must not acquire NCCL merely because its world
size is greater than one.

## Runtime contract

The runtime loader remains generic and explicit. One process per rank calls the
same `trtmc` executable, passes the same bundle and runtime root, and the family
runtime maps each rank to its owned plan and device.

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

The process count and visible devices must match the family manifest. Families
that use collectives require a discoverable NCCL runtime and a rendezvous path
visible to every rank.

## Find supported model contracts

```bash
rg -l '"(tensor_parallel_size|context_parallel_size)"[[:space:]]*:[[:space:]]*[2-9]' \
  families --glob '**/tests/manifests/*.json' | sort
```

Inspect the selected JSON rather than copying a global matrix into code. The
manifest provides the checkpoint, task, precision, topology, and testcases.

## Validation

Run only the owning family's test on a compatible host. For Qwen, an explicit
selection looks like:

```bash
TRTMC_E2E=1 \
TRTMC_BINARY=/path/to/trtmc \
TRTMC_RUNTIME_ROOT=/path/to/runtime \
PYTHONPATH=core/builder:. \
python3 -m pytest families/qwen/tests/test_e2e.py \
  --e2e-testcase qwen3-0.6b-fp16-tp4 -v
```

This page documents the contract only. It does not claim that these commands
were run on the machine rendering the documentation.

Keep the evidence levels separate:

- source and unit checks prove ownership and local rules;
- a build proves TensorRT accepted one exact distributed graph;
- a complete multi-rank task run proves launch and coordination;
- the family-owned oracle determines output parity or quality;
- matched repeated runs are required for a performance claim.

## Current limits

- Topology is fixed in the bundle and is not a request-time switch.
- Support is exact to the family manifest; another family's topology does not
  generalize.
- Distributed builds and runs need the libraries required by that family.
- The public loader does not schedule ranks or discover a cluster.
- BYOK graph transforms and distributed graphs remain independent features; a
  family or application must prove their exact combination before claiming it.

See [Run Inference on Multiple GPUs](../tutorials/advanced/multi-device-inference.md)
for worked examples.

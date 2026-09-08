---
title: Multi-Device Inference
---

# Multi-device inference

Tensor and context parallelism are build-time topology choices. A family owns
the graph partitioning, communication, rank behavior, runtime orchestration,
and validation for each topology it supports.

## Build a topology-specific bundle

```bash
python -m tensorrt_model_connect build Qwen/Qwen3-0.6B \
  -o qwen-tp4.bundle \
  --precision bf16 \
  --max-sequence-length 256 \
  --tensor-parallel-size 4
```

For a diffusion family that supports context parallelism:

```bash
python -m tensorrt_model_connect build black-forest-labs/FLUX.1-schnell \
  -o flux-cp4.bundle \
  --precision bf16 \
  --context-parallel-size 4
```

Unsupported topology combinations fail inside the selected family. The shared
builder passes explicit TP and CP sizes but does not implement model-specific
partitioning.

## Launch the required ranks

Use the MPI launcher and rank count required by the family-owned manifest:

```bash
mpirun -n 4 trtmc run qwen-tp4.bundle \
  --prompt "Explain tensor parallelism." \
  --max-new-tokens 64 \
  --seed 1234
```

For output-producing image or video tasks, follow the owning family's launcher
and output-path contract so ranks do not race on one file. Do not invent rank
mapping, environment variables, or synchronization rules outside that family.

## Validate topology and a control

Build a single-device bundle from the same revision and request, then compare
against the family thresholds. Run the checked-in topology case through its
family E2E entry point:

```bash
export TRTMC_BINARY="$PWD/build/apps/cli/trtmc"
export TRTMC_RUNTIME_ROOT="$PWD/build/install/lib"

PYTHONPATH=core/builder:. python3 -m pytest \
  families/qwen/tests/test_e2e.py \
  --e2e-testcase <tp4-manifest-name> \
  -q
```

Replace the selector with an actual manifest in that checkout. Evidence should
include the bundle, manifest, model revision, world size, visible devices,
launcher command, target, TensorRT version, output, and comparison report.

Topology bugs belong to the family unless evidence shows a failure in a
model-agnostic shared communication or loading contract.

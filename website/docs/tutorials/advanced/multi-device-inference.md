---
title: "Run Inference on Multiple GPUs"
---

This lab follows two family-owned examples: Qwen tensor parallelism and FLUX
context parallelism. It documents the commands and evidence boundary; it does
not claim that the host rendering this page has a working multi-GPU fabric.

## Learning objectives

- build one topology through the shared `BuildRequest`;
- inspect rank-specific, family-owned bundle sections;
- launch one process per rank with an explicit runtime root;
- distinguish a documented command from completed target-hardware validation.

## Before you start

Use a host with the required number of visible NVIDIA GPUs, TensorRT, the
family's dependencies, Open MPI, and NCCL when that family actually performs
collectives. Build and install the requested family DSO with the native runtime.

```bash
nvidia-smi -L
mpirun --version
python -m pip install -r families/qwen/requirements.txt
```

Set paths for your own installation:

```bash
export TRTMC_BINARY=/path/to/trtmc
export TRTMC_RUNTIME_ROOT=/path/to/runtime
```

## Level 1: tensor-parallel Qwen3

The exact checked-in example is
`families/qwen/tests/manifests/qwen3-0.6b-fp16-tp4.json`.

### 1. Build a TP4 bundle

```bash
python -m tensorrt_model_connect build Qwen/Qwen3-0.6B \
  --output qwen3-0.6b-fp16-tp4.bundle \
  --precision fp16 \
  --max-sequence-length 256 \
  --tensor-parallel-size 4
```

Inspect the result:

```bash
"$TRTMC_BINARY" inspect qwen3-0.6b-fp16-tp4.bundle
```

The current Qwen family writes `engine.rank0.plan` through
`engine.rank3.plan`. These names belong to Qwen; another family may use
different sections.

### 2. Launch exactly four ranks

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TRTMC_NCCL_RENDEZVOUS="$PWD/qwen-tp4.nccl"

mpirun --tag-output -np 4 \
  -x LD_LIBRARY_PATH \
  -x CUDA_VISIBLE_DEVICES \
  -x TRTMC_NCCL_RENDEZVOUS \
  "$TRTMC_BINARY" run qwen3-0.6b-fp16-tp4.bundle \
    --runtime-root "$TRTMC_RUNTIME_ROOT" \
    --prompt "What is the capital of France? Answer in one word." \
    --max-new-tokens 10 \
    --use-chat-template true \
    --enable-thinking false
```

Do not treat rank 0 text as a pass until every rank exits successfully.

### 3. Compare with a single-device control

Build the matching single-device manifest and use the same prompt, precision,
and token budget. Compare semantic output first. Measure performance only in a
separate, synchronized experiment with the same scope.

## Level 2: context-parallel FLUX

The checked-in CP4 example is
`families/flux/tests/manifests/flux-schnell-l0-cp4.json`.

### 4. Build a CP4 bundle

```bash
python -m pip install -r families/flux/requirements.txt
python -m tensorrt_model_connect build black-forest-labs/FLUX.1-schnell \
  --output flux-schnell-cp4.bundle \
  --precision fp32 \
  --image-height 384 \
  --image-width 384 \
  --context-parallel-size 4
```

The current FLUX family writes its collective denoiser as
`denoiser.cp.plan` and keeps the remaining component plans in the same family
bundle.

### 5. Launch the family task

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TRTMC_NCCL_RENDEZVOUS="$PWD/flux-cp4.nccl"

mpirun --tag-output -np 4 \
  -x LD_LIBRARY_PATH \
  -x CUDA_VISIBLE_DEVICES \
  -x TRTMC_NCCL_RENDEZVOUS \
  "$TRTMC_BINARY" generate-image flux-schnell-cp4.bundle \
    --runtime-root "$TRTMC_RUNTIME_ROOT" \
    --prompt "A photo of a cat sitting on a windowsill at sunset" \
    --output flux-cp4.png \
    --num-steps 20 \
    --height 384 \
    --width 384
```

Artifact ownership is family-specific. Verify that all ranks completed and that
the declared rank writes the final output before interpreting the image.

## Level 3: follow model-owned topology

Discover exact multi-device manifests rather than assuming Qwen or FLUX layout:

```bash
rg -l '"(tensor_parallel_size|context_parallel_size)"[[:space:]]*:[[:space:]]*[2-9]' \
  families --glob '**/tests/manifests/*.json' | sort
```

For a selected manifest, read its family `model.py`, `runtime/`, and
`tests/test_e2e.py`. The shared core carries topology values but owns none of
the partitioning or communicator policy.

## Run the model-owned E2E contract

On a compatible four-GPU host:

```bash
TRTMC_E2E=1 \
TRTMC_BINARY="$TRTMC_BINARY" \
TRTMC_RUNTIME_ROOT="$TRTMC_RUNTIME_ROOT" \
PYTHONPATH=core/builder:. \
python3 -m pytest families/qwen/tests/test_e2e.py \
  --e2e-testcase qwen3-0.6b-fp16-tp4 -v
```

No multi-device command in this page was executed as part of documentation-only
validation. Record your own GPU model, topology, software versions, exact
command, and all-rank result.

## Troubleshooting checklist

- Does visible GPU count match the manifest world size?
- Is every rank using the same bundle and explicit runtime root?
- Does the runtime root contain the selected family and backend DSOs?
- Is the rendezvous path shared and unique to this job?
- Does this family actually use collectives and therefore require NCCL?
- Did every rank exit cleanly?

## Self-check

1. Which file proves that Qwen TP4 is an intended exact test case?
2. Why are bundle section names family-owned?
3. When is NCCL a legitimate family dependency?
4. Why is a rank-0 artifact insufficient evidence?

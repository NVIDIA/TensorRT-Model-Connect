---
title: "Run Inference on Multiple GPUs"
description: Build, launch, validate, and benchmark model-owned tensor- and context-parallel TensorRT bundles.
---

import Diagram from '@site/src/components/Diagram';

This tutorial starts with Qwen3-0.6B tensor parallelism, then applies context
parallelism to the denoiser in FLUX.1-schnell. Both paths build one `.trtfb`
bundle and launch one process per visible GPU, but they partition different
model dimensions and store different engine sections.

For the topology contract, supported limits, and model-discovery workflow, see
the [Multi-Device Execution feature reference](../../features/multi-device.md).

Use the mode declared by the model family and its E2E manifest:

| Level | Model and mode | What it teaches |
| --- | --- | --- |
| Recommended | Qwen3-0.6B with TP4 | Rank-specific weight shards and decoder engines. |
| Advanced | FLUX.1-schnell with CP4 | One sequence-sharded denoiser graph shared by all ranks. |
| Model-owned extension | Cosmos3-Nano with CP4 or CP8 | A requested CP world can contain smaller denoiser and classifier-free-guidance subgroups on a qualified target. |

The CLI flags select a requested topology. They do not make every model
support every mode. The selected family must implement that topology, and an
exact multi-device manifest is the repository's executable support contract.

<Diagram
  src="/img/diagrams/tutorials/advanced/multi-device-topologies.svg"
  alt="A single build command producing either rank-specific tensor-parallel plans or one shared context-parallel denoiser plan, followed by one runtime process per GPU joined through NCCL"
  caption="The bundle fixes the topology at build time. At runtime, the launcher creates one rank per GPU and the model-owned pipeline selects the correct plan and collectives."
/>

## Before you start

Complete [Installation](../../getting-started/installation.md), clone this
repository, and run the commands below from its root. You need:

- a native build using TensorRT 11.0 or newer;
- NCCL available as `libnccl.so.2` or `libnccl.so`;
- Open MPI with `mpirun` on `PATH`;
- four visible NVIDIA GPUs for the TP4 and CP4 examples;
- enough aggregate device memory and disk for the selected model;
- access to gated model assets when the checkpoint requires it.

Set up one shell:

```bash
export TRTMC="${TRTMC:-$PWD/build/trtmc}"
export WORK="${WORK:-$PWD/artifacts/multi-device}"
export CUDA_VISIBLE_DEVICES=0,1,2,3
mkdir -p "$WORK"

test -x "$TRTMC"
test "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" -ge 4
mpirun --version
```

The repository's current E2E manifests name model IDs but do not pin revisions
for these two examples. For reproducible qualification, set an immutable
Hugging Face commit before building:

```bash
# Optional for an exploratory run; required for a reproducible report.
export MODEL_REVISION="${MODEL_REVISION:-}"
REVISION_ARGS=()
if test -n "$MODEL_REVISION"; then
  REVISION_ARGS=(--model-revision "$MODEL_REVISION")
fi
```

:::warning Single-node runtime
The current runtime maps global rank `N` to visible CUDA device ordinal `N`.
These examples are single-node launches; they are not a multi-node recipe.
:::

## Level 1: tensor-parallel Qwen3

Tensor parallelism shards model weights and projection work. Qwen builds one
decoder engine per rank and stores the plans as
`engine_plan_tp_rank0` through `engine_plan_tp_rank3`.

### 1. Build a TP4 bundle

Use the same model and reduced cache capacity as the repository's
`qwen3-0.6b-fp16-tp4` multi-device manifest:

```bash
"$TRTMC" build Qwen/Qwen3-0.6B \
  "${REVISION_ARGS[@]}" \
  --precision fp16 \
  --max-cache-length 256 \
  --tensor-parallel-size 4 \
  -o "$WORK/qwen3-0.6b-tp4.trtfb"
```

The build itself is one process. It compiles all four rank plans in order and
packages them in one bundle. A TP build can therefore take substantially
longer and use more disk than its single-device counterpart even though only
one rank plan is loaded by each runtime process.

Inspect the result and list its engine sections:

```bash
"$TRTMC" inspect "$WORK/qwen3-0.6b-tp4.trtfb"
"$TRTMC" inspect "$WORK/qwen3-0.6b-tp4.trtfb" --list-engines
```

Confirm that all four rank engine sections are present. The bundle metadata
also records `parallel_mode=tensor_parallel` and `tensor_parallel_size=4` for
the runtime. The topology is part of the bundle; `trtmc run` has no runtime
flag that turns a single-device bundle into TP4.

### 2. Launch exactly four ranks

Every launch needs one rendezvous path shared by its ranks. Use a different
path for another simultaneous job and remove a stale file before reusing a
path:

```bash
export TRTMC_NCCL_RENDEZVOUS="$WORK/qwen3-tp4.nccl"
rm -f -- "$TRTMC_NCCL_RENDEZVOUS"

mpirun --tag-output -np 4 \
  -x LD_LIBRARY_PATH \
  -x CUDA_VISIBLE_DEVICES \
  -x TRTMC_NCCL_RENDEZVOUS \
  "$TRTMC" run "$WORK/qwen3-0.6b-tp4.trtfb" \
    --prompt "What is the capital of France? Answer in one word." \
    --max-new-tokens 10 \
    --greedy \
  | tee "$WORK/qwen3-tp4.stdout.log"
```

`--tag-output` prefixes each stream with its MPI rank. Rank 0 owns the
user-facing generated result; the other ranks still execute their engine
shards and participate in collectives. Treat a clean launcher exit as
necessary but not sufficient: also inspect rank 0's text and the TensorRT/NCCL
diagnostics from every rank.

If rank and device counts disagree, the runtime fails instead of silently
placing two ranks on one GPU. If a rank times out waiting for the rendezvous
file, verify that all ranks received the same `TRTMC_NCCL_RENDEZVOUS` value and
that its parent directory is writable.

### 3. Compare with a single-device control

Build the control from the same checkpoint revision, precision, and cache
capacity. Change only the topology:

```bash
"$TRTMC" build Qwen/Qwen3-0.6B \
  "${REVISION_ARGS[@]}" \
  --precision fp16 \
  --max-cache-length 256 \
  -o "$WORK/qwen3-0.6b-single.trtfb"

"$TRTMC" run "$WORK/qwen3-0.6b-single.trtfb" \
  --prompt "What is the capital of France? Answer in one word." \
  --max-new-tokens 10 \
  --greedy \
  | tee "$WORK/qwen3-single.stdout.log"
```

For this smoke input, both runs should satisfy the same one-word response
contract. A plausible answer does not establish full numerical parity. Use the
model-owned E2E comparison for exact checkpoint and oracle coverage.

To measure performance, keep the prompt, output length, precision, cache
capacity, GPU cohort, warmup, and measured iterations fixed. Launch the TP
benchmark through `mpirun`, just like normal TP inference:

```bash
rm -f -- "$TRTMC_NCCL_RENDEZVOUS"
mpirun --tag-output -np 4 \
  -x LD_LIBRARY_PATH \
  -x CUDA_VISIBLE_DEVICES \
  -x TRTMC_NCCL_RENDEZVOUS \
  "$TRTMC" run "$WORK/qwen3-0.6b-tp4.trtfb" \
    --prompt "Explain tensor parallelism in one sentence." \
    --max-new-tokens 64 --greedy \
    --warmup 3 --benchmark 10 \
  > "$WORK/qwen3-tp4.perf.log" 2>&1
```

More GPUs do not by themselves prove a speedup. Report the slowest-rank or
end-to-end request boundary, and include process startup and bundle loading
only when they belong to the deployment metric.

## Level 2: context-parallel FLUX

Context parallelism shards the sequence handled by a diffusion denoiser. For
FLUX.1, the model-owned builder creates one `denoiser_plan_cp` graph containing
Ulysses all-to-all collectives. All ranks load that shared graph; rank identity
controls which sequence shard each process owns.

### 4. Build a CP4 bundle

The following dimensions and step count mirror the reduced
`flux-schnell-l0-cp4` model contract:

```bash
"$TRTMC" build black-forest-labs/FLUX.1-schnell \
  "${REVISION_ARGS[@]}" \
  --precision fp16 \
  --image-height 384 \
  --image-width 384 \
  --num-inference-steps 20 \
  --context-parallel-size 4 \
  -o "$WORK/flux-schnell-cp4.trtfb"

"$TRTMC" inspect "$WORK/flux-schnell-cp4.trtfb"
"$TRTMC" inspect "$WORK/flux-schnell-cp4.trtfb" --list-engines
```

Confirm that `denoiser_plan_cp` is present. The bundle metadata also records
`parallel_mode=context_parallel` and `context_parallel_size=4`. Do not add
`--tensor-parallel-size`: TP and CP are mutually exclusive, and FLUX owns
different graphs for the two modes.

### 5. Give every rank a separate output directory

Only global rank 0 performs the final VAE decode for the current distributed
FLUX pipeline. Separate rank directories keep the command safe for model paths
where a non-output rank still creates an empty result:

```bash
export TRTMC_NCCL_RENDEZVOUS="$WORK/flux-cp4.nccl"
rm -f -- "$TRTMC_NCCL_RENDEZVOUS"
export TRTMC WORK

mpirun --tag-output -np 4 \
  -x LD_LIBRARY_PATH \
  -x CUDA_VISIBLE_DEVICES \
  -x TRTMC_NCCL_RENDEZVOUS \
  -x TRTMC \
  -x WORK \
  bash -lc '
    rank="${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-${RANK:-0}}}"
    output="$WORK/flux-cp4-output/rank_$rank"
    mkdir -p "$output"
    exec "$TRTMC" generate-video "$WORK/flux-schnell-cp4.trtfb" \
      --prompt "A photo of a cat sitting on a windowsill at sunset" \
      --output "$output" \
      --num-steps 20 \
      --seed 42
  '
```

Success produces the image artifact under
`$WORK/flux-cp4-output/rank_0`. The other ranks must still complete cleanly;
an empty nonzero-rank directory is expected and is not a failed collective.

Compare against a single-device bundle using the same prompt, seed, dimensions,
step count, checkpoint revision, and precision. Distributed graph partitioning
can change floating-point operation order, so use the family oracle's image
quality policy rather than requiring byte-identical PNG files unless that
exact equality is part of the model contract.

## Level 3: follow the model-owned topology

Do not generalize the Qwen and FLUX commands by changing only the model ID.
Start from an exact multi-device manifest:

```bash
rg -l '"ci_tier"\s*:\s*"multi_device"' \
  tests/e2e/models --glob '*.json' | sort
```

Representative contracts in the current tree include:

| Workload | Declared model | Mode |
| --- | --- | --- |
| Text generation | `Qwen/Qwen3-0.6B` | TP4 |
| Image diffusion | `black-forest-labs/FLUX.1-schnell` | TP4 and CP4 |
| Video diffusion | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | TP4 |
| Video diffusion | `nvidia/Cosmos3-Nano` | CP4 |
| Speech recognition | `openai/whisper-tiny` | TP2 |
| Vision-language generation | `OpenGVLab/InternVL3-2B-hf` | TP2 |

The list is illustrative. The exact manifest supplies the required world size,
task input, gating prerequisites, runtime strategy, and comparison policy.

### Cosmos3's qualified B200 topology

Cosmos3 demonstrates why the requested size and internal topology are not
always the same thing. The user still builds with `--context-parallel-size 4`
or `8`. On a qualified B200 target, the model-owned builder records:

| Requested world | Denoiser CP group | Classifier-free parallel groups |
| ---: | ---: | ---: |
| CP2 | 2 | 1 |
| CP4 | 2 | 2 |
| CP8 | 4 | 2 |

For CP4 and CP8, half of the ranks execute the conditional denoiser branch and
half execute the unconditional branch. Paired classifier-free groups exchange
the predictions before the scheduler update. The same builder preserves pure
CP on unqualified targets, so users do not select this split with another CLI
flag.

The optimized B200 graph also uses a rank input for local replicated-row
selection and packs multiple Ulysses tensors into fewer all-to-all exchanges.
Those are model implementation details carried by the bundle and runtime; they
are not generic settings to copy into another family.

## Run the model-owned E2E contract

After a manual smoke run, use the harness so model selection, launcher size,
rank environment, task oracle, and artifact policy come from the same
manifest:

```bash
pytest tests/test_e2e.py \
  --multi-device-only \
  --e2e-model qwen3-0.6b-fp16-tp4 \
  --engine-dir "$WORK" \
  --trtmc-binary "$TRTMC" \
  --model-plugin-dir "$PWD/build"
```

Add `--rebuild-engines` only when the harness should build the bundle itself.
The exact environment may also require `--hf-python` and model-specific gated
assets. A skipped preflight is not a passing model result.

## Troubleshooting checklist

| Symptom | Check |
| --- | --- |
| World-size error | Launch `-np` equal to the bundle's requested TP or CP size. |
| CUDA ordinal error | Expose at least one device per rank and keep rank ordinals contiguous within `CUDA_VISIBLE_DEVICES`. |
| Rendezvous timeout | Export one writable, unique `TRTMC_NCCL_RENDEZVOUS` path to every rank. |
| Missing NCCL symbol | Make `libnccl.so.2` or `libnccl.so` visible through the runtime library path. |
| Missing rank plan | Rebuild with the declared topology; do not launch a single-device bundle with multiple ranks. |
| Build rejects the mode | Confirm the exact family manifest supports TP or CP and its dimensions divide by the requested size. |
| Only rank 0 writes media | Expected for current distributed diffusion pipelines; inspect all ranks for successful completion. |
| Output differs from single-device | Apply the model-owned numerical or task-quality oracle before treating non-bit-identical output as a regression. |

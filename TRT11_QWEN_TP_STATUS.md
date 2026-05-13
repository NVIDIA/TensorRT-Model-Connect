# TRT 11 Multi-Device Model Status

Date: 2026-05-13

## Goal

Install the TensorRT 11 package from the internal CUDA repository, run a smoke
test for TensorRT-Model-Connect against TRT 11, and prototype TensorRT 11
multi-device execution for supported model families with tensor parallel sizes
2, 4, and 8 on the A100-x8 host.

The active first implementation target is Qwen tensor parallelism. FLUX stays
as the next diffusion consumer after the Qwen path proves the sharding,
collective, runtime, and test patterns.

External Edge-LLM ONNX references are not part of the active gap list for this
work. The checked-in Qwen TP path is being validated directly through rank-local
bundle sections, TRT 11 collectives, the C++ runtime, and E2E parity artifacts.

## Repository Findings

- The repo builds HuggingFace-style checkpoints into `.trtfb` bundles through
the `trtmc-build` Python CLI and runs them through the C++ `./build/trtmc`
runtime.
- CMake already searches for TensorRT 11 libraries (`libnvinfer.so.11`) and
derives the backend ABI suffix from TensorRT headers.
- Qwen family coverage exists under
`tensorrt_model_connect/tensorrt_model_connect/families/qwen.py`,
`qwen_vl.py`, `qwen_moe.py`, and `qwen3_5.py`.
- FLUX coverage exists under
`tensorrt_model_connect/tensorrt_model_connect/families/flux.py` and the C++
runtime pipeline is under `src/runtime/pipelines/flux_pipeline.cpp`.
- The first FLUX E2E target is `tests/e2e/models/flux-schnell-l0.json`
(`black-forest-labs/FLUX.1-schnell`, `runtime_strategy="diffusion_flux"`,
384 px, 20 denoising steps). The manifest is marked `gated`, so access depends
on available HF credentials/cache.
- Accuracy/performance hooks exist:
  - Text models including Qwen: `tests/e2e/test_full_pipeline.py` runs C++
  inference, logit parity through `tools/diff_logits.py`, and optional
  `tools/perf_compare.py`.
  - Runner parity: `tests/e2e/test_runner_parity.py`.
  - Diffusion: `tests/e2e/test_diffusion_pipeline.py` plus
  `tests/e2e_harness/plugins/diffusion.py` check component health, frame/image
  statistics, and optional PSNR/SSIM.
- Website testing docs live at `website/docs/reference/testing.md`; update this
  page whenever multi-device test entry points, CI lanes, or validation
  expectations change.

## Repository Docker Runtime

The supported repository workflow is the Docker dev image, not an ad hoc TRT
11 Python venv. `Dockerfile.gb300` builds a CUDA 13 Ubuntu 24.04 image, creates
`/opt/venv`, installs the Python builder/test stack there, and includes Torch,
Transformers, HuggingFace tooling, diffusers, pytest, and the other packages
needed for Qwen reference and E2E tests. `scripts/docker_run_gb300.sh` mounts
the repo, a persistent engine directory, and the host HF cache into that
container.

For TRT 11 validation, the image must install or mount the TRT 11 package into
the container and point the build/runtime at that package. The current
`Dockerfile.gb300` installs TensorRT from the `tensorrt_cu13` wheels and even
creates a `libnvinfer.so` symlink to `libnvinfer.so.10`, so it is not enough by
itself for TRT 11 multi-device validation. The TRT 11 path should be handled by
either a TRT 11 Dockerfile variant or a mounted/extracted SDK inside the
container, with `TRT_LIB_DIR`, `TRT_INC_DIR`, and `LD_LIBRARY_PATH` adjusted
accordingly. NCCL 2.30 or newer also needs to be present in the image or mounted
into it before the TRT 11 libraries in `LD_LIBRARY_PATH`.

## Current Plan

1. Use the repo Docker dev workflow and install or expose TRT 11 from the
   internal package path inside that container. Do not use the throwaway TRT 11
   venv for repo E2E, accuracy, or performance tests.
2. Rebuild the project with TRT 11 and keep the standard single-device backend
   load path unchanged.
3. Run a small TRT 11 smoke:
  - Python TensorRT import/version check.
  - `./build/trtmc version`.
  - Build a tiny synthetic Qwen TP bundle with `tp_size=2` to validate TRT 11
  `add_dist_collective` graph construction.
  - Run that tiny bundle under `mpirun -np 2` once NCCL is available.
4. Keep the Qwen TP sharding rules validated through repo-owned bundle metadata,
  rank-local engine sections, and runtime parity artifacts.
5. Add a minimal TP metadata/build path for Qwen text models, then run TP=2,
  TP=4, and TP=8 with `mpirun`.
6. Extend the E2E pipeline with a gated multi-device lane that:
  - runs a default single-device Qwen baseline,
  - builds or reuses the TP bundle,
  - runs TP=2/4/8 via `mpirun`,
  - records accuracy, memory, and latency artifacts,
  - does not enter the normal single-device parallel E2E scheduler unless
  explicitly enabled.
7. Update `website/docs/reference/testing.md` and any relevant architecture docs
  with the multi-device testing flow and single-device guardrail after local
  smoke, accuracy, and performance validation are working.
8. Validate accuracy with existing logit/runner parity and record memory/perf
  deltas from `trtmc profile`, `tools/perf_compare.py`, and `nvidia-smi` where
   applicable.

## Future CI Enablement

The multi-device CI path is intentionally present but default-off. Current CI
runners are still single-device scheduled, so the normal selective and full E2E
lanes explicitly exclude manifests with `ci_tier="multi_device"`.

When CI runners are provisioned for multi-device testing, enable the lane with
repository or workflow environment variables:

```bash
TRTMC_MULTI_DEVICE_E2E=true
TRTMC_MULTI_DEVICE_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
TRTMC_NCCL_LIB_DIR=/path/to/nccl-2.30-or-newer/lib
MULTI_DEVICE_E2E_TIMEOUT=120m
```

The enabled workflow stage runs `.github/scripts/run-gha-stage.sh
multi-device-e2e`, which calls `.github/scripts/run-trtmc-ci.sh` and collects
all manifests marked `ci_tier="multi_device"`.

The runner requirements are:

- At least two visible NVIDIA GPUs for the TP=2 lane, four for TP=4, and eight
  for TP=8.
- `mpirun` available on `PATH`.
- TensorRT 11.0+ runtime libraries in `LD_LIBRARY_PATH`.
- NCCL 2.30 or newer first in `LD_LIBRARY_PATH`; the apt-provided NCCL 2.18 in
  this container does not export `ncclAlltoAll`, which TRT 11 requires.
- A HuggingFace cache or online access sufficient to build the Qwen checkpoint
  and run the HF/Torch reference.

Manual CI-equivalent invocation:

```bash
TRTMC_MULTI_DEVICE_E2E=true \
TRTMC_MULTI_DEVICE_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
TRTMC_NCCL_LIB_DIR=/opt/nccl-2.30/lib \
ENGINE_DIR=/tmp/trtmc-engines \
HF_PYTHON=/opt/venv/bin/python \
bash .github/scripts/run-trtmc-ci.sh multi-device-e2e
```

Until CI has runners with enough visible GPUs for the selected rank sizes, this
stage may still report skipped E2E cases from `gpu_count_min` preflight. Once
multi-GPU CI is available, make the lane fail if all selected multi-device cases
skip instead of treating that as coverage.

Keep this lane separate from `scripts/run_e2e_parallel.sh`: the existing
parallel E2E scheduler pins one worker per visible GPU, while TP needs one test
process to own multiple GPUs through `mpirun`.

## Full Qwen Accuracy/Performance E2E Status

The original full-Qwen blocker is resolved for TP=2, TP=4, and TP=8 on this
A100 x86 host. The repo Docker dev flow now has an A100 x86 TRT 11 wrapper
around the GB300 image stack:

- `scripts/docker_build_a100x86_trt11.sh` builds the GB300 dependency image
  with `/usr/include/x86_64-linux-gnu` and no GB300 cuBLAS preload.
- `Dockerfile.a100x86-trt11` adds the thin OpenMPI launcher layer.
- `scripts/docker_run_a100x86_trt11.sh` mounts the local TRT 11 SDK and NCCL
  2.30.4 extraction into `/opt/tensorrt-11` and `/opt/nccl-2.30`, then runs
  through `/opt/venv/bin/python`.
- `scripts/bootstrap_trt11_container.sh` installs the mounted TRT 11 Python
  wheels into `/opt/venv` and uses `PYTHONPATH` plus a temporary
  `trtmc-build` wrapper because editable install metadata writes fail on the
  mounted workspace.

Clean TP=8 build-and-run validation passed with:

```bash
sudo -n env \
  TRTMC_DOCKER_CONTAINER=trtmc-dev-a100x86-trt11-e2e-tp8 \
  TRTMC_STORAGE_ROOT=/tmp/trtmc-storage \
  TRTMC_HF_CACHE=/tmp/trtmc-hf-cache/hub \
  ./scripts/docker_run_a100x86_trt11.sh bash -lc '
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    python3 -m pytest tests/test_e2e.py::test_e2e[qwen3-0.6b-fp16-tp8] -v \
      --engine-dir /tmp/trtmc-storage/engines \
      --trtmc-binary /tmp/trtmc-storage/build-a100x86-trt11/trtmc \
      --hf-python /opt/venv/bin/python \
      --e2e-artifacts-dir /tmp/trtmc-storage/e2e-artifacts \
      --rebuild-engines'
```

Latest artifact results:

- Pytest: TP=2 cached rerun `1 passed, 1 warning in 44.73s`; TP=4 rebuild
  `1 passed, 1 warning in 247.87s`; TP=8 rebuild
  `1 passed, 1 warning in 418.92s`. The warning is only pytest cache
  write permission under the mounted workspace.
- Engine build: TP=2 `109.32s` total / `93.64s` TRT compile; TP=4 `194.38s`
  total / `178.14s` TRT compile; TP=8 `355.01s` total / `340.60s` TRT compile.
- Runtime: TP=2 C++ distributed stage returned `0` in `6.98s`; TP=4 returned
  `0` in `9.18s`; TP=8 returned `0` in `14.73s`.
- GPU memory capture: per-case `gpu_memory_samples.csv` files are written.
  Peak visible memory sums were TP=2 `4220 MiB`, TP=4 `7989 MiB`, and TP=8
  `15473 MiB`.
- Accuracy: all TP lanes reported normalized text edit distance `0.0`; TRT
  rank-0 text was
  `The capital of France is Paris.\nAnswer:\nParis`, matching the HF reference
  after normalization.
- Post-review cached accuracy reruns after the final guard/comment cleanup:
  TP=2 passed in `56.70s` with `logit_cosine_p5=0.999951`,
  `logit_rel_l2_p95=0.014935`, `token_agreement_rate=1.0`, and normalized
  text edit distance `0.0`; TP=4 passed in `61.33s` with
  `logit_cosine_p5=0.999958`, `logit_rel_l2_p95=0.008944`,
  `token_agreement_rate=1.0`, and normalized text edit distance `0.0`; TP=8
  passed in `75.23s` with `logit_cosine_p5=0.999951`,
  `logit_rel_l2_p95=0.010559`, `token_agreement_rate=1.0`, and normalized text
  edit distance `0.0`.

No parity gap remains. Distributed rank-0 debug logits are available and have
passed cached TP=2, TP=4, and TP=8 E2E parity. Broader performance comparison
beyond the single-prompt smoke recorded in `result.json` is optional follow-up
characterization, not an accuracy blocker.

## Compatibility Guardrail

Multi-device support must be additive and gated. Existing single-device bundle
builds, bundle metadata, runtime loading, CLI defaults, and E2E manifests should
continue to behave exactly as they do today unless a caller explicitly requests
tensor parallel or multi-device execution.

Validation will include single-device baseline checks before and after any TP
changes:

- Build/run `flux-schnell-l0` with the default single-device path.
- Run the existing diffusion health/component checks for the default path.
- Build/run Qwen with the default single-device path.
- Repeat the same single-device checks after TP changes are present.
- Only then run TP=2/4/8 with `mpirun`.

## Status

- Branch: `trt11-qwen-tp`.
- Sandbox note: local command sandboxing fails during `bwrap` loopback setup in
this container, so repo inspection commands are being run with approved
escalation.
- TRT 11 install: completed locally under
`/tmp/trt11-install/external/TensorRT-11.0.0.103`.
- TRT 11 Python package: installed into `/tmp/trt11-install/venv` only for an
  early API smoke. That venv is not the repo-supported path for Qwen E2E,
  accuracy, or performance validation.
- Repository runtime correction: inspected `Dockerfile.gb300`,
  `scripts/docker_run_gb300.sh`, and the installation docs. Full validation now
  uses the Docker dev image path and `/opt/venv/bin/python`, with an A100 x86
  TRT11 wrapper around the GB300 dependency image.
- Current shell environment: host is x86_64 with eight
  `NVIDIA A100-SXM4-40GB` GPUs. The user is not in the `docker` group, but
  `sudo -n docker ...` works, so the A100 x86 Docker validation is no longer
  blocked.
- TRT 11 smoke: Python import/builder smoke passed
(`tensorrt.__version__ == 11.0.0.103`, minimal serialized engine = 2556 bytes).
- TRT 11.0+ API check: Python exposes
  `INetworkDefinition.add_dist_collective`; C++ runtime exposes
  `IExecutionContext::setCommunicator(void*)`.
- Build tools: installed `cmake`, `ninja`, `mpirun`, `python3-pip`, NCCL
runtime/dev packages, and CUDA runtime headers/libs from the available apt
repos.
- NCCL compatibility: the apt-provided NCCL 2.18 package does not export
  `ncclAlltoAll`, which TRT 11 resolves during `setCommunicator`. A matching
  NCCL 2.30.4 CUDA 13.2/r595 package was extracted locally under
  `/tmp/trt11-install/external/nccl_2.30.4-1+cuda13.2_x86_64` and is required
  first in `LD_LIBRARY_PATH` for multi-device TRT 11 runtime smoke tests.
- CMake configure: passed for `build-trt11` with TRT 11 SDK and standard TRT
backend DSO enabled. CUDA kernels are disabled because no `nvcc` is installed.
- Project build: `cmake --build build-trt11 -j` passed against TRT 11.
- Project smoke: `./build-trt11/trtmc version` passed with TRT support enabled.
- C++ compatibility patch: `src/runtime/core/trt_common.cpp` now supports CUDA
  11.x and CUDA 12.x `cudaGraphInstantiate` signatures.
- Qwen TP build path: local TP=2 smoke passed. The Python builder creates
  rank-local Qwen engine sections (`engine_plan_tp_rankN`) when
  `--set parallel.mode=tensor_parallel --set parallel.tp_size=N` is used.
  TRT 11 `IDistCollectiveLayer.num_ranks` must be set to `tp_size`; without it,
  TRT builds ordinary single-device engines and rejects `setCommunicator`.
- Runtime path: local TP=2 smoke passed. The decoder runtime selects
  rank-local engine sections under `mpirun`, initializes NCCL through a
  runtime-loaded `libnccl`, and attaches the communicator to TRT 11 execution
  contexts.
- Runtime shutdown fix: the NCCL communicator owner now lives on the text
  pipeline and is destroyed after the TRT modules, matching the ordering in
  TensorRT's `sampleDistCollective`. Explicit `ncclCommDestroy` is the default
  path again; `TRTMC_NCCL_SKIP_DESTROY=1` remains only as an escape hatch.
- Python debug parity path: the debug runner can now open rank-local TP engine
  sections (`engine_plan_tp_rankN`), initialize NCCL from Python through
  `ctypes`, attach the communicator with TensorRT 11
  `IExecutionContext.set_communicator`, and write rank-0 per-step logits for
  the existing text comparator. The default single-device
  `runner_from_bundle(bundle_path)` path remains unchanged.
- Tiny Qwen TP bundle smoke: passed for build/packaging with TP=2:
  `/tmp/trtmc-tiny-qwen-tp2.trtfb` contains `engine_plan_tp_rank0` and
  `engine_plan_tp_rank1`.
- Tiny Qwen TP runtime smoke: passed with
  `mpirun --tag-output -np 2` using GPUs 0 and 1 plus NCCL 2.30.4. Both ranks
  loaded their rank-local engine sections and reported
  `backend=trt_new_runtime`.
- Tiny Qwen single-device guardrail: passed. A default build produced the
  normal `engine_plan` section and the C++ runtime loaded it without trying to
  initialize distributed runtime.
- E2E/CI pipeline extension: added a gated `ci_tier="multi_device"` Qwen TP
  manifest, manifest-driven `--set parallel.*` build config propagation,
  `mpirun` wrapping in the text-generation E2E runner, rank-0 stdout selection,
  MPI-tagged timing parsing, and a default-off `multi-device-e2e` CI stage.
  Normal selective/full E2E lanes explicitly exclude `ci_tier="multi_device"`.
- A100 x86 Docker continuation: added an x86/TRT11 wrapper path around the
  GB300 image stack. The GB300 Dockerfile now accepts build args for the
  TensorRT include directory and cuBLAS preload, while
  `scripts/docker_build_a100x86_trt11.sh` builds the same dependency image with
  `/usr/include/x86_64-linux-gnu` and no GB300 cuBLAS preload.
  `scripts/docker_run_a100x86_trt11.sh` mounts the local TRT 11 SDK and NCCL
  2.30.4 extraction into `/opt/tensorrt-11` and `/opt/nccl-2.30`, overrides
  CMake/runtime env vars, and bootstraps the TRT 11 Python wheels into
  `/opt/venv` before entering the container command. The bootstrap uses
  `PYTHONPATH` plus a temporary `trtmc-build` wrapper instead of `pip install
  -e`, because the mounted workspace rejects root-owned `egg-info` timestamp
  writes under Docker on this host.
- The x86/TRT11 Docker image now has a thin final layer,
  `Dockerfile.a100x86-trt11`, that installs `openmpi-bin` for the TP launcher
  and sets the OpenMPI root-allow env used by the root container workflow.
- Validation results:
  - Python compile check passed for modified builder modules.
  - Focused Python tests passed:
    `tests/builder/test_parallel_config.py`,
    `tests/builder/test_family_plugins.py::TestQwenPlugin::test_tensor_parallel_shards_qwen_projection_weights`,
    `tests/tools/test_multi_device_e2e.py`, and
    `tests/tools/test_github_actions_ci.py`.
  - C++ build passed after runtime communicator changes.
  - `trtmc-build inspect --list-engines` now lists TP rank engine sections.
  - Direct A100 x86 TRT11 Docker TP=2 `mpirun` smoke passed with both ranks
    producing the expected Paris answer and exiting cleanly.
  - Full Qwen TP=2 E2E with `--rebuild-engines` passed inside
    `trtmc-dev-a100x86-trt11`: normalized text edit distance `0.0`, C++
    distributed return code `0`, engine build `109.32s`.
  - Full Qwen TP=4 E2E with `--rebuild-engines` passed inside
    `trtmc-dev-a100x86-trt11`: `1 passed, 1 warning in 247.87s`, normalized
    text edit distance `0.0`, C++ distributed return code `0`, engine build
    `194.38s`, distributed C++ stage `9.18s`, peak visible memory sum
    `7989 MiB`.
  - Full Qwen TP=8 E2E with `--rebuild-engines` passed inside
    `trtmc-dev-a100x86-trt11`: `1 passed, 1 warning in 418.92s`, normalized
    text edit distance `0.0`, C++ distributed return code `0`, engine build
    `355.01s`, distributed C++ stage `14.73s`, peak visible memory sum
    `15473 MiB`.
  - Cached Full Qwen TP=2/TP=4/TP=8 E2E with distributed debug logits passed
    inside `trtmc-dev-a100x86-trt11`. The comparator used logit metrics instead
    of text-only fallback in all three lanes. TP=2 passed in `66.30s` with
    `logit_cosine_p5=0.999951`, `logit_rel_l2_p95=0.014935`,
    `token_agreement_rate=1.0`, and normalized text edit distance `0.0`.
    TP=4 passed in `61.36s` with `logit_cosine_p5=0.999958`,
    `logit_rel_l2_p95=0.008944`, `token_agreement_rate=1.0`, and normalized
    text edit distance `0.0`. TP=8 passed in `75.20s` with
    `logit_cosine_p5=0.999951`, `logit_rel_l2_p95=0.010559`,
    `token_agreement_rate=1.0`, and normalized text edit distance `0.0`.

## Open Questions

- Need add fuller perf comparison around the multi-device E2E command beyond
  the single prompt recorded in `result.json`.

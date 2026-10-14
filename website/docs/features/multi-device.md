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

## Native Windows

NVIDIA does not publish NCCL binaries for Windows, so you build `nccl.dll`
from the upstream source. Requirements:

- Two or more GPUs in TCC mode (`nvidia-smi -g <i> -dm 1` from an elevated
  prompt). Under MCDM or WDDM, the GPUs report no peer access and NCCL has no
  working transport.
- TensorRT 11.4 or newer, or TensorRT-RTX 1.7.1 or newer. Older Windows
  releases do not load `nccl.dll`.
- Visual Studio 2022 (MSVC v143), CMake 3.25 or newer, Ninja, Git, Python 3,
  and a CUDA 13.x toolkit. CUDA 12.9 also works; see the note below.

Build it in PowerShell. Set `CMAKE_CUDA_ARCHITECTURES` to your GPUs' compute
capabilities, for example `120` for RTX PRO 6000 Blackwell or
`86-real;89-real;120-real` for a mix. If you leave it out, you get nvcc's default
architecture instead of your GPUs'.

```powershell
git clone --branch v2.32.3-1 https://github.com/NVIDIA/nccl.git C:\nccl\src
& "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64 -HostArch amd64 -SkipAutomaticLocation
$cuda = $env:CUDA_PATH_V13_3 -replace '\\', '/'
cmake -S C:\nccl\src -B C:\nccl\build -G Ninja -DCMAKE_BUILD_TYPE=Release `
  "-DCMAKE_CUDA_COMPILER=$cuda/bin/nvcc.exe" "-DCUDAToolkit_ROOT=$cuda" `
  "-DCMAKE_CUDA_ARCHITECTURES=120" -DCMAKE_INSTALL_PREFIX=C:\nccl\install
cmake --build C:\nccl\build --parallel
cmake --install C:\nccl\build
```

With CUDA 12.9, NCCL v2.32.3-1 needs two extra configure settings. Its CUDA 12
build is C++14, which MSVC rejects in `sym_kernels.h`. And it links
`cudart64_12.dll` dynamically, which no TensorRT package ships. Write a CMake
include that switches the build to C++17, and link the static runtime:

```powershell
@'
set(CMAKE_CUDA14_STANDARD_COMPILE_OPTION "-std=c++17")
set(CMAKE_CUDA14_EXTENSION_COMPILE_OPTION "-std=c++17")
set(CMAKE_CXX14_STANDARD_COMPILE_OPTION "-std:c++17")
set(CMAKE_CXX14_EXTENSION_COMPILE_OPTION "-std:c++17")
'@ | Set-Content -Encoding ascii C:\nccl\cuda12_windows.cmake
# add to the cmake configure line above:
#   "-DCUDA_cudart_LIBRARY=$cuda/lib/x64/cudart_static.lib"
#   -DCMAKE_PROJECT_NCCL_INCLUDE=C:/nccl/cuda12_windows.cmake
```

Point the runtime at the result. Either set `TRTMC_NCCL_LIBRARY` to
`C:\nccl\install\bin\nccl.dll` or put `C:\nccl\install\bin` first on `PATH`.
TensorRT loads `nccl.dll` by name and gets the module that TRTMC already
loaded. Native Windows has no `mpirun`, so `tools/launch_ranks.py` starts the
ranks and sets the same rank and rendezvous variables:

```powershell
python tools\launch_ranks.py -n 2 --gpus 0,1 `
  --nccl-library C:\nccl\install\bin\nccl.dll `
  --library-dir <TensorRT bin directory> `
  -- trtmc generate-video model-cp2.bundle --runtime-root <trtmc runtime root> ...
```

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

---
title: Prerequisites and Environment
description: Select and verify the environment before installing TensorRT-Model-Connect.
---

Complete this page before [Installation](installation.md). It helps you select
an install path; it does not install the project or build a model.

## 1. Choose an environment

TensorRT-Model-Connect is not a CPU-only Python package. Building and running a
bundle requires a compatible NVIDIA GPU software stack.

| Path | Use it when | Current boundary |
| --- | --- | --- |
| Published aarch64 wheel | You have a matching Linux aarch64 host and the release artifacts. | Python 3.10 or 3.12, glibc 2.39 or newer, and the matching official TensorRT 11.1.0.106 cohort. |
| Repository dev container | You are developing from source in the repository's GB300/aarch64 environment. | Linux host, NVIDIA driver, Docker, NVIDIA Container Toolkit, and enough access to build the image. |

Do not combine binaries, bundles, or TensorRT libraries from different
cohorts. A bundle contains TensorRT compatibility metadata, but it does not
contain the host driver, CUDA runtime, dynamic loader, or every system library.

:::info Why x86_64 qualification is not a newcomer path

The retained x86_64 profile metadata covers exact Qwen revisions, A100 SM80,
FP16, an external Edge-LLM implementation, and profile-specific options. The
public source tree publishes the pinned TensorRT-Edge-LLM dependency lock and
profile metadata, but it does not vendor that dependency or publish the former
target-hardware runner and qualification artifacts. This is not the native
BF16/full-context command used by the site's first-inference path, and it does
not establish a generally downloadable x86_64 wheel. See the [advanced profile
boundary](installation.md#x86_64-optimized-profiles) only when reviewing that
exact tuple.

:::

## 2. Check the host

On the host, record:

```bash
uname -m
getconf GNU_LIBC_VERSION
python3 --version
nvidia-smi
docker --version
```

Expected signals depend on the path you selected:

- The architecture printed by `uname -m` must match the aarch64 wheel or
  repository-container path used by this newcomer flow.
- The published wheel path requires glibc 2.39 or newer; for example,
  `getconf GNU_LIBC_VERSION` must report `glibc 2.39` or a later version.
- Python must be 3.10 or 3.12 for the documented wheel paths.
- `nvidia-smi` must show the target GPU and a working driver.
- Docker is required for the repository dev-container path, not for an already
  installed wheel.

For the container path, the NVIDIA Container Toolkit must also allow a
container to see the GPU. If the repository container launches but
`nvidia-smi` fails inside it, fix the host/container boundary before building
TensorRT-Model-Connect.

## 3. Prepare the source container

Skip this section when you are installing a published wheel.

Clone the repository, then run the repository scripts from its root:

```bash
git clone https://github.com/NVIDIA/TensorRT-Model-Connect.git
cd TensorRT-Model-Connect

./scripts/docker_build_gb300.sh
./scripts/docker_run_gb300.sh
```

`docker_run_gb300.sh` starts an interactive shell in the container with the
repository mounted at `/workspace/tensorrt-model-connect`, the Hugging Face
cache mounted from the host, and GPU access requested through Docker. Keep that
shell open for the source-install and Quick Start commands.
In parallel agent workspaces, the matching container may be named
`trtmc-dev-gb300-agent-N` instead of `trtmc-dev-gb300`.

:::warning Host versus container
Source-build commands in this site assume you are inside that container. A
source-built `./build/trtmc` can fail on the host with a missing shared library
even when it works in the development environment.
:::

## 4. Budget for the first model

The Quick Start builds:

```text
Qwen/Qwen3-0.6B
```

The first build may download checkpoint files and compile multiple TensorRT
plans. It needs network access or a populated Hugging Face cache, write access
for the bundle, and substantially more time than normal application startup.

The default native path uses the checkpoint's complete 40,960-token context.
Its fixed BF16 KV allocation is 4.375 GiB by itself. This number excludes model
weights, TensorRT plans, build workspace, runtime allocations, and host-side
cache files; do not use it as a total-memory estimate.

## 5. Know the failure boundaries

| Symptom | Likely boundary | Next check |
| --- | --- | --- |
| `nvidia-smi` fails | Host driver or GPU access | Fix the host before starting installation. |
| Docker cannot see the GPU | NVIDIA Container Toolkit | Verify the Docker GPU runtime configuration. |
| Wheel architecture is incompatible | Install-path selection | Match Python, architecture, glibc, and TensorRT cohort. |
| Hugging Face 401/403/not found | Model resolution | Check model ID, network, auth, and gated-model access. |
| CMake cannot find CUDA or TensorRT | Native build environment | Use the repository container or provide the matching development files explicitly. |
| `libtorch.so` or another DSO is missing | Runtime library environment | Run in the installed/container environment instead of moving one executable by itself. |
| TensorRT ABI mismatch | Bundle/runtime cohort | Build and run with compatible TensorRT environments. |

Continue to [Installation](installation.md). Return here instead of changing
model flags when the failure is an environment mismatch.

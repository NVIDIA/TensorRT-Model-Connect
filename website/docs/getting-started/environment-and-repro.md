---
title: System Requirements
description: Verify the host before installing TensorRT-Model-Connect.
---

TensorRT-Model-Connect requires a compatible NVIDIA GPU software stack. Choose
one installation path:

| Path | Current boundary |
| --- | --- |
| Release wheel | Linux aarch64, Python 3.10 or 3.12, glibc 2.39 or newer, and official TensorRT 11.1.0.106. |
| Source build | Linux x86_64 or aarch64 with Docker, NVIDIA Container Toolkit, and enough disk space for the image and bundle. |

x86_64 release wheels are not published yet. x86_64 users should use the
source-build path.

## Check the host

All users:

```bash
uname -m
nvidia-smi
```

Source users also need Docker:

```bash
docker --version
```

Wheel users following the first-time Python 3.12 path instead check:

```bash
getconf GNU_LIBC_VERSION
python3.12 --version
```

Confirm that:

- the host architecture matches the selected path;
- the NVIDIA driver sees the target GPU;
- Docker and NVIDIA Container Toolkit are available for a source build; and
- the wheel's Python and glibc requirements are met for a wheel install.

Do not mix binaries, bundles, or TensorRT libraries from different TensorRT
cohorts.

Source users should continue to [Build from Source](source-build.md), where the
target GPU and compute capability are selected once. Wheel users should
continue to [Installation](installation.md).

## Common boundaries

| Symptom | Check first |
| --- | --- |
| `nvidia-smi` fails | Host driver or GPU access. |
| Docker cannot see the GPU | NVIDIA Container Toolkit configuration. |
| Wheel is incompatible | Architecture, Python, glibc, and TensorRT cohort. |
| Hugging Face returns 401/403/not found | Model ID, network, authentication, and gated access. |
| CMake cannot find CUDA or TensorRT | Use the repository source container. |
| TensorRT reports an ABI mismatch | Build and run in a compatible TensorRT cohort. |

{/* Collaborative review anchor: batch 2. */}

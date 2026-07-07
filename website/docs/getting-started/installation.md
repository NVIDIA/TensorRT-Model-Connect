---
title: Installation
---

TensorRT-Model-Connect has three install paths:

1. Simple pip install from a published wheel.
2. Build the wheel from source, then install it.
3. Advanced Python-only editable install for developers.

## Requirements

- Linux aarch64 with an NVIDIA GPU.
- Python 3.10 or Python 3.12.
- NVIDIA driver and CUDA runtime libraries.
- glibc 2.39 or newer for the published `manylinux_2_39_aarch64` wheels.

Building from source also needs the repository dev container or an equivalent
CUDA/TensorRT build environment with CMake, Ninja, Conan, CUDA headers and
libraries, TensorRT headers and libraries, `patchelf`, and `auditwheel`.

Building the dev container requires read access to the repository-linked,
access-controlled TensorRT SDK image in GitHub Container Registry. Authenticate
Docker with a GitHub token that has `read:packages` access before the first build:

```bash
gh auth token | docker login ghcr.io \
  --username "$(gh api user --jq .login)" --password-stdin
```

GitHub Actions uses its scoped `GITHUB_TOKEN`; no NVIDIA Artifactory credential
is stored in GitHub.

Maintainers publishing a manually selected TensorRT build use:

```bash
TRTMC_ARTIFACTORY_CREDENTIAL_FILE=/secure/path/to/artifactory-credentials \
  ./scripts/publish_tensorrt_sdk.sh
```

The credential file contains the Artifactory username and password on separate
lines. The publishing script downloads and verifies the pinned SDK, builds the
GHCR image, and pushes the versioned tag. Copy the digest printed by the script
into `TENSORRT_SDK_IMAGE` in the main `Dockerfile` as part of the version bump.
Do not add the credential file to the repository or a Docker build context.

## 1. Simple pip install

Install a published wheel that matches your Python version:

```bash
python3.12 -m venv .venv-trtmc
. .venv-trtmc/bin/activate

pip install ./tensorrt_model_connect-0.1.0-py312-none-manylinux_2_39_aarch64.whl

trtmc version
trtmc build --help
```

Use the `py310` wheel with Python 3.10 and the `py312` wheel with Python 3.12.
The wheel installs:

- the Python builder package,
- the native `trtmc` executable,
- packaged TensorRT backend DSOs,
- the pinned `tensorrt==11.2.0.113` Python dependency.

TensorRT 11.2.0.113 is a nightly SDK rather than a public PyPI release. The
repository dev container installs its matching Python wheel under
`/opt/tensorrt/python` and configures that directory as a pip package source.
Outside the dev container, install the matching TensorRT wheel from the SDK
before installing TensorRT-Model-Connect.

Quick smoke test:

```bash
trtmc build Qwen/Qwen3-0.6B -o /tmp/qwen3.trtfb --max-cache-length 256
trtmc run /tmp/qwen3.trtfb \
  --prompt "The capital of France is" \
  --max-new-tokens 20 \
  --greedy
```

## 2. Build wheel from source

Use this path when you need the same pip-installable artifact that CI and
nightly releases produce.

```bash
git clone https://github.com/NVIDIA/TensorRT-Model-Connect.git
cd TensorRT-Model-Connect

./scripts/docker_build_gb300.sh
./scripts/docker_run_gb300.sh
```

Inside the dev container, build one Python 3.12 wheel:

```bash
python -m pip install --upgrade build auditwheel
rm -rf dist /tmp/trtmc-conan-py-wheel-py312

CONAN_PY_BUILD_PROFILE_AUTODETECT=1 \
TRTMC_TRT_INCLUDE_DIR="${TRT_INC_DIR:-/opt/tensorrt/include}" \
TRTMC_TRT_LIBRARY="${TRT_LIB_DIR:-/opt/tensorrt/lib}/libnvinfer.so" \
TRTMC_CUDA_INCLUDE_DIR=/usr/local/cuda/include \
TRTMC_CUDART_LIBRARY=/usr/local/cuda/lib64/libcudart.so \
WHEEL_PYVER=py312 \
WHEEL_ABI=none \
WHEEL_ARCH=manylinux_2_39_aarch64 \
python -m build --wheel --outdir "$PWD/dist" \
  -C build-dir=/tmp/trtmc-conan-py-wheel-py312 \
  .

python -m auditwheel show dist/tensorrt_model_connect-*-py312-none-manylinux_2_39_aarch64.whl
```

For Python 3.10, use `WHEEL_PYVER=py310` and a matching Python 3.10 build
environment. Install the built wheel in a fresh environment:

```bash
python3.12 -m venv /tmp/trtmc-wheel-smoke
/tmp/trtmc-wheel-smoke/bin/python -m pip install --upgrade pip
/tmp/trtmc-wheel-smoke/bin/python -m pip install \
  dist/tensorrt_model_connect-0.1.0-py312-none-manylinux_2_39_aarch64.whl
/tmp/trtmc-wheel-smoke/bin/trtmc version
```

Run wheel builds from the repository root. Do not point `python -m build` at a
package subdirectory.

## 3. Advanced editable Python-only install

Use this only for local development of the Python builder. It does not run
Conan, does not run CMake, and does not install the native `trtmc` executable or
backend DSOs.

```bash
git clone https://github.com/NVIDIA/TensorRT-Model-Connect.git
cd TensorRT-Model-Connect

./scripts/docker_build_gb300.sh
./scripts/docker_run_gb300.sh
```

Inside the dev container:

```bash
pip install -e . -C py-only=true
```

If the container already has the declared dependencies installed and you want
to avoid dependency resolution:

```bash
pip install --no-deps -e . -C py-only=true
```

This editable install points Python at `python/tensorrt_model_connect/`. Use it
with a separate source build when you need the native CLI:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DTRTMC_TRT_INCLUDE_DIR="${TRT_INC_DIR:-/opt/tensorrt/include}" \
  -DTRTMC_TRT_LIBRARY="${TRT_LIB_DIR:-/opt/tensorrt/lib}/libnvinfer.so" \
  -DTRTMC_CUDA_INCLUDE_DIR=/usr/local/cuda/include \
  -DTRTMC_CUDART_LIBRARY=/usr/local/cuda/lib64/libcudart.so
cmake --build build -j

./build/trtmc version
```

CI and release validation use the wheel path, not the Python-only editable
install.

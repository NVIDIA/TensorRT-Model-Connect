---
title: Installation
---

TensorRT-Model-Connect has three install paths:

1. Install an official release wheel when one is published.
2. Build the wheel from source, then install it.
3. Advanced Python-only editable install for developers.

## Requirements

- Linux x86_64 or aarch64 with a compatible NVIDIA GPU for the selected build
  or qualified profile.
- Python 3.10 or Python 3.12.
- NVIDIA driver and CUDA runtime libraries.
- glibc 2.39 or newer for a `manylinux_2_39_*` wheel.

The artifact and TensorRT cohort are architecture-specific:

| Path | Current boundary |
| --- | --- |
| Release/source-built wheel examples below | Linux aarch64, `manylinux_2_39_aarch64`, official TensorRT 11.1.0.106. |
| Qualified Qwen x Edge-LLM profiles | Linux x86_64, A100 PCIe 80 GB (SM80), FP16, TensorRT 11.1.0.106, and exact pinned model revisions/options. Source does not publish the former target-hardware runner or its artifacts. |
| Python-only editable install | Installs no native CLI or backend DSO; its Python dependencies select official TensorRT 11.1.0.106 on aarch64 and x86_64. |

Building from source also needs the repository dev container or an equivalent
CUDA/TensorRT build environment with CMake, Ninja, Conan, CUDA headers and
libraries, TensorRT headers and libraries, `patchelf`, and `auditwheel`.

Do not treat a profile marked `qualified` as a promise that a matching public
wheel or optimized-runtime dependency is downloadable. Artifact
availability and exact-profile qualification are separate evidence.

## 1. Install an official aarch64 release wheel

When an official release provides an aarch64 TensorRT-Model-Connect wheel,
download the wheel matching your Python version. Pip resolves its pinned
TensorRT dependency from NVIDIA's official TensorRT 11.1.0.106 distribution:

```bash
python3.12 -m venv .venv-trtmc
. .venv-trtmc/bin/activate

pip install ./tensorrt_model_connect-0.1.0-py312-none-manylinux_2_39_aarch64.whl

trtmc version
trtmc build --help
```

To create Wan2.2 bundles, install the same wheel with its build-only extra:

```bash
pip install './tensorrt_model_connect-0.1.0-py312-none-manylinux_2_39_aarch64.whl[wan]'
```

PyTorch reads the official checkpoint during `trtmc build`; it is not used by
the native C++ video-generation runtime.

Use the TensorRT-Model-Connect `py310` wheel with Python 3.10 and the `py312`
wheel with Python 3.12. The TensorRT-Model-Connect wheel installs:

- the Python builder package,
- the native `trtmc` executable,
- packaged TensorRT backend DSOs,
- the pinned official `tensorrt==11.1.0.106` Python dependency.

Installation is complete when `trtmc version` and `trtmc build --help`
succeed. Continue to
[Your First NLP Inference](quick-start.md) for the single canonical
build-inspect-run smoke test; it is intentionally not duplicated here.

### x86_64 optimized profiles

The three Qwen x Edge-LLM profile TOMLs describe exact qualified tuples, not a
general x86_64 install promise. This repository does not publish an A100
qualification runner or hardware-dispatch route. Do not infer support for a
different x86_64 GPU, model revision, precision, or engine configuration.

## 2. Build the aarch64 wheel from source

Use this path when you need the same pip-installable artifact that release
validation produces. The commands below are the current GB300/aarch64 release
path.

Continue in the repository-root dev-container shell opened by
[System Requirements](environment-and-repro.md#3-prepare-the-source-container).
Do not clone again or try to start Docker from inside that container. Build one
Python 3.12 wheel there:

```bash
python -m pip install --upgrade build auditwheel
rm -rf dist /tmp/trtmc-conan-py-wheel-py312

CONAN_PY_BUILD_PROFILE_AUTODETECT=1 \
TRTMC_TRT_INCLUDE_DIR="${TRT_INC_DIR:-/usr/include/aarch64-linux-gnu}" \
TRTMC_TRT_LIBRARY="${TRT_LIB_DIR:-/usr/lib/aarch64-linux-gnu}/libnvinfer.so" \
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
. /tmp/trtmc-wheel-smoke/bin/activate
trtmc version
```

Keep this virtual environment active while following
[Your First NLP Inference](quick-start.md), where `TRTMC=trtmc` will now refer
to this installed wheel.

Run wheel builds from the repository root. Do not point `python -m build` at a
package subdirectory.

## 3. Advanced editable Python-only install

Use this only for local development of the Python builder. It does not run
Conan, does not run CMake, and does not install the native `trtmc` executable or
backend DSOs. The container commands below use the current GB300/aarch64
development environment; an x86_64 environment must provide its matching
TensorRT/CUDA cohort.

Continue in the same repository-root dev-container shell prepared on the
previous page:

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
  -DTRTMC_TRT_INCLUDE_DIR="${TRT_INC_DIR:-/usr/include/aarch64-linux-gnu}" \
  -DTRTMC_TRT_LIBRARY="${TRT_LIB_DIR:-/usr/lib/aarch64-linux-gnu}/libnvinfer.so" \
  -DTRTMC_CUDA_INCLUDE_DIR=/usr/local/cuda/include \
  -DTRTMC_CUDART_LIBRARY=/usr/local/cuda/lib64/libcudart.so
cmake --build build -j

./build/trtmc version
```

CI and release validation use the wheel path, not the Python-only editable
install.

{/* Collaborative review anchor. */}

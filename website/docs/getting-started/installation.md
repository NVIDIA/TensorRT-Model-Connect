---
title: Installation
---

TensorRT-Model-Connect has three install paths:

1. Simple pip install from a published wheel.
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
| Published/nightly wheel examples below | Linux aarch64, `manylinux_2_39_aarch64`, TensorRT 11.2.0.113. |
| Qualified Qwen x Edge-LLM profiles | Linux x86_64, A100 PCIe 80 GB (SM80), FP16, TensorRT 11.1.0.106, and exact pinned model revisions/options. The qualification runner builds its own `manylinux_2_39_x86_64` wheel from the tested source. |
| Python-only editable install | Installs no native CLI or backend DSO; its Python dependencies still select TensorRT 11.2 on aarch64 and TensorRT 11.1 on x86_64. |

Building from source also needs the repository dev container or an equivalent
CUDA/TensorRT build environment with CMake, Ninja, Conan, CUDA headers and
libraries, TensorRT headers and libraries, `patchelf`, and `auditwheel`.

Do not treat a profile marked `qualified` as a promise that a matching public
wheel or private optimized-runtime dependency is downloadable. Artifact
availability and exact-profile qualification are separate evidence.

## 1. Simple pip install for the published aarch64 wheel

Install the TensorRT 11.2 Python wheel from the TensorRT 11.2.0.113 SDK, then
install the published aarch64 TensorRT-Model-Connect wheel that matches your
Python version:

```bash
python3.12 -m venv .venv-trtmc
. .venv-trtmc/bin/activate

pip install ./tensorrt-11.2.0.113-cp312-none-linux_aarch64.whl
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

Use the TensorRT `cp310` and TensorRT-Model-Connect `py310` wheels with Python
3.10. Use the `cp312` and `py312` wheels with Python 3.12. The
TensorRT-Model-Connect wheel installs:

- the Python builder package,
- the native `trtmc` executable,
- packaged TensorRT backend DSOs,
- the pinned `tensorrt==11.2.0.113` Python dependency.

Quick smoke test:

```bash
trtmc build Qwen/Qwen3-0.6B
trtmc run Qwen3-0.6B.trtfb \
  --prompt "The capital of France is" \
  --max-new-tokens 20 \
  --greedy
```

### x86_64 optimized qualification path

The three current Qwen x Edge-LLM profiles are not a general x86_64 install
promise. Their family-owned profile TOMLs require exact Qwen revisions, A100
SM80, FP16, cache/batch options, and the pinned private Edge-LLM dependency.
The model-owned
`tests/e2e/models/qwen/edge_llm_adapter/run_a100_ci.sh` qualification runner
builds and audits a `manylinux_2_39_x86_64` wheel with TensorRT 11.1.0.106,
installs it in a clean environment, and runs the matching profile proof.

Use the optimized-runtime proof workflow and its retained artifacts for that
path. Do not substitute the aarch64 wheels above or infer support for another
x86_64 GPU, model revision, precision, or engine configuration.

## 2. Build the aarch64 wheel from source

Use this path when you need the same pip-installable artifact that CI and
nightly releases produce. The commands below are the current GB300/aarch64
release path; the controlled x86_64 qualification path is described above.

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
TRTMC_TRT_INCLUDE_DIR="${TRT_INC_DIR:-/usr/include/aarch64-linux-gnu}" \
TRTMC_TRT_LIBRARY="${TRT_LIB_DIR:-/opt/venv/lib/python3.12/site-packages/tensorrt_libs}/libnvinfer.so" \
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
  /opt/tensorrt/python/tensorrt-11.2.0.113-cp312-none-linux_aarch64.whl
/tmp/trtmc-wheel-smoke/bin/python -m pip install \
  dist/tensorrt_model_connect-0.1.0-py312-none-manylinux_2_39_aarch64.whl
/tmp/trtmc-wheel-smoke/bin/trtmc version
```

Run wheel builds from the repository root. Do not point `python -m build` at a
package subdirectory.

## 3. Advanced editable Python-only install

Use this only for local development of the Python builder. It does not run
Conan, does not run CMake, and does not install the native `trtmc` executable or
backend DSOs. The container commands below use the current GB300/aarch64
development environment; an x86_64 environment must provide its matching
TensorRT/CUDA cohort.

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
  -DTRTMC_TRT_INCLUDE_DIR="${TRT_INC_DIR:-/usr/include/aarch64-linux-gnu}" \
  -DTRTMC_TRT_LIBRARY="${TRT_LIB_DIR:-/opt/venv/lib/python3.12/site-packages/tensorrt_libs}/libnvinfer.so" \
  -DTRTMC_CUDA_INCLUDE_DIR=/usr/local/cuda/include \
  -DTRTMC_CUDART_LIBRARY=/usr/local/cuda/lib64/libcudart.so
cmake --build build -j

./build/trtmc version
```

CI and release validation use the wheel path, not the Python-only editable
install.

---
title: Installation
---

TensorRT-Model-Connect can be installed in two ways:

- Simple pip install from a release wheel.
- Source install from the repository.

## Host requirements

- Linux with an NVIDIA GPU.
- A compatible NVIDIA driver and CUDA runtime libraries.
- Linux aarch64 for the published wheels.
- Docker and NVIDIA Container Toolkit for the source workflow.
- CUDA development files, TensorRT headers, and TensorRT libraries when
  building from source.
- Optional TensorRT-RTX SDK when building the RTX backend DSO from source.

Use one of the two paths below.

## Simple pip install

Nightly GitHub Releases publish Linux aarch64 wheels for Python 3.10 and
Python 3.12. Download the wheel asset that matches your interpreter tag
(`py310` or `py312`) and install it in a fresh environment:

```bash
python3.12 -m venv .venv-trtmc
. .venv-trtmc/bin/activate
pip install ./tensorrt_model_connect-0.1.0-py312-none-manylinux_2_35_aarch64.whl
trtmc version
trtmc build --help
```

If you do not activate the environment, call the executable by path:

```bash
.venv-trtmc/bin/trtmc version
```

Use the `py310` wheel with Python 3.10 and the `py312` wheel with Python 3.12.
Do not install one tag into the other interpreter version.
The published wheel platform tag is `manylinux_2_35_aarch64`, matching the
TensorRT CUDA 13 aarch64 pip wheels and requiring a glibc 2.35 or newer Linux
host.

The wheel installs the native `trtmc` executable into the environment's
`bin/` directory, the Python builder package, declared Python dependencies
including `tensorrt>=10.16`, and the packaged TensorRT backend DSOs. For
`trtmc build`, the executable uses the Python interpreter from the same
environment when it is installed next to `python3`; source-built binaries fall
back to `python3` from the user's shell. CUDA driver/runtime libraries still
come from the host environment.

```bash
trtmc build Qwen/Qwen3-0.6B -o /tmp/qwen3.trtfb --max-cache-length 256
trtmc run /tmp/qwen3.trtfb \
  --prompt "The capital of France is" \
  --max-new-tokens 20 \
  --greedy
```

## Install from source

Use this path for development or when you need to modify native/runtime code.
From the repository root on the host:

```bash
git clone https://github.com/NVIDIA/TensorRT-Model-Connect.git
cd TensorRT-Model-Connect

./scripts/docker_build_gb300.sh
./scripts/docker_run_gb300.sh
```

Then, inside the dev container:

```bash
pip install -e tensorrt_model_connect/
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DTRTMC_TRT_INCLUDE_DIR="${TRT_INC_DIR:-/usr/include/aarch64-linux-gnu}" \
  -DTRTMC_TRT_LIBRARY="${TRT_LIB_DIR:-/opt/venv/lib/python3.12/site-packages/tensorrt_libs}/libnvinfer.so" \
  -DTRTMC_CUDA_INCLUDE_DIR=/usr/local/cuda/include \
  -DTRTMC_CUDART_LIBRARY=/usr/local/cuda/lib64/libcudart.so
cmake --build build -j

./build/trtmc version
./build/trtmc build Qwen/Qwen3-0.6B -o /tmp/qwen3.trtfb --max-cache-length 256
./build/trtmc run /tmp/qwen3.trtfb \
  --prompt "The capital of France is" \
  --max-new-tokens 20 \
  --greedy
```

The configure step should print `TRT backend DSO: enabled`. If it says the
TensorRT backend was skipped, fix `TRTMC_TRT_INCLUDE_DIR` and
`TRTMC_TRT_LIBRARY` before building bundles.

Use `pip install --no-deps -e tensorrt_model_connect/` only when the container
already has the declared dependencies installed and you intentionally want to
avoid dependency resolution. In a fresh Python environment, install with
dependencies so packages such as `transformers`, `onnxscript`, and `tensorrt`
are present.

Run source-built `./build/trtmc` inside the dev container unless you have
exported equivalent runtime library paths. A host-side failure such as
`libtorch.so: cannot open shared object file` usually means the executable was
built against libraries available in the container environment but not on the
host loader path.

The core runtime does not directly link `libnvinfer`. TensorRT execution lives behind backend DSOs loaded at runtime. The standard DSO is `libtrtmc_backend_trt.so`, with an ABI-suffixed alias when TensorRT headers expose a major/minor version.

## Build a release wheel

Use this path when you need to produce the same pip-installable artifact that
nightly CI publishes. Run it from the repository root in an aarch64 CUDA
development environment with Python 3.12, CMake, Ninja, Conan, `patchelf`,
CUDA headers/libraries, TensorRT headers, and a TensorRT `libnvinfer.so`
available. Build on the CI image or another glibc 2.35-compatible aarch64
environment; building on a newer host can produce a wheel that auditwheel
correctly reports as requiring a newer `manylinux` tag. The provided Docker
images install the expected prerequisites.

The release wheel build uses the repository-root `pyproject.toml`, not
`tensorrt_model_connect/pyproject.toml`. The root build invokes
`conan-py-build`, which runs the root `conanfile.py`, drives CMake, stages the
native runtime artifacts under `tensorrt_model_connect/bin/`, and installs the
native `trtmc` executable directly into the wheel's scripts directory.

Build one py312 wheel:

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
WHEEL_ARCH=manylinux_2_35_aarch64 \
python -m build --wheel --outdir "$PWD/dist" \
  -C build-dir=/tmp/trtmc-conan-py-wheel-py312 \
  .

python -m auditwheel show dist/tensorrt_model_connect-*-py312-none-manylinux_2_35_aarch64.whl
```

Build both release tags:

```bash
python -m pip install --upgrade build auditwheel
rm -rf dist /tmp/trtmc-conan-py-wheel

for tag in py310 py312; do
  CONAN_PY_BUILD_PROFILE_AUTODETECT=1 \
  TRTMC_TRT_INCLUDE_DIR="${TRT_INC_DIR:-/usr/include/aarch64-linux-gnu}" \
  TRTMC_TRT_LIBRARY="${TRT_LIB_DIR:-/opt/venv/lib/python3.12/site-packages/tensorrt_libs}/libnvinfer.so" \
  TRTMC_CUDA_INCLUDE_DIR=/usr/local/cuda/include \
  TRTMC_CUDART_LIBRARY=/usr/local/cuda/lib64/libcudart.so \
  WHEEL_PYVER="$tag" \
  WHEEL_ABI=none \
  WHEEL_ARCH=manylinux_2_35_aarch64 \
  python -m build --wheel --outdir "$PWD/dist" \
    -C "build-dir=/tmp/trtmc-conan-py-wheel/$tag" \
    .
done
```

The expected outputs are:

```text
dist/tensorrt_model_connect-0.1.0-py310-none-manylinux_2_35_aarch64.whl
dist/tensorrt_model_connect-0.1.0-py312-none-manylinux_2_35_aarch64.whl
```

Install and smoke-test a built wheel in a fresh environment:

```bash
python3.12 -m venv /tmp/trtmc-wheel-smoke
/tmp/trtmc-wheel-smoke/bin/python -m pip install --upgrade pip
/tmp/trtmc-wheel-smoke/bin/python -m pip install \
  dist/tensorrt_model_connect-0.1.0-py312-none-manylinux_2_35_aarch64.whl
/tmp/trtmc-wheel-smoke/bin/trtmc version
```

That smoke path intentionally does not activate the venv or prepend
`/tmp/trtmc-wheel-smoke/bin` to `PATH`; it matches a user invoking the installed
executable directly.

Do not use `python -m build tensorrt_model_connect/` for release wheels. That
subdirectory is the editable developer package; it does not run the Conan/CMake
native build or stage the release wheel runtime artifacts.

## Optional build switches

| Switch | Default | Purpose |
| --- | --- | --- |
| `TRTMC_ENABLE_TRT` | `ON` | Enable TensorRT backend DSO integration. |
| `TRTMC_BUILD_BACKEND_TRT` | `ON` | Build the standard TensorRT backend DSO when headers and libraries exist. |
| `TRTMC_BUILD_BACKEND_RTX` | `OFF` | Build the TensorRT-RTX backend DSO. |
| `TRTMC_ENABLE_TVM_FFI` | `ON` | Enable the optional TVM-FFI module loader and plugin when available. |

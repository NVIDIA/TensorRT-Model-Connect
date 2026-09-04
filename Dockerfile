# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Repository-wide ARM64 development container. Common CUDA, Python, tooling,
# and model execution profiles are built once; the final stage adds one exact
# TensorRT version as an immutable overlay. Project source is mounted at
# runtime, so the image does not contain a model or choose an application.

ARG CUDA_IMAGE=nvidia/cuda:13.3.0-devel-ubuntu24.04@sha256:ef2203909e80b8b976cfc672f7e2ae2b00bc0e25c404ee86d89e10a3802f1c52
ARG TENSORRT_VERSION=11.1.0.106
ARG TENSORRT_APT_VERSION=11.1.0.106-1+cuda13.3
FROM ${CUDA_IMAGE} AS ci-common-base

ARG PYTORCH_CUDA_INDEX=https://download.pytorch.org/whl/cu130
ARG TORCH_VERSION=2.12.0+cu130
ARG TORCHVISION_VERSION=0.27.0+cu130
ARG TORCHAUDIO_VERSION=2.11.0+cu130
ARG MODELOPT_VERSION=0.44.0
ARG TRTMC_TORCH_CUDA_ARCH_LIST=10.0

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST=${TRTMC_TORCH_CUDA_ARCH_LIST}

# ── System packages ──────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    gnupg \
    build-essential \
    cmake \
    ninja-build \
    patchelf \
    git \
    pkg-config \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    lcov \
    nlohmann-json3-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Python venv with all deps ───────────────────────────────────────────────
ENV VIRTUAL_ENV=/opt/venv
RUN python3.12 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install -U pip

# CUDA Python bindings (needed by debug_runner.py / diff tools). Match the
# CUDA 13.0 Python stack pulled by PyTorch.
RUN pip install "cuda-python==13.0.3"

# Apache TVM-FFI: the kernel-bridge ABI that lets compiled CUDA modules
# (FlashInfer, vendored diffusion kernels) be called from TRT plugins
# without a Python callback. Headers + libtvm_ffi.so ship in the wheel at
# /opt/venv/lib/python3.12/site-packages/tvm_ffi/{include,lib}/. CMake
# discovers them via a Python-spec lookup in the venv (see CMakeLists.txt).
RUN pip install "apache-tvm-ffi==0.1.12"

# Core Python deps
RUN pip install \
    "transformers==5.2.0" \
    tokenizers \
    safetensors \
    sentencepiece \
    huggingface_hub \
    ml_dtypes \
    datasets

# PyTorch ecosystem. The development image is Ubuntu 24.04 to match the
# manylinux_2_39/glibc floor used by the native wheel.
RUN pip install \
      "torch==${TORCH_VERSION}" \
      "torchvision==${TORCHVISION_VERSION}" \
      "torchaudio==${TORCHAUDIO_VERSION}" \
      --index-url "${PYTORCH_CUDA_INDEX}" && \
    pip install "setuptools>=80,<82"

# Quantized model support
RUN pip install "nvidia-modelopt==${MODELOPT_VERSION}"

# ML / testing / utilities
RUN pip install \
    "pytest<9" \
    pytest-cov \
    "jsonschema==4.26.0" \
    coverage \
    gcovr \
    pytest-xdist \
    lizard \
    accelerate \
    diffusers \
    protobuf \
    scipy \
    librosa \
    soundfile \
    sentencepiece \
    ftfy

# CLIP semantic metrics for the Flux diffusion E2E comparator.
# open-clip-torch must be pinned to a CPU-compatible version; it will use
# the torch installation already present in this image for GPU inference.
RUN pip install "open-clip-torch>=2.20"

# NeMo currently declares transformers~=4.57; force the runtime pin we need.
RUN pip install "nemo_toolkit[tts]==2.7.0" && \
    pip install --upgrade "transformers==5.2.0" && \
    python3 -c "import transformers; assert transformers.__version__ == '5.2.0', transformers.__version__" && \
    python3 -c "import diffusers, ftfy; print('deps_ok', diffusers.__version__)"

# Upgrade NeMo to a main-branch SHA that ships
# `nemo.collections.asr.models.rnnt_bpe_models_prompt.EncDecRNNTBPEModelWithPrompt`,
# required to load `nvidia/nemotron-3.5-asr-streaming-0.6b` as the HF/NeMo
# reference backend in E2E. PyPI's latest (2.7.3) doesn't ship this module yet;
# bump the SHA when 2.7.4+ lands the class. --no-deps keeps the rest of the
# dependency graph pinned to what 2.7.0 set up.
RUN pip install --no-deps \
        "git+https://github.com/NVIDIA/NeMo.git@c9040511b" && \
    python3 -c "from nemo.collections.asr.models.rnnt_bpe_models_prompt import EncDecRNNTBPEModelWithPrompt; print('NeMo prompt RNN-T class loaded')"

# NeMo may adjust the torch stack through transitive dependencies. Reinstall the
# exact CUDA 13 stack that this image is meant to test.
RUN pip install --force-reinstall \
      "torch==${TORCH_VERSION}" \
      "torchvision==${TORCHVISION_VERSION}" \
      "torchaudio==${TORCHAUDIO_VERSION}" \
      --index-url "${PYTORCH_CUDA_INDEX}" && \
    pip install "setuptools>=80,<82" && \
    python3 -c "import torch; assert torch.__version__ == '${TORCH_VERSION}', torch.__version__" && \
    python3 -c "import importlib.metadata as m; assert m.version('nvidia-modelopt') == '${MODELOPT_VERSION}', m.version('nvidia-modelopt')"

# ── Environment ─────────────────────────────────────────────────────────────
# ARM64 Blackwell targets use the system CUDA 13 cuBLAS kernels. Keep PyTorch/HF
# reference inference on the system cuBLAS instead of pip-installed CUDA libs.
ENV LD_PRELOAD=/usr/local/cuda/lib64/libcublas.so.13

# This model-agnostic downloader is part of the reviewed base runtime. It may
# fetch exact public PyPI artifacts, but never imports or builds package code.
COPY tools/ci/profile_downloader.py /opt/trtmc-profile-downloader.py

# Coverage tooling verification (run inside container):
#   python3 -m coverage --version && pytest --version && \
#   python3 -m pytest --help | grep -- '--cov' && \
#   gcovr --version && lcov --version && genhtml --version

# Keep the reusable runtime independent of family-owned Python environments.
# Online CI preparation materializes the selected family's exact-pinned
# profiles before a network-disabled proof and mounts them read-only there.
FROM ci-common-base AS ci-common

# Keep the reusable common layer independent of every TensorRT release. The
# version overlay below is the only stage allowed to add bindings, headers, or
# native runtime libraries.
RUN python3 -c \
      'import importlib.metadata as m, importlib.util; assert importlib.util.find_spec("tensorrt") is None; missing = ("tensorrt", "tensorrt_cu13", "tensorrt_cu13_bindings", "tensorrt_cu13_libs"); assert all(not tuple(m.distributions(name=name)) for name in missing)' && \
    test ! -e /usr/include/aarch64-linux-gnu/NvInferVersion.h && \
    test ! -e /usr/include/aarch64-linux-gnu/NvOnnxParser.h && \
    test -z "$(find /opt/venv -name 'libnvinfer.so*' -print -quit)" && \
    test -z "$(find /opt/venv -name 'libnvonnxparser.so*' -print -quit)"

WORKDIR /workspace/tensorrt-model-connect
CMD ["bash"]

# Add one exact TensorRT toolchain without rebuilding common dependencies or
# execution profiles. The same target is built with different version arguments
# to produce immutable TRT 11.1 and TRT 11.2 overlays.
FROM ci-common AS ci-runtime

ARG TENSORRT_VERSION
ARG TENSORRT_APT_VERSION

ENV TRT_LIB_DIR=/opt/venv/lib/python3.12/site-packages/tensorrt_libs
ENV TRT_INC_DIR=/usr/include/aarch64-linux-gnu
ENV LD_LIBRARY_PATH="$TRT_LIB_DIR:/usr/local/cuda/lib64"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      "libnvinfer-dev=${TENSORRT_APT_VERSION}" \
      "libnvinfer-headers-dev=${TENSORRT_APT_VERSION}" \
      "libnvinfer-headers-plugin-dev=${TENSORRT_APT_VERSION}" \
      "libnvinfer-safe-headers-dev=${TENSORRT_APT_VERSION}" \
      "libnvinfer11=${TENSORRT_APT_VERSION}" \
      "libnvonnxparsers-dev=${TENSORRT_APT_VERSION}" \
      "libnvonnxparsers11=${TENSORRT_APT_VERSION}" && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir "tensorrt==${TENSORRT_VERSION}" && \
    TENSORRT_MAJOR="${TENSORRT_VERSION%%.*}" && \
    test -f "$TRT_LIB_DIR/libnvinfer.so.$TENSORRT_MAJOR" && \
    test -f "$TRT_LIB_DIR/libnvonnxparser.so.$TENSORRT_MAJOR" && \
    { test -e "$TRT_LIB_DIR/libnvinfer.so" || \
      ln -s "libnvinfer.so.$TENSORRT_MAJOR" "$TRT_LIB_DIR/libnvinfer.so"; } && \
    { test -e "$TRT_LIB_DIR/libnvonnxparser.so" || \
      ln -s "libnvonnxparser.so.$TENSORRT_MAJOR" "$TRT_LIB_DIR/libnvonnxparser.so"; }

RUN EXPECTED_TENSORRT_VERSION="$TENSORRT_VERSION" python3 -c \
      'import ctypes, importlib.metadata as m, os; from pathlib import Path; expected = os.environ["EXPECTED_TENSORRT_VERSION"]; versions = tuple(map(int, expected.split("."))); assert all(m.version(name) == expected for name in ("tensorrt", "tensorrt_cu13", "tensorrt_cu13_bindings", "tensorrt_cu13_libs")); import tensorrt; assert tensorrt.__version__ == expected; include = Path("/usr/include/aarch64-linux-gnu"); header = (include / "NvInferVersion.h").read_text().splitlines(); assert (include / "NvOnnxParser.h").is_file(); assert all(f"#define TRT_{name}_ENTERPRISE {value}" in header for name, value in zip(("MAJOR", "MINOR", "PATCH", "BUILD"), versions)); major = versions[0]; libraries = Path(m.distribution("tensorrt_cu13_libs").locate_file("tensorrt_libs")); versioned = tuple(libraries / f"{name}.so.{major}" for name in ("libnvinfer", "libnvonnxparser")); linker = tuple(libraries / f"{name}.so" for name in ("libnvinfer", "libnvonnxparser")); assert all(path.is_file() for path in versioned); assert all(path.is_symlink() and path.resolve() == target.resolve() for path, target in zip(linker, versioned)); assert next(libraries.glob("libnvinfer_builder_resource_sm110.so.*"), None) is not None; library = ctypes.CDLL(str(versioned[0])); functions = tuple(getattr(library, f"getInferLib{name}Version") for name in ("Major", "Minor", "Patch", "Build")); [setattr(function, "restype", ctypes.c_int32) for function in functions]; assert tuple(function() for function in functions) == versions' && \
    printf '#include <NvInferRuntime.h>\n#include <NvOnnxParser.h>\nint main() { return getInferLibVersion() > 0 ? 0 : 1; }\n' | \
      c++ -x c++ - -I"$TRT_INC_DIR" -I/usr/local/cuda/include \
        -Wl,-rpath,"$TRT_LIB_DIR" -Wl,--no-as-needed \
        -x none \
        "$TRT_LIB_DIR/libnvinfer.so" "$TRT_LIB_DIR/libnvonnxparser.so" \
        -o /tmp/trtmc-trt-link-probe && \
    /tmp/trtmc-trt-link-probe && \
    rm -f /tmp/trtmc-trt-link-probe

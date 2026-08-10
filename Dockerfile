# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Repository-wide ARM64 development container. It provides the TensorRT,
# CUDA, Python, and all declared execution-profile dependencies used to develop
# and build any TRTMC model supported by the selected target. Project source is
# mounted at runtime; this image does not contain a model or choose a runtime.

ARG TENSORRT_IMAGE=nvcr.io/nvidia/tensorrt:26.07-py3@sha256:f794a79e8b996d16dbc2e5884e19d8e2269a51c960106c9b49b0061a6926c541
FROM ${TENSORRT_IMAGE} AS ci-base

ARG TENSORRT_VERSION=11.1.0.106
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
    && rm -rf /var/lib/apt/lists/*

# The pinned NVIDIA NGC image is the official TensorRT 11.1 ARM64 development
# release. Validate the Python binding, C++ headers, runtime library, and Thor
# (SM110) builder resource before adding any project dependencies.
RUN python3.12 -c \
      "import tensorrt; assert tensorrt.__version__ == '${TENSORRT_VERSION}', tensorrt.__version__" && \
    grep -q "#define TRT_MAJOR_ENTERPRISE 11" \
      /usr/include/aarch64-linux-gnu/NvInferVersion.h && \
    grep -q "#define TRT_MINOR_ENTERPRISE 1" \
      /usr/include/aarch64-linux-gnu/NvInferVersion.h && \
    grep -q "#define TRT_PATCH_ENTERPRISE 0" \
      /usr/include/aarch64-linux-gnu/NvInferVersion.h && \
    grep -q "#define TRT_BUILD_ENTERPRISE 106" \
      /usr/include/aarch64-linux-gnu/NvInferVersion.h && \
    test -f /usr/lib/aarch64-linux-gnu/libnvinfer.so.11 && \
    test -f /usr/lib/aarch64-linux-gnu/libnvinfer_builder_resource_sm110.so.11.1.0

# ── Python venv with all deps ───────────────────────────────────────────────
ENV VIRTUAL_ENV=/opt/venv
RUN python3.12 -m venv --system-site-packages $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# The venv inherits the official TensorRT binding from the NGC image. CMake
# derives the backend ABI alias from the matching NvInferVersion.h.
RUN pip install -U pip && \
    python3.12 -c \
      "import tensorrt; assert tensorrt.__version__ == '${TENSORRT_VERSION}', tensorrt.__version__"

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
    "transformers==5.5.0" \
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

# NeMo currently declares transformers~=4.57; force the secure runtime pin we need.
RUN pip install "nemo_toolkit[tts]==2.7.0" && \
    pip install --upgrade "transformers==5.5.0" && \
    python3 -c "import transformers; assert transformers.__version__ == '5.5.0', transformers.__version__" && \
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
    LD_LIBRARY_PATH="/opt/venv/lib/python3.12/site-packages/tensorrt_libs:/usr/local/cuda/lib64" \
      python3 -c "import tensorrt; assert tensorrt.__version__ == '${TENSORRT_VERSION}', tensorrt.__version__" && \
    python3 -c "import torch; assert torch.__version__ == '${TORCH_VERSION}', torch.__version__" && \
    python3 -c "import importlib.metadata as m; assert m.version('nvidia-modelopt') == '${MODELOPT_VERSION}', m.version('nvidia-modelopt')"

# ── Environment ─────────────────────────────────────────────────────────────
# Pre-compute paths so cmake / runtime find TRT without manual exports
ENV TRT_LIB_DIR=/usr/lib/aarch64-linux-gnu
ENV TRT_INC_DIR=/usr/include/aarch64-linux-gnu
ENV LD_LIBRARY_PATH="$TRT_LIB_DIR:/usr/local/cuda/lib64"

# ARM64 Blackwell targets use the system CUDA 13 cuBLAS kernels. Keep PyTorch/HF
# reference inference on the system cuBLAS instead of pip-installed CUDA libs.
ENV LD_PRELOAD=/usr/local/cuda/lib64/libcublas.so.13

# Coverage tooling verification (run inside container):
#   python3 -m coverage --version && pytest --version && \
#   python3 -m pytest --help | grep -- '--cov' && \
#   gcovr --version && lcov --version && genhtml --version

# Keep the final layer small and cache-friendly. Isolated validation runs have
# networking disabled, so CMake must find nlohmann/json in the image instead of
# falling back to FetchContent during each isolated scratch build.
RUN apt-get update && \
    apt-get install -y --no-install-recommends nlohmann-json3-dev && \
    rm -rf /var/lib/apt/lists/*

# Build every family-declared Python execution profile while network access is
# available. The family-owned lock and verification files are the only package
# source of truth; python_profiles.py additionally rejects non-exact pins and
# verifies every installed distribution before marking a profile ready.
FROM ci-base AS python-profile-builder

ENV PYTHONPATH=/opt/trtmc-profile-source
ENV TRTMC_PYTHON_PROFILE_ROOT=/opt/trtmc-python-profiles
# sphn publishes no aarch64 wheel. Keep its Rust build toolchain in this
# throwaway builder stage; the final development stage receives only the
# verified profile.
RUN apt-get update && \
    apt-get install -y --no-install-recommends cargo rustc && \
    rm -rf /var/lib/apt/lists/* && \
    pip install "maturin==1.14.1"
# Avoid compiling profile-local CUDA extensions for every architecture known
# to a GPU-less Docker build. Keep 10.0 as the existing default; Jetson AGX
# Thor callers pass 11.0 for SM110.
COPY python/tensorrt_model_connect /opt/trtmc-profile-source/tensorrt_model_connect
COPY .github/scripts/build-python-profiles.py /opt/trtmc-build-python-profiles.py
RUN python3 /opt/trtmc-build-python-profiles.py \
    && chmod -R a+rX /opt/trtmc-python-profiles

# Do not retain the full builder source tree in the development image. Only the
# verified virtual environments cross the stage boundary, so sibling model
# implementations cannot satisfy imports in an isolated source projection.
FROM ci-base AS ci-runtime

COPY --from=python-profile-builder \
    /opt/trtmc-python-profiles /opt/trtmc-python-profiles
ENV TRTMC_PYTHON_PROFILE_ROOT=/opt/trtmc-python-profiles
# Execution-profile environments are part of the dev-image contract. Rebuild
# the image after changing their lock or verification files.
ENV TRTMC_PYTHON_PROFILE_PREBUILT_ONLY=1

WORKDIR /workspace/tensorrt-model-connect

CMD ["bash"]

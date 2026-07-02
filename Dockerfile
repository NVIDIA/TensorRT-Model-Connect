# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM nvidia/cuda:13.0.0-devel-ubuntu24.04

ARG TENSORRT_VERSION=11.0.0.114
ARG TENSORRT_DEB_VERSION=11.0.0.114-1+cuda13.2
ARG PYTORCH_CUDA_INDEX=https://download.pytorch.org/whl/cu130
ARG TORCH_VERSION=2.12.0+cu130
ARG TORCHVISION_VERSION=0.27.0+cu130
ARG TORCHAUDIO_VERSION=2.11.0+cu130
ARG MODELOPT_VERSION=0.44.0

ENV DEBIAN_FRONTEND=noninteractive

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
    "libnvinfer-headers-dev=${TENSORRT_DEB_VERSION}" \
    && rm -rf /var/lib/apt/lists/*

# ── Python venv with all deps ───────────────────────────────────────────────
ENV VIRTUAL_ENV=/opt/venv
RUN python3.12 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# TensorRT Python/runtime libraries. Keep this in sync with the headers above:
# CMake derives the backend ABI alias from NvInferVersion.h.
RUN pip install -U pip && \
    pip install --no-cache-dir "tensorrt_cu13==${TENSORRT_VERSION}" && \
    pip install --no-cache-dir "tensorrt==${TENSORRT_VERSION}" --no-deps

# CUDA Python bindings (needed by debug_runner.py / diff tools). Match the CUDA
# 13.0 Python stack pulled by PyTorch.
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

# PyTorch ecosystem. The CI image is Ubuntu 24.04 to match the
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

# NeMo currently declares transformers~=4.57; force the runtime pin we need.
RUN pip install "nemo_toolkit[tts]==2.7.0" && \
    pip install --upgrade "transformers==5.2.0" && \
    python3 -c "import transformers; assert transformers.__version__ == '5.2.0', transformers.__version__" && \
    python3 -c "import diffusers, ftfy; print('deps_ok', diffusers.__version__)"

# NeMo may adjust the torch stack through transitive dependencies. Reinstall the
# exact CUDA 13 stack that this image is meant to test.
RUN pip install --force-reinstall \
      "torch==${TORCH_VERSION}" \
      "torchvision==${TORCHVISION_VERSION}" \
      "torchaudio==${TORCHAUDIO_VERSION}" \
      --index-url "${PYTORCH_CUDA_INDEX}" && \
    pip install "setuptools>=80,<82" && \
    python3 -c "import tensorrt; assert tensorrt.__version__ == '${TENSORRT_VERSION}', tensorrt.__version__" && \
    python3 -c "import torch; assert torch.__version__ == '${TORCH_VERSION}', torch.__version__" && \
    python3 -c "import importlib.metadata as m; assert m.version('nvidia-modelopt') == '${MODELOPT_VERSION}', m.version('nvidia-modelopt')"

# Create libnvinfer.so symlink (pip ships the versioned libnvinfer.so.11 only)
RUN TRT_LIB=$(python3 -c \
      "import importlib.util; s=importlib.util.find_spec('tensorrt_libs'); print(s.submodule_search_locations[0])") && \
    [ ! -f "$TRT_LIB/libnvinfer.so" ] && ln -sf libnvinfer.so.11 "$TRT_LIB/libnvinfer.so" || true && \
    echo "$TRT_LIB" > /etc/ld.so.conf.d/tensorrt.conf && \
    ldconfig

# ── Environment ─────────────────────────────────────────────────────────────
# Pre-compute paths so cmake / runtime find TRT without manual exports
ENV TRT_LIB_DIR=/opt/venv/lib/python3.12/site-packages/tensorrt_libs
ENV TRT_INC_DIR=/usr/include/aarch64-linux-gnu
ENV LD_LIBRARY_PATH="$TRT_LIB_DIR:/usr/local/cuda/lib64"

# GB300/Blackwell uses the system CUDA 13 cuBLAS kernels. Keep PyTorch/HF
# reference inference on the system cuBLAS instead of pip-installed CUDA libs.
ENV LD_PRELOAD=/usr/local/cuda/lib64/libcublas.so.13

# Coverage tooling verification (run inside container):
#   python3 -m coverage --version && pytest --version && \
#   python3 -m pytest --help | grep -- '--cov' && \
#   gcovr --version && lcov --version && genhtml --version

WORKDIR /workspace/tensorrt-model-connect

CMD ["bash"]

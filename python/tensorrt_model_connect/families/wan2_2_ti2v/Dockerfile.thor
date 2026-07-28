# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Official NVIDIA NGC TensorRT 11.1.0.106 image, pinned to its ARM64 image
# digest. This is the only source of TensorRT in the customer image.
ARG TENSORRT_IMAGE=nvcr.io/nvidia/tensorrt:26.07-py3@sha256:f794a79e8b996d16dbc2e5884e19d8e2269a51c960106c9b49b0061a6926c541
FROM ${TENSORRT_IMAGE}

ARG TENSORRT_VERSION=11.1.0.106
ARG TORCH_VERSION=2.12.0+cu130
ARG PYTORCH_CUDA_INDEX=https://download.pytorch.org/whl/cu130
ARG TRANSFORMERS_VERSION=5.2.0
ARG TRTMC_SOURCE_REVISION=unknown

LABEL org.opencontainers.image.source="https://github.com/NVIDIA/TensorRT-Model-Connect"
LABEL org.opencontainers.image.description="TensorRT-Model-Connect Wan2.2 deployment for Jetson AGX Thor"
LABEL org.opencontainers.image.revision="${TRTMC_SOURCE_REVISION}"
LABEL com.nvidia.tensorrt.version="${TENSORRT_VERSION}"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    git \
    ninja-build \
    nlohmann-json3-dev \
    pkg-config \
    python3-pip \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

# Fail the build if the official image ever stops resolving to the exact
# TensorRT release and SM110 SDK expected by this customer package.
RUN python3.12 -c \
      "import importlib.metadata as m, tensorrt; \
assert tensorrt.__version__ == '${TENSORRT_VERSION}', tensorrt.__version__; \
assert m.version('tensorrt') == '${TENSORRT_VERSION}', m.version('tensorrt')" && \
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

ENV VIRTUAL_ENV=/opt/venv
RUN python3.12 -m venv --system-site-packages "${VIRTUAL_ENV}"
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# The venv inherits the official TensorRT Python distribution and libraries
# from the NGC image; it does not download or copy a second TensorRT package.
RUN python -m pip install --no-cache-dir --upgrade pip "setuptools>=80,<82" && \
    python -c \
      "import importlib.metadata as m, tensorrt; \
assert tensorrt.__version__ == '${TENSORRT_VERSION}', tensorrt.__version__; \
assert m.version('tensorrt') == '${TENSORRT_VERSION}', m.version('tensorrt')"

RUN python -m pip install --no-cache-dir \
      "torch==${TORCH_VERSION}" \
      --index-url "${PYTORCH_CUDA_INDEX}" && \
    python -m pip install --no-cache-dir \
      "transformers==${TRANSFORMERS_VERSION}"

ENV TRT_LIB_DIR=/usr/lib/aarch64-linux-gnu
ENV TRT_INC_DIR=/usr/include/aarch64-linux-gnu
ENV LD_LIBRARY_PATH="${TRT_LIB_DIR}:/usr/local/cuda/lib64"
ENV LD_PRELOAD=/usr/local/cuda/lib64/libcublas.so.13

COPY . /opt/tensorrt-model-connect
WORKDIR /opt/tensorrt-model-connect

RUN chmod -R a+rX /opt/tensorrt-model-connect && \
    printf '%s\n' "${TRTMC_SOURCE_REVISION}" > /opt/trtmc-source-revision && \
    python -m pip install --no-cache-dir -e '.[wan]' -C py-only=true

RUN cmake -S . -B build-thor -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=110 \
      -DTRTMC_BUILD_TESTS=OFF \
      -DTRTMC_BUILD_BENCHMARKS=OFF \
      -DTRTMC_ENABLE_LIBTORCH_MULTINOMIAL=OFF \
      -DTRTMC_ENABLE_TVM_FFI=OFF \
      -DTRTMC_BUILD_DIFFUSION_KERNELS=OFF \
      -DTRTMC_SOURCE_REVISION="${TRTMC_SOURCE_REVISION}" \
      -DTRTMC_TRT_INCLUDE_DIR="${TRT_INC_DIR}" \
      -DTRTMC_TRT_LIBRARY="${TRT_LIB_DIR}/libnvinfer.so" \
      -DTRTMC_CUDA_INCLUDE_DIR=/usr/local/cuda/include \
      -DTRTMC_CUDART_LIBRARY=/usr/local/cuda/lib64/libcudart.so && \
    cmake --build build-thor --parallel "$(nproc)" --target \
      trtmc \
      trtmc_backend_trt \
      trtmc_model_wan2_2_ti2v && \
    ln -s /opt/tensorrt-model-connect/build-thor/trtmc /usr/local/bin/trtmc

RUN python -c "import tensorrt, torch, transformers; \
assert tensorrt.__version__ == '${TENSORRT_VERSION}'; \
assert torch.__version__ == '${TORCH_VERSION}'; \
assert transformers.__version__ == '${TRANSFORMERS_VERSION}'" && \
    trtmc version && \
    test -f build-thor/libtrtmc_backend_trt.so && \
    test -f build-thor/models/wan2_2_ti2v/libtrtmc_model_wan2_2_ti2v.so && \
    for artifact in \
      build-thor/trtmc \
      build-thor/libtrtmc_core.so \
      build-thor/libtrtmc_backend_trt.so \
      build-thor/models/wan2_2_ti2v/libtrtmc_model_wan2_2_ti2v.so; do \
        ldd "${artifact}" > /tmp/trtmc-ldd.txt || exit 1; \
        if grep -q "not found" /tmp/trtmc-ldd.txt; then \
          cat /tmp/trtmc-ldd.txt; \
          exit 1; \
        fi; \
    done

ENV HF_HOME=/data/huggingface
ENV PYTHONUNBUFFERED=1
WORKDIR /data

ENTRYPOINT ["trtmc"]

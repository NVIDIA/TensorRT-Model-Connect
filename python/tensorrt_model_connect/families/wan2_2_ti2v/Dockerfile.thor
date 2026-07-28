# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

ARG TENSORRT_SDK_IMAGE=ghcr.io/nvidia/tensorrt-model-connect/tensorrt-sdk:11.2.0.113@sha256:18c12935c4e7f507d8719d44509cb5623b17f298ee951d844bd9e558a8309929
ARG CUDA_IMAGE=nvidia/cuda:13.3.0-devel-ubuntu24.04@sha256:ef2203909e80b8b976cfc672f7e2ae2b00bc0e25c404ee86d89e10a3802f1c52

FROM ${TENSORRT_SDK_IMAGE} AS tensorrt_sdk

FROM ${CUDA_IMAGE}

ARG TENSORRT_VERSION=11.2.0.113
ARG TORCH_VERSION=2.12.0+cu130
ARG PYTORCH_CUDA_INDEX=https://download.pytorch.org/whl/cu130
ARG TRANSFORMERS_VERSION=5.2.0
ARG TRTMC_SOURCE_REVISION=unknown

LABEL org.opencontainers.image.source="https://github.com/NVIDIA/TensorRT-Model-Connect"
LABEL org.opencontainers.image.description="TensorRT-Model-Connect Wan2.2 deployment for Jetson AGX Thor"
LABEL org.opencontainers.image.revision="${TRTMC_SOURCE_REVISION}"

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

ENV VIRTUAL_ENV=/opt/venv
RUN python3.12 -m venv "${VIRTUAL_ENV}"
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

COPY --from=tensorrt_sdk /opt/tensorrt/include/ /usr/include/aarch64-linux-gnu/
COPY --from=tensorrt_sdk /opt/tensorrt/lib/ \
    /opt/venv/lib/python3.12/site-packages/tensorrt_libs/
COPY --from=tensorrt_sdk \
    /opt/tensorrt/python/tensorrt-${TENSORRT_VERSION}-cp312-none-linux_aarch64.whl \
    /opt/tensorrt/python/

RUN python -m pip install --no-cache-dir --upgrade pip "setuptools>=80,<82" && \
    python -m pip install --no-cache-dir \
      "/opt/tensorrt/python/tensorrt-${TENSORRT_VERSION}-cp312-none-linux_aarch64.whl" && \
    ln -sf libnvinfer.so.11 \
      /opt/venv/lib/python3.12/site-packages/tensorrt_libs/libnvinfer.so && \
    echo /opt/venv/lib/python3.12/site-packages/tensorrt_libs \
      > /etc/ld.so.conf.d/tensorrt.conf && \
    ldconfig

RUN python -m pip install --no-cache-dir \
      "torch==${TORCH_VERSION}" \
      --index-url "${PYTORCH_CUDA_INDEX}" && \
    python -m pip install --no-cache-dir \
      "transformers==${TRANSFORMERS_VERSION}"

ENV TRT_LIB_DIR=/opt/venv/lib/python3.12/site-packages/tensorrt_libs
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

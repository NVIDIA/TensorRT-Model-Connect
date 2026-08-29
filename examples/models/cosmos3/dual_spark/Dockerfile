# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This digest is the repository-pinned aarch64 TensorRT 11.1 development image.
ARG TENSORRT_IMAGE=nvcr.io/nvidia/tensorrt:26.07-py3@sha256:f794a79e8b996d16dbc2e5884e19d8e2269a51c960106c9b49b0061a6926c541

FROM ${TENSORRT_IMAGE} AS builder

ARG PYTORCH_CPU_INDEX=https://download.pytorch.org/whl/cpu
ARG TORCH_VERSION=2.12.0+cpu
ARG TRTMC_CUDA_ARCHITECTURES=121-real

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      cmake \
      git \
      ninja-build \
      nlohmann-json3-dev \
      patchelf \
      pkg-config \
      python3.12-dev \
      python3.12-venv && \
    rm -rf /var/lib/apt/lists/*

RUN target_triplet="$(gcc -dumpmachine)" && \
    install -d -m 0755 /opt/trtmc/platform && \
    ln -s "/usr/lib/${target_triplet}" /opt/trtmc/platform/lib && \
    ln -s "/usr/include/${target_triplet}" /opt/trtmc/platform/include && \
    test -f /opt/trtmc/platform/lib/libnvinfer.so.11 && \
    test -f /opt/trtmc/platform/include/NvInferVersion.h

RUN python3.12 -m venv --system-site-packages "${VIRTUAL_ENV}" && \
    ln -s /usr/bin/cmake "${VIRTUAL_ENV}/bin/cmake"
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# Torch remains CPU-only: TensorRT owns GPU execution, while safetensors uses
# Torch to read checkpoints during the one-time bundle build.
RUN pip install -U pip && \
    pip install \
      "cuda-python>=13.0.3,<14" \
      "huggingface_hub>=0.23" \
      "ml_dtypes>=0.4" \
      "numpy>=1.24,<2.5" \
      "onnx>=1.16" \
      "onnxscript>=0.2" \
      "PyYAML>=6.0" \
      "safetensors>=0.4" \
      "sentencepiece>=0.1.99" \
      "transformers==5.2.0" && \
    pip install "torch==${TORCH_VERSION}" --index-url "${PYTORCH_CPU_INDEX}" && \
    pip install "setuptools>=80,<82"

ENV TRT_LIB_DIR=/opt/trtmc/platform/lib \
    TRT_INC_DIR=/opt/trtmc/platform/include \
    LD_LIBRARY_PATH=/opt/trtmc/platform/lib:/usr/local/cuda/lib64

WORKDIR /src
COPY . /src

# Dockerfile.dockerignore excludes every other model family.
# TRTMC_MODEL_PROOF_MODEL makes that isolation a configure-time invariant.
RUN pip install --no-deps -e . -C py-only=true && \
    cmake -S . -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES="${TRTMC_CUDA_ARCHITECTURES}" \
      -DTRTMC_BUILD_BACKEND_TRT=ON \
      -DTRTMC_BUILD_BACKEND_RTX=OFF \
      -DTRTMC_BUILD_TESTS=OFF \
      -DTRTMC_BUILD_BENCHMARKS=OFF \
      -DTRTMC_ENABLE_LIBTORCH_MULTINOMIAL=OFF \
      -DTRTMC_ENABLE_TVM_FFI=OFF \
      -DTRTMC_MODEL_PROOF_MODEL=cosmos3 && \
    cmake --build build --parallel "$(nproc)" --target \
      trtmc \
      trtmc_backend_trt \
      trtmc_model_cosmos3 && \
    install -d /opt/trtmc/bin/models/cosmos3 && \
    install -m 0755 build/trtmc /opt/trtmc/bin/trtmc && \
    install -m 0755 build/libtrtmc_core.so /opt/trtmc/bin/libtrtmc_core.so && \
    install -m 0755 build/libtrtmc_backend_trt.so /opt/trtmc/bin/libtrtmc_backend_trt.so && \
    find build -maxdepth 1 -type f -name 'libtrtmc_backend_trt_*.so' \
      -exec install -m 0755 '{}' /opt/trtmc/bin/ \; && \
    install -m 0755 \
      build/models/cosmos3/libtrtmc_model_cosmos3.so \
      /opt/trtmc/bin/models/cosmos3/libtrtmc_model_cosmos3.so && \
    /opt/trtmc/bin/trtmc --version

FROM ${TENSORRT_IMAGE} AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/trtmc/bin:/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=/opt/trtmc/bin:/opt/trtmc/platform/lib:/usr/local/cuda/lib64 \
    TRTMC_BIN=/opt/trtmc/bin/trtmc \
    TRTMC_MODEL_PLUGIN_DIR=/opt/trtmc/bin/models \
    TRTMC_TRT_LIBRARY_DIR=/opt/trtmc/platform/lib \
    HF_HOME=/root/.cache/huggingface \
    HF_HUB_CACHE=/root/.cache/huggingface/hub \
    HF_HUB_DISABLE_TELEMETRY=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      ffmpeg \
      ibverbs-providers \
      libibverbs1 \
      rdma-core && \
    rm -rf /var/lib/apt/lists/* && \
    install -d -m 0755 \
      /models \
      /outputs \
      /opt/trtmc/licenses \
      /root/.cache/huggingface/hub

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/trtmc /opt/trtmc
# Preserve the source target recorded by the Python-only editable install.
COPY --from=builder /src/python /src/python
COPY --from=builder /src/LICENSE /src/NOTICE /opt/trtmc/licenses/

RUN test -x /opt/trtmc/bin/trtmc && \
    test -f /opt/trtmc/bin/models/cosmos3/libtrtmc_model_cosmos3.so && \
    ffmpeg -version >/dev/null

ARG TRTMC_SOURCE_REVISION=unknown
LABEL org.opencontainers.image.source="https://github.com/NVIDIA/TensorRT-Model-Connect" \
      org.opencontainers.image.revision="${TRTMC_SOURCE_REVISION}"

WORKDIR /work

ENTRYPOINT ["/opt/trtmc/bin/trtmc"]
CMD ["--help"]

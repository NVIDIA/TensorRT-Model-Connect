# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The default digest matches Dockerfile.dev.x86. Pass the ARM digest documented
# in the sample README for aarch64. TensorRT engines are built on first start
# because they are specific to the visible GPU.
ARG TENSORRT_IMAGE=nvcr.io/nvidia/tensorrt:26.07-py3@sha256:b82db1abc23750ab0069abc99bbe4ea29138dbdc23ea39861199e2346638b48a

FROM ${TENSORRT_IMAGE} AS trtmc-builder

ARG PYTORCH_CPU_INDEX=https://download.pytorch.org/whl/cpu
ARG TORCH_VERSION=2.12.0+cpu
ARG TRTMC_CUDA_ARCHITECTURES=80-real;89-real;90-real;100-real;103-real;120-real;121-real

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

# Match the development image's builder stack. Torch remains CPU-only:
# TensorRT owns GPU execution, while safetensors uses Torch to read checkpoints.
RUN pip install -U pip && \
    pip install \
      "apache-tvm-ffi==0.1.12" \
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
COPY CMakeLists.txt pyproject.toml _pyproject_backend.py README.md LICENSE NOTICE ASSET_LICENSES.md ./
COPY cmake ./cmake
COPY include ./include
COPY python ./python
COPY src ./src
# Runtime manifests validate their model-owned test source paths at configure time,
# even when test targets are disabled.
COPY tests/cpp/models ./tests/cpp/models
# TVM-FFI declares this EXCLUDE_FROM_ALL target during CMake generation.
COPY examples/byok/identity_copy_kernel.cpp ./examples/byok/identity_copy_kernel.cpp
COPY third_party ./third_party

RUN pip install --no-deps -e . -C py-only=true && \
    cmake -S . -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES="${TRTMC_CUDA_ARCHITECTURES}" \
      -DTRTMC_BUILD_BACKEND_TRT=ON \
      -DTRTMC_BUILD_BACKEND_RTX=OFF \
      -DTRTMC_BUILD_TESTS=OFF \
      -DTRTMC_BUILD_BENCHMARKS=OFF \
      -DTRTMC_ENABLE_LIBTORCH_MULTINOMIAL=OFF && \
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

FROM ${TENSORRT_IMAGE} AS story-scene

# The NGC base exports OPAL_PREFIX for HPC-X; this stage uses apt Open MPI.
ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/trtmc/bin:/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    PYTHONPATH=/opt/app \
    TRTMC_BIN=/opt/trtmc/bin/trtmc \
    TRTMC_MODEL_PLUGIN_DIR=/opt/trtmc/bin/models \
    TRTMC_TRT_LIBRARY_DIR=/opt/trtmc/platform/lib \
    HF_HOME=/root/.cache/huggingface \
    HF_HUB_CACHE=/root/.cache/huggingface/hub \
    HF_HUB_DISABLE_TELEMETRY=1 \
    COSMOS3_BUNDLE=/models/cosmos3.trtfb \
    COSMOS3_CP_SIZE=1 \
    OUTPUT_ROOT=/outputs \
    HOST=0.0.0.0 \
    PORT=8080 \
    PYTHONUNBUFFERED=1 \
    OMPI_ALLOW_RUN_AS_ROOT=1 \
    OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
    OPAL_PREFIX=

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      ffmpeg \
      fontconfig \
      fonts-dejavu-core \
      fonts-liberation2 \
      fonts-noto-color-emoji \
      openmpi-bin && \
    rm -rf /var/lib/apt/lists/* && \
    install -d -m 0755 \
      /models \
      /outputs \
      /root/.cache/huggingface/hub \
      /opt/app

COPY --from=trtmc-builder /opt/venv /opt/venv
COPY --from=trtmc-builder /opt/trtmc /opt/trtmc
# Preserve the source target recorded by the Python-only editable install.
COPY --from=trtmc-builder /src/python /src/python
COPY examples/cosmos3_story_scene/app/ /opt/app/
COPY examples/cosmos3_story_scene/scripts/entrypoint.sh /usr/local/bin/cosmos3-entrypoint

RUN chmod 0755 /usr/local/bin/cosmos3-entrypoint && \
    fc-cache -f && \
    test -x /opt/trtmc/bin/trtmc && \
    test -f /opt/trtmc/bin/models/cosmos3/libtrtmc_model_cosmos3.so && \
    /usr/bin/mpirun -np 1 /usr/bin/true && \
    ffmpeg -version >/dev/null

WORKDIR /opt/app

# A long start period is intentional: the first health transition waits for
# checkpoint download and hardware-specific TensorRT engine compilation.
HEALTHCHECK --interval=30s --timeout=5s --start-period=12h --retries=3 \
  CMD ["python3", "-c", "import json, os, urllib.request; data=json.load(urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8080') + '/healthz', timeout=3)); assert data.get('status') == 'ok' and data.get('worker_ready') is True"]

EXPOSE 8080
VOLUME ["/root/.cache/huggingface", "/models", "/outputs"]

ENTRYPOINT ["/usr/local/bin/cosmos3-entrypoint"]
CMD ["python3", "-m", "story_scene"]

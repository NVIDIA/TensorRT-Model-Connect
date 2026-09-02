# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This digest is the repository-pinned aarch64 TensorRT 11.1 development image.
# The README gives the matching pinned x86_64 override.
ARG TENSORRT_IMAGE=nvcr.io/nvidia/tensorrt:26.07-py3@sha256:f794a79e8b996d16dbc2e5884e19d8e2269a51c960106c9b49b0061a6926c541

FROM ${TENSORRT_IMAGE} AS builder

ARG TRTMC_CUDA_ARCHITECTURES=103-real

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      cmake \
      libasound2-dev \
      ninja-build \
      nlohmann-json3-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . /src

# Dockerfile.dockerignore projects the native source tree while excluding every
# other model family. TRTMC_MODEL_PROOF_MODEL makes that model isolation a
# configure-time invariant instead of relying on the build target alone.
RUN cmake \
      -S /src/examples/models/nemotron_voicechat/full_duplex \
      -B /build \
      -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES="${TRTMC_CUDA_ARCHITECTURES}" \
      -DTRTMC_BUILD_BACKEND_TRT=ON \
      -DTRTMC_BUILD_BACKEND_RTX=OFF \
      -DTRTMC_BUILD_TESTS=OFF \
      -DTRTMC_BUILD_BENCHMARKS=OFF \
      -DTRTMC_ENABLE_LIBTORCH_MULTINOMIAL=OFF \
      -DTRTMC_ENABLE_TVM_FFI=OFF \
      -DTRTMC_BUILD_DIFFUSION_KERNELS=OFF \
      -DTRTMC_DISTRIBUTABLE_BUILD=ON \
      -DTRTMC_MODEL_PROOF_MODEL=nemotron_voicechat && \
    cmake --build /build --parallel --target \
      trtmc_voicechat_full_duplex \
      test_trtmc_voicechat_playback_queue \
      trtmc_backend_trt \
      trtmc_model_nemotron_voicechat && \
    ctest --test-dir /build --output-on-failure \
      --tests-regex '^trtmc_voicechat_playback_queue$' && \
    /build/trtmc_voicechat_full_duplex --help >/dev/null

FROM ${TENSORRT_IMAGE} AS runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      libasound2-data \
      libasound2t64 && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p /opt/trtmc/bin /opt/trtmc/licenses

# The repository-wide install target includes every model DSO. Stage only the
# application and the three native runtime layers this example actually uses.
COPY --from=builder /build/trtmc_voicechat_full_duplex /opt/trtmc/bin/
COPY --from=builder /build/trtmc/libtrtmc_core.so /opt/trtmc/bin/
COPY --from=builder /build/libtrtmc_backend_trt.so /opt/trtmc/bin/
COPY --from=builder /build/libtrtmc_backend_trt_11_1.so /opt/trtmc/bin/
COPY --from=builder /build/models/nemotron_voicechat/libtrtmc_model_nemotron_voicechat.so /opt/trtmc/bin/
COPY --from=builder /src/LICENSE /src/NOTICE /opt/trtmc/licenses/

ENV PATH="/opt/trtmc/bin:${PATH}" \
    LD_LIBRARY_PATH="/opt/trtmc/bin:${LD_LIBRARY_PATH}" \
    TRTMC_MODEL_PLUGIN_DIR=/opt/trtmc/bin \
    TRTMC_MODEL_PLUGIN_STRICT=1

ARG TRTMC_SOURCE_REVISION=unknown
LABEL org.opencontainers.image.source="https://github.com/NVIDIA/TensorRT-Model-Connect" \
      org.opencontainers.image.revision="${TRTMC_SOURCE_REVISION}"

WORKDIR /models

ENTRYPOINT ["/opt/trtmc/bin/trtmc_voicechat_full_duplex"]
CMD ["/models/model.bundle"]

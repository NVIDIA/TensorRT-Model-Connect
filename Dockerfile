FROM nvidia/cuda:13.0.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# ── System packages ──────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    gnupg \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Ubuntu 22.04 provides the glibc 2.35 floor used by the release wheel tag.
# Python 3.12 comes from deadsnakes; TensorRT CUDA 13 headers come from NVIDIA's
# Ubuntu 24.04 CUDA repo because CUDA 13 TensorRT header packages are not
# published in the Ubuntu 22.04 repo.
RUN add-apt-repository -y ppa:deadsnakes/ppa && \
    echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/ /" \
      > /etc/apt/sources.list.d/cuda-ubuntu2404.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    git \
    pkg-config \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    lcov \
    libnvinfer-headers-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Python venv with all deps ───────────────────────────────────────────────
ENV VIRTUAL_ENV=/opt/venv
RUN python3.12 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# TensorRT (auto-selects cu13 wheels for CUDA 13.x)
RUN pip install -U pip && \
    pip install tensorrt_cu13 && \
    pip install tensorrt --no-deps

# CUDA Python bindings (needed by debug_runner.py / diff tools)
RUN pip install cuda-python

# Core Python deps
RUN pip install \
    "transformers==5.2.0" \
    tokenizers \
    safetensors \
    sentencepiece \
    huggingface_hub \
    ml_dtypes \
    datasets

# PyTorch ecosystem — install torch first, then derive the CUDA variant tag
# so torchvision/torchaudio/torch_tensorrt all use the same CUDA build.
RUN pip install torch --index-url https://download.pytorch.org/whl/cu130 && \
    TORCH_CUDA=$(python3 -c "import torch; print(torch.version.cuda.replace('.','')[:3])") && \
    echo "Detected torch CUDA variant: cu${TORCH_CUDA}" && \
    pip install torchvision torchaudio \
        --index-url "https://download.pytorch.org/whl/cu${TORCH_CUDA}"

# Torch-TRT (compiles HF models to TRT-optimized TorchScript via torch.export + dynamo)
# Use the same CUDA variant index as torch to avoid version mismatch.
RUN TORCH_CUDA=$(python3 -c "import torch; print(torch.version.cuda.replace('.','')[:3])") && \
    pip install torch_tensorrt \
        --extra-index-url "https://download.pytorch.org/whl/cu${TORCH_CUDA}"

# Quantized model support for Torch-TRT
RUN pip install nvidia-modelopt

# ML / testing / utilities
RUN pip install \
    pytest \
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

# NeMo currently declares transformers~=4.57; force the runtime pin we need.
RUN pip install "nemo_toolkit[tts]==2.7.0" && \
    pip install --upgrade "transformers==5.2.0" && \
    python3 -c "import transformers; assert transformers.__version__ == '5.2.0', transformers.__version__" && \
    python3 -c "import diffusers, ftfy; print('deps_ok', diffusers.__version__)"

# Create libnvinfer.so symlink (pip ships libnvinfer.so.10 only)
RUN TRT_LIB=$(python3 -c \
      "import importlib.util; s=importlib.util.find_spec('tensorrt_libs'); print(s.submodule_search_locations[0])") && \
    [ ! -f "$TRT_LIB/libnvinfer.so" ] && ln -sf libnvinfer.so.10 "$TRT_LIB/libnvinfer.so" || true && \
    echo "$TRT_LIB" > /etc/ld.so.conf.d/tensorrt.conf && \
    ldconfig

# ── Environment ─────────────────────────────────────────────────────────────
# Pre-compute paths so cmake / runtime find TRT without manual exports
ENV TRT_LIB_DIR=/opt/venv/lib/python3.12/site-packages/tensorrt_libs
ENV TRT_INC_DIR=/usr/include/aarch64-linux-gnu
ENV LD_LIBRARY_PATH="$TRT_LIB_DIR:/usr/local/cuda/lib64"

# Coverage tooling verification (run inside container):
#   python3 -m coverage --version && pytest --version && \
#   python3 -m pytest --help | grep -- '--cov' && \
#   gcovr --version && lcov --version && genhtml --version

WORKDIR /workspace/tensorrt-model-connect

CMD ["bash"]

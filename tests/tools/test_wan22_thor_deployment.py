# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static contracts for the customer-facing Wan2.2 Thor deployment."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
WAN22_FAMILY = (
    REPO_ROOT / "python/tensorrt_model_connect/families/wan2_2_ti2v"
)
DOCKERFILE = WAN22_FAMILY / "Dockerfile.thor"
DOCKERIGNORE = WAN22_FAMILY / "Dockerfile.thor.dockerignore"
GUIDE = REPO_ROOT / "website/docs/getting-started/wan2-2-thor-from-scratch.md"


def test_wan22_thor_image_uses_only_official_tensorrt_11_1() -> None:
    assert (DOCKERFILE.parent / "MODEL.toml").is_file()
    text = DOCKERFILE.read_text(encoding="utf-8")

    required = (
        "nvcr.io/nvidia/tensorrt:26.07-py3@sha256:",
        "TENSORRT_VERSION=11.1.0.106",
        "com.nvidia.tensorrt.version",
        "TRT_MAJOR_ENTERPRISE 11",
        "TRT_MINOR_ENTERPRISE 1",
        "TRT_PATCH_ENTERPRISE 0",
        "TRT_BUILD_ENTERPRISE 106",
        "libnvinfer.so.11",
        "libnvinfer_builder_resource_sm110.so.11.1.0",
        "m.version('tensorrt') == '${TENSORRT_VERSION}'",
        "TORCH_VERSION=2.12.0+cu130",
        "TRANSFORMERS_VERSION=5.2.0",
        "-DCMAKE_CUDA_ARCHITECTURES=110",
        "-DTRTMC_BUILD_TESTS=OFF",
        "-DTRTMC_BUILD_BENCHMARKS=OFF",
        "-DTRTMC_ENABLE_LIBTORCH_MULTINOMIAL=OFF",
        "-DTRTMC_ENABLE_TVM_FFI=OFF",
        '-DTRTMC_SOURCE_REVISION="${TRTMC_SOURCE_REVISION}"',
        "trtmc_model_wan2_2_ti2v",
        "chmod -R a+rX /opt/tensorrt-model-connect",
        'grep -q "not found" /tmp/trtmc-ldd.txt',
        'ENTRYPOINT ["trtmc"]',
    )
    for token in required:
        assert token in text

    forbidden = (
        "11.2.0.113",
        "ghcr.io/",
        "artifactory",
        "tensorrt_sdk",
        "TENSORRT_SDK_IMAGE",
        "tensorrt-sdk",
        "/opt/tensorrt/python",
        "TORCH_CUDA_ARCH_LIST=10.0",
        "TRTMC_WAN22_",
        "--no-deps",
        "GITHUB_TOKEN",
        "GHCR_TOKEN",
    )
    for token in forbidden:
        assert token not in text


def test_wan22_thor_context_excludes_credentials_and_build_outputs() -> None:
    text = DOCKERIGNORE.read_text(encoding="utf-8")

    for pattern in (
        ".*",
        ".git",
        ".env",
        ".env.*",
        ".netrc",
        ".npmrc",
        ".pypirc",
        ".ssh",
        "*.key",
        "*.pem",
        "artifacts",
        "build",
    ):
        assert f"\n{pattern}\n" in text
    assert "\n*\n" not in text


def test_wan22_thor_guide_covers_official_11_1_full_generation() -> None:
    text = GUIDE.read_text(encoding="utf-8")

    required = (
        "TensorRT 11.1.0.106",
        "nvcr.io/nvidia/tensorrt:26.07-py3",
        "nvidia-ctk runtime configure --runtime=docker",
        "nvidia/cuda:13.3.0-base-ubuntu24.04 nvidia-smi",
        "Jetson AGX Thor Quick Start Guide",
        "python/tensorrt_model_connect/families/wan2_2_ti2v/Dockerfile.thor",
        "--model-revision 921dbaf3f1674a56f47e83fb80a34bac8a8f203e",
        "--set wan2_2_ti2v.easycache_enabled=true",
        "--set wan2_2_ti2v.easycache_threshold=1.0",
        "--set wan2_2_ti2v.easycache_max_consecutive_reuse=4",
        "--set wan2_2_ti2v.late_cfg_enabled=true",
        "1280x704",
        "121 PNG files",
        "wan22-thor.effective_config.json",
        "wan22-thor.mp4",
    )
    for token in required:
        assert token in text

    for token in ("11.2.0.113", "ghcr.io/", "artifactory", "TRTMC_WAN22_"):
        assert token not in text

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The small host-to-container environment contract used by active CI."""

from __future__ import annotations


COMMON_ENVIRONMENT = (
    "CI_BASE_REF",
    "GITHUB_EVENT_NAME",
    "GITHUB_REF_NAME",
    "GITHUB_RUN_ID",
    "GITHUB_RUN_ATTEMPT",
    "TRTMC_PREMERGE_UNIT_SCOPE",
    "SOURCE_QUALITY_TIMEOUT",
    "PYTHON_UNIT_TIMEOUT",
    "CPP_BUILD_TIMEOUT",
    "CPP_UNIT_TIMEOUT",
)

TRUSTED_ENVIRONMENT = COMMON_ENVIRONMENT + (
    "TRTMC_BINARY",
    "TRTMC_RUNTIME_ROOT",
    "TRTMC_NATIVE_BUILD_DIR",
    "TRTMC_E2E_TIMEOUT",
    "TRTMC_CI_STATE_DIR",
    "TRT_ROOT",
    "CMAKE_CUDA_ARCHITECTURES",
    "HF_HOME",
    "HF_HUB_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "HF_MODULES_CACHE",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
)

OPTIONAL_HUGGING_FACE_ENVIRONMENT = (
    "HF_HUB_DISABLE_XET",
    "HF_HUB_DOWNLOAD_TIMEOUT",
    "HF_HUB_ETAG_TIMEOUT",
)


def forwarded_environment(names: tuple[str, ...], env: dict[str, str]) -> tuple[str, ...]:
    model_directories = sorted(
        name
        for name, value in env.items()
        if name.startswith("TRTMC_") and name.endswith("_MODEL_DIR") and value
    )
    return tuple(dict.fromkeys((*names, *model_directories)))

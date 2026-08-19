# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Import-only decord compatibility for Lance's official x2t_image path.

The upstream validation dataset imports decord before it dispatches by input
modality. PyPI does not publish decord 0.6.0 for Linux aarch64, and image-only
inference never calls its video APIs. Keep those APIs fail-closed so this shim
cannot silently make a video workload look supported.
"""

from __future__ import annotations


_ERROR = (
    "decord video APIs are unavailable in the image-only Lance reference; "
    "install upstream decord and use a video-capable reference environment"
)


class VideoReader:
    def __init__(self, *_args, **_kwargs) -> None:
        raise RuntimeError(_ERROR)


def cpu(*_args, **_kwargs):
    raise RuntimeError(_ERROR)


__all__ = ["VideoReader", "cpu"]

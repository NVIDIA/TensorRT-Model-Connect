# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve the packaged Wan2.2 AOT plugin companion for VAE builds."""

from pathlib import Path

from .cuda_plugin_companion import load_wan22_plugin_companion


def ensure_vae_cuda_plugin(*, verbose: bool = False) -> Path:
    """Return the packaged companion DSO; never compile source at build time."""

    return load_wan22_plugin_companion(verbose=verbose).load_path


__all__ = ["ensure_vae_cuda_plugin"]

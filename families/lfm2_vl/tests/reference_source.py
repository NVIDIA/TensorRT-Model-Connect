# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Truthful source metadata for regenerating this family's reference fixtures."""

from pathlib import Path

CHECKPOINT = "LiquidAI/LFM2.5-VL-1.6B"
REVISION = "919fde3d022e3f90a4716006f993938ee8c2eb97"


def reference_source(checkpoint):
    if Path(checkpoint).is_dir():
        return {"local_files_only": True}, {
            "checkpoint": "local",
            "revision": None,
            "provenance": {"source": "local", "revision_verified": False},
        }
    if checkpoint != CHECKPOINT:
        raise ValueError("expected the pinned repository ID or an existing local directory")
    return {"revision": REVISION}, {
        "checkpoint": CHECKPOINT,
        "revision": REVISION,
        "provenance": {"source": "hugging_face", "revision_verified": True},
    }

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Flux-owned family plugin contracts."""

from __future__ import annotations

try:
    from tensorrt_model_connect.config import ModelConfig
    from tensorrt_model_connect.families.flux import model as flux_mod
except (ImportError, ModuleNotFoundError):
    import pytest
    pytest.skip("tensorrt_model_connect requires tensorrt", allow_module_level=True)


def test_flux_pipeline_classes_resolve_to_flux_plugin() -> None:
    """Flux owns the real Diffusers pipeline class mapping for Flux models."""
    for pipeline_class in ("FluxPipeline", "Flux2Pipeline"):
        config = ModelConfig(model_type="flux", raw={"_class_name": pipeline_class})
        assert flux_mod.matches(config)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused checks for the family-owned FLUX TensorRT build policy."""

from pathlib import Path

import pytest


FAMILY_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize(
    ("builder", "config_count"),
    (
        ("clip_encoder_builder.py", 1),
        ("t5_encoder_builder.py", 2),
        ("mistral_encoder_builder.py", 1),
        ("flux_vae_builder.py", 1),
        ("flux_dit_builder.py", 2),
        ("flux_dit_tp_builder.py", 1),
        ("flux_dit_cp_builder.py", 1),
        ("flux2_dit_builder.py", 1),
        ("flux2_dit_tp_builder.py", 1),
    ),
)
def test_every_flux_builder_config_uses_optimization_level_one(
    builder: str, config_count: int
) -> None:
    source = (FAMILY_ROOT / builder).read_text(encoding="utf-8")

    assert source.count("config = builder.create_builder_config()") == config_count
    assert source.count("config.builder_optimization_level = 1") == config_count


def test_preprocessor_keeps_core_weights_required() -> None:
    source = (FAMILY_ROOT / "runtime/diffusion_helpers.cpp").read_text(encoding="utf-8")

    for key in (
        "patch_embedding.weight",
        "condition_embedder.time_embedding.0.weight",
        "condition_embedder.time_embedding.2.weight",
        "context_embedder.weight",
    ):
        call = "\n".join(source.split(f'"{key}"', maxsplit=1)[0].splitlines()[-2:])
        assert "load_preprocessor_floats" in call
        assert "load_optional_preprocessor_floats" not in call

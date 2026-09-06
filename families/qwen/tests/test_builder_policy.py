# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused checks for the family-owned Qwen TensorRT build policy."""

from pathlib import Path


FAMILY_ROOT = Path(__file__).resolve().parent.parent


def test_qwen_builders_preserve_mainline_optimization_level() -> None:
    for filename in (
        "dual_profile_decoder_builder.py",
        "dual_profile_decoder_tp_builder.py",
        "standard_decoder_builder.py",
    ):
        source = (FAMILY_ROOT / filename).read_text(encoding="utf-8")
        assert source.count("create_builder_config()") == 1
        assert source.count("builder_optimization_level = 3") == 1


def test_qwen_runtime_uses_the_checkpoint_chat_template_without_fallback() -> None:
    plugin = (FAMILY_ROOT / "runtime/plugin.cpp").read_text(encoding="utf-8")
    model = (FAMILY_ROOT / "model.py").read_text(encoding="utf-8")

    assert 'require_text_section(bundle, "tokenizer_config.json")' in plugin
    assert 'config.find("chat_template")' in plugin
    assert "chat_template.jinja" not in plugin
    assert '"chat_template.jinja"' not in model

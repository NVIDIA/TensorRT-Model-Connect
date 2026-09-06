# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused checks for the family-owned Qwen3-Omni Thinker build policy."""

from pathlib import Path


FAMILY_ROOT = Path(__file__).resolve().parent.parent


def test_builds_only_the_qualified_thinker_text_engine() -> None:
    thinker = (FAMILY_ROOT / "model.py").read_text(encoding="utf-8")

    assert thinker.count("create_builder_config()") == 1
    assert thinker.count("builder_optimization_level = 3") == 1
    assert 'request.task != "text_generation"' in thinker
    assert 'writer.add_bytes("thinker.plan", thinker_plan)' in thinker
    assert "talker.plan" not in thinker
    assert "code2wav.plan" not in thinker


def test_thinker_preserves_mainline_bf16_graph_boundaries() -> None:
    thinker = (FAMILY_ROOT / "model.py").read_text(encoding="utf-8")

    assert "work_np_dtype = np.float16" in thinker
    assert "router_logits = network.add_cast" not in thinker
    assert "routed_output = network.add_cast" not in thinker
    assert "fp32_opmath=True" not in thinker

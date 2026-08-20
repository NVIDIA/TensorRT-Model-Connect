# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tensorrt_model_connect import engine_builder


def _select(monkeypatch: pytest.MonkeyPatch, plugin: object) -> None:
    monkeypatch.setattr(engine_builder, "_setup_trt_import", lambda _rtx: None)
    monkeypatch.setattr(engine_builder, "_resolve_diffusion_entrypoint", lambda _path: None)
    monkeypatch.setattr(engine_builder.ModelConfig, "from_dir", lambda _path: object())
    monkeypatch.setattr(engine_builder, "find_plugin", lambda _config: plugin)


def test_declared_model_owned_build_receives_complete_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class Plugin:
        name = "pilot"

        def build(self, model_dir: str, output_path: str, *, options: dict[str, object]) -> None:
            captured.update(
                model_dir=model_dir,
                output_path=output_path,
                options=options,
            )

    _select(monkeypatch, Plugin())
    monkeypatch.setattr(
        engine_builder,
        "normalize_parallel_config",
        lambda _value: pytest.fail("legacy orchestration must not run"),
    )

    engine_builder.build_bundle(
        "/models/pilot",
        "/tmp/pilot.bundle",
        max_cache_length=64,
        precision="fp16",
        family_build_options={"pilot.option": True},
    )

    assert captured["model_dir"] == "/models/pilot"
    assert captured["output_path"] == "/tmp/pilot.bundle"
    options = captured["options"]
    assert isinstance(options, dict)
    assert options["max_cache_length"] == 64
    assert options["precision"] == "fp16"
    assert options["family_build_options"] == {"pilot.option": True}
    assert "model_dir" not in options
    assert "output_path" not in options


def test_magic_mock_does_not_invent_model_owned_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _select(monkeypatch, MagicMock(name="legacy_plugin"))

    def legacy_marker(_value: object) -> object:
        raise RuntimeError("entered legacy orchestration")

    monkeypatch.setattr(engine_builder, "normalize_parallel_config", legacy_marker)

    with pytest.raises(RuntimeError, match="entered legacy orchestration"):
        engine_builder.build_bundle("/models/legacy", "/tmp/legacy.bundle")


def test_declared_noncallable_build_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Plugin:
        name = "broken"
        build = "not-callable"

    _select(monkeypatch, Plugin())

    with pytest.raises(TypeError, match="build attribute is not callable"):
        engine_builder.build_bundle("/models/broken", "/tmp/broken.bundle")


@pytest.mark.parametrize(
    "relative",
    (
        "python/tensorrt_model_connect/families/bert/model/model.py",
        "python/tensorrt_model_connect/families/gpt2/model.py",
        "python/tensorrt_model_connect/families/timm_vit/model/model.py",
    ),
)
def test_migrated_model_builds_do_not_import_engine_builder(relative: str) -> None:
    root = Path(__file__).resolve().parents[2]
    tree = ast.parse((root / relative).read_text(encoding="utf-8"))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert not [name for name in imported if name.endswith("engine_builder")]

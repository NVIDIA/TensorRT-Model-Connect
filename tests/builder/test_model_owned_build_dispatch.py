# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded contracts for the model-owned build dispatch bridge."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
import tomllib
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tensorrt_model_connect import engine_builder


REPO_ROOT = Path(__file__).resolve().parents[2]
FAMILIES_ROOT = REPO_ROOT / "python" / "tensorrt_model_connect" / "families"
ENGINE_BUILDER = REPO_ROOT / "python" / "tensorrt_model_connect" / "engine_builder.py"
MODEL_OWNED_BUILD = "model_owned_build"


class OwnerFailure(RuntimeError):
    """Sentinel proving an owner failure is not replaced by legacy dispatch."""


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        model_type="marked_owner",
        architectures=[],
        raw={},
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
    )


def _select(
    monkeypatch: pytest.MonkeyPatch,
    plugin: object,
    *,
    marked: bool = True,
) -> None:
    config = _config()
    monkeypatch.setattr(engine_builder, "_setup_trt_import", lambda _rtx: None)
    monkeypatch.setattr(
        engine_builder.trt_compat,
        "resolved_summary",
        lambda: "mock TensorRT",
    )
    monkeypatch.setattr(engine_builder, "_resolve_diffusion_entrypoint", lambda _path: None)
    monkeypatch.setattr(engine_builder.ModelConfig, "from_dir", lambda _path: config)
    monkeypatch.setattr(engine_builder, "_apply_family_builder_capabilities", lambda _config: None)
    monkeypatch.setattr(engine_builder, "find_plugin", lambda _config: plugin)

    def has_capability(_config, capability: str) -> bool:
        assert capability == MODEL_OWNED_BUILD
        return marked

    monkeypatch.setattr(engine_builder, "family_has_capability", has_capability)
    monkeypatch.setattr(engine_builder, "_new_build_timing", lambda _path: {"phases": {}})
    monkeypatch.setattr(engine_builder, "_write_build_timing", lambda _timing: None)


def _forbid_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    def legacy(*_args, **_kwargs):
        raise AssertionError("marked owner must not enter legacy orchestration")

    monkeypatch.setattr(engine_builder, "_load_plugin_weights", legacy)
    monkeypatch.setattr(engine_builder, "write_bundle", legacy)


def test_marked_owner_receives_every_original_option_without_legacy_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def build(model_dir: str, output_path: str, **options: object) -> None:
        captured.update(
            model_dir=model_dir,
            output_path=output_path,
            options=options,
        )

    plugin = SimpleNamespace(name="marked_owner")
    plugin.build = build
    _select(monkeypatch, plugin)
    _forbid_legacy(monkeypatch)

    engine_builder.build_bundle(
        "/models/marked",
        "/tmp/marked.bundle",
        max_cache_length=0,
        precision=None,
        quant_calibration_samples=0,
        kernel_artifacts=[],
        family_build_options={},
        diffusion_overrides={},
        build_timing_path="/tmp/marked-timing.json",
        max_batch_size=0,
        tokenizer_source_model_id_or_path="source/model",
        tokenizer_source_revision="revision",
    )

    assert captured["model_dir"] == "/models/marked"
    assert captured["output_path"] == "/tmp/marked.bundle"
    options = captured["options"]
    assert isinstance(options, dict)
    expected = set(inspect.signature(engine_builder.build_bundle).parameters) - {
        "model_dir",
        "output_path",
    }
    assert set(options) == expected
    assert options["max_cache_length"] == 0
    assert options["precision"] is None
    assert options["quant_calibration_samples"] == 0
    assert options["max_batch_size"] == 0
    assert options["tokenizer_source_model_id_or_path"] == "source/model"
    assert options["tokenizer_source_revision"] == "revision"


def _missing_plugin() -> object:
    return SimpleNamespace(name="marked_owner")


def _none_plugin() -> object:
    return SimpleNamespace(name="marked_owner", build=None)


def _noncallable_plugin() -> object:
    return SimpleNamespace(name="marked_owner", build="not callable")


def _dynamic_plugin() -> object:
    plugin = MagicMock()
    plugin.name = "marked_owner"
    return plugin


@pytest.mark.parametrize(
    "plugin_factory",
    (_missing_plugin, _none_plugin, _noncallable_plugin, _dynamic_plugin),
    ids=("missing", "none", "noncallable", "dynamic-magic-mock"),
)
def test_marked_owner_requires_a_concrete_direct_callable(
    monkeypatch: pytest.MonkeyPatch,
    plugin_factory,
) -> None:
    _select(monkeypatch, plugin_factory())
    _forbid_legacy(monkeypatch)

    with pytest.raises(TypeError, match="concrete build binding|not callable"):
        engine_builder.build_bundle("/models/marked", "/tmp/marked.bundle")


def test_marked_owner_exception_propagates_without_legacy_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def build(_model_dir: str, _output_path: str, **_options: object) -> None:
        raise OwnerFailure("owner build failed")

    plugin = SimpleNamespace(name="marked_owner")
    plugin.build = build
    _select(monkeypatch, plugin)
    _forbid_legacy(monkeypatch)

    with pytest.raises(OwnerFailure, match="owner build failed"):
        engine_builder.build_bundle("/models/marked", "/tmp/marked.bundle")


def test_structurally_selected_owner_uses_plugin_id_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def build(_model_dir: str, _output_path: str, **_options: object) -> None:
        nonlocal called
        called = True

    plugin = SimpleNamespace(name="marked_owner")
    plugin.build = build
    _select(monkeypatch, plugin)
    _forbid_legacy(monkeypatch)
    monkeypatch.setattr(
        engine_builder,
        "family_has_capability",
        lambda owner, capability: capability == MODEL_OWNED_BUILD and owner == "marked_owner",
    )

    engine_builder.build_bundle("/models/structural", "/tmp/structural.bundle")

    assert called


def test_unmarked_plugin_keeps_the_legacy_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def build(*_args, **_kwargs):
        raise AssertionError("unmarked plugin build binding must be ignored")

    plugin = SimpleNamespace(
        name="legacy_owner",
        runtime_strategy="",
        default_build_precision="fp32",
    )
    plugin.build = build
    _select(monkeypatch, plugin, marked=False)

    def reached_legacy(*_args, **_kwargs):
        raise OwnerFailure("entered legacy orchestration")

    monkeypatch.setattr(engine_builder, "_load_plugin_weights", reached_legacy)

    with pytest.raises(OwnerFailure, match="entered legacy orchestration"):
        engine_builder.build_bundle("/models/legacy", "/tmp/legacy.bundle")


def _marked_owner_dirs() -> list[Path]:
    owners = []
    for descriptor_path in sorted(FAMILIES_ROOT.glob("*/MODEL.toml")):
        with descriptor_path.open("rb") as stream:
            descriptor = tomllib.load(stream)
        capabilities = descriptor.get("capabilities", ())
        if isinstance(capabilities, str):
            capabilities = (capabilities,)
        if MODEL_OWNED_BUILD in capabilities:
            owners.append(descriptor_path.parent)
    return owners


def _imports_engine_builder(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.endswith("engine_builder") for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.endswith("engine_builder"):
                return True
            if any(alias.name == "engine_builder" for alias in node.names):
                return True
    return False


def _direct_build_bindings(path: Path) -> list[ast.Assign]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    bindings = []
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "plugin"
            and target.attr == "build"
            for target in node.targets
        ):
            bindings.append(node)
    return bindings


def test_marked_owner_builds_are_direct_and_self_contained() -> None:
    owners = _marked_owner_dirs()
    assert owners

    violations = []
    for owner in owners:
        plugin_path = owner / "plugin.py"
        bindings = _direct_build_bindings(plugin_path)
        if len(bindings) != 1 or not isinstance(bindings[0].value, (ast.Name, ast.Attribute)):
            violations.append(f"{owner.name}: plugin.build must have one direct binding")
        for source in sorted(owner.rglob("*.py")):
            if "tests" in source.relative_to(owner).parts:
                continue
            if _imports_engine_builder(source):
                violations.append(
                    f"{owner.name}: {source.relative_to(REPO_ROOT)} imports engine_builder"
                )

    engine_tree = ast.parse(
        ENGINE_BUILDER.read_text(encoding="utf-8"),
        filename=str(ENGINE_BUILDER),
    )
    owner_names = {owner.name for owner in owners}
    leaked_names = sorted(
        {
            node.value
            for node in ast.walk(engine_tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value in owner_names
        }
    )
    if leaked_names:
        violations.append(f"engine_builder contains marked owner literals: {leaked_names}")

    assert not violations, "\n".join(violations)

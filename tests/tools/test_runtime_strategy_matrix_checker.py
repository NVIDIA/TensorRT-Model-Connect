# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the owner-derived runtime-strategy control-plane checker."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.e2e_harness.runtime_strategy_metadata import (
    clear_runtime_strategy_metadata_cache,
    load_runtime_strategy_catalog,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _checker():
    return importlib.import_module("check_runtime_strategy_matrix")


def _write_owner(
    models_dir: Path,
    owner: str,
    strategies: tuple[str, ...],
    task_strategy: str,
    *,
    performance_mode: str | None = None,
    diff_checks: tuple[str, ...] = (),
    runner: bool = True,
    comparator: bool = True,
) -> Path:
    owner_dir = models_dir / owner
    manifests_dir = owner_dir / "tests" / "manifests"
    manifests_dir.mkdir(parents=True)
    descriptor = [
        f'id = "{owner}"',
        "runtime_strategies = [",
        *(f'  "{strategy}",' for strategy in strategies),
        "]",
        "test_manifests = [",
        *(f'  "tests/manifests/{index}.json",' for index in range(len(strategies))),
        "]",
    ]
    if performance_mode is not None:
        descriptor.append(f'performance_mode = "{performance_mode}"')
    if diff_checks:
        descriptor.extend(
            [
                "diff_framework_check_classes = [",
                *(f'  "{name}",' for name in diff_checks),
                "]",
            ]
        )
    (owner_dir / "MODEL.toml").write_text("\n".join(descriptor) + "\n")

    for index, strategy in enumerate(strategies):
        (manifests_dir / f"{index}.json").write_text(
            json.dumps(
                {
                    "name": f"{owner}-{index}",
                    "hf_id": f"unit/{owner}",
                    "family": owner,
                    "runtime_strategy": strategy,
                    "task_strategy": task_strategy,
                }
            )
        )

    plugins = owner_dir / "tests" / "e2e_plugins"
    plugins.mkdir(parents=True)
    if runner:
        (plugins / "runner.py").write_text(
            f'class UnitRunner:\n    def strategy_name(self):\n        return "{task_strategy}"\n'
        )
    if comparator:
        (plugins / "comparator.py").write_text(
            "class UnitComparator:\n"
            "    def task_strategy(self):\n"
            f'        return "{task_strategy}"\n'
        )
    return owner_dir


def _write_diff_check(checks_dir: Path, class_name: str) -> None:
    checks_dir.mkdir(parents=True, exist_ok=True)
    (checks_dir / f"{class_name.lower()}.py").write_text(
        f'class {class_name}:\n    name = "unit"\n'
    )


def test_repository_has_no_central_runtime_strategy_matrix() -> None:
    assert not (REPO_ROOT / "tests" / "runtime_strategy_matrix.yaml").exists()


def test_repository_control_plane_is_consistent() -> None:
    assert _checker().validate_control_plane_paths() == []


def test_catalog_derives_task_defaults_and_owner_performance_override(
    tmp_path: Path,
) -> None:
    models = tmp_path / "models"
    _write_owner(models, "decoder", ("decoder_runtime",), "text_generation_causal")
    _write_owner(
        models,
        "seq2seq",
        ("seq2seq_runtime",),
        "text_generation_causal",
        performance_mode="enc_dec",
    )
    clear_runtime_strategy_metadata_cache()
    catalog = load_runtime_strategy_catalog(models)

    assert catalog["decoder_runtime"].performance_mode == "decode"
    assert catalog["decoder_runtime"].cli_commands == ("run",)
    assert catalog["seq2seq_runtime"].performance_mode == "enc_dec"


def test_catalog_rejects_duplicate_runtime_strategy_owners(tmp_path: Path) -> None:
    models = tmp_path / "models"
    _write_owner(models, "alpha", ("duplicate_runtime",), "embedding")
    _write_owner(models, "beta", ("duplicate_runtime",), "embedding")
    clear_runtime_strategy_metadata_cache()
    with pytest.raises(ValueError, match="declared by both"):
        load_runtime_strategy_catalog(models)


def test_catalog_rejects_manifest_strategy_outside_owner(tmp_path: Path) -> None:
    models = tmp_path / "models"
    owner = _write_owner(models, "alpha", ("alpha_runtime",), "embedding")
    manifest = owner / "tests" / "manifests" / "0.json"
    payload = json.loads(manifest.read_text())
    payload["runtime_strategy"] = "foreign_runtime"
    manifest.write_text(json.dumps(payload))
    clear_runtime_strategy_metadata_cache()
    with pytest.raises(ValueError, match="is not declared by owner"):
        load_runtime_strategy_catalog(models)


def test_catalog_rejects_missing_task_strategy(tmp_path: Path) -> None:
    models = tmp_path / "models"
    owner = _write_owner(models, "alpha", ("alpha_runtime",), "embedding")
    manifest = owner / "tests" / "manifests" / "0.json"
    payload = json.loads(manifest.read_text())
    del payload["task_strategy"]
    manifest.write_text(json.dumps(payload))
    clear_runtime_strategy_metadata_cache()
    with pytest.raises(ValueError, match="task_strategy is required"):
        load_runtime_strategy_catalog(models)


def test_catalog_rejects_performance_sidecar_for_foreign_strategy(
    tmp_path: Path,
) -> None:
    models = tmp_path / "models"
    owner = _write_owner(models, "alpha", ("alpha_runtime",), "embedding")
    (owner / "tests" / "perf_validation.json").write_text(
        json.dumps({"models": [{"model": "unit/alpha", "pipeline_type": "foreign_runtime"}]})
    )
    clear_runtime_strategy_metadata_cache()
    with pytest.raises(ValueError, match="is not declared by owner"):
        load_runtime_strategy_catalog(models)


def test_checker_requires_owner_runner_and_comparator(tmp_path: Path) -> None:
    models = tmp_path / "models"
    _write_owner(
        models,
        "alpha",
        ("alpha_runtime",),
        "embedding",
        runner=False,
        comparator=False,
    )
    errors = _checker().validate_control_plane_paths(
        models_dir=models, diff_checks_dir=tmp_path / "checks"
    )
    assert any("has no runner class" in error for error in errors)
    assert any("has no comparator class" in error for error in errors)


def test_checker_accepts_owner_cli_exemption_when_no_generic_command() -> None:
    metadata = SimpleNamespace(
        owner="specialized",
        task_strategy="prompted_segmentation",
        cli_commands=(),
        cli_exemption="Uses a model-owned public C ABI.",
        diff_framework_check_classes=(),
    )

    assert _checker().validate_control_plane_data(
        catalog={"specialized_runtime": metadata},
        runner_classes_by_owner_task={
            "specialized": {"prompted_segmentation": {"SpecializedRunner"}}
        },
        comparator_classes_by_owner_task={
            "specialized": {"prompted_segmentation": {"SpecializedComparator"}}
        },
        diff_check_classes=set(),
    ) == []


def test_checker_requires_cli_exemption_when_no_generic_command() -> None:
    metadata = SimpleNamespace(
        owner="specialized",
        task_strategy="prompted_segmentation",
        cli_commands=(),
        cli_exemption=None,
        diff_framework_check_classes=(),
    )

    errors = _checker().validate_control_plane_data(
        catalog={"specialized_runtime": metadata},
        runner_classes_by_owner_task={
            "specialized": {"prompted_segmentation": {"SpecializedRunner"}}
        },
        comparator_classes_by_owner_task={
            "specialized": {"prompted_segmentation": {"SpecializedComparator"}}
        },
        diff_check_classes=set(),
    )

    assert errors == [
        "specialized_runtime: cli_exemption is required when no CLI command exists"
    ]


def test_checker_rejects_unknown_owner_diff_check(tmp_path: Path) -> None:
    models = tmp_path / "models"
    _write_owner(
        models,
        "alpha",
        ("alpha_runtime",),
        "embedding",
        diff_checks=("MissingCheck",),
    )
    errors = _checker().validate_control_plane_paths(
        models_dir=models, diff_checks_dir=tmp_path / "checks"
    )
    assert errors == [
        "alpha_runtime: owner descriptor references unknown diff-framework classes ['MissingCheck']"
    ]


def test_checker_accepts_local_diff_check_and_nested_task_plugins(
    tmp_path: Path,
) -> None:
    models = tmp_path / "models"
    owner = _write_owner(
        models,
        "alpha",
        ("alpha_runtime",),
        "embedding",
        diff_checks=("EmbeddingCheck",),
    )
    plugins = owner / "tests" / "e2e_plugins"
    (plugins / "runner.py").unlink()
    (plugins / "comparator.py").unlink()
    (plugins / "runners").mkdir()
    (plugins / "comparators").mkdir()
    (plugins / "runners" / "embedding.py").write_text(
        'class EmbeddingRunner:\n    def strategy_name(self):\n        return "embedding"\n'
    )
    (plugins / "comparators" / "embedding.py").write_text(
        'class EmbeddingComparator:\n    def task_strategy(self):\n        return "embedding"\n'
    )
    checks = tmp_path / "checks"
    _write_diff_check(checks, "EmbeddingCheck")

    assert _checker().validate_control_plane_paths(models_dir=models, diff_checks_dir=checks) == []


def test_checker_cli_has_no_matrix_or_legacy_source_options() -> None:
    parser = _checker().build_arg_parser()
    option_strings = {option for action in parser._actions for option in action.option_strings}
    assert "--models-dir" in option_strings
    assert "--matrix" not in option_strings
    assert "--runtime-models-dir" not in option_strings
    assert "--e2e-models-dir" not in option_strings

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tools import model_plugin_isolation
from tools.ci.model_proof_selection import ModelProofSelector
from tools.ci.process import CiError


REPO_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_CMAKE = REPO_ROOT / "cmake" / "trtmc_pipeline_plugins.cmake"
REVISION = "a" * 40


def _make_project(tmp_path: Path, descriptor_tail: str = "") -> Path:
    source = tmp_path / "source"
    model = source / "python" / "tensorrt_model_connect" / "models" / "alpha"
    (model / "runtime").mkdir(parents=True)
    (model / "runtime" / "plugin.cpp").write_text("// fixture\n", encoding="utf-8")
    (model / "runtime" / "CMakeLists.txt").write_text(
        "add_library(trtmc_model_alpha SHARED plugin.cpp)\n", encoding="utf-8"
    )
    (model / "MODEL.toml").write_text(
        'id = "alpha"\n'
        'runtime_plugins = ["plugin.cpp|register_alpha"]\n'
        'runtime_strategies = ["alpha_runtime"]\n'
        f"{descriptor_tail}",
        encoding="utf-8",
    )

    manifests = model / "tests" / "manifests"
    manifests.mkdir(parents=True)
    (manifests / "alpha-small.json").write_text(
        json.dumps(
            {
                "name": "alpha-small",
                "family": "alpha",
                "runtime_strategy": "alpha_runtime",
                "testcases": [{"name": "alpha-small", "ci_tier": "l0_only"}],
            }
        ),
        encoding="utf-8",
    )
    (model / "tests" / "test_alpha_e2e.py").write_text("# fixture\n", encoding="utf-8")
    (source / ".trtmc-model-projection.json").write_text(
        json.dumps(
            {
                "revision": REVISION,
                "model": "alpha",
                "runtime_model": "alpha",
                "e2e_family": "alpha",
            }
        ),
        encoding="utf-8",
    )
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.20)\n"
        "project(runtime_descriptor_contract NONE)\n"
        f'include("{PLUGIN_CMAKE}")\n',
        encoding="utf-8",
    )
    return source


def _configure(source: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(source / "build"),
            "-DTRTMC_MODEL_PROOF_MODEL=alpha",
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def _select(source: Path) -> dict[str, object]:
    output = source / "selection.json"
    selection = ModelProofSelector("alpha", "premerge", REVISION, source).select(output)
    return selection.payload


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("runtime_library", '"libcustom.so"'),
        ("runtime_strategy", '"alpha_runtime"'),
        ("default_runtime_strategy", '"alpha_runtime"'),
        ("legacy_runtime_strategy_aliases", '["old||||alpha_runtime"]'),
        ("runtime_tests", '["test_alpha|test_alpha.cpp|_|_|_"]'),
        ("runtime_link_libraries", '["cublas"]'),
        ("gnu_warning_suppressed_sources", '["plugin.cpp"]'),
    ),
)
def test_retired_fields_fail_closed_across_runtime_consumers(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    source = _make_project(tmp_path, f"{field} = {value}\n")

    with pytest.raises(SystemExit, match=field):
        model_plugin_isolation.discover_runtime_plugins(source)
    with pytest.raises(CiError, match=field):
        _select(source)
    configured = _configure(source)

    assert configured.returncode != 0
    assert field in configured.stdout + configured.stderr
    assert "Retired" in configured.stdout + configured.stderr


def test_missing_runtime_strategies_fails_closed_across_runtime_consumers(
    tmp_path: Path,
) -> None:
    source = _make_project(tmp_path)
    descriptor = source / "python" / "tensorrt_model_connect" / "models" / "alpha" / "MODEL.toml"
    descriptor.write_text(
        descriptor.read_text(encoding="utf-8").replace(
            'runtime_strategies = ["alpha_runtime"]\n', ""
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="runtime_strategies"):
        model_plugin_isolation.discover_runtime_plugins(source)
    with pytest.raises(CiError, match="runtime_strategies"):
        _select(source)
    configured = _configure(source)

    assert configured.returncode != 0
    assert "runtime_strategies" in configured.stdout + configured.stderr


def test_runtime_library_is_owner_derived_across_runtime_consumers(tmp_path: Path) -> None:
    source = _make_project(tmp_path)
    expected = "libtrtmc_model_alpha.so"

    assert model_plugin_isolation.discover_runtime_plugins(source)["alpha"].library == expected
    assert _select(source)["runtime_library"] == expected

    configured = _configure(source)
    assert configured.returncode == 0, configured.stdout + configured.stderr
    generated_index = source / "build" / "generated" / "model_plugin_index.cpp"
    assert expected in generated_index.read_text(encoding="utf-8")

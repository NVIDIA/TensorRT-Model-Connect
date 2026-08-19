# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

from tools import family_specialization as specialization


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _demo_repo(tmp_path: Path) -> Path:
    family = (
        tmp_path
        / "python"
        / "tensorrt_model_connect"
        / "families"
        / "demo"
    )
    _write(
        family / "MODEL.toml",
        'id = "demo"\nplugin = "demo"\nmodule = "plugin"\n'
        'debug_runner = "model/runtime.py|runner_from_bundle"\n',
    )
    _write(
        family / "__init__.py",
        'from .plugin import plugin\n\n__all__ = ["plugin"]\n',
    )
    _write(
        family / "plugin.py",
        "from .model.model import build_decoder\n\n"
        "class DemoPlugin:\n"
        "    def build_engine(self):\n"
        "        return build_decoder(norm_type='rmsnorm')\n\n"
        "plugin = DemoPlugin()\n",
    )
    _write(family / "config.py", "class ModelConfig:\n    pass\n")
    _write(family / "weights/__init__.py", "class WeightDict(dict):\n    pass\n")
    _write(family / "model/__init__.py", '"""Demo model package."""\n')
    _write(
        family / "model/model.py",
        "def used_helper():\n"
        "    return 1\n\n"
        "def unused_helper():\n"
        "    return 2\n\n"
        "def build_decoder(*, norm_type='rmsnorm'):\n"
        "    if norm_type == 'rmsnorm':\n"
        "        return used_helper()\n"
        "    return 0\n",
    )
    _write(
        family / "model/runtime.py",
        "def runner_from_bundle():\n"
        "    return 'runner'\n\n"
        "def unused_runtime_helper():\n"
        "    return 'unused'\n",
    )
    _write(family / "model/dead.py", "def dead():\n    return None\n")
    _write(
        family / "model/tool_runner.py",
        "class ToolRunner:\n    pass\n",
    )
    _write(
        tmp_path / "tools/families/demo/use_runner.py",
        "from tensorrt_model_connect.families.demo.model.tool_runner "
        "import ToolRunner\n",
    )
    return tmp_path


def test_audit_classifies_production_tool_and_unreachable_modules(
    tmp_path: Path,
) -> None:
    repo = _demo_repo(tmp_path)

    report = specialization.audit_repo(repo, ("demo",))
    family = report["families"][0]

    assert family["production_modules"] == [
        "tensorrt_model_connect.families.demo",
        "tensorrt_model_connect.families.demo.model",
        "tensorrt_model_connect.families.demo.model.model",
        "tensorrt_model_connect.families.demo.model.runtime",
        "tensorrt_model_connect.families.demo.plugin",
    ]
    assert family["tool_test_only_modules"] == [
        "tensorrt_model_connect.families.demo.model.tool_runner"
    ]
    assert family["unreachable_modules"] == [
        "tensorrt_model_connect.families.demo.config",
        "tensorrt_model_connect.families.demo.model.dead",
        "tensorrt_model_connect.families.demo.weights",
    ]
    assert family["missing_dynamic_entrypoints"] == []


def test_audit_reports_symbols_switches_and_layout_violations(tmp_path: Path) -> None:
    repo = _demo_repo(tmp_path)

    family = specialization.audit_repo(repo, ("demo",))["families"][0]

    unreachable = {
        (item["path"], item["symbol"])
        for item in family["unreachable_symbols"]
    }
    assert ("model/model.py", "unused_helper") in unreachable
    assert ("model/runtime.py", "unused_runtime_helper") in unreachable
    assert family["fixed_strategy_switches"] == [
        {
            "function": "build_decoder",
            "parameter": "norm_type",
            "value": "rmsnorm",
            "definitions": [
                "tensorrt_model_connect.families.demo.model.model"
            ],
            "call_sites": [{"path": "plugin.py", "line": 5}],
        }
    ]
    assert family["noncanonical_model_paths"] == [
        "model/dead.py",
        "model/tool_runner.py",
    ]
    assert {
        item["kind"] for item in family["violations"]
    } >= {
        "fixed_strategy_switch",
        "noncanonical_model_path",
        "tool_test_only_model_module",
        "unreachable_module",
        "unreachable_symbol",
    }


def test_audit_does_not_fix_switch_with_default_only_call(tmp_path: Path) -> None:
    repo = _demo_repo(tmp_path)
    family = repo / "python/tensorrt_model_connect/families/demo"
    _write(
        family / "plugin.py",
        "from .model.model import build_decoder\n\n"
        "class DemoPlugin:\n"
        "    def build_engine(self, explicit):\n"
        "        if explicit:\n"
        "            return build_decoder(norm_type='rmsnorm')\n"
        "        return build_decoder()\n\n"
        "plugin = DemoPlugin()\n",
    )

    result = specialization.audit_repo(repo, ("demo",))["families"][0]

    assert result["fixed_strategy_switches"] == []


def test_audit_reports_missing_manifest_paths_and_sibling_imports(
    tmp_path: Path,
) -> None:
    repo = _demo_repo(tmp_path)
    family = (
        repo
        / "python"
        / "tensorrt_model_connect"
        / "families"
        / "demo"
    )
    _write(
        family / "MODEL.toml",
        'id = "demo"\nplugin = "demo"\nmodule = "plugin"\n'
        'debug_runner = "model/missing.py|runner_from_bundle"\n',
    )
    _write(
        family / "plugin.py",
        "from tensorrt_model_connect.families.other.model import build\n\n"
        "plugin = build\n",
    )

    result = specialization.audit_repo(repo, ("demo",))["families"][0]

    assert result["missing_dynamic_entrypoints"] == [
        {
            "source": "MODEL.toml",
            "path": "model/missing.py",
            "symbol": "runner_from_bundle",
            "reason": "missing_path",
        }
    ]
    assert result["sibling_family_imports"] == [
        {
            "path": "plugin.py",
            "line": 1,
            "target": "tensorrt_model_connect.families.other.model",
        }
    ]


def test_audit_reports_missing_manifest_symbols(tmp_path: Path) -> None:
    repo = _demo_repo(tmp_path)
    family = (
        repo
        / "python"
        / "tensorrt_model_connect"
        / "families"
        / "demo"
    )
    _write(
        family / "MODEL.toml",
        'id = "demo"\nplugin = "demo"\nmodule = "plugin"\n'
        'debug_runner = "model/runtime.py|missing_runner"\n',
    )

    result = specialization.audit_repo(repo, ("demo",))["families"][0]

    assert result["missing_dynamic_entrypoints"] == [
        {
            "source": "MODEL.toml",
            "path": "model/runtime.py",
            "symbol": "missing_runner",
            "reason": "missing_symbol",
        }
    ]


def test_audit_does_not_treat_profile_boolean_as_python_symbol(
    tmp_path: Path,
) -> None:
    repo = _demo_repo(tmp_path)
    family = repo / "python/tensorrt_model_connect/families/demo"
    _write(
        family / "MODEL.toml",
        'id = "demo"\nplugin = "demo"\nmodule = "plugin"\n'
        'python_profile_specs = ["demo|requirements.txt|model/runtime.py|true"]\n',
    )
    _write(family / "requirements.txt", "demo==1\n")

    result = specialization.audit_repo(repo, ("demo",))["families"][0]

    assert result["missing_dynamic_entrypoints"] == []
    entry = next(
        item
        for item in result["entrypoints"]["dynamic"]
        if item["path"] == "model/runtime.py"
    )
    assert entry["symbol"] is None


def test_audit_resolves_generic_vision_language_runner_convention(
    tmp_path: Path,
) -> None:
    repo = _demo_repo(tmp_path)
    family = (
        repo
        / "python"
        / "tensorrt_model_connect"
        / "families"
        / "demo"
    )
    _write(
        family / "plugin.py",
        "from .model.model import build_decoder\n\n"
        "class DemoPlugin:\n"
        "    runtime_strategy = 'demo_vision_language'\n\n"
        "    def build_engine(self):\n"
        "        return build_decoder(norm_type='rmsnorm')\n\n"
        "plugin = DemoPlugin()\n",
    )
    _write(
        repo / "tools/families/demo/vl_debug_runner.py",
        "class VLTrtRunner:\n    pass\n",
    )

    result = specialization.audit_repo(repo, ("demo",))["families"][0]
    assert result["missing_dynamic_entrypoints"] == []


def test_audit_requires_generic_vision_language_runner_convention(
    tmp_path: Path,
) -> None:
    repo = _demo_repo(tmp_path)
    family = (
        repo
        / "python"
        / "tensorrt_model_connect"
        / "families"
        / "demo"
    )
    _write(
        family / "plugin.py",
        "class DemoPlugin:\n"
        "    runtime_strategy = 'demo_vision_language'\n\n"
        "plugin = DemoPlugin()\n",
    )

    result = specialization.audit_repo(repo, ("demo",))["families"][0]

    assert {
        "source": "tools/diff_vl.py::<family-dispatch>",
        "path": "tools/families/demo/vl_debug_runner.py",
        "reason": "missing_path",
    } in result["missing_dynamic_entrypoints"]


def test_inventory_report_is_deterministic_and_serializable(tmp_path: Path) -> None:
    repo = _demo_repo(tmp_path)

    first = specialization.audit_repo(repo, ("demo",))
    second = specialization.audit_repo(repo, ("demo",))

    assert first == second
    assert json.loads(json.dumps(first, sort_keys=True)) == first
    assert first["schema_version"] == specialization.SCHEMA_VERSION


def test_repository_registers_all_current_families() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    families = specialization.family_dirs(repo_root, ())

    assert len(families) == 83
    assert any(family.name == "dinov3" for family in families)
    assert any(family.name == "fast_foundation_stereo" for family in families)
    assert any(family.name == "lfm2" for family in families)
    assert any(family.name == "minimax_h3" for family in families)
    assert any(family.name == "sam2" for family in families)

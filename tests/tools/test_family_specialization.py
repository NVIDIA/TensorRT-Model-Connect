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
    family = tmp_path / "python/tensorrt_model_connect/families/demo"
    _write(
        family / "MODEL.toml",
        'id = "demo"\ndebug_runner = "runtime.py|runner_from_bundle"\n',
    )
    _write(family / "__init__.py", '"""Model family package."""\n')
    _write(
        family / "model.py",
        "def used_helper():\n"
        "    return 1\n\n"
        "def unused_helper():\n"
        "    return 2\n\n"
        "def build_decoder(*, norm_type='rmsnorm'):\n"
        "    if norm_type == 'rmsnorm':\n"
        "        return used_helper()\n"
        "    return 0\n\n"
        "def matches(config):\n"
        "    return True\n\n"
        "def build(model_dir, output_path, **options):\n"
        "    return build_decoder(norm_type='rmsnorm')\n",
    )
    _write(family / "config.py", "class ModelConfig:\n    pass\n")
    _write(family / "weights.py", "class WeightDict(dict):\n    pass\n")
    _write(
        family / "runtime.py",
        "def runner_from_bundle():\n"
        "    return 'runner'\n\n"
        "def unused_runtime_helper():\n"
        "    return 'unused'\n",
    )
    _write(family / "dead.py", "def dead():\n    return None\n")
    _write(family / "tool_runner.py", "class ToolRunner:\n    pass\n")
    _write(
        tmp_path / "tools/families/demo/use_runner.py",
        "from tensorrt_model_connect.families.demo.tool_runner import ToolRunner\n",
    )
    return tmp_path


def test_audit_classifies_production_tool_and_unreachable_modules(tmp_path: Path) -> None:
    family = specialization.audit_repo(_demo_repo(tmp_path), ("demo",))["families"][0]

    assert family["production_modules"] == [
        "tensorrt_model_connect.families.demo",
        "tensorrt_model_connect.families.demo.model",
        "tensorrt_model_connect.families.demo.runtime",
    ]
    assert family["tool_test_only_modules"] == [
        "tensorrt_model_connect.families.demo.tool_runner"
    ]
    assert family["unreachable_modules"] == [
        "tensorrt_model_connect.families.demo.config",
        "tensorrt_model_connect.families.demo.dead",
        "tensorrt_model_connect.families.demo.weights",
    ]
    assert family["missing_dynamic_entrypoints"] == []


def test_audit_reports_symbols_and_fixed_switches(tmp_path: Path) -> None:
    family = specialization.audit_repo(_demo_repo(tmp_path), ("demo",))["families"][0]
    unreachable = {(item["path"], item["symbol"]) for item in family["unreachable_symbols"]}

    assert ("model.py", "unused_helper") in unreachable
    assert ("runtime.py", "unused_runtime_helper") in unreachable
    assert family["fixed_strategy_switches"] == [
        {
            "function": "build_decoder",
            "parameter": "norm_type",
            "value": "rmsnorm",
            "definitions": ["tensorrt_model_connect.families.demo.model"],
            "call_sites": [{"path": "model.py", "line": 16}],
        }
    ]
    assert family["noncanonical_model_paths"] == []
    assert {item["kind"] for item in family["violations"]} >= {
        "fixed_strategy_switch",
        "tool_test_only_model_module",
        "unreachable_module",
        "unreachable_symbol",
    }


def test_audit_does_not_fix_switch_with_default_only_call(tmp_path: Path) -> None:
    repo = _demo_repo(tmp_path)
    model = repo / "python/tensorrt_model_connect/families/demo/model.py"
    _write(
        model,
        "def build_decoder(*, norm_type='rmsnorm'):\n    return norm_type\n\n"
        "def matches(config):\n    return True\n\n"
        "def build(model_dir, output_path, **options):\n"
        "    if options.get('explicit'):\n"
        "        return build_decoder(norm_type='rmsnorm')\n"
        "    return build_decoder()\n",
    )

    result = specialization.audit_repo(repo, ("demo",))["families"][0]

    assert result["fixed_strategy_switches"] == []


def test_audit_reports_missing_manifest_paths_and_sibling_imports(tmp_path: Path) -> None:
    repo = _demo_repo(tmp_path)
    family = repo / "python/tensorrt_model_connect/families/demo"
    _write(family / "MODEL.toml", 'id = "demo"\ndebug_runner = "missing.py|runner"\n')
    _write(
        family / "model.py",
        "from tensorrt_model_connect.families.other.model import build as other_build\n\n"
        "def matches(config):\n    return True\n\n"
        "def build(model_dir, output_path, **options):\n    return other_build\n",
    )

    result = specialization.audit_repo(repo, ("demo",))["families"][0]

    assert result["missing_dynamic_entrypoints"] == [
        {
            "source": "MODEL.toml",
            "path": "missing.py",
            "symbol": "runner",
            "reason": "missing_path",
        }
    ]
    assert result["sibling_family_imports"] == [
        {
            "path": "model.py",
            "line": 1,
            "target": "tensorrt_model_connect.families.other.model",
        }
    ]


def test_audit_reports_missing_model_build_symbol(tmp_path: Path) -> None:
    repo = _demo_repo(tmp_path)
    model = repo / "python/tensorrt_model_connect/families/demo/model.py"
    _write(model, "def matches(config):\n    return True\n")

    result = specialization.audit_repo(repo, ("demo",))["families"][0]

    assert {
        "source": "family model convention",
        "path": "model.py",
        "symbol": "build",
        "reason": "missing_symbol",
    } in result["missing_dynamic_entrypoints"]


def test_audit_resolves_and_requires_vision_language_runner_convention(
    tmp_path: Path,
) -> None:
    repo = _demo_repo(tmp_path)
    family = repo / "python/tensorrt_model_connect/families/demo"
    model = (family / "model.py").read_text(encoding="utf-8")
    _write(family / "model.py", "runtime_strategy = 'demo_vision_language'\n\n" + model)
    tool = repo / "tools/families/demo/vl_debug_runner.py"
    _write(tool, "class VLTrtRunner:\n    pass\n")

    assert specialization.audit_repo(repo, ("demo",))["families"][0][
        "missing_dynamic_entrypoints"
    ] == []

    tool.unlink()
    missing = specialization.audit_repo(repo, ("demo",))["families"][0][
        "missing_dynamic_entrypoints"
    ]
    assert {
        "source": "tools/diff_vl.py::<family-dispatch>",
        "path": "tools/families/demo/vl_debug_runner.py",
        "reason": "missing_path",
    } in missing


def test_audit_tracks_self_module_graph_aliases(tmp_path: Path) -> None:
    repo = _demo_repo(tmp_path)
    model = repo / "python/tensorrt_model_connect/families/demo/model.py"
    _write(
        model,
        "import sys\n\n"
        "graph_ops = sys.modules[__name__]\n\n"
        "def add_constant():\n    return 1\n\n"
        "class DemoModel:\n"
        "    def run(self):\n        return graph_ops.add_constant()\n\n"
        "_model = DemoModel()\n\n"
        "def matches(config):\n    return True\n\n"
        "def build(model_dir, output_path, **options):\n    return _model.run()\n",
    )

    result = specialization.audit_repo(repo, ("demo",))["families"][0]

    assert not any(
        row["symbol"] == "add_constant" for row in result["unreachable_symbols"]
    )


def test_inventory_report_is_deterministic_and_serializable(tmp_path: Path) -> None:
    repo = _demo_repo(tmp_path)

    first = specialization.audit_repo(repo, ("demo",))
    second = specialization.audit_repo(repo, ("demo",))

    assert first == second
    assert json.loads(json.dumps(first, sort_keys=True)) == first
    assert first["schema_version"] == specialization.SCHEMA_VERSION


def test_repository_registers_all_current_family_models() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    families = specialization.family_dirs(repo_root, ())

    assert families
    assert all((family / "model.py").is_file() for family in families)
    assert any(family.name == "dinov3" for family in families)
    assert any(family.name == "minimax_h3" for family in families)

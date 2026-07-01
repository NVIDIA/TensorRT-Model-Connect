from __future__ import annotations

from pathlib import Path

from tools import family_specialization
from tools import specialize_family_switches


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_apply_report_removes_switch_and_unsupported_branch(tmp_path: Path) -> None:
    family = tmp_path / "python/tensorrt_model_connect/families/demo"
    _write(family / "MODEL.toml", 'id = "demo"\nmodule = "plugin"\n')
    _write(family / "__init__.py", "from .plugin import plugin\n")
    _write(
        family / "plugin.py",
        "from .model.model import build_decoder\n\n"
        "plugin = build_decoder(norm_type='rmsnorm')\n",
    )
    _write(family / "config.py", "class ModelConfig:\n    pass\n")
    _write(family / "weights/__init__.py", "class WeightDict(dict):\n    pass\n")
    _write(family / "model/__init__.py", '"""Model package."""\n')
    _write(
        family / "model/model.py",
        "def rms_norm():\n"
        "    return 'rms'\n\n"
        "def layer_norm():\n"
        "    return 'layer'\n\n"
        "def build_decoder(norm_type='layernorm'):\n"
        "    if norm_type == 'rmsnorm':\n"
        "        return rms_norm()\n"
        "    return layer_norm()\n",
    )
    report = family_specialization.audit_repo(tmp_path, ("demo",))

    changed = specialize_family_switches.apply_report(tmp_path, report)

    assert changed == 1
    plugin = (family / "plugin.py").read_text(encoding="utf-8")
    model = (family / "model/model.py").read_text(encoding="utf-8")
    assert "norm_type" not in plugin
    assert "norm_type" not in model
    assert model.count("layer_norm()") == 1  # Definition remains for reachability pruning.
    assert "return rms_norm()" in model
    assert not family_specialization.audit_repo(
        tmp_path, ("demo",)
    )["families"][0]["fixed_strategy_switches"]

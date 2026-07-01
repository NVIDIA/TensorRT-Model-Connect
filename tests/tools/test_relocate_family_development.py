from __future__ import annotations

from pathlib import Path

from tools import relocate_family_development as relocation


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_apply_relocations_moves_owner_tool_and_rewrites_imports(
    tmp_path: Path,
    monkeypatch,
) -> None:
    family = tmp_path / "python/tensorrt_model_connect/families/demo"
    source = family / "model/debug_runner.py"
    _write(source, "class Runner:\n    pass\n")
    _write(
        tmp_path / "tools/families/demo/use.py",
        "from tensorrt_model_connect.families.demo.model.debug_runner "
        "import Runner\n",
    )
    planned = relocation.Relocation(
        "demo", "model/debug_runner.py", "tools/families/demo/debug_runner.py"
    )
    monkeypatch.setattr(relocation, "RELOCATIONS", (planned,))

    relocation.apply_relocations(tmp_path, frozenset({"demo"}))

    destination = tmp_path / "tools/families/demo/debug_runner.py"
    assert destination.is_file()
    assert not source.exists()
    assert "from tools.families.demo.debug_runner import Runner" in (
        tmp_path / "tools/families/demo/use.py"
    ).read_text(encoding="utf-8")
    assert relocation.pending_relocations(tmp_path, frozenset({"demo"})) == []

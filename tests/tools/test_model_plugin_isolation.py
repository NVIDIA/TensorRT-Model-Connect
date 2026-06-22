from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL = REPO_ROOT / "tools" / "model_plugin_isolation.py"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(TOOL), *args],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )


def test_targets_resolve_e2e_model_to_runtime_plugin_owner() -> None:
    result = _run("targets", "--model", "qwen3-0.6b-fp16")

    assert result.stdout.splitlines() == ["trtmc_model_text_generation"]


def test_targets_resolve_model_owned_node_id_from_tests_file(tmp_path: Path) -> None:
    tests_file = tmp_path / "tests.txt"
    tests_file.write_text(
        "tests/e2e/models/qwen/test_qwen_e2e.py::test_model_e2e[qwen3-0.6b-fp16]\n",
        encoding="utf-8",
    )

    result = _run("targets", "--tests-file", str(tests_file))

    assert result.stdout.splitlines() == ["trtmc_model_text_generation"]


def test_prepare_copies_only_selected_runtime_plugin(tmp_path: Path) -> None:
    build_dir = tmp_path / "build"
    source_dir = build_dir / "models" / "text_generation"
    source_dir.mkdir(parents=True)
    source = source_dir / "libtrtmc_model_text_generation.so"
    source.write_bytes(b"fake-so")

    output_dir = tmp_path / "only-selected"
    result = _run(
        "prepare",
        "--model",
        "qwen3-0.6b-fp16",
        "--build-dir",
        str(build_dir),
        "--output-dir",
        str(output_dir),
    )

    copied = output_dir / "text_generation" / "libtrtmc_model_text_generation.so"
    assert copied.read_bytes() == b"fake-so"
    assert result.stdout.splitlines() == [f"trtmc_model_text_generation {copied}"]

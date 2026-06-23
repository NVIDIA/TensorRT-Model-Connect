from __future__ import annotations

import subprocess
import sys
import json
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


def _make_repo(tmp_path: Path) -> Path:
    repo_root = tmp_path / "repo"
    manifests_dir = repo_root / "tests" / "e2e" / "models" / "decoder_family" / "manifests"
    manifests_dir.mkdir(parents=True)
    (manifests_dir / "decoder-small.json").write_text(
        json.dumps({
            "name": "decoder-small",
            "family": "decoder_family",
            "runtime_strategy": "decoder_kv_cache",
        }),
        encoding="utf-8",
    )
    runtime_dir = repo_root / "src" / "runtime" / "models" / "text_generation"
    runtime_dir.mkdir(parents=True)
    (runtime_dir / "MODEL.toml").write_text(
        'id = "text_generation"\n'
        'runtime_strategies = ["decoder_kv_cache"]\n',
        encoding="utf-8",
    )
    return repo_root


def test_targets_resolve_e2e_model_to_runtime_plugin_owner(tmp_path: Path) -> None:
    repo_root = _make_repo(tmp_path)
    result = _run(
        "targets",
        "--repo-root",
        str(repo_root),
        "--model",
        "decoder-small",
    )

    assert result.stdout.splitlines() == ["trtmc_model_text_generation"]


def test_targets_resolve_model_owned_node_id_from_tests_file(tmp_path: Path) -> None:
    repo_root = _make_repo(tmp_path)
    tests_file = tmp_path / "tests.txt"
    tests_file.write_text(
        "tests/e2e/models/decoder_family/test_decoder_family_e2e.py::test_model_e2e[decoder-small]\n",
        encoding="utf-8",
    )

    result = _run(
        "targets",
        "--repo-root",
        str(repo_root),
        "--tests-file",
        str(tests_file),
    )

    assert result.stdout.splitlines() == ["trtmc_model_text_generation"]


def test_prepare_copies_only_selected_runtime_plugin(tmp_path: Path) -> None:
    repo_root = _make_repo(tmp_path)
    build_dir = tmp_path / "build"
    source_dir = build_dir / "models" / "text_generation"
    source_dir.mkdir(parents=True)
    source = source_dir / "libtrtmc_model_text_generation.so"
    source.write_bytes(b"fake-so")

    output_dir = tmp_path / "only-selected"
    result = _run(
        "prepare",
        "--repo-root",
        str(repo_root),
        "--model",
        "decoder-small",
        "--build-dir",
        str(build_dir),
        "--output-dir",
        str(output_dir),
    )

    copied = output_dir / "text_generation" / "libtrtmc_model_text_generation.so"
    assert copied.read_bytes() == b"fake-so"
    assert result.stdout.splitlines() == [f"trtmc_model_text_generation {copied}"]

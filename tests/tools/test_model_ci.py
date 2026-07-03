#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL = REPO_ROOT / "tools" / "model_ci.py"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write(repo: Path, relative: str, content: str) -> None:
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _add_model(
    repo: Path,
    logical_id: str,
    *,
    runtime_id: str | None = None,
    strategy: str | None = None,
) -> None:
    runtime_id = runtime_id or logical_id
    strategy = strategy or f"{runtime_id}_runtime"
    _write(
        repo,
        f"python/tensorrt_model_connect/families/{logical_id}/MODEL.toml",
        f'id = "{logical_id}"\n',
    )
    _write(
        repo,
        f"python/tensorrt_model_connect/families/{logical_id}/plugin.py",
        f'MODEL = "{logical_id}"\n',
    )
    _write(
        repo,
        f"src/runtime/models/{runtime_id}/MODEL.toml",
        f'id = "{runtime_id}"\n'
        f'runtime_library = "libtrtmc_model_{runtime_id}.so"\n'
        f'runtime_strategies = ["{strategy}"]\n',
    )
    _write(
        repo,
        f"src/runtime/models/{runtime_id}/plugin.cpp",
        f"// {runtime_id}\n",
    )
    _write(
        repo,
        f"tests/e2e/models/{logical_id}/MODEL.toml",
        f'id = "{logical_id}"\ntest_manifests = ["manifests/{logical_id}.json"]\n',
    )
    _write(
        repo,
        f"tests/e2e/models/{logical_id}/manifests/{logical_id}.json",
        json.dumps(
            {
                "name": logical_id,
                "family": logical_id,
                "runtime_strategy": strategy,
            }
        )
        + "\n",
    )
    _write(
        repo,
        f"tests/cpp/models/{runtime_id}/test_{runtime_id}.cpp",
        f"// {runtime_id} test\n",
    )


def _make_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Model CI Test")
    _git(repo, "config", "user.email", "model-ci@example.com")
    _add_model(repo, "model_a")
    _add_model(repo, "model_b")
    _write(repo, "python/tensorrt_model_connect/families/__init__.py", "# registry\n")
    _write(repo, "CMakeLists.txt", "# platform build\n")
    _write(repo, "src/runtime/core/core.cpp", "// platform core\n")
    _write(repo, "README.md", "# Documentation\n")
    _write(repo, "tests/__init__.py", "")
    _write(repo, "tests/runtime_strategy_matrix.yaml", "strategies: []\n")
    _write(repo, ".github/scripts/run-model-proof.sh", "#!/usr/bin/env bash\n")
    os.chmod(repo / ".github/scripts/run-model-proof.sh", 0o755)
    _write(
        repo,
        ".github/scripts/write-model-proof-fallback-report.py",
        "#!/usr/bin/env python3\n",
    )
    _write(repo, "scripts/generate_e2e_report.py", "# report generator\n")
    _write(repo, "scripts/generate_e2e_report_assets/e2e_report.css", "/* report */\n")
    _write(repo, "scripts/generate_e2e_report_assets/e2e_report.js", "// report\n")
    _write(repo, "scripts/reporting/vlm_assessment.py", "# report component\n")
    return repo, _commit(repo, "initial")


def _run(
    repo: Path,
    *args: str,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(TOOL), *args, "--repo-root", str(repo)],
        check=check,
        capture_output=True,
        text=True,
    )


def _impact(repo: Path, base: str, head: str) -> dict[str, object]:
    result = _run(repo, "impact", "--base", base, "--head", head)
    return json.loads(result.stdout)


def test_validate_and_all_emit_deterministic_matrix_and_github_outputs(
    tmp_path: Path,
) -> None:
    repo, revision = _make_repo(tmp_path)
    validated = json.loads(_run(repo, "validate", "--revision", revision).stdout)
    output = tmp_path / "github-output"

    result = json.loads(
        _run(
            repo,
            "all",
            "--revision",
            revision,
            "--github-output",
            str(output),
        ).stdout
    )

    assert validated["models"] == ["model_a", "model_b"]
    assert result["matrix"] == {"include": [{"model": "model_a"}, {"model": "model_b"}]}
    assert output.read_text(encoding="utf-8").splitlines() == [
        'matrix={"include":[{"model":"model_a"},{"model":"model_b"}]}',
        "has_models=true",
        'affected_models=["model_a","model_b"]',
        "expected_count=2",
        "mode=all",
    ]


def test_impact_selects_only_model_a(tmp_path: Path) -> None:
    repo, base = _make_repo(tmp_path)
    _write(
        repo,
        "python/tensorrt_model_connect/families/model_a/plugin.py",
        'MODEL = "model_a_changed"\n',
    )
    head = _commit(repo, "change a")

    result = _impact(repo, base, head)

    assert result["mode"] == "models"
    assert result["affected_models"] == ["model_a"]
    assert result["matrix"] == {"include": [{"model": "model_a"}]}


def test_impact_selects_each_modified_model_once(tmp_path: Path) -> None:
    repo, base = _make_repo(tmp_path)
    _write(repo, "src/runtime/models/model_a/plugin.cpp", "// changed a\n")
    _write(
        repo,
        "tests/e2e/models/model_b/manifests/model_b.json",
        json.dumps(
            {
                "name": "model_b",
                "family": "model_b",
                "runtime_strategy": "model_b_runtime",
                "changed": True,
            }
        )
        + "\n",
    )
    head = _commit(repo, "change a and b")

    result = _impact(repo, base, head)

    assert result["affected_models"] == ["model_a", "model_b"]
    assert result["expected_count"] == 2


def test_impact_treats_legal_and_docs_as_no_model_change(tmp_path: Path) -> None:
    repo, base = _make_repo(tmp_path)
    _write(repo, "README.md", "# Updated documentation\n")
    _write(repo, "NOTICE", "Legal notice\n")
    head = _commit(repo, "docs")

    result = _impact(repo, base, head)

    assert result["mode"] == "none"
    assert result["has_models"] is False
    assert result["matrix"] == {"include": []}


def test_impact_treats_platform_change_as_all_models(tmp_path: Path) -> None:
    repo, base = _make_repo(tmp_path)
    _write(repo, "src/runtime/core/core.cpp", "// changed platform core\n")
    head = _commit(repo, "platform")

    result = _impact(repo, base, head)

    assert result["mode"] == "all"
    assert result["affected_models"] == ["model_a", "model_b"]


def test_impact_treats_shared_family_registry_as_platform(tmp_path: Path) -> None:
    repo, base = _make_repo(tmp_path)
    _write(repo, "python/tensorrt_model_connect/families/__init__.py", "# changed registry\n")
    head = _commit(repo, "shared family registry")

    result = _impact(repo, base, head)

    assert result["mode"] == "all"
    assert result["affected_models"] == ["model_a", "model_b"]


def test_impact_includes_deletions_and_both_sides_of_rename(tmp_path: Path) -> None:
    repo, base = _make_repo(tmp_path)
    (repo / "tests/cpp/models/model_a/test_model_a.cpp").unlink()
    _git(
        repo,
        "mv",
        "python/tensorrt_model_connect/families/model_a/plugin.py",
        "python/tensorrt_model_connect/families/model_b/from_a.py",
    )
    head = _commit(repo, "delete and rename")

    result = _impact(repo, base, head)

    assert result["affected_models"] == ["model_a", "model_b"]
    assert any(change["status"] == "D" for change in result["changes"])
    assert any(str(change["status"]).startswith("R") for change in result["changes"])


def test_impact_rejects_unknown_source_path(tmp_path: Path) -> None:
    repo, base = _make_repo(tmp_path)
    _write(repo, "mystery/implementation.py", "VALUE = 1\n")
    head = _commit(repo, "unknown source")

    result = _run(
        repo,
        "impact",
        "--base",
        base,
        "--head",
        head,
        check=False,
    )

    assert result.returncode == 2
    assert "has no model, platform, CI, legal, or docs owner" in result.stderr


def test_validate_rejects_overlapping_runtime_ownership(tmp_path: Path) -> None:
    repo, _ = _make_repo(tmp_path)
    manifest = repo / "tests/e2e/models/model_b/manifests/model_b.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["runtime_strategy"] = "model_a_runtime"
    manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    _commit(repo, "introduce overlap")

    result = _run(repo, "validate", check=False)

    assert result.returncode == 2
    assert "depends on multiple runtime models" in result.stderr


def test_projection_contains_only_selected_model_and_stable_git_blobs(
    tmp_path: Path,
) -> None:
    repo, revision = _make_repo(tmp_path)
    source = repo / "python/tensorrt_model_connect/families/model_a/plugin.py"
    expected = source.read_bytes()
    source.write_text('MODEL = "dirty_worktree_value"\n', encoding="utf-8")
    output = tmp_path / "projection"

    manifest = json.loads(
        _run(
            repo,
            "project",
            "--revision",
            revision,
            "--model",
            "model_a",
            "--output-dir",
            str(output),
        ).stdout
    )

    copied = output / "python/tensorrt_model_connect/families/model_a/plugin.py"
    assert copied.read_bytes() == expected
    assert not (output / "python/tensorrt_model_connect/families/model_b").exists()
    assert not (output / "src/runtime/models/model_b").exists()
    assert not (output / "tests/e2e/models/model_b").exists()
    assert (output / "src/runtime/core/core.cpp").is_file()
    assert (output / "python/tensorrt_model_connect/families/__init__.py").is_file()
    assert (output / "tests/__init__.py").is_file()
    assert (output / "tests/runtime_strategy_matrix.yaml").is_file()
    assert (output / ".github/scripts/run-model-proof.sh").is_file()
    assert os.access(output / ".github/scripts/run-model-proof.sh", os.X_OK)
    fallback = output / ".github/scripts/write-model-proof-fallback-report.py"
    assert fallback.is_file()
    assert not os.access(fallback, os.X_OK)
    for report_path in (
        "scripts/generate_e2e_report.py",
        "scripts/generate_e2e_report_assets/e2e_report.css",
        "scripts/generate_e2e_report_assets/e2e_report.js",
        "scripts/reporting/vlm_assessment.py",
    ):
        assert (output / report_path).is_file()
    assert manifest["runtime_model"] == "model_a"
    assert manifest["build_target"] == "trtmc_model_model_a"
    entry = next(
        item for item in manifest["files"] if item["path"] == copied.relative_to(output).as_posix()
    )
    assert entry["sha256"] == hashlib.sha256(expected).hexdigest()


def test_projection_and_impact_normalize_logical_runtime_owner(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Model CI Test")
    _git(repo, "config", "user.email", "model-ci@example.com")
    _add_model(repo, "logical_model", runtime_id="runtime_model", strategy="runtime_strategy")
    _write(repo, "CMakeLists.txt", "# platform\n")
    base = _commit(repo, "initial")
    _write(repo, "src/runtime/models/runtime_model/plugin.cpp", "// changed runtime\n")
    head = _commit(repo, "runtime change")

    impact = _impact(repo, base, head)
    output = tmp_path / "projection"
    projection = json.loads(
        _run(
            repo,
            "project",
            "--revision",
            head,
            "--model",
            "logical_model",
            "--output-dir",
            str(output),
        ).stdout
    )

    assert impact["affected_models"] == ["logical_model"]
    assert projection["runtime_model"] == "runtime_model"
    assert projection["build_target"] == "trtmc_model_runtime_model"
    assert (output / "src/runtime/models/runtime_model/plugin.cpp").is_file()


def test_projection_rejects_symlink_that_escapes_allowlist(tmp_path: Path) -> None:
    repo, revision = _make_repo(tmp_path)
    link = repo / "python/tensorrt_model_connect/families/model_a/escape"
    link.symlink_to("/etc/passwd")
    revision = _commit(repo, "escaping symlink")

    result = _run(
        repo,
        "project",
        "--revision",
        revision,
        "--model",
        "model_a",
        "--output-dir",
        str(tmp_path / "projection"),
        check=False,
    )

    assert result.returncode == 2
    assert "symlink escapes projection" in result.stderr

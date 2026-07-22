# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for generic, descriptor-driven optimized-runtime CI selection."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tools.ci import optimized_runtime_qualifications as qualifications


_DIGEST_IMAGE = f"example.invalid/runtime:1@sha256:{'a' * 64}"


def _write(path: Path, text: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _profile(
    repository: Path,
    family: str,
    adapter: str,
    name: str,
    gpu: str = "a100",
    state: str = "qualified",
) -> str:
    relative = f"python/tensorrt_model_connect/families/{family}/{adapter}/profiles/{name}.toml"
    _write(
        repository / relative,
        f'qualification_state = "{state}"\n[target]\ngpu = "{gpu}"\n',
    )
    return relative


def _descriptor(
    repository: Path,
    family: str,
    adapter: str,
    gpu: str = "a100",
    *,
    runtime_id: str | None = None,
    representative: bool = True,
) -> str:
    root = f"tests/e2e/models/{family}/{adapter}"
    entrypoint = f"{root}/run.sh"
    _write(repository / entrypoint, "#!/bin/sh\n").chmod(0o755)
    relative = f"{root}/QUALIFICATION.{gpu}.toml"
    shared_triggers = '["shared/**"]' if representative else "[]"
    _write(
        repository / relative,
        f"""
schema_version = 2
kind = "producer"
id = "{family}-{adapter}-{gpu}"
runtime_id = "{runtime_id or adapter}"
representative = {str(representative).lower()}
entrypoint = "{entrypoint}"
container_image = "{_DIGEST_IMAGE}"
runner_labels = ["self-hosted", "{gpu}"]
profile_glob = "python/tensorrt_model_connect/families/{family}/{adapter}/profiles/**/*.toml"
trigger_globs = [
  "python/tensorrt_model_connect/families/{family}/{adapter}/**",
  "src/runtime/models/{family}/{adapter}/**",
  "tests/e2e/models/{family}/{adapter}/**",
]
representative_trigger_globs = {shared_triggers}

[profile_target]
gpu = "{gpu}"
""".lstrip(),
    )
    return relative


@pytest.fixture
def repository(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    _descriptor(tmp_path, "family-a", "runtime-a")
    paths = {
        "target_a": _profile(tmp_path, "family-a", "runtime-a", "target-a"),
        "target_b": _profile(tmp_path, "family-a", "runtime-a", "target-b"),
        "wrong_target": _profile(
            tmp_path, "family-a", "runtime-a", "other-gpu", "h100", state="candidate"
        ),
    }
    return tmp_path, paths


def _selection(repository: Path, *paths: str) -> list[dict[str, object]]:
    return qualifications.select_qualifications(repository, paths)["producers"]["include"]


def test_adapter_or_generic_change_selects_the_whole_target_suite(
    repository: tuple[Path, dict[str, str]],
) -> None:
    root, _ = repository
    for changed in (
        "src/runtime/models/family-a/runtime-a/adapter.cpp",
        "shared/host.cpp",
    ):
        assert _selection(root, changed)[0]["profile_files"] == ""


def test_family_resolver_change_selects_the_runtime_representative() -> None:
    repository = Path(__file__).resolve().parents[2]
    matrix = qualifications.select_qualifications(
        repository,
        ["python/tensorrt_model_connect/families/__init__.py"],
    )
    assert [item["id"] for item in matrix["producers"]["include"]] == [
        "qwen-tensorrt-edge-llm-a100-sm80"
    ]


def test_profile_only_change_selects_only_that_target_profile(
    repository: tuple[Path, dict[str, str]],
) -> None:
    root, paths = repository
    selected = _selection(root, paths["target_b"])
    assert len(selected) == 1
    assert selected[0]["profile_files"] == "target-b.toml"


def test_non_target_profile_and_unrelated_changes_select_nothing(
    repository: tuple[Path, dict[str, str]],
) -> None:
    root, paths = repository
    assert _selection(root, paths["wrong_target"]) == []
    assert _selection(root, "docs/readme.md") == []


def test_candidate_profile_change_does_not_schedule_hardware(tmp_path: Path) -> None:
    _descriptor(tmp_path, "family-a", "runtime-a")
    _profile(tmp_path, "family-a", "runtime-a", "qualified")
    candidate = _profile(tmp_path, "family-a", "runtime-a", "candidate", state="candidate")
    assert _selection(tmp_path, candidate) == []


def test_every_qualified_profile_requires_exactly_one_producer(tmp_path: Path) -> None:
    _descriptor(tmp_path, "family-a", "runtime-a")
    _profile(tmp_path, "family-a", "runtime-a", "covered")
    _profile(tmp_path, "family-a", "runtime-a", "uncovered", gpu="h100")
    with pytest.raises(qualifications.QualificationError, match="exactly one producer"):
        _selection(tmp_path, "shared/host.cpp")

    _descriptor(
        tmp_path,
        "family-a",
        "runtime-a",
        gpu="a100-copy",
        representative=False,
    )
    duplicate = tmp_path / ("tests/e2e/models/family-a/runtime-a/QUALIFICATION.a100-copy.toml")
    duplicate.write_text(
        duplicate.read_text(encoding="utf-8").replace(
            '[profile_target]\ngpu = "a100-copy"',
            '[profile_target]\ngpu = "a100"',
        ),
        encoding="utf-8",
    )
    (
        tmp_path
        / ("python/tensorrt_model_connect/families/family-a/runtime-a/profiles/uncovered.toml")
    ).unlink()
    with pytest.raises(qualifications.QualificationError, match="exactly one producer"):
        _selection(tmp_path, "shared/host.cpp")


def test_candidate_profile_may_have_no_producer(tmp_path: Path) -> None:
    _descriptor(tmp_path, "family-a", "runtime-a")
    _profile(tmp_path, "family-a", "runtime-a", "qualified")
    _profile(tmp_path, "family-a", "runtime-a", "future", gpu="h100", state="candidate")
    assert len(_selection(tmp_path, "shared/host.cpp")) == 1


def test_profile_target_matching_is_type_strict(tmp_path: Path) -> None:
    descriptor = _descriptor(tmp_path, "family-a", "runtime-a")
    profile = _profile(tmp_path, "family-a", "runtime-a", "qualified")
    descriptor_path = tmp_path / descriptor
    descriptor_path.write_text(
        descriptor_path.read_text(encoding="utf-8").replace(
            '[profile_target]\ngpu = "a100"',
            "[profile_target]\ngpu = false",
        ),
        encoding="utf-8",
    )
    (tmp_path / profile).write_text(
        'qualification_state = "qualified"\n[target]\ngpu = 0\n',
        encoding="utf-8",
    )
    with pytest.raises(qualifications.QualificationError, match="exactly one producer"):
        _selection(tmp_path, "shared/host.cpp")


def test_descriptor_rejects_nonfinite_target_and_symlink_profile(tmp_path: Path) -> None:
    descriptor = _descriptor(tmp_path, "family-a", "runtime-a")
    profile = _profile(tmp_path, "family-a", "runtime-a", "a")
    descriptor_path = tmp_path / descriptor
    original = descriptor_path.read_text(encoding="utf-8")
    descriptor_path.write_text(
        original.replace('[profile_target]\ngpu = "a100"', "[profile_target]\ngpu = nan"),
        encoding="utf-8",
    )
    with pytest.raises(qualifications.QualificationError, match="JSON scalar"):
        _selection(tmp_path, "shared/host.cpp")

    descriptor_path.write_text(original, encoding="utf-8")
    source = tmp_path / profile
    source.unlink()
    source.symlink_to(tmp_path / "outside.toml")
    _write(tmp_path / "outside.toml", 'qualification_state="qualified"\n[target]\ngpu="a100"\n')
    with pytest.raises(qualifications.QualificationError, match="must not be a symlink"):
        _selection(tmp_path, "shared/host.cpp")


def test_deleted_profile_conservatively_selects_the_whole_target_suite(
    repository: tuple[Path, dict[str, str]],
) -> None:
    root, _ = repository
    assert (
        _selection(
            root,
            "python/tensorrt_model_connect/families/family-a/runtime-a/profiles/deleted.toml",
        )[0]["profile_files"]
        == ""
    )


def test_two_descriptors_select_independently(tmp_path: Path) -> None:
    _descriptor(tmp_path, "family-a", "runtime-a")
    _profile(tmp_path, "family-a", "runtime-a", "a")
    _descriptor(tmp_path, "family-b", "runtime-b", "h100")
    target_b = _profile(tmp_path, "family-b", "runtime-b", "b", "h100")

    selected = _selection(tmp_path, target_b)
    assert [(item["id"], item["profile_files"]) for item in selected] == [
        ("family-b-runtime-b-h100", "b.toml")
    ]


def test_shared_change_selects_one_representative_per_runtime(tmp_path: Path) -> None:
    _descriptor(tmp_path, "family-a", "adapter-a", runtime_id="runtime-a")
    _profile(tmp_path, "family-a", "adapter-a", "a")
    _descriptor(
        tmp_path,
        "family-b",
        "adapter-b",
        runtime_id="runtime-a",
        representative=False,
    )
    _profile(tmp_path, "family-b", "adapter-b", "b")

    assert [item["id"] for item in _selection(tmp_path, "shared/host.cpp")] == [
        "family-a-adapter-a-a100"
    ]
    assert [
        item["id"]
        for item in _selection(tmp_path, "src/runtime/models/family-b/adapter-b/adapter.cpp")
    ] == ["family-b-adapter-b-a100"]


@pytest.mark.parametrize("shared_glob", ["**", "src/**", "src/**/flux/**"])
def test_shared_trigger_cannot_select_a_cross_family_change(
    tmp_path: Path, shared_glob: str
) -> None:
    descriptor = _descriptor(tmp_path, "family-a", "runtime-a")
    _profile(tmp_path, "family-a", "runtime-a", "a")
    path = tmp_path / descriptor
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            'representative_trigger_globs = ["shared/**"]',
            f'representative_trigger_globs = ["{shared_glob}"]',
        ),
        encoding="utf-8",
    )

    assert _selection(tmp_path, "src/runtime/models/flux/adapter/source.cpp") == []


def test_each_runtime_requires_exactly_one_representative(tmp_path: Path) -> None:
    _descriptor(tmp_path, "family-a", "adapter-a", runtime_id="runtime-a")
    _profile(tmp_path, "family-a", "adapter-a", "a")
    _descriptor(tmp_path, "family-b", "adapter-b", runtime_id="runtime-a")
    _profile(tmp_path, "family-b", "adapter-b", "b")
    with pytest.raises(qualifications.QualificationError, match="exactly one representative"):
        _selection(tmp_path, "shared/host.cpp")


def test_select_all_schedules_every_descriptor_without_fake_paths(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _descriptor(tmp_path, "family-a", "runtime-a")
    _profile(tmp_path, "family-a", "runtime-a", "a")
    _descriptor(tmp_path, "family-b", "runtime-b", "h100")
    _profile(tmp_path, "family-b", "runtime-b", "b", "h100")

    matrix = qualifications.select_qualifications(tmp_path, (), select_all=True)
    assert [(item["id"], item["profile_files"]) for item in matrix["producers"]["include"]] == [
        ("family-a-runtime-a-a100", ""),
        ("family-b-runtime-b-h100", ""),
    ]
    assert qualifications.main(["--repository", str(tmp_path), "--all"]) == 0
    output = capsys.readouterr().out
    assert '"producers":{"include":' in output
    assert '"profile_files":""' in output


def test_matrix_limit_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _descriptor(tmp_path, "family-a", "runtime-a")
    _profile(tmp_path, "family-a", "runtime-a", "a")
    _descriptor(tmp_path, "family-b", "runtime-b")
    _profile(tmp_path, "family-b", "runtime-b", "b")
    monkeypatch.setattr(qualifications, "_MAX_MATRIX_ENTRIES", 1)
    with pytest.raises(qualifications.QualificationError, match="GitHub matrix limit"):
        qualifications.select_qualifications(tmp_path, (), select_all=True)


def test_invalid_descriptor_and_duplicate_profile_basenames_fail(tmp_path: Path) -> None:
    descriptor = _descriptor(tmp_path, "family-a", "runtime-a")
    _profile(tmp_path, "family-a", "runtime-a", "a")
    text = (tmp_path / descriptor).read_text(encoding="utf-8")
    (tmp_path / descriptor).write_text(text.replace("@sha256:", ":"), encoding="utf-8")
    with pytest.raises(qualifications.QualificationError, match="pinned by sha256"):
        _selection(tmp_path, descriptor)

    (tmp_path / descriptor).write_text(text, encoding="utf-8")
    _write(
        tmp_path
        / "python/tensorrt_model_connect/families/family-a/runtime-a/profiles/nested/a.toml",
        'qualification_state="qualified"\n[target]\ngpu="a100"',
    )
    with pytest.raises(qualifications.QualificationError, match="basenames must be unique"):
        _selection(tmp_path, descriptor)


def test_descriptor_requires_bounded_id_and_executable_entrypoint(tmp_path: Path) -> None:
    descriptor = _descriptor(tmp_path, "family-a", "runtime-a")
    _profile(tmp_path, "family-a", "runtime-a", "a")
    entrypoint = tmp_path / "tests/e2e/models/family-a/runtime-a/run.sh"
    entrypoint.chmod(0o644)
    with pytest.raises(qualifications.QualificationError, match="must be executable"):
        _selection(tmp_path, descriptor)

    entrypoint.chmod(0o755)
    text = (tmp_path / descriptor).read_text(encoding="utf-8")
    long_id = "a" * 65
    (tmp_path / descriptor).write_text(
        text.replace('id = "family-a-runtime-a-a100"', f'id = "{long_id}"'),
        encoding="utf-8",
    )
    with pytest.raises(qualifications.QualificationError, match="id is invalid"):
        _selection(tmp_path, descriptor)


def test_descriptor_paths_cannot_escape_model_owned_roots(tmp_path: Path) -> None:
    descriptor = _descriptor(tmp_path, "family-a", "runtime-a")
    _profile(tmp_path, "family-a", "runtime-a", "a")
    path = tmp_path / descriptor
    original = path.read_text(encoding="utf-8")
    path.write_text(
        original.replace(
            "python/tensorrt_model_connect/families/family-a/runtime-a/profiles/**/*.toml",
            "python/tensorrt_model_connect/families/family-b/runtime-b/profiles/**/*.toml",
        ),
        encoding="utf-8",
    )
    with pytest.raises(qualifications.QualificationError, match="profile_glob must stay"):
        _selection(tmp_path, descriptor)

    path.write_text(
        original.replace(
            '  "src/runtime/models/family-a/runtime-a/**",',
            '  "src/runtime/models/family-b/runtime-b/**",',
        ),
        encoding="utf-8",
    )
    with pytest.raises(qualifications.QualificationError, match="trigger_globs must stay"):
        _selection(tmp_path, descriptor)

    for cross_family_glob in (
        "src/runtime/models/family-b/**",
        "src/runtime/models/**",
        "python/tensorrt_model_connect/families/**",
        "tests/e2e/models/**",
    ):
        path.write_text(
            original.replace(
                'representative_trigger_globs = ["shared/**"]',
                f'representative_trigger_globs = ["{cross_family_glob}"]',
            ),
            encoding="utf-8",
        )
        with pytest.raises(qualifications.QualificationError, match="must not claim model-owned"):
            _selection(tmp_path, descriptor)


def test_git_diff_disables_rename_detection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: list[str] = []

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[bytes]:
        captured.extend(command)
        return subprocess.CompletedProcess(command, 0, b"old.toml\0new.toml\0", b"")

    monkeypatch.setattr(qualifications.subprocess, "run", fake_run)
    assert qualifications.git_changed_paths(tmp_path, "base", "head") == [
        "old.toml",
        "new.toml",
    ]
    assert "--no-renames" in captured

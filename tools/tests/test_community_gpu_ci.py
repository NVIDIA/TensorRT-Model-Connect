# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for isolated Community GPU family planning and orchestration."""

from __future__ import annotations

import shlex
from concurrent.futures import ThreadPoolExecutor
import time
from tools import brev_exec as brev_provision

import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import community_gpu_ci
from tools.ci.process import CiError as RemoteTaskError
from tools.community_gpu_ci import CommunityGpuError as CiError
from tools.ci import context as ci_context, e2e as ci_e2e


def _family(repository: Path, name: str, manifests: list[dict[str, object]]) -> Path:
    """Create the physical files required by one family-owned GPU plan."""
    root = repository / "families" / name
    (root / "tests/manifests").mkdir(parents=True)
    (root / "model.py").write_text("# model\n", encoding="utf-8")
    (root / "tests/test_e2e.py").write_text("# e2e\n", encoding="utf-8")
    for index, manifest in enumerate(manifests):
        (root / f"tests/manifests/{index}.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_shared_gpu_plan_includes_direct_and_new_families() -> None:
    """Shared changes retain smoke coverage and every directly affected owner."""
    selected = community_gpu_ci.selected_families(
        "all",
        '["bert","gpt2","llama"]',
        '["llama"]',
        '["new_family"]',
    )

    assert selected == tuple(
        sorted((*community_gpu_ci.SHARED_SMOKE_FAMILIES, "llama", "new_family"))
    )


@pytest.mark.parametrize(
    ("scope", "families", "direct", "added"),
    [
        ("docs", "[]", "[]", "[]"),
        ("families", "[]", "[]", "[]"),
        ("families", '["bert"]', '["bert"]', '["new_family"]'),
        ("families", '["bert"]', "[]", "[]"),
        ("all", '["bert"]', '["bert"]', '["bert"]'),
        ("all", '["bert"]', '["gpt2"]', "[]"),
        ("all", '["not-valid"]', "[]", "[]"),
        ("all", '["gpt2","bert"]', "[]", "[]"),
    ],
)
def test_gpu_plan_rejects_malformed_or_inconsistent_selection(
    scope: str,
    families: str,
    direct: str,
    added: str,
) -> None:
    """Selection crossing into the isolated runner stays fail closed."""
    with pytest.raises(CiError):
        community_gpu_ci.selected_families(scope, families, direct, added)


def test_family_plan_selects_only_explicit_premerge_cases(tmp_path: Path) -> None:
    """The runner passes explicit case names and immutable checkpoint inputs."""
    _family(
        tmp_path,
        "alpha",
        [
            {
                "family": "alpha",
                "hf_id": "example/alpha",
                "hf_revision": "a" * 40,
                "hf_dependencies": [{"repo_id": "example/dependency", "revision": "b" * 40}],
                "testcases": [
                    {"name": "alpha-smoke", "premerge": True},
                    {"name": "alpha-nightly", "premerge": False},
                ],
            },
            {
                "family": "alpha",
                "testcases": [{"name": "alpha-local", "premerge": True}],
            },
        ],
    )

    plan = community_gpu_ci.family_plan(tmp_path, "alpha")

    assert plan.family == "alpha"
    assert plan.testcases == ("alpha-local", "alpha-smoke")
    assert plan.checkpoints == (
        ("example/alpha", "a" * 40),
        ("example/dependency", "b" * 40),
    )


@pytest.mark.parametrize(
    "manifests",
    [
        [{"family": "alpha", "testcases": [{"name": "nightly"}]}],
        [
            {
                "family": "alpha",
                "testcases": [
                    {"name": "duplicate", "premerge": True},
                    {"name": "duplicate", "premerge": True},
                ],
            }
        ],
        [
            {
                "family": "alpha",
                "hf_dependencies": [{"repo_id": "example/dependency", "revision": ""}],
                "testcases": [{"name": "invalid-dependency", "premerge": True}],
            }
        ],
    ],
)
def test_family_plan_rejects_missing_or_duplicate_premerge_cases(
    tmp_path: Path,
    manifests: list[dict[str, object]],
) -> None:
    """No selected family can pass without one unambiguous executed case."""
    _family(tmp_path, "alpha", manifests)

    with pytest.raises(CiError):
        community_gpu_ci.family_plan(tmp_path, "alpha")


@pytest.mark.parametrize("with_byok", [False, True])
def test_runtime_root_requires_and_links_native_artifacts(tmp_path: Path, with_byok: bool) -> None:
    """The staged native tree must satisfy the real E2E runtime consumer."""
    build = tmp_path / "build"
    build.mkdir()
    plan = community_gpu_ci.FamilyPlan("alpha", ("alpha-smoke",), ())
    required = (
        "libtrtmc_core.so",
        "libtrtmc_runtime.so",
        "libtrtmc_c.so",
        "libtrtmc_c.so.1",
        "libtrtmc_backend_trt.so",
        "libtrtmc_model_alpha.so",
    )
    for name in required:
        (build / name).write_bytes(b"native")

    if with_byok:
        (build / "libtrtmc_byok_tvm_ffi.so").write_bytes(b"native")
    runtime = community_gpu_ci._runtime_root(build, plan)
    runner = ci_e2e.E2ERunner(ci_context.CiContext(tmp_path, {}))
    with runner._isolated_runtime_root(runtime, plan.family) as isolated:
        for name in required:
            assert (isolated / name).resolve() == (build / name).resolve()
        assert (isolated / "libtrtmc_byok_tvm_ffi.so").is_file() is with_byok


@pytest.mark.parametrize("missing", ["libtrtmc_runtime.so", "libtrtmc_c.so", "libtrtmc_c.so.1"])
def test_runtime_root_rejects_missing_public_runtime_libraries(
    tmp_path: Path, missing: str
) -> None:
    """Incomplete public runtime artifacts cannot reach family E2E execution."""
    build = tmp_path / "build"
    build.mkdir()
    for name in (
        "libtrtmc_core.so",
        "libtrtmc_runtime.so",
        "libtrtmc_c.so",
        "libtrtmc_c.so.1",
        "libtrtmc_backend_trt.so",
        "libtrtmc_model_alpha.so",
    ):
        if name != missing:
            (build / name).write_bytes(b"native")
    plan = community_gpu_ci.FamilyPlan("alpha", ("alpha-smoke",), ())
    with pytest.raises(CiError, match="native Community GPU build is missing"):
        community_gpu_ci._runtime_root(build, plan)


@pytest.mark.parametrize("state", ["python", "native", "missing_declaration", "missing_adapter"])
def test_runtime_root_preserves_selected_family_cli(tmp_path: Path, state: str) -> None:
    """A family CLI declaration and its optional native adapter reach E2E."""
    build = tmp_path / "build"
    build.mkdir()
    plan = community_gpu_ci.FamilyPlan("alpha", ("alpha-smoke",), ())
    for name in (
        "libtrtmc_core.so",
        "libtrtmc_runtime.so",
        "libtrtmc_c.so",
        "libtrtmc_c.so.1",
        "libtrtmc_backend_trt.so",
        "libtrtmc_model_alpha.so",
        "libtrtmc_cli_beta.so",
    ):
        (build / name).write_bytes(b"native")
    source = tmp_path / "families/alpha/cli.json"
    source.parent.mkdir(parents=True)
    executor = "native" if state in {"native", "missing_adapter"} else "python"
    source.write_text(json.dumps({"version": 1, "commands": [{"executor": executor}]}))
    if state == "native":
        (build / "libtrtmc_cli_alpha.so").write_bytes(b"adapter")
    for family in ("alpha", "beta"):
        if family == "alpha" and state == "missing_declaration":
            continue
        declaration = build / "families" / family / "cli.json"
        declaration.parent.mkdir(parents=True)
        declaration.write_bytes(source.read_bytes())

    if state.startswith("missing_"):
        message = (
            "no CLI declaration for alpha"
            if state == "missing_declaration"
            else "libtrtmc_cli_alpha.so"
        )
        with pytest.raises(CiError, match=message):
            community_gpu_ci._runtime_root(build, plan, tmp_path)
        return

    runtime = community_gpu_ci._runtime_root(build, plan, tmp_path)
    assert (runtime / "families/alpha/cli.json").read_bytes() == source.read_bytes()
    assert not (runtime / "families/beta").exists()
    assert (runtime / "libtrtmc_cli_alpha.so").is_file() == (state == "native")
    assert not (runtime / "libtrtmc_cli_beta.so").exists()


def test_checkpoint_staging_verifies_the_resolved_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A moving checkpoint cannot enter offline E2E execution."""
    revision = "a" * 40
    snapshot = tmp_path / revision
    snapshot.mkdir()
    (snapshot / "checkpoint.pt").write_bytes(b"weights")
    calls: list[tuple[str, str | None, Path]] = []

    class FakeApi:
        """Return one immutable revision for the requested model."""

        def model_info(self, _repo_id: str, revision: str | None):
            assert revision is None
            return SimpleNamespace(sha=revision_value)

    def download(*, repo_id: str, revision: str | None, cache_dir: Path) -> str:
        calls.append((repo_id, revision, cache_dir))
        return str(snapshot)

    revision_value = revision
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(HfApi=FakeApi, snapshot_download=download),
    )
    plan = community_gpu_ci.FamilyPlan(
        "alpha",
        ("alpha-smoke",),
        (("example/alpha", None),),
    )

    community_gpu_ci._stage_checkpoints((plan,), tmp_path / "cache")

    assert calls == [("example/alpha", None, tmp_path / "cache")]

    revision_value = "b" * 40
    with pytest.raises(CiError, match="revision changed"):
        community_gpu_ci._stage_checkpoints((plan,), tmp_path / "cache")


@pytest.mark.parametrize("executor", [None, "python", "native"])
def test_gpu_run_builds_native_contract_before_family_e2e(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    executor: str | None,
) -> None:
    """The entrypoint supplies every native path required by E2ERunner."""
    _family(
        tmp_path,
        "alpha",
        [{"family": "alpha", "testcases": [{"name": "alpha-smoke", "premerge": True}]}],
    )
    if executor is not None:
        (tmp_path / "families/alpha/cli.json").write_text(
            json.dumps({"version": 1, "commands": [{"executor": executor}]})
        )
    build = Path("/tmp") / f"{tmp_path.name}-native-build"
    commands: list[list[str]] = []
    e2e_calls: list[tuple[dict[str, str], tuple[str, ...], tuple[str, ...]]] = []

    class FakeContext:
        """Record orchestration without requiring CUDA or a compiler."""

        def __init__(self, repository: Path, env: dict[str, str]):
            self.repository = repository
            self.env = env

        def run(self, command, **_kwargs) -> subprocess.CompletedProcess[str]:
            commands.append([str(argument) for argument in command])
            return subprocess.CompletedProcess(command, 0)

    class FakeE2ERunner:
        """Capture the exact family and testcase contract passed by the entrypoint."""

        def __init__(self, context: FakeContext):
            self.context = context

        def _run(self, families: tuple[str, ...], testcases: tuple[str, ...]) -> None:
            e2e_calls.append((self.context.env, families, testcases))

    monkeypatch.setattr(ci_context, "CiContext", FakeContext)
    monkeypatch.setattr(ci_e2e, "E2ERunner", FakeE2ERunner)
    monkeypatch.setattr(community_gpu_ci, "_install_family_requirements", lambda *_args: None)
    monkeypatch.setattr(community_gpu_ci, "_stage_checkpoints", lambda *_args: None)
    monkeypatch.setattr(
        community_gpu_ci,
        "_runtime_root",
        lambda native_build, _plans, _repository: native_build / "runtime",
    )

    community_gpu_ci.run(
        tmp_path,
        {
            "TRTMC_GPU_SCOPE": "families",
            "TRTMC_GPU_FAMILIES": '["alpha"]',
            "TRTMC_GPU_DIRECT_FAMILIES": '["alpha"]',
            "TRTMC_GPU_ADDED_FAMILIES": "[]",
            "TRTMC_NATIVE_BUILD_DIR": str(build),
        },
        "alpha",
    )

    assert any(command[:2] == ["cmake", "-S"] for command in commands)
    native_builds = [command for command in commands if command[:2] == ["cmake", "--build"]]
    assert "trtmc" in native_builds[0]
    assert "trtmc_backend_trt" in native_builds[0]
    assert "trtmc_runtime" in native_builds[0]
    assert "trtmc_c" in native_builds[0]
    expected_targets = ["trtmc_model_alpha"]
    if executor == "native":
        expected_targets.append("trtmc_cli_alpha")
    assert native_builds[1][native_builds[1].index("--target") + 1 :] == expected_targets
    assert len(e2e_calls) == 1
    runtime, families, testcases = e2e_calls[0]
    assert families == ("alpha",)
    assert testcases == ("alpha-smoke",)
    assert runtime["TRTMC_BINARY"] == str(build / "trtmc")
    assert runtime["TRTMC_RUNTIME_ROOT"] == str(build / "runtime")
    assert runtime["TRTMC_NATIVE_BUILD_DIR"] == str(build)
    assert runtime["HF_HUB_OFFLINE"] == "1"


def _container_receipt(command: list[str], failed: bool = False) -> None:
    """Emulate a completed container, including its selected E2E evidence."""
    cache = next(value for value in command if value.endswith(":/tmp/trtmc-community-huggingface"))
    family = command[-1]
    (Path(cache.rsplit(":", 1)[0]) / "result.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "family": family,
                "status": "failed" if failed else "passed",
                "phase": "validation" if failed else "complete",
                "failure_class": "validation" if failed else None,
                "requested_cases": [family],
                "cases": {family: "failed" if failed else "passed"},
            }
        )
    )


def _planned_owners(repository: Path, *families: str) -> None:
    for family in families:
        _family(
            repository,
            family,
            [
                {
                    "family": family,
                    "testcases": [
                        {"name": family, "premerge": True},
                    ],
                }
            ],
        )


@pytest.mark.parametrize("failed_family", [None, "alpha"])
def test_containers_are_sequential_and_failures_do_not_skip_families(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_family: str | None,
) -> None:
    """Each execution is removed before the next family starts, including failures."""
    events = []
    staged = []
    _planned_owners(tmp_path, "alpha", "beta", "gamma")
    image = "sha256:" + "a" * 64

    def docker(command, **kwargs):
        if command[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(command, 0, stdout=image + "\n")
        if "--stage-family" in command:
            family = command[command.index("--stage-family") + 1]
            assert command[:2] == [sys.executable, str(Path(community_gpu_ci.__file__).resolve())]
            assert command[command.index("--repository") + 1] == str(tmp_path)
            assert kwargs["env"] == {}
            staged.append(family)
            return subprocess.CompletedProcess(command, 0)
        if command[:2] == ["docker", "run"]:
            family = command[-1]
            assert command[-2] == "--family"
            assert image in command
            # Retain exit/OOM state until inspection, then explicitly remove it.
            assert "--rm" not in command
            assert "--memory" in command and "--memory-swap" in command
            assert f"{tmp_path}:/src:ro" in command
            assert (
                f"{Path(community_gpu_ci.__file__).resolve()}:/opt/community_gpu_ci.py:ro"
                in command
            )
            assert "PYTHONPATH=/src" in command
            assert command[-4:] == ["python3.12", "/opt/community_gpu_ci.py", "--family", family]
            assert not any("HF_TOKEN" in value or "docker.sock" in value for value in command)
            events.append(("start", family, command[command.index("--name") + 1]))
            _container_receipt(command, family == failed_family)
            return subprocess.CompletedProcess(command, 17 if family == failed_family else 0)
        if command[:2] == ["docker", "inspect"]:
            return subprocess.CompletedProcess(command, 0, stdout='{"OOMKilled":false}')
        assert command[:3] == ["docker", "rm", "--force"]
        assert command[-1] == events[-1][2]
        events.append(("remove", events[-1][1], command[-1]))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(community_gpu_ci.subprocess, "run", docker)
    env = {
        "TRTMC_GPU_SCOPE": "families",
        "TRTMC_GPU_FAMILIES": '["alpha","beta","gamma"]',
        "TRTMC_GPU_DIRECT_FAMILIES": '["alpha","beta","gamma"]',
        "TRTMC_GPU_ADDED_FAMILIES": "[]",
    }
    if failed_family:
        with pytest.raises(CiError, match="alpha: container exited 17"):
            community_gpu_ci.run_containers(tmp_path, env, "test-image")
    else:
        community_gpu_ci.run_containers(tmp_path, env, "test-image")
    assert [(kind, family) for kind, family, _ in events] == [
        (kind, family) for family in ("alpha", "beta", "gamma") for kind in ("start", "remove")
    ]
    assert staged == ["alpha", "beta", "gamma"]
    assert len({name for kind, _, name in events if kind == "start"}) == 3


@pytest.mark.parametrize("token_from_file", [False, True])
def test_checkpoint_staging_forwards_only_network_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    token_from_file: bool,
) -> None:
    """Only trusted staging receives credentials, never contributor containers."""
    image = "sha256:" + "c" * 64
    runs: list[tuple[list[str], dict]] = []
    token_file = tmp_path / "checkpoint-token"
    _planned_owners(tmp_path, "alpha")

    def docker(command, **kwargs):
        if command[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(command, 0, stdout=image + "\n")
        if "--stage-family" in command or command[:2] == ["docker", "run"]:
            assert not token_file.exists()
            if token_from_file:
                assert "HF_TOKEN" not in os.environ
            runs.append((command, kwargs))
            if command[:2] == ["docker", "run"]:
                _container_receipt(command)
            return subprocess.CompletedProcess(command, 0)
        if command[:2] == ["docker", "inspect"]:
            return subprocess.CompletedProcess(command, 0, stdout='{"OOMKilled":false}')
        assert command[:3] == ["docker", "rm", "--force"]
        return subprocess.CompletedProcess(command, 1, stderr=f"No such container: {command[-1]}")

    monkeypatch.setattr(community_gpu_ci.subprocess, "run", docker)
    env = {
        "TRTMC_GPU_SCOPE": "families",
        "TRTMC_GPU_FAMILIES": '["alpha"]',
        "TRTMC_GPU_DIRECT_FAMILIES": '["alpha"]',
        "TRTMC_GPU_ADDED_FAMILIES": "[]",
        "HTTPS_PROXY": "https://proxy.example",
        "UNRELATED_SECRET": "secret-value",
    }
    if token_from_file:
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        for key in (
            "HF_TOKEN",
            "HF_ENDPOINT",
            "HTTP_PROXY",
            "NO_PROXY",
            "REQUESTS_CA_BUNDLE",
            "SSL_CERT_FILE",
        ):
            monkeypatch.delenv(key, raising=False)
        token_file.write_text("checkpoint-secret", encoding="utf-8")
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "community_gpu_ci.py",
                "--containers",
                "--repository",
                str(tmp_path),
                "--checkpoint-token-file",
                str(token_file),
            ],
        )
        assert community_gpu_ci.main() == 0
        assert not token_file.exists()
    else:
        env["HF_TOKEN"] = "checkpoint-secret"
        community_gpu_ci.run_containers(tmp_path, env, "test-image")

    assert len(runs) == 2
    (stage, stage_options), (family, family_options) = runs
    assert stage[:2] == [sys.executable, str(Path(community_gpu_ci.__file__).resolve())]
    assert "--stage-family" in stage
    assert "secret-value" not in stage
    assert stage_options["env"] == {
        "HTTPS_PROXY": "https://proxy.example",
        "HF_TOKEN": "checkpoint-secret",
    }
    assert "checkpoint-secret" not in str(stage)
    assert "--family" in family
    assert "env" not in family_options
    assert not any("UNRELATED_SECRET" in value or "secret-value" in value for value in family)
    assert "HF_TOKEN" not in str(family)
    assert "checkpoint-secret" not in str(family)
    assert str(token_file) not in str(family)
    assert "TRTMC_CHECKPOINTS_PRESTAGED=1" in family
    stage_cache = str(Path(stage[stage.index("--cache-dir") + 1]).parent)
    family_cache = next(
        value for value in family if value.endswith(":/tmp/trtmc-community-huggingface")
    ).split(":", 1)[0]
    assert family_cache == stage_cache


def test_dependency_build_uses_the_family_container_environment(tmp_path: Path) -> None:
    """Native package build hooks can import the base image's existing torch."""
    root = _family(tmp_path, "alpha", [])
    (root / "requirements.txt").write_text("example==1.0\n")
    calls = []
    context = SimpleNamespace(
        repository=tmp_path, run=lambda command, **kwargs: calls.append(command)
    )
    community_gpu_ci._install_family_requirements(
        context, (community_gpu_ci.FamilyPlan("alpha", (), ()),)
    )
    assert len(calls) == 1
    assert "--no-build-isolation" in calls[0]
    assert calls[0][-1] == root / "requirements.txt"


@pytest.mark.parametrize("already_removed", [False, True])
def test_cleanup_failure_cannot_leave_overlapping_families(tmp_path, monkeypatch, already_removed):
    """Only confirmed absence may ignore a failed explicit container removal."""
    started = []
    _planned_owners(tmp_path, "alpha", "beta")

    def docker(command, **kwargs):
        if command[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(command, 0, stdout="sha256:" + "b" * 64)
        if "--stage-family" in command:
            return subprocess.CompletedProcess(command, 0)
        if command[:2] == ["docker", "run"]:
            started.append(command[-1])
            _container_receipt(command)
            return subprocess.CompletedProcess(command, 0)
        if command[:2] == ["docker", "inspect"]:
            return subprocess.CompletedProcess(command, 0, stdout='{"OOMKilled":false}')
        error = f"No such container: {command[-1]}" if already_removed else "daemon unavailable"
        return subprocess.CompletedProcess(command, 1, stderr=error)

    monkeypatch.setattr(community_gpu_ci.subprocess, "run", docker)
    env = {
        "TRTMC_GPU_SCOPE": "families",
        "TRTMC_GPU_FAMILIES": '["alpha","beta"]',
        "TRTMC_GPU_DIRECT_FAMILIES": '["alpha","beta"]',
        "TRTMC_GPU_ADDED_FAMILIES": "[]",
    }
    if already_removed:
        community_gpu_ci.run_containers(tmp_path, env, "image")
        assert started == ["alpha", "beta"]
    else:
        with pytest.raises(CiError, match="Cannot remove.*daemon unavailable"):
            community_gpu_ci.run_containers(tmp_path, env, "image")
        assert started == ["alpha"]


def test_host_coordinator_does_not_import_source_code(tmp_path: Path) -> None:
    """An untrusted tools package cannot execute in the VM coordinator."""
    _family(
        tmp_path,
        "alpha",
        [{"family": "alpha", "testcases": [{"name": "alpha", "premerge": True}]}],
    )
    tools = tmp_path / "tools"
    tools.mkdir()
    sentinel = tmp_path / "imported"
    (tools / "__init__.py").write_text(
        f"from pathlib import Path; Path({str(sentinel)!r}).touch(); raise RuntimeError('untrusted')\n"
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text(
        "#!/usr/bin/env python3\nimport json, sys\nfrom pathlib import Path\n"
        "if sys.argv[1] == 'image': print('sha256:' + 'a' * 64)\n"
        "elif sys.argv[1] == 'inspect': print('{}')\n"
        "elif sys.argv[1] == 'run':\n"
        "    cache = next(v for v in sys.argv if v.endswith(':/tmp/trtmc-community-huggingface'))\n"
        "    (Path(cache.rsplit(':', 1)[0]) / 'result.json').write_text(json.dumps({\n"
        "        'schema_version': 1, 'family': 'alpha', 'status': 'passed',\n"
        "        'phase': 'complete', 'failure_class': None,\n"
        "        'requested_cases': ['alpha'], 'cases': {'alpha': 'passed'}}))\n"
    )
    docker.chmod(0o755)
    result = subprocess.run(
        [sys.executable, community_gpu_ci.__file__, "--containers", "--repository", str(tmp_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
            "PYTHONPATH": str(tmp_path),
            "TRTMC_GPU_SCOPE": "families",
            "TRTMC_GPU_FAMILIES": '["alpha"]',
            "TRTMC_GPU_DIRECT_FAMILIES": '["alpha"]',
            "TRTMC_GPU_ADDED_FAMILIES": "[]",
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not sentinel.exists()


@pytest.mark.skipif(
    not os.environ.get("TRTMC_COMMUNITY_CONTAINER_TEST_IMAGE"),
    reason="requires an explicitly selected local Docker image",
)
def test_real_containers_do_not_share_family_state(tmp_path: Path, monkeypatch) -> None:
    """A failed family cannot contaminate later families' packages, builds, or cache."""
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import pathlib, sys, sysconfig\n"
        "family = sys.argv[-1]\n"
        "paths = [pathlib.Path(sysconfig.get_paths()['purelib']) / 'trtmc_isolation_probe.py', "
        "pathlib.Path('/tmp/trtmc-community-gpu-build/probe'), "
        "pathlib.Path('/tmp/trtmc-community-huggingface/probe')]\n"
        "for path in paths:\n"
        "    assert not path.exists(), f'{family} inherited {path}'\n"
        "    path.parent.mkdir(parents=True, exist_ok=True)\n"
        "    path.write_text(family)\n"
        "assert not pathlib.Path('/src/should-not-exist').exists()\n"
        "try:\n"
        "    pathlib.Path('/src/should-not-exist').touch()\n"
        "except OSError:\n"
        "    pass\n"
        "else:\n"
        "    raise AssertionError('Source mount is writable')\n"
        "print(f'{family}: fresh Python environment, build, cache, and read-only source', flush=True)\n"
        "sys.exit(17 if family == 'bert' else 0)\n"
    )
    monkeypatch.setattr(community_gpu_ci, "__file__", str(probe))
    with pytest.raises(CiError) as error:
        community_gpu_ci.run_containers(
            tmp_path,
            {
                "TRTMC_GPU_SCOPE": "all",
                "TRTMC_GPU_FAMILIES": "[]",
                "TRTMC_GPU_DIRECT_FAMILIES": "[]",
                "TRTMC_GPU_ADDED_FAMILIES": "[]",
            },
            os.environ["TRTMC_COMMUNITY_CONTAINER_TEST_IMAGE"],
        )
    assert str(error.value) == "Community GPU family failures: bert: container exited 17"


def test_brev_wrapper_caches_application_failure_without_retry(tmp_path: Path) -> None:
    """A nonzero remote application is reported once without Brev rerunning it."""
    from tools import brev_exec

    app = tmp_path / "app.sh"
    counter = tmp_path / "counter"
    result_file = tmp_path / "result"
    app.write_text('#!/bin/sh\nprintf run >> "$1"\nexit 17\n', encoding="utf-8")
    app.chmod(0o755)
    marker = "__TRTMC_TEST_EXIT__="
    wrapper = brev_exec.remote_wrapper((str(app), str(counter)), result_file, marker)

    first = subprocess.run(["bash", "-c", wrapper], check=False, capture_output=True, text=True)
    second = subprocess.run(["bash", "-c", wrapper], check=False, capture_output=True, text=True)

    assert first.returncode == 0
    assert second.returncode == 0
    assert first.stdout.endswith(f"{marker}17\n")
    assert second.stdout == f"{marker}17\n"
    assert counter.read_text(encoding="utf-8") == "run"
    assert brev_exec.parse_remote_status(first.stdout.splitlines(), marker) == 17


def test_eagle_vlm_declares_remote_processor_http_dependency() -> None:
    """The official checkpoint processor can import its requests dependency."""
    requirements = (Path(__file__).parents[2] / "families/eagle_vlm/requirements.txt").read_text(
        encoding="utf-8"
    )

    assert "requests==2.32.5" in requirements.splitlines()


def _row(name: str, **changes: str) -> dict[str, str]:
    row = {
        "name": name,
        "id": f"allocation-{name}",
        "status": "RUNNING",
        "build_status": "COMPLETED",
        "shell_status": "READY",
        "health_status": "HEALTHY",
        "instance_type": brev_provision.DEFAULT_INSTANCE_TYPE,
    }
    row.update(changes)
    return row


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        assert seconds >= 0
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    result = FakeClock()
    monkeypatch.setattr(brev_provision.time, "monotonic", result.monotonic)
    monkeypatch.setattr(brev_provision.time, "sleep", result.sleep)
    return result


class FakeBrev:
    """Exercise lifecycle decisions without a cloud allocation or remote command."""

    def __init__(self, directory):
        self.output = directory / "github-output"
        self.lease = directory / "lease.json"
        self.name = None
        self.sku = brev_provision.DEFAULT_INSTANCE_TYPE
        self.commands = []
        self.command_times = []
        self.events = []
        self.creates = []
        self.states = []
        self.probe_results = []
        self.delete_failures = 0
        self.delete_succeeds = True
        self.inventory_result = None
        self.create_code = 0
        self.native_ready = False
        self.catalog = [
            {"type": sku, "provider": provider, "disk_min_gb": 50, "disk_max_gb": 2560}
            for sku, provider in (
                (brev_provision.DEFAULT_INSTANCE_TYPE, "aws"),
                ("g6e.2xlarge", "aws"),
                (brev_provision.NEBIUS_INSTANCE_TYPE, "nebius"),
            )
        ]

    def provision(self, **options):
        return brev_provision.provision("gpu", lease_file=self.lease, **options)

    def __call__(self, command, deadline, cap=brev_provision.CLI_TIMEOUT, *, input_text=None):
        assert deadline > brev_provision.time.monotonic()
        assert cap > 0
        self.commands.append(command)
        self.command_times.append((command, brev_provision.time.monotonic()))
        action = command[1]
        code, stdout = 0, ""
        if action == "search":
            stdout = json.dumps(self.catalog)
        elif action == "create":
            name = command[2]
            lease = json.loads(self.lease.read_text())
            assert lease["name"] == name and lease["phase"] == "creating"
            assert lease["organization_id"] == "org-test"
            assert lease["allocation_pending"] and lease["instance_id"] is None
            assert command[3:] == ["--detached", "--mode", "vm"]
            specs = json.loads(input_text)
            assert len(specs) == 1 and specs[0]["target_disk_gb"] == 500
            published = dict(line.split("=", 1) for line in self.output.read_text().splitlines())
            assert published["instance_name"] == name
            assert published["instance_type"] == specs[0]["type"]
            assert published["allocation_requested"] == "true"
            self.name, self.sku = name, specs[0]["type"]
            self.creates.append(name)
            self.events.append("create")
            code = self.create_code
        elif action == "ls":
            if self.inventory_result is not None:
                stdout = self.inventory_result
            else:
                rows = []
                if self.name is not None:
                    row = self.states[0] if self.states else _row(self.name)
                    if len(self.states) > 1:
                        self.states.pop(0)
                    row = {**row, "name": self.name, "instance_type": self.sku}
                    self.native_ready = (
                        row["status"] == "RUNNING"
                        and row["build_status"] == "COMPLETED"
                        and row["shell_status"] == "READY"
                        and row["health_status"] not in {"UNHEALTHY", "UNAVAILABLE"}
                    )
                    rows.append(row)
                    self.events.append("inventory:" + row["build_status"])
                else:
                    self.events.append("inventory:absent")
                stdout = json.dumps({"workspaces": rows})
        elif action == "refresh":
            assert self.native_ready, "SSH configuration changed before native readiness"
            self.events.append("refresh")
        elif action == "exec":
            assert self.native_ready, "host command executed before native readiness"
            script = command[-1]
            if "TRTMC_DIAG_" in script:
                self.events.append("diagnostic")
                stdout = "TRTMC_DIAG_CLOUD_STATE=done\nTRTMC_DIAG_HOST_GPU_EXIT=0\n"
            else:
                assert not any(
                    mutation in script
                    for mutation in ("reset-failed", "daemon-reload", "systemctl restart")
                ), "provisioning mutated host services"
                self.events.append("probe")
                outcome = self.probe_results.pop(0) if self.probe_results else "ready"
                marker = shlex.split(script.splitlines()[-1])[-1]
                stdout = "\n".join(
                    f"TRTMC_PHASE_{phase}=0"
                    for phase in ("sudo", "docker", "disk_headroom", "host_gpu", "container_gpu")
                )
                stdout += (
                    "\nTRTMC_PHASE_cloud_init=1\nTRTMC_DISK_AVAILABLE_BYTES="
                    + str(450 * 1024**3)
                    + "\n"
                )
                if outcome == "missing-marker":
                    pass
                elif outcome == "marker-only":
                    stdout = marker + "\n"
                elif outcome == "disk-small":
                    stdout = stdout.replace(str(450 * 1024**3), str(100 * 1024**3)) + marker + "\n"
                elif outcome == "duplicate-marker":
                    stdout += marker + "\n" + marker + "\n"
                elif outcome == "ready":
                    stdout += marker + "\n"
                else:
                    code = 1
        elif action == "delete":
            self.events.append("delete:" + command[2])
            if self.delete_failures:
                self.delete_failures -= 1
                code = 1
            elif self.delete_succeeds:
                self.name, self.states = None, []
            else:
                code = 1
        else:
            raise AssertionError(action)
        return subprocess.CompletedProcess(command, code, stdout, "private-cli-token")


class FakeCleanupAPI:
    """An independent provider boundary: delete acceptance and visibility differ."""

    def __init__(self, brev):
        self.brev = brev
        self.requests = []
        self.responses = {"GET": [], "DELETE": [], "AUTH": [], "LIST": []}
        self.body_changes = {}
        self.auth = (200, {"id": "org-test"})
        self.outage_until = 0

    def workspace(self):
        row = self.brev.states[0] if self.brev.states else _row("gpu")
        return {
            "id": row["id"],
            "name": self.brev.name or "gpu",
            "organizationId": "org-test",
            "instanceType": self.brev.sku,
            "status": "RUNNING",
            **self.body_changes,
        }

    def __call__(self, method, identity, deadline):
        assert method in {"GET", "DELETE", "AUTH", "LIST"}
        assert deadline is None or deadline > brev_provision.time.monotonic()
        self.requests.append((method, identity, deadline, brev_provision.time.monotonic()))
        if brev_provision.time.monotonic() < self.outage_until:
            raise brev_provision.InventoryError("private-api-token temporary outage")
        if self.responses[method]:
            reply = self.responses[method].pop(0)
            if callable(reply):
                return reply()
            if isinstance(reply, BaseException):
                raise reply
            return reply
        if method == "AUTH":
            return self.auth
        if method == "GET":
            return (200, self.workspace()) if self.brev.name is not None else (404, None)
        if method == "LIST":
            return 200, {"items": [self.workspace()] if self.brev.name is not None else []}
        self.brev.events.append("delete:" + identity)
        if self.brev.delete_failures:
            self.brev.delete_failures -= 1
            return 503, None
        if not self.brev.delete_succeeds:
            return 503, None
        body = self.workspace()
        self.brev.name, self.brev.states = None, []
        return 202, body


@pytest.fixture
def fake(monkeypatch, tmp_path, clock):
    result = FakeBrev(tmp_path)
    result.api = FakeCleanupAPI(result)
    monkeypatch.setenv("GITHUB_OUTPUT", str(result.output))
    monkeypatch.setattr(brev_provision, "_run", result)
    monkeypatch.setattr(brev_provision, "_cleanup_request", result.api)
    monkeypatch.setattr(brev_provision, "_allocation_organization", lambda deadline: "org-test")
    return result


def test_exact_catalog_json_disk_and_default_vm_are_used(fake):
    assert fake.provision() == "gpu"
    assert fake.creates == ["gpu"]
    assert fake.events.index("refresh") < fake.events.index("probe")
    lease = json.loads(fake.lease.read_text())
    assert lease["sku"] == "g6.4xlarge" and lease["requested_disk_gb"] == 500
    assert lease["phase"] == "ready" and lease["instance_id"] == "allocation-gpu"
    assert lease["metadata_ready_elapsed_seconds"] <= lease["ready_elapsed_seconds"]
    assert lease["disk_available_bytes"] >= 200 * 1024**3
    assert lease["phases"]["cloud_init"] == "1"
    assert not any(command[1] == "delete" for command in fake.commands)


@pytest.mark.parametrize(
    "catalog", [[], [{"type": "other"}], [{"type": "g6.4xlarge"}, {"type": "g6.4xlarge"}]]
)
def test_invalid_or_ambiguous_catalog_never_creates(fake, catalog):
    fake.catalog = catalog
    with pytest.raises(brev_provision.ProvisionError):
        fake.provision()
    assert not fake.creates
    lease = json.loads(fake.lease.read_text())
    assert not lease["create_started"] and not lease["allocation_pending"]
    assert brev_provision.cleanup("gpu", lease_file=fake.lease) == "gpu"
    assert not any(command[1] == "delete" for command in fake.commands)


def test_insufficient_disk_catalog_never_allocates(fake):
    fake.catalog[0]["disk_max_gb"] = 120
    with pytest.raises(brev_provision.ProvisionError, match="requested disk"):
        fake.provision()
    assert not fake.creates


@pytest.mark.parametrize(
    "state",
    [
        {"status": "FAILURE", "build_status": "PENDING"},
        {"build_status": "CREATE_FAILED"},
    ],
)
def test_create_exit_zero_then_terminal_failure_retains_lease_without_replacement(fake, state):
    fake.states = [_row("gpu", **state)]
    with pytest.raises(brev_provision.ProvisionError):
        fake.provision()
    assert fake.creates == ["gpu"] and "probe" not in fake.events
    assert fake.name == "gpu" and json.loads(fake.lease.read_text())["phase"] == "provision_failed"
    assert not any(command[1] == "delete" for command in fake.commands)
    assert not any(command[1] in {"refresh", "exec"} for command in fake.commands)


def test_building_is_gated_before_actual_probe(fake, clock):
    fake.states = [_row("gpu", build_status="BUILDING", shell_status="NOT READY"), _row("gpu")]
    assert fake.provision() == "gpu"
    assert "diagnostic" not in fake.events
    assert fake.events.index("inventory:COMPLETED") < fake.events.index("refresh")
    assert clock.now == brev_provision.POLL_INTERVAL


@pytest.mark.parametrize("health", ["UNAVAILABLE", "UNHEALTHY"])
def test_initial_health_sync_waits_without_an_extra_vm(fake, health):
    fake.states = [_row("gpu", status="STARTING", health_status=health), _row("gpu")]
    assert fake.provision() == "gpu"
    assert fake.creates == ["gpu"]


@pytest.mark.parametrize(
    "outcome",
    [
        "SSH failure",
        "Docker failure",
        "GPU failure",
        "missing-marker",
        "marker-only",
        "duplicate-marker",
        "disk-small",
    ],
)
def test_probe_failure_and_false_receipts_retry_only_the_same_vm(fake, outcome):
    fake.probe_results = [outcome, "ready"]
    assert fake.provision() == "gpu"
    assert fake.creates == ["gpu"] and fake.events.count("probe") == 2


def test_failed_create_records_partial_allocation_id_for_cleanup(fake):
    fake.create_code = 1
    with pytest.raises(brev_provision.ProvisionError, match="create failed"):
        fake.provision()
    lease = json.loads(fake.lease.read_text())
    assert lease["instance_id"] == "allocation-gpu" and not lease["allocation_pending"]
    assert brev_provision.cleanup("gpu", lease_file=fake.lease) == "gpu"
    assert "delete:allocation-gpu" in fake.events
    assert json.loads(fake.lease.read_text())["phase"] == "deleted"


def test_failed_delete_is_retried_and_cannot_claim_success_while_visible(fake, clock):
    assert fake.provision() == "gpu"
    fake.delete_succeeds = False
    with pytest.raises(brev_provision.ProvisionError, match="unconfirmed"):
        brev_provision.cleanup("gpu", lease_file=fake.lease, timeout=130)
    assert clock.now <= 130 and fake.name == "gpu" and fake.creates == ["gpu"]


def test_delete_requires_two_valid_absence_reads_and_persists_confirmation(fake):
    assert fake.provision() == "gpu"
    fake.delete_failures = 2
    assert brev_provision.cleanup("gpu", lease_file=fake.lease) == "gpu"
    assert fake.events.count("delete:allocation-gpu") == 3
    deletion = next(i for i, request in enumerate(fake.api.requests) if request[0] == "DELETE")
    assert [r[0] for r in fake.api.requests[deletion + 1 :]].count("AUTH") >= 2
    assert [r[0] for r in fake.api.requests[deletion + 1 :]].count("LIST") >= 2
    lease = json.loads(fake.lease.read_text())
    assert lease["cleanup_confirmed"] and lease["phase"] == "deleted"
    assert brev_provision.cleanup("gpu", lease_file=fake.lease) == "gpu"


def test_missing_lease_absence_does_not_hide_a_late_allocation(fake, clock):
    with pytest.raises(brev_provision.ProvisionError, match="unconfirmed"):
        brev_provision.cleanup("gpu", lease_file=fake.lease, timeout=130)
    assert clock.now <= 130 and not fake.creates
    lease = json.loads(fake.lease.read_text())
    assert lease["allocation_pending"] and not lease["cleanup_confirmed"]


def test_missing_owned_lease_does_not_authorize_a_same_name_vm(fake, tmp_path):
    fake.name = "gpu"
    path = tmp_path / "missing-artifact" / "nested" / "lease.json"
    with pytest.raises(brev_provision.ProvisionError, match="unconfirmed"):
        brev_provision.cleanup("gpu", lease_file=path, timeout=130)
    assert path.is_file() and not fake.api.requests
    assert fake.name == "gpu" and not any(event.startswith("delete:") for event in fake.events)


def test_original_unknown_id_creation_is_reconciled_before_direct_deletion(fake, tmp_path):
    fake.name = "gpu"
    path = tmp_path / "original-owned-lease.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "gpu",
                "sku": "g6.4xlarge",
                "instance_id": None,
                "create_started": True,
                "organization_id": "org-test",
                "allocation_pending": True,
            }
        )
    )
    assert brev_provision.cleanup("gpu", lease_file=path) == "gpu"
    assert "delete:allocation-gpu" in fake.events
    assert json.loads(path.read_text())["cleanup_confirmed"]


def test_unknown_id_cannot_bind_foreign_inventory_when_original_org_auth_is_denied(fake, clock):
    fake.name = "gpu"
    fake.lease.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "gpu",
                "sku": "g6.4xlarge",
                "instance_id": None,
                "create_started": True,
                "allocation_pending": True,
                "organization_id": "org-original",
            }
        )
    )
    fake.api.auth = (403, None)
    with pytest.raises(brev_provision.ProvisionError, match="unconfirmed"):
        brev_provision.cleanup("gpu", lease_file=fake.lease, timeout=130)
    assert not fake.commands
    assert all(
        request[0] == "AUTH" and request[1] == "org-original" for request in fake.api.requests
    )
    assert json.loads(fake.lease.read_text())["instance_id"] is None
    assert fake.name == "gpu" and clock.now <= 130


def test_unknown_id_without_original_org_never_discovers_foreign_same_name_instance(
    fake, monkeypatch
):
    fake.name = "gpu"
    fake.lease.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "gpu",
                "sku": "g6.4xlarge",
                "instance_id": None,
                "create_started": True,
                "allocation_pending": True,
            }
        )
    )
    fake.api.auth = (200, {"id": "org-foreign"})
    monkeypatch.setattr(
        brev_provision,
        "_instance",
        lambda *args, **kwargs: pytest.fail(
            "missing original scope must block inventory discovery"
        ),
    )
    with pytest.raises(brev_provision.ProvisionError, match="unconfirmed"):
        brev_provision.cleanup("gpu", lease_file=fake.lease, timeout=130)
    assert not fake.commands and not fake.api.requests
    assert fake.name == "gpu" and json.loads(fake.lease.read_text())["instance_id"] is None


def test_ready_lease_initial_absence_still_requests_deletion_by_id(fake, monkeypatch):
    assert fake.provision() == "gpu"
    fake.name = None
    assert brev_provision.cleanup("gpu", lease_file=fake.lease) == "gpu"
    assert "delete:allocation-gpu" in fake.events


def test_other_cleanup_can_delete_the_same_leased_id_idempotently(fake, monkeypatch):
    assert fake.provision() == "gpu"
    assert brev_provision.cleanup("gpu", lease_file=fake.lease) == "gpu"
    deletes = [r for r in fake.api.requests if r[0] == "DELETE"]
    assert brev_provision.cleanup("gpu", lease_file=fake.lease) == "gpu"
    assert [r for r in fake.api.requests if r[0] == "DELETE"] == deletes
    assert json.loads(fake.lease.read_text())["cleanup_confirmed"]


@pytest.mark.parametrize("change", [{"name": "other"}, {"instance_id": "other"}, {"sku": "other"}])
def test_cleanup_identity_mismatch_never_deletes(fake, change):
    assert fake.provision() == "gpu"
    lease = json.loads(fake.lease.read_text())
    lease.update(change)
    fake.lease.write_text(json.dumps(lease))
    with pytest.raises(brev_provision.ProvisionError):
        brev_provision.cleanup("gpu", lease_file=fake.lease)
    assert not any(r[0] == "DELETE" for r in fake.api.requests)


def test_cleanup_inventory_errors_cannot_count_as_absence(fake, clock, monkeypatch):
    assert fake.provision() == "gpu"

    def unavailable(command, deadline, cap=brev_provision.CLI_TIMEOUT, **kwargs):
        return subprocess.CompletedProcess(command, 1, "", "private-token")

    monkeypatch.setattr(brev_provision, "_run", unavailable)
    fake.api.outage_until = clock.now + 300
    with pytest.raises(brev_provision.ProvisionError, match="unconfirmed"):
        brev_provision.cleanup("gpu", lease_file=fake.lease, timeout=120)
    assert clock.now <= 120
    assert not json.loads(fake.lease.read_text())["cleanup_confirmed"]


def test_known_owned_id_is_deleted_even_when_full_inventory_is_unavailable(fake, monkeypatch):
    assert fake.provision() == "gpu"

    def blocked(*args, **kwargs):
        raise AssertionError("known-ID cleanup must not depend on CLI inventory or CLI delete")

    monkeypatch.setattr(brev_provision, "_run", blocked)
    assert brev_provision.cleanup("gpu", lease_file=fake.lease) == "gpu"
    assert [
        (method, identity) for method, identity, _, _ in fake.api.requests if method == "DELETE"
    ] == [("DELETE", "allocation-gpu")]
    assert json.loads(fake.lease.read_text())["cleanup_confirmed"]


@pytest.mark.parametrize("get_status", [500, 503, 401, 403])
def test_unavailable_id_read_can_use_two_authenticated_post_delete_org_lists(
    fake, monkeypatch, get_status
):
    assert fake.provision() == "gpu"
    fake.api.responses["GET"] = [(get_status, None)] * 100

    def no_cli(*args, **kwargs):
        pytest.fail("known-ID deletion must not require CLI inventory or CLI delete")

    monkeypatch.setattr(brev_provision, "_run", no_cli)
    assert brev_provision.cleanup("gpu", lease_file=fake.lease) == "gpu"
    requests = fake.api.requests
    deletion = next(i for i, request in enumerate(requests) if request[0] == "DELETE")
    lists = [(i, request) for i, request in enumerate(requests) if request[0] == "LIST"]
    assert len([r for r in requests if r[0] == "DELETE"]) == 1
    assert len(lists) == 2 and all(i > deletion for i, _ in lists)
    assert all(request[1] == "org-test" for _, request in lists)
    assert lists[1][1][3] - lists[0][1][3] >= 2
    assert [r[0] for r in requests[deletion + 1 :]].count("AUTH") >= 2
    lease = json.loads(fake.lease.read_text())
    assert lease["phase"] == "deleted" and lease["cleanup_confirmed"]


@pytest.mark.parametrize("listed_status", ["RUNNING", "DELETING", "STOPPED"])
def test_accepted_delete_waits_for_owned_instance_to_disappear_from_org_list(fake, listed_status):
    assert fake.provision() == "gpu"
    body = {**fake.api.workspace(), "status": listed_status}
    fake.api.responses["GET"] = [(500, None)] * 100

    def still_visible():
        assert not json.loads(fake.lease.read_text()).get("cleanup_confirmed", False)
        return 200, {"items": [body]}

    fake.api.responses["LIST"] = [
        still_visible,
        still_visible,
        (200, {"items": []}),
        (200, {"items": []}),
    ]
    assert brev_provision.cleanup("gpu", lease_file=fake.lease) == "gpu"
    assert len([r for r in fake.api.requests if r[0] == "DELETE"]) == 1
    lists = [r for r in fake.api.requests if r[0] == "LIST"]
    assert len(lists) == 4 and lists[-1][3] - lists[-2][3] >= 2
    assert json.loads(fake.lease.read_text())["cleanup_confirmed"]


@pytest.mark.parametrize(
    "invalid",
    [
        (500, None),
        (401, None),
        (403, None),
        (200, None),
        (200, {}),
        (200, {"items": {}}),
        (200, {"items": [{}]}),
    ],
)
def test_invalid_org_list_resets_consecutive_absence_proof(fake, invalid):
    assert fake.provision() == "gpu"
    fake.api.responses["GET"] = [(500, None)] * 100
    fake.api.responses["LIST"] = [
        (200, {"items": []}),
        invalid,
        (200, {"items": []}),
        (200, {"items": []}),
    ]
    assert brev_provision.cleanup("gpu", lease_file=fake.lease) == "gpu"
    assert len([r for r in fake.api.requests if r[0] == "LIST"]) == 4
    assert json.loads(fake.lease.read_text())["cleanup_confirmed"]


@pytest.mark.parametrize("auth", [(401, None), (403, None), (200, {}), (200, {"id": "org-other"})])
def test_empty_list_does_not_confirm_when_original_org_cannot_be_authenticated(fake, auth):
    assert fake.provision() == "gpu"
    fake.api.responses["GET"] = [(500, None)] * 100
    fake.api.auth = auth
    fake.api.responses["LIST"] = [(200, {"items": []})] * 100
    with pytest.raises(brev_provision.ProvisionError, match="unconfirmed"):
        brev_provision.cleanup("gpu", lease_file=fake.lease, timeout=130)
    assert any(r[0] == "DELETE" for r in fake.api.requests)
    assert not any(r[0] == "LIST" for r in fake.api.requests)
    assert not json.loads(fake.lease.read_text())["cleanup_confirmed"]


@pytest.mark.parametrize(
    "change",
    [
        {"organizationId": "org-other"},
        {"id": "other-id"},
        {"name": "other-name"},
        {"instanceType": "other-sku"},
    ],
)
def test_org_list_identity_conflict_cannot_be_treated_as_owned_instance_absence(fake, change):
    assert fake.provision() == "gpu"
    conflicting = {**fake.api.workspace(), **change}
    fake.api.responses["GET"] = [(500, None)] * 100
    fake.api.responses["LIST"] = [(200, {"items": [conflicting]})] * 100
    with pytest.raises(brev_provision.ProvisionError, match="unconfirmed"):
        brev_provision.cleanup("gpu", lease_file=fake.lease, timeout=130)
    assert not json.loads(fake.lease.read_text())["cleanup_confirmed"]
    assert all(r[1] == "allocation-gpu" for r in fake.api.requests if r[0] == "DELETE")


def test_visible_direct_owned_instance_cannot_be_overruled_by_empty_org_list(fake):
    assert fake.provision() == "gpu"
    body = fake.api.workspace()
    fake.api.responses["GET"] = [(200, body)] * 100
    fake.api.responses["LIST"] = [(200, {"items": []})] * 100
    with pytest.raises(brev_provision.ProvisionError, match="unconfirmed"):
        brev_provision.cleanup("gpu", lease_file=fake.lease, timeout=130)
    assert not any(r[0] == "LIST" for r in fake.api.requests)
    assert not json.loads(fake.lease.read_text())["cleanup_confirmed"]


def test_lost_delete_ack_and_broken_id_route_can_confirm_using_original_org_list(fake):
    assert fake.provision() == "gpu"
    fake.api.responses["GET"] = [(500, None)] * 100

    def accepted_but_lost():
        fake.name = None
        raise brev_provision.InventoryError("delete acknowledgement lost")

    fake.api.responses["DELETE"] = [accepted_but_lost, (404, None)]
    assert brev_provision.cleanup("gpu", lease_file=fake.lease) == "gpu"
    requests = fake.api.requests
    deletion = next(i for i, request in enumerate(requests) if request[0] == "DELETE")
    lists = [(i, r) for i, r in enumerate(requests) if r[0] == "LIST"]
    assert len(lists) == 2 and all(i > deletion and r[1] == "org-test" for i, r in lists)
    assert json.loads(fake.lease.read_text())["cleanup_confirmed"]


def test_failed_delete_is_retried_before_org_list_outage_recovers(fake):
    assert fake.provision() == "gpu"
    fake.api.responses["GET"] = [(500, None)] * 100
    fake.api.responses["DELETE"] = [(503, None)]
    fake.api.responses["LIST"] = [(503, None)] * 4
    assert brev_provision.cleanup("gpu", lease_file=fake.lease) == "gpu"
    requests = fake.api.requests
    deletes = [r for r in requests if r[0] == "DELETE"]
    assert len(deletes) == 2 and deletes[1][3] <= 180
    lists = [r for r in requests if r[0] == "LIST"]
    assert len(lists) == 6 and deletes[1][3] < lists[4][3]
    assert fake.name is None and json.loads(fake.lease.read_text())["cleanup_confirmed"]


def test_accepted_delete_is_not_confirmation_while_the_instance_remains_visible(fake):
    assert fake.provision() == "gpu"
    body = fake.api.workspace()
    fake.api.responses["GET"] = [(200, body), (200, body)]
    assert brev_provision.cleanup("gpu", lease_file=fake.lease) == "gpu"
    assert len([r for r in fake.api.requests if r[0] == "DELETE"]) == 1
    deletion = next(i for i, request in enumerate(fake.api.requests) if request[0] == "DELETE")
    assert [r[0] for r in fake.api.requests[deletion + 1 :]].count("AUTH") >= 2
    assert [r[0] for r in fake.api.requests[deletion + 1 :]].count("LIST") >= 2


def test_pre_delete_notfound_read_does_not_replace_two_post_delete_reads(fake):
    assert fake.provision() == "gpu"
    fake.name = None
    fake.api.responses["DELETE"] = [(202, fake.api.workspace())]
    assert brev_provision.cleanup("gpu", lease_file=fake.lease) == "gpu"
    methods = [r[0] for r in fake.api.requests]
    deletion = methods.index("DELETE")
    assert methods[:deletion] == ["GET"]
    assert methods[deletion + 1 :].count("LIST") == 2
    assert methods[deletion + 1 :].count("AUTH") >= 2


def test_until_deleted_outlasts_old_900_second_budget_and_retries_the_same_id(fake, clock):
    assert fake.provision() == "gpu"
    started = clock.now
    fake.api.outage_until = clock.now + 1080
    assert brev_provision.cleanup("gpu", lease_file=fake.lease, until_deleted=True) == "gpu"
    assert clock.now - started > 900
    assert fake.creates == ["gpu"] and fake.name is None
    assert all(deadline is None for _, _, deadline, _ in fake.api.requests)
    assert all(
        identity == "allocation-gpu"
        for method, identity, _, _ in fake.api.requests
        if method in {"GET", "DELETE"}
    )
    assert all(
        identity == "org-test" for method, identity, _, _ in fake.api.requests if method == "LIST"
    )
    assert json.loads(fake.lease.read_text())["cleanup_confirmed"]


@pytest.mark.parametrize("status", [401, 403])
def test_auth_errors_and_masked_notfound_cannot_confirm_cleanup(fake, clock, status):
    assert fake.provision() == "gpu"
    fake.api.responses["GET"] = [(404, None)] * 10
    fake.api.responses["DELETE"] = [(status, None)] * 10
    fake.api.auth = (status, None)
    with pytest.raises(brev_provision.ProvisionError, match="unconfirmed"):
        brev_provision.cleanup("gpu", lease_file=fake.lease, timeout=130)
    assert fake.name == "gpu" and clock.now <= 130
    assert not json.loads(fake.lease.read_text())["cleanup_confirmed"]


def test_notfound_without_owned_scope_or_accepted_delete_is_ambiguous(fake, clock):
    assert fake.provision() == "gpu"
    fake.api.responses["GET"] = [(404, None)] * 10
    fake.api.responses["DELETE"] = [(404, None)] * 10
    fake.api.auth = (200, {})
    with pytest.raises(brev_provision.ProvisionError, match="unconfirmed"):
        brev_provision.cleanup("gpu", lease_file=fake.lease, timeout=130)
    assert fake.name == "gpu" and clock.now <= 130
    assert not json.loads(fake.lease.read_text())["cleanup_confirmed"]


def test_legacy_lease_without_original_org_cannot_use_wrong_org_notfound_as_absence(fake, clock):
    assert fake.provision() == "gpu"
    lease = json.loads(fake.lease.read_text())
    lease.pop("organization_id")
    fake.lease.write_text(json.dumps(lease))
    fake.api.responses["GET"] = [(404, None)] * 10
    fake.api.responses["DELETE"] = [(404, None)] * 10
    fake.api.auth = (200, {"id": "org-other"})
    with pytest.raises(brev_provision.ProvisionError, match="unconfirmed"):
        brev_provision.cleanup("gpu", lease_file=fake.lease, timeout=130)
    assert fake.name == "gpu" and clock.now <= 130
    assert not json.loads(fake.lease.read_text())["cleanup_confirmed"]


def test_legacy_lease_without_original_org_cannot_bind_scope_from_empty_list(fake):
    assert fake.provision() == "gpu"
    lease = json.loads(fake.lease.read_text())
    lease.pop("organization_id")
    fake.lease.write_text(json.dumps(lease))
    fake.api.responses["GET"] = [(500, None)] * 100
    fake.api.responses["DELETE"] = [(404, None)] * 100
    fake.api.responses["LIST"] = [(200, {"items": []})] * 100
    with pytest.raises(brev_provision.ProvisionError, match="unconfirmed"):
        brev_provision.cleanup("gpu", lease_file=fake.lease, timeout=130)
    assert not any(r[0] == "LIST" for r in fake.api.requests)
    lease = json.loads(fake.lease.read_text())
    assert not lease["cleanup_confirmed"] and "organization_id" not in lease


def test_legacy_no_org_lease_cannot_confirm_without_positive_owned_response(fake, clock):
    assert fake.provision() == "gpu"
    lease = json.loads(fake.lease.read_text())
    lease.pop("organization_id")
    fake.lease.write_text(json.dumps(lease))
    fake.api.outage_until = clock.now + 300
    with pytest.raises(brev_provision.ProvisionError, match="unconfirmed"):
        brev_provision.cleanup("gpu", lease_file=fake.lease, timeout=130)
    assert fake.name == "gpu"
    current = json.loads(fake.lease.read_text())
    assert not current["cleanup_confirmed"] and "organization_id" not in current


def test_malformed_direct_get_does_not_block_known_owned_id_delete(fake):
    assert fake.provision() == "gpu"
    fake.api.responses["GET"] = [(200, {"name": "gpu"})]
    assert brev_provision.cleanup("gpu", lease_file=fake.lease) == "gpu"
    assert any(r[0] == "DELETE" for r in fake.api.requests)
    assert json.loads(fake.lease.read_text())["cleanup_confirmed"]


@pytest.mark.parametrize("body", [{"id": "a-different-instance"}, {"name": "someone-else"}])
def test_explicit_identity_conflict_blocks_delete_even_when_other_fields_are_missing(fake, body):
    assert fake.provision() == "gpu"
    fake.api.responses["GET"] = [(200, body)] * 10
    with pytest.raises(brev_provision.ProvisionError, match="unconfirmed"):
        brev_provision.cleanup("gpu", lease_file=fake.lease, timeout=130)
    assert not any(r[0] == "DELETE" for r in fake.api.requests)
    assert fake.name == "gpu"


def test_absence_requires_authenticated_original_organization(fake):
    assert fake.provision() == "gpu"
    fake.api.auth = (200, {"id": "a-different-organization"})
    with pytest.raises(brev_provision.ProvisionError, match="unconfirmed"):
        brev_provision.cleanup("gpu", lease_file=fake.lease, timeout=130)
    assert not json.loads(fake.lease.read_text())["cleanup_confirmed"]


def test_lost_delete_ack_can_recover_from_prior_owned_read_and_authenticated_absence(fake):
    assert fake.provision() == "gpu"

    def accepted_but_lost():
        fake.name = None
        raise brev_provision.InventoryError("delete acknowledgement lost")

    fake.api.responses["DELETE"] = [accepted_but_lost, (404, None)]
    assert brev_provision.cleanup("gpu", lease_file=fake.lease) == "gpu"
    assert fake.name is None and json.loads(fake.lease.read_text())["cleanup_confirmed"]
    methods = [r[0] for r in fake.api.requests]
    first_delete = methods.index("DELETE")
    assert methods[first_delete + 1 :].count("LIST") >= 2
    assert methods[first_delete + 1 :].count("AUTH") >= 2


def test_lost_delete_ack_with_first_get_unavailable_uses_the_trusted_original_lease(fake):
    assert fake.provision() == "gpu"
    fake.api.responses["GET"] = [brev_provision.InventoryError("direct GET unavailable")]

    def accepted_but_lost():
        fake.name = None
        raise brev_provision.InventoryError("delete acknowledgement lost")

    fake.api.responses["DELETE"] = [accepted_but_lost, (404, None)]
    assert brev_provision.cleanup("gpu", lease_file=fake.lease) == "gpu"
    lease = json.loads(fake.lease.read_text())
    assert lease["cleanup_confirmed"] and lease["organization_id"] == "org-test"
    methods = [r[0] for r in fake.api.requests]
    first_delete = methods.index("DELETE")
    assert methods[first_delete + 1 :].count("LIST") >= 2
    assert methods[first_delete + 1 :].count("AUTH") >= 2


@pytest.mark.parametrize("change", [{"id": "wrong"}, {"name": "other"}, {"instanceType": "other"}])
def test_direct_workspace_identity_mismatch_never_requests_deletion(fake, change):
    assert fake.provision() == "gpu"
    fake.api.body_changes = change
    with pytest.raises(brev_provision.ProvisionError, match="unconfirmed"):
        brev_provision.cleanup("gpu", lease_file=fake.lease, timeout=130)
    assert not any(r[0] == "DELETE" for r in fake.api.requests)
    assert fake.name == "gpu"


@pytest.mark.parametrize("malformed_saved", [False, True])
def test_cleanup_environment_key_precedes_saved_credentials(monkeypatch, tmp_path, malformed_saved):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    credentials = tmp_path / ".brev/credentials.json"
    credentials.parent.mkdir()
    credentials.write_text(
        "invalid-json"
        if malformed_saved
        else json.dumps(
            {"api_key": "bak-saved", "access_token": "oauth-saved", "api_key_org_id": "org-test"}
        )
    )
    monkeypatch.setenv("BREV_API_KEY", "bak-environment")
    token, org = brev_provision._cleanup_credentials()
    assert token == "bak-environment"
    assert org == ("" if malformed_saved else "org-test")


def test_wrong_scope_credentials_cannot_start_an_allocation(monkeypatch, tmp_path):
    cli = FakeBrev(tmp_path)
    api_calls = []

    def denied(method, identity, deadline):
        api_calls.append((method, identity))
        return 403, None

    monkeypatch.setattr(brev_provision, "_run", cli)
    monkeypatch.setattr(
        brev_provision, "_cleanup_credentials", lambda: ("bak-other-org-key", "org-original")
    )
    monkeypatch.setattr(brev_provision, "_cleanup_request", denied)
    with pytest.raises(brev_provision.ProvisionError):
        brev_provision.provision("gpu", lease_file=tmp_path / "lease.json")
    assert not cli.creates
    assert api_calls == [("AUTH", "org-original")]


@pytest.mark.parametrize(
    "origin",
    [
        "http://brevapi.us-west-2-prod.control-plane.brev.dev",
        "https://untrusted.example",
        "https://brevapi.us-west-2-prod.control-plane.brev.dev/private",
    ],
)
def test_cleanup_credentials_are_never_sent_to_an_untrusted_origin(monkeypatch, origin):
    import urllib.request

    monkeypatch.setenv("BREV_API_URL", origin)
    monkeypatch.setattr(brev_provision, "_cleanup_credentials", lambda: ("bak-private", "org-test"))
    monkeypatch.setattr(
        urllib.request,
        "build_opener",
        lambda *args: pytest.fail("origin must be rejected before opening HTTP"),
    )
    with pytest.raises(brev_provision.InventoryError, match="official HTTPS"):
        brev_provision._cleanup_http("DELETE", "allocation-gpu")


def test_direct_http_uses_official_origin_bounded_timeout_and_no_credential_redirect(monkeypatch):
    import urllib.request

    monkeypatch.delenv("BREV_API_URL", raising=False)
    monkeypatch.setattr(brev_provision, "_cleanup_credentials", lambda: ("bak-private", "org-test"))

    class Response:
        status = 202

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, bound):
            assert bound == 1024 * 1024 + 1
            return json.dumps(
                {
                    "id": "allocation-gpu",
                    "name": "gpu",
                    "organizationId": "org-test",
                    "instanceType": "g6.4xlarge",
                    "status": "DELETING",
                    "token": "private-response-token",
                }
            ).encode()

    class Opener:
        def open(self, request, timeout):
            assert (
                request.full_url
                == "https://brevapi.us-west-2-prod.control-plane.brev.dev/api/workspaces/allocation-gpu"
            )
            assert request.get_method() == "DELETE"
            assert request.get_header("Authorization") == "Bearer bak-private"
            assert timeout == 30
            return Response()

    def opener(handler):
        assert (
            handler.redirect_request(None, None, 302, "redirect", {}, "https://untrusted.example")
            is None
        )
        return Opener()

    monkeypatch.setattr(urllib.request, "build_opener", opener)
    status, body = brev_provision._cleanup_http("DELETE", "allocation-gpu")
    assert status == 202 and body["organizationId"] == "org-test"
    assert "private-response-token" not in json.dumps(body)


@pytest.mark.parametrize("credential", [("bak-private", "org-test"), ("oauth-private", "")])
def test_org_list_http_preserves_full_native_and_legacy_scope_and_filters_secrets(
    monkeypatch, credential
):
    import urllib.request

    monkeypatch.delenv("BREV_API_URL", raising=False)
    monkeypatch.setattr(brev_provision, "_cleanup_credentials", lambda: credential)
    records = [
        {
            "id": "allocation-gpu",
            "name": "gpu",
            "organizationId": "org-test",
            "instanceType": "g6.4xlarge",
            "status": "DELETING",
            "createdByUserId": "current-user",
            "workspaceClassId": "native",
            "password": "private-password",
        },
        {
            "id": "another-user-instance",
            "name": "another-name",
            "organizationId": "org-test",
            "instanceType": "g6e.2xlarge",
            "status": "RUNNING",
            "createdByUserId": "another-user",
            "workspaceClassId": "legacy",
            "startupScript": "private-userdata",
        },
    ]

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, bound):
            assert bound == 1024 * 1024 + 1
            return json.dumps(records).encode()

    class Opener:
        def open(self, request, timeout):
            assert request.full_url == (
                "https://brevapi.us-west-2-prod.control-plane.brev.dev"
                "/api/organizations/org-test/workspaces"
            )
            assert request.get_method() == "GET"
            assert request.get_header("Authorization") == "Bearer " + credential[0]
            assert timeout == 30
            return Response()

    def opener(handler):
        assert (
            handler.redirect_request(None, None, 302, "redirect", {}, "https://untrusted.example")
            is None
        )
        return Opener()

    monkeypatch.setattr(urllib.request, "build_opener", opener)
    status, body = brev_provision._cleanup_http("LIST", "org-test")
    assert status == 200
    assert [row["id"] for row in body["items"]] == ["allocation-gpu", "another-user-instance"]
    assert all(set(row) == {"id", "name", "organizationId"} for row in body["items"])
    assert "private" not in json.dumps(body)


@pytest.mark.parametrize("document", [{"workspaces": []}, None, [None]])
def test_org_list_http_rejects_a_non_array_or_invalid_workspace_element(monkeypatch, document):
    import urllib.request

    monkeypatch.delenv("BREV_API_URL", raising=False)
    monkeypatch.setattr(brev_provision, "_cleanup_credentials", lambda: ("bak-private", "org-test"))

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, bound):
            return json.dumps(document).encode()

    monkeypatch.setattr(
        urllib.request,
        "build_opener",
        lambda *args: SimpleNamespace(open=lambda *args, **kwargs: Response()),
    )
    with pytest.raises(brev_provision.InventoryError):
        brev_provision._cleanup_http("LIST", "org-test")


@pytest.mark.parametrize("returned_org", ["org-test", "org-other"])
def test_authenticated_http_organization_response_is_validated(monkeypatch, returned_org):
    import urllib.request

    monkeypatch.delenv("BREV_API_URL", raising=False)
    monkeypatch.setattr(brev_provision, "_cleanup_credentials", lambda: ("bak-private", "org-test"))

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, bound):
            return json.dumps({"id": returned_org, "token": "private-token"}).encode()

    class Opener:
        def open(self, request, timeout):
            assert request.full_url.endswith("/api/organizations/org-test")
            return Response()

    monkeypatch.setattr(urllib.request, "build_opener", lambda *args: Opener())
    if returned_org == "org-test":
        assert brev_provision._cleanup_http("AUTH", "") == (200, {"id": "org-test"})
    else:
        with pytest.raises(brev_provision.InventoryError, match="organization is invalid"):
            brev_provision._cleanup_http("AUTH", "")


def test_cleanup_child_argv_and_failed_request_never_disclose_credentials(monkeypatch):
    token = "bak-private-api-token"
    monkeypatch.setenv("BREV_API_KEY", token)
    monkeypatch.setattr(brev_provision.time, "monotonic", lambda: 10)

    def failure(command, deadline, *args, **kwargs):
        assert token not in " ".join(command)
        assert deadline == 40
        raise brev_provision.ProvisionError("upstream error includes " + token)

    monkeypatch.setattr(brev_provision, "_run", failure)
    with pytest.raises(brev_provision.InventoryError) as error:
        brev_provision._cleanup_request("DELETE", "allocation-gpu", None)
    assert token not in str(error.value)


def test_uncertain_create_absence_never_confirms_cleanup(fake, clock):
    fake.lease.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "gpu",
                "sku": "g6.4xlarge",
                "instance_id": None,
                "allocation_pending": True,
                "create_started": True,
                "organization_id": "org-test",
            }
        )
    )
    with pytest.raises(brev_provision.ProvisionError, match="unconfirmed"):
        brev_provision.cleanup("gpu", lease_file=fake.lease, timeout=130)
    assert clock.now <= 130
    assert not json.loads(fake.lease.read_text()).get("cleanup_confirmed", False)


@pytest.mark.parametrize("point", ["initial", "before_probe", "after_probe"])
@pytest.mark.parametrize("bad_json", [False, True])
def test_inventory_errors_preserve_the_same_allocation(fake, monkeypatch, point, bad_json, capsys):
    failed = False

    def run(command, deadline, cap=brev_provision.CLI_TIMEOUT, **kwargs):
        nonlocal failed
        reached = {
            "initial": fake.name is None,
            "before_probe": fake.name is not None and "probe" not in fake.events,
            "after_probe": "probe" in fake.events,
        }[point]
        if command[1] == "ls" and reached and not failed:
            failed = True
            return subprocess.CompletedProcess(
                command, 0 if bad_json else 1, "private-token-invalid-json", "private-token"
            )
        return fake(command, deadline, cap, **kwargs)

    monkeypatch.setattr(brev_provision, "_run", run)
    assert fake.provision() == "gpu" and failed and fake.creates == ["gpu"]
    assert "private-token" not in capsys.readouterr().err


@pytest.mark.parametrize("action", ["refresh", "exec"])
def test_transient_ssh_stage_timeout_retries_the_same_vm(fake, monkeypatch, action):
    failed = False

    def run(command, deadline, cap=brev_provision.CLI_TIMEOUT, **kwargs):
        nonlocal failed
        if command[1] == action and not failed:
            failed = True
            raise brev_provision.ProvisionError("bounded wait")
        return fake(command, deadline, cap, **kwargs)

    monkeypatch.setattr(brev_provision, "_run", run)
    assert fake.provision() == "gpu" and failed and fake.creates == ["gpu"]


@pytest.mark.parametrize(
    "document",
    [
        "banner\n{}",
        "[]",
        "{}",
        '{"workspaces": {}}',
        '{"workspaces": [{}]}',
        '{"workspaces":[],"workspaces":[]}',
    ],
)
def test_invalid_inventory_is_never_absence(fake, document):
    fake.inventory_result = document
    with pytest.raises(brev_provision.InventoryError):
        brev_provision._instance("gpu", 10)


def test_existing_exact_name_is_not_reused_or_deleted(fake):
    fake.name = "gpu"
    with pytest.raises(brev_provision.ProvisionError, match="already exists"):
        fake.provision()
    assert not fake.creates and not any(
        command[1] in {"exec", "delete"} for command in fake.commands
    )
    assert brev_provision.cleanup("gpu", lease_file=fake.lease) == "gpu"
    assert fake.name == "gpu"
    assert not any(command[1] == "delete" for command in fake.commands)


def test_replaced_instance_id_cannot_inherit_probe_or_be_deleted(fake):
    fake.states = [_row("gpu"), _row("gpu", id="different-allocation")]
    with pytest.raises(brev_provision.ProvisionError, match="ID changed"):
        fake.provision()
    with pytest.raises(brev_provision.ProvisionError, match="unconfirmed"):
        brev_provision.cleanup("gpu", lease_file=fake.lease)
    assert not any(command[1] == "delete" for command in fake.commands)


def test_ambiguous_inventory_never_deletes(fake):
    fake.inventory_result = json.dumps({"workspaces": [_row("gpu"), _row("gpu", id="other")]})
    fake.lease.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "gpu",
                "sku": "g6.4xlarge",
                "instance_id": None,
                "create_started": True,
                "organization_id": "org-test",
                "allocation_pending": True,
            }
        )
    )
    with pytest.raises(brev_provision.ProvisionError):
        brev_provision.cleanup("gpu", lease_file=fake.lease, timeout=130)
    assert all(request[0] == "AUTH" for request in fake.api.requests)


def test_probe_deadline_keeps_lease_without_reallocation(fake, clock):
    fake.probe_results = ["GPU failure"] * 30
    with pytest.raises(brev_provision.ProvisionError, match="deadline"):
        fake.provision(timeout=120)
    assert clock.now <= 120 and fake.creates == ["gpu"] and fake.name == "gpu"
    assert json.loads(fake.lease.read_text())["instance_id"] == "allocation-gpu"


def test_interruption_retains_lease_and_never_creates_a_replacement(fake, monkeypatch):
    def run(command, deadline, cap=brev_provision.CLI_TIMEOUT, **kwargs):
        if command[1] == "exec":
            raise KeyboardInterrupt
        return fake(command, deadline, cap, **kwargs)

    monkeypatch.setattr(brev_provision, "_run", run)
    with pytest.raises(KeyboardInterrupt):
        fake.provision()
    assert fake.creates == ["gpu"] and fake.name == "gpu"
    assert json.loads(fake.lease.read_text())["phase"] == "interrupted"


@pytest.mark.parametrize("provider", ["aws", "nebius"])
@pytest.mark.parametrize("ready_after", [720, 1260, 2640])
def test_slow_native_bootstrap_waits_passively_on_one_allocation(
    fake, clock, provider, ready_after
):
    pending_reads = ready_after // brev_provision.POLL_INTERVAL
    fake.states = [
        _row("gpu", build_status="BUILDING", shell_status="NOT READY")
        for _ in range(int(pending_reads))
    ] + [_row("gpu")]
    assert fake.provision(provider=provider) == "gpu"
    assert clock.now == ready_after and fake.creates == ["gpu"]
    assert fake.events.count("refresh") == 1 and fake.events.count("probe") == 1
    assert "diagnostic" not in fake.events
    assert all(
        timestamp >= ready_after
        for command, timestamp in fake.command_times
        if command[1] in {"refresh", "exec"}
    )
    assert all(
        command[1] in {"ls", "search", "create", "refresh", "exec"} for command in fake.commands
    )
    lease = json.loads(fake.lease.read_text())
    assert lease["metadata_ready_elapsed_seconds"] == ready_after
    assert lease["phase"] == "ready" and lease["instance_id"] == "allocation-gpu"


@pytest.mark.parametrize("provider", ["aws", "nebius"])
@pytest.mark.parametrize(
    "not_ready",
    [
        {"status": "STARTING"},
        {"status": "FUTURE_STATE"},
        {"build_status": "BUILDING"},
        {"build_status": "PENDING"},
        {"build_status": "FUTURE_BUILD_STATE"},
        {"shell_status": "NOT READY"},
        {"health_status": "UNAVAILABLE"},
        {"health_status": "UNHEALTHY"},
    ],
)
def test_every_native_admission_field_prevents_early_host_commands(
    fake, clock, provider, not_ready
):
    fake.states = [_row("gpu", **not_ready), _row("gpu")]
    assert fake.provision(provider=provider) == "gpu"
    assert fake.creates == ["gpu"] and "diagnostic" not in fake.events
    assert all(
        timestamp >= brev_provision.POLL_INTERVAL
        for command, timestamp in fake.command_times
        if command[1] in {"refresh", "exec"}
    )
    assert clock.now == brev_provision.POLL_INTERVAL


@pytest.mark.parametrize("provider", ["aws", "nebius"])
def test_passive_bootstrap_timeout_retains_one_vm_for_confirmed_cleanup(fake, clock, provider):
    fake.states = [_row("gpu", build_status="BUILDING", shell_status="NOT READY")]
    with pytest.raises(brev_provision.ProvisionError, match="deadline"):
        fake.provision(provider=provider, timeout=660)
    assert clock.now == 660 and fake.creates == ["gpu"]
    assert not any(command[1] in {"refresh", "exec", "delete"} for command in fake.commands)
    lease = json.loads(fake.lease.read_text())
    assert lease["phase"] == "provision_failed" and lease["instance_id"] == "allocation-gpu"
    assert brev_provision.cleanup("gpu", lease_file=fake.lease) == "gpu"
    assert fake.creates == ["gpu"] and fake.name is None
    assert json.loads(fake.lease.read_text())["cleanup_confirmed"]


@pytest.mark.parametrize("bad_json", [False, True])
def test_api_transient_during_slow_bootstrap_retains_vm_without_early_ssh(
    fake, clock, monkeypatch, bad_json
):
    fake.states = [
        _row("gpu", build_status="BUILDING", shell_status="NOT READY") for _ in range(12)
    ] + [_row("gpu")]
    failed = False

    def read(command, deadline, cap=brev_provision.CLI_TIMEOUT, **kwargs):
        nonlocal failed
        if command[1] == "ls" and fake.name is not None and clock.now >= 300 and not failed:
            failed = True
            return subprocess.CompletedProcess(
                command, 0 if bad_json else 1, "private-invalid-json", "private-api-token"
            )
        return fake(command, deadline, cap, **kwargs)

    monkeypatch.setattr(brev_provision, "_run", read)
    assert fake.provision() == "gpu" and failed and fake.creates == ["gpu"]
    assert clock.now >= 720 and "diagnostic" not in fake.events
    assert all(
        timestamp >= 720
        for command, timestamp in fake.command_times
        if command[1] in {"refresh", "exec"}
    )
    assert not any(command[1] == "delete" for command in fake.commands)


def test_provision_cli_defaults_to_one_45_minute_allocation(monkeypatch, tmp_path):
    captured = {}

    def provision(**options):
        captured.update(options)
        return "gpu"

    monkeypatch.setattr(brev_provision, "provision", provision)
    assert (
        brev_provision.main(
            ["provision", "--instance", "gpu", "--lease-file", str(tmp_path / "lease.json")]
        )
        == 0
    )
    assert captured["timeout"] == 2700 and captured["attempts"] == 1


def test_removed_recovery_option_cannot_allocate(fake):
    with pytest.raises(SystemExit) as error:
        brev_provision.main(
            [
                "provision",
                "--instance",
                "gpu",
                "--lease-file",
                str(fake.lease),
                "--recover-nebius-start-limit",
            ]
        )
    assert error.value.code == 2 and not fake.creates and not fake.commands


@pytest.mark.parametrize("phase", ["sudo", "docker", "host_gpu", "container_gpu", "disk_headroom"])
def test_actual_probe_cannot_hide_functional_failure(phase):
    stubs = r"""
cloud-init() { printf 'private-probe-token\n'; return 1; }
sudo() { shift; if [ "$FAIL_PHASE" = sudo ]; then return 13; fi; "$@"; }
docker() { printf 'private-probe-token\n'; if [ "$FAIL_PHASE" = docker ] && [ "$1" = info ]; then return 3; fi; if [ "$FAIL_PHASE" = container_gpu ] && [ "$1" = run ]; then return 4; fi; }
nvidia-smi() { printf 'private-probe-token\n'; test "$FAIL_PHASE" != host_gpu || return 14; }
python3() { if [ "$FAIL_PHASE" = disk_headroom ]; then printf '0\n'; else printf '500000000000\n'; fi; }
"""
    result = subprocess.run(
        ["bash", "-c", stubs + brev_provision._probe("image", "ready")],
        env={**os.environ, "FAIL_PHASE": phase},
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode != 0 and "ready" not in result.stdout.splitlines()
    assert "private-probe-token" not in result.stdout + result.stderr


def test_actual_probe_accepts_historical_cloud_init_error_but_checks_workload():
    stubs = r"""
cloud-init() { return 1; }
sudo() { shift; "$@"; }
docker() { return 0; }
nvidia-smi() { return 0; }
python3() { printf '500000000000\n'; }
"""
    result = subprocess.run(
        ["bash", "-c", stubs + brev_provision._probe("image", "ready")],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0
    receipt = brev_provision._probe_receipt(result.stdout, "ready", 200)
    assert receipt and receipt["phases"]["cloud_init"] == "1"


def test_public_receipt_fields_filter_private_diagnostic_content():
    stdout = "password=private-token\nTRTMC_DIAG_CLOUD_STATE=done\nTRTMC_DIAG_CLOUD_EXIT=1\nTRTMC_DIAG_UNIT_docker_ActiveState=active\nTRTMC_DIAG_UNIT_instance-oneshot_Environment=private-token\n"
    lines = list(brev_provision._safe_lines(stdout))
    assert lines == [
        "TRTMC_DIAG_CLOUD_STATE=done",
        "TRTMC_DIAG_CLOUD_EXIT=1",
        "TRTMC_DIAG_UNIT_docker_ActiveState=active",
    ]
    assert "private-token" not in "\n".join(lines)


@pytest.mark.parametrize("value", [None, 42, "private\ntoken"])
def test_optional_metadata_cannot_inject_logs(value):
    assert (
        json.loads(brev_provision._state(_row("gpu", instance_type=value)))["instance_type"]
        == "UNKNOWN"
    )


def test_cli_failure_and_interruption_are_nonzero(monkeypatch, tmp_path):
    args = ["provision", "--instance", "gpu", "--lease-file", str(tmp_path / "lease.json")]

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(brev_provision, "provision", interrupted)
    assert brev_provision.main(args) == 130

    def failed(*args, **kwargs):
        raise brev_provision.ProvisionError("not ready")

    monkeypatch.setattr(brev_provision, "provision", failed)
    assert brev_provision.main(args) == 75


@pytest.mark.skipif(
    os.name != "posix" or not Path("/proc").exists(), reason="Linux process groups required"
)
def test_hung_cli_and_ssh_child_are_killed_at_subprocess_deadline(tmp_path: Path) -> None:
    child_pid = tmp_path / "child-pid"
    fake_cli = tmp_path / "hung_brev.py"
    fake_cli.write_text(
        "import subprocess, sys, time\n"
        "from pathlib import Path\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"Path({str(child_pid)!r}).write_text(str(child.pid))\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    start = time.monotonic()

    with pytest.raises(brev_provision.ProvisionError, match="bounded wait"):
        brev_provision._run([sys.executable, str(fake_cli)], start + 0.5)

    assert time.monotonic() - start < 2
    child_stat = Path(f"/proc/{child_pid.read_text(encoding='utf-8')}/stat")
    # SIGKILL delivery is asynchronous; retain the original total termination bound.
    while child_stat.exists() and time.monotonic() < start + 2:
        if child_stat.read_text(encoding="utf-8").split()[2] == "Z":
            break
        time.sleep(0.005)
    if child_stat.exists():
        assert child_stat.read_text(encoding="utf-8").split()[2] == "Z"
    assert time.monotonic() - start < 2


@pytest.mark.skipif(os.name != "posix", reason="POSIX process sessions required")
def test_timeout_does_not_wait_for_escaped_child_holding_pipes(tmp_path):
    child_pid = tmp_path / "escaped-pid"
    cli = tmp_path / "escaped_cli.py"
    cli.write_text(
        "import subprocess,sys,time\nfrom pathlib import Path\n"
        "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'], "
        "start_new_session=True)\n"
        f"Path({str(child_pid)!r}).write_text(str(child.pid))\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    started = time.monotonic()
    try:
        with pytest.raises(brev_provision.ProvisionError, match="bounded wait"):
            brev_provision._run([sys.executable, str(cli)], started + 0.5)
        assert time.monotonic() - started < 2
    finally:
        if child_pid.exists():
            try:
                os.kill(int(child_pid.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.fixture
def local_transport(monkeypatch, tmp_path):
    config = tmp_path / ".brev/ssh_config"
    config.parent.mkdir()
    config.touch()
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path))

    def transport(instance, script, deadline):
        assert instance == "gpu"
        assert deadline > time.monotonic()
        return subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=5)

    monkeypatch.setattr(brev_provision, "_ssh", transport)
    return transport


def _task_command(counter: Path, *, status: int = 0, delay: float = 0.05, output: str = "hello\n"):
    program = (
        "import pathlib,sys,time; "
        "p=pathlib.Path(sys.argv[1]); p.write_text(p.read_text()+'run\\n' if p.exists() else 'run\\n'); "
        "time.sleep(float(sys.argv[2])); sys.stdout.write(sys.argv[3]); sys.exit(int(sys.argv[4]))"
    )
    return [sys.executable, "-c", program, str(counter), str(delay), output, str(status)]


@pytest.mark.parametrize("status", [0, 17])
def test_application_result_is_cached_without_rerun(local_transport, tmp_path, status):
    counter = tmp_path / "counter"
    argv = _task_command(counter, status=status)
    result = tmp_path / "result"
    log = tmp_path / "output.log"
    assert brev_provision.execute("gpu", argv, log, result, 5, 0.02) == status
    assert brev_provision.execute("gpu", argv, log, result, 5, 0.02) == status
    assert counter.read_text() == "run\n"
    assert log.read_text() == "hello\n"


def test_lost_launch_acknowledgment_does_not_duplicate_work(local_transport, monkeypatch, tmp_path):
    calls = 0

    def dropped_ack(instance, script, deadline):
        nonlocal calls
        response = local_transport(instance, script, deadline)
        calls += 1
        if calls == 1:
            return subprocess.CompletedProcess([], 255, "", "connection lost")
        return response

    monkeypatch.setattr(brev_provision, "_ssh", dropped_ack)
    counter = tmp_path / "counter"
    assert (
        brev_provision.execute(
            "gpu",
            _task_command(counter, status=17, delay=0.15),
            tmp_path / "log",
            tmp_path / "result",
            5,
            0.02,
        )
        == 17
    )
    assert counter.read_text() == "run\n"


def test_concurrent_launches_only_execute_once(local_transport, tmp_path):
    counter = tmp_path / "counter"
    result = tmp_path / "result"
    payload = brev_provision._payload(_task_command(counter, status=17, delay=0.2), result, 5)
    marker = "receipt="
    script = brev_provision.launch_script(payload, marker)
    with ThreadPoolExecutor(max_workers=6) as pool:
        responses = list(
            pool.map(lambda _: local_transport("gpu", script, time.monotonic() + 5), range(6))
        )
    assert all(
        brev_provision._record(response.stdout, marker)["state"] == "accepted"
        for response in responses
    )
    deadline = time.monotonic() + 5
    while not result.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert result.read_text() == "17\n"
    assert counter.read_text() == "run\n"


def test_completed_command_cannot_be_replaced(local_transport, tmp_path):
    counter = tmp_path / "counter"
    result = tmp_path / "result"
    assert (
        brev_provision.execute("gpu", _task_command(counter), tmp_path / "log", result, 5, 0.02)
        == 0
    )
    with pytest.raises(RemoteTaskError, match="different task"):
        brev_provision.execute(
            "gpu", _task_command(counter, status=17), tmp_path / "log", result, 5, 0.02
        )
    assert counter.read_text() == "run\n"


def test_large_remote_log_is_fully_drained(local_transport, tmp_path, capsys):
    content = "0123456789\n" * 15000
    # Keep argv within Linux's per-argument limit while producing >2 log chunks.
    argv = [sys.executable, "-c", "print('0123456789\\n' * 15000, end='')"]
    log = tmp_path / "log"
    assert brev_provision.execute("gpu", argv, log, tmp_path / "result", 5, 0.02) == 0
    assert log.read_text() == content
    assert capsys.readouterr().out == content


def test_remote_timeout_has_one_persisted_application_result(local_transport, tmp_path):
    counter = tmp_path / "counter"
    result = tmp_path / "result"
    assert (
        brev_provision.execute(
            "gpu", _task_command(counter, delay=10), tmp_path / "log", result, 0.15, 0.02
        )
        == 124
    )
    assert result.read_text() == "124\n"
    assert counter.read_text() == "run\n"


def test_lost_worker_is_not_relaunched(local_transport, tmp_path):
    counter = tmp_path / "counter"
    argv = _task_command(counter)
    result = tmp_path / "result"
    payload = brev_provision._payload(argv, result, 5)
    root = Path(payload["directory"])
    root.mkdir()
    (root / "fingerprint").write_text(payload["fingerprint"])
    (root / "started").write_text("started\n")
    with pytest.raises(RemoteTaskError, match="no trustworthy result.*lost"):
        brev_provision.execute("gpu", argv, tmp_path / "log", result, 5, 0.02)
    assert not counter.exists()


def test_invalid_existing_exit_status_cannot_pass(local_transport, tmp_path):
    counter = tmp_path / "counter"
    argv = _task_command(counter)
    result = tmp_path / "result"
    assert brev_provision.execute("gpu", argv, tmp_path / "log", result, 5, 0.02) == 0
    result.write_text("success\n")
    with pytest.raises(RemoteTaskError, match="no trustworthy result.*invalid"):
        brev_provision.execute("gpu", argv, tmp_path / "log", result, 5, 0.02)
    assert counter.read_text() == "run\n"


def _gpu_environment(*families: str) -> dict[str, str]:
    owners = json.dumps(sorted(families))
    return {
        "TRTMC_GPU_SCOPE": "families",
        "TRTMC_GPU_FAMILIES": owners,
        "TRTMC_GPU_DIRECT_FAMILIES": owners,
        "TRTMC_GPU_ADDED_FAMILIES": "[]",
    }


def test_execution_budget_scales_without_consuming_cleanup_budget():
    assert community_gpu_ci.execution_budget_seconds(_gpu_environment("alpha")) == 3600
    assert community_gpu_ci.execution_budget_seconds(_gpu_environment("alpha", "beta")) == 7200
    assert (
        community_gpu_ci.execution_budget_seconds(
            _gpu_environment(*(f"family_{chr(97 + i)}" for i in range(20)))
        )
        == 10800
    )


def test_family_result_preserves_failed_skipped_and_unrun_cases(tmp_path):
    build = tmp_path / "build"
    build.mkdir()
    (build / "trtmc-alpha-e2e-junit.xml").write_text(
        "<testsuites><testsuite>"
        '<testcase name="test_official_checkpoint_e2e[one]" />'
        '<testcase name="test_other_e2e[two]"><failure /></testcase>'
        '<testcase name="test_other_e2e[three]"><skipped /></testcase>'
        "</testsuite></testsuites>"
    )
    destination = tmp_path / "result.json"
    env = {"TRTMC_NATIVE_BUILD_DIR": str(build), "TRTMC_GPU_RESULT_FILE": str(destination)}
    with pytest.raises(RuntimeError, match="original validation error"):
        with community_gpu_ci._family_result(env, "alpha") as record:
            record["requested_cases"] = ["one", "two", "three", "four"]
            community_gpu_ci._phase(record, env, "validation", "validation")
            raise RuntimeError("original validation error")
    result = json.loads(destination.read_text())
    assert result["status"] == "failed" and result["failure_class"] == "validation"
    assert result["cases"] == {
        "one": "passed",
        "two": "failed",
        "three": "skipped",
        "four": "not_run",
    }
    assert result["evidence"] == "original validation error"


@pytest.mark.parametrize("mutation", ["missing", "empty", "skipped", "wrong_owner", "symlink"])
def test_container_exit_zero_requires_complete_owned_e2e_evidence(tmp_path, mutation):
    path = tmp_path / "result.json"
    record = {
        "schema_version": 1,
        "family": "alpha",
        "status": "passed",
        "phase": "complete",
        "requested_cases": ["one"],
        "cases": {"one": "passed"},
        "failure_class": None,
    }
    if mutation == "empty":
        record["cases"] = {}
    elif mutation == "skipped":
        record["cases"] = {"one": "skipped"}
    elif mutation == "wrong_owner":
        record["family"] = "beta"
    if mutation != "missing":
        path.write_text(json.dumps(record))
    if mutation == "symlink":
        outside = tmp_path / "outside.json"
        path.rename(outside)
        path.symlink_to(outside)
    result = community_gpu_ci._container_result(path, "alpha", 0, {})
    assert result["status"] == "failed"
    assert result["failure_class"] == "unknown"


def test_confirmed_container_oom_is_a_resource_failure(tmp_path):
    path = tmp_path / "result.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "family": "alpha",
                "status": "running",
                "phase": "validation",
                "failure_class": "validation",
                "requested_cases": ["one"],
                "cases": {"one": "not_run"},
            }
        )
    )
    result = community_gpu_ci._container_result(path, "alpha", 137, {"OOMKilled": True})
    assert result["failure_class"] == "resource" and result["status"] == "failed"
    assert result["phase"] == "validation" and result["cases"]["one"] == "not_run"


def test_exhausted_coordinator_budget_marks_every_unstarted_family(tmp_path, monkeypatch):
    _planned_owners(tmp_path, "alpha", "beta")
    env = {**_gpu_environment("alpha", "beta"), "TRTMC_GPU_RESULTS_DIR": str(tmp_path / "results")}
    calls = []

    def inspect_only(command, **kwargs):
        calls.append(command)
        assert command[:3] == ["docker", "image", "inspect"]
        return subprocess.CompletedProcess(command, 0, stdout="sha256:" + "a" * 64)

    monkeypatch.setattr(community_gpu_ci.subprocess, "run", inspect_only)
    monkeypatch.setattr(community_gpu_ci, "execution_budget_seconds", lambda _env: 0)
    with pytest.raises(CiError, match="remaining families were not run"):
        community_gpu_ci.run_containers(tmp_path, env, "image")
    report = json.loads((tmp_path / "results/summary.json").read_text())
    assert not report["passed"] and not report["complete"]
    assert [row["family"] for row in report["families"]] == ["alpha", "beta"]
    assert all(
        row["status"] == "not_run" and row["failure_class"] == "budget"
        for row in report["families"]
    )
    assert len(calls) == 1


def test_container_cannot_replace_the_host_selected_case_inventory(tmp_path):
    path = tmp_path / "result.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "family": "alpha",
                "status": "passed",
                "phase": "complete",
                "failure_class": None,
                "requested_cases": ["easy"],
                "cases": {"easy": "passed"},
            }
        )
    )
    result = community_gpu_ci._container_result(path, "alpha", 0, {}, ("required", "other"))
    assert result["status"] == "failed" and result["failure_class"] == "unknown"
    assert result["cases"] == {"required": "not_run", "other": "not_run"}


def test_fifo_result_cannot_block_host_cleanup(tmp_path):
    path = tmp_path / "result.json"
    os.mkfifo(path)
    start = time.monotonic()
    result = community_gpu_ci._container_result(path, "alpha", 124, {}, ("required",))
    assert time.monotonic() - start < 1
    assert result["status"] == "failed" and result["cases"] == {"required": "not_run"}


def test_real_container_client_timeout_cleans_up_before_next_family(tmp_path, monkeypatch):
    for family in ("alpha", "beta"):
        _family(
            tmp_path,
            family,
            [{"family": family, "testcases": [{"name": family, "premerge": True}]}],
        )
    events = tmp_path / "events"
    binary = tmp_path / "bin"
    binary.mkdir()
    docker = binary / "docker"
    docker.write_text(
        f"#!{sys.executable}\nimport json, sys, time\nfrom pathlib import Path\n"
        f"events = Path({str(events)!r})\n"
        "if sys.argv[1] == 'image': print('sha256:' + 'a' * 64)\n"
        "elif sys.argv[1] == 'inspect': print('{}')\n"
        "elif sys.argv[1] == 'rm':\n"
        "    with events.open('a') as f: f.write('remove ' + sys.argv[-1].rsplit('-',1)[1] + '\\n')\n"
        "elif sys.argv[1] == 'run':\n"
        "    family = sys.argv[-1]\n"
        "    with events.open('a') as f: f.write('start ' + family + '\\n')\n"
        "    if family == 'alpha': time.sleep(10)\n"
        "    cache = next(v for v in sys.argv if v.endswith(':/tmp/trtmc-community-huggingface'))\n"
        "    (Path(cache.rsplit(':',1)[0]) / 'result.json').write_text(json.dumps({\n"
        "        'schema_version':1,'family':family,'status':'passed','phase':'complete',\n"
        "        'failure_class':None,'requested_cases':[family],'cases':{family:'passed'}}))\n"
    )
    docker.chmod(0o755)
    monkeypatch.setenv("PATH", str(binary) + os.pathsep + os.environ["PATH"])
    monkeypatch.setattr(community_gpu_ci, "FAMILY_TIMEOUT_SECONDS", 0.2)
    monkeypatch.setattr(community_gpu_ci, "STAGING_TIMEOUT_SECONDS", 2)
    env = {**_gpu_environment("alpha", "beta"), "TRTMC_GPU_RESULTS_DIR": str(tmp_path / "results")}
    with pytest.raises(CiError, match="alpha: family execution timed out"):
        community_gpu_ci.run_containers(tmp_path, env, "image")
    assert events.read_text().splitlines() == [
        "start alpha",
        "remove alpha",
        "start beta",
        "remove beta",
    ]
    result = json.loads((tmp_path / "results/summary.json").read_text())
    assert result["families"][0]["failure_class"] == "budget"
    assert result["families"][1]["status"] == "passed"
    assert not result["passed"] and not result["complete"]


def test_log_summary_preserves_last_complete_record_after_interruption(tmp_path):
    path = tmp_path / "output.log"
    expected = {
        "schema_version": 1,
        "complete": False,
        "passed": False,
        "families": [
            {
                "family": "alpha",
                "status": "running",
                "phase": "container",
                "failure_class": "unknown",
                "cases": {},
            },
            {
                "family": "beta",
                "status": "not_run",
                "phase": "pending",
                "failure_class": None,
                "cases": {},
            },
        ],
    }
    path.write_text(
        "ordinary output\n"
        + community_gpu_ci.SUMMARY_PREFIX
        + json.dumps(expected)
        + "\n"
        + community_gpu_ci.SUMMARY_PREFIX
        + '{"schema_'
    )
    rendered = tmp_path / "summary.txt"
    assert community_gpu_ci.summarize_log(path, rendered) == expected
    assert "| beta | not_run |" in rendered.read_text()
    assert "All selected Community E2Es executed: False" in rendered.read_text()


def test_community_routing_preserves_small_case_and_reports_large_case(tmp_path):
    _family(
        tmp_path,
        "alpha",
        [
            {
                "family": "alpha",
                "hf_id": "example/small",
                "testcases": [
                    {"name": "small", "premerge": True},
                ],
            },
            {
                "family": "alpha",
                "hf_id": "example/large",
                "testcases": [
                    {"name": "large", "premerge": True, "community_gpu": False},
                ],
            },
        ],
    )
    plan = community_gpu_ci.family_plan(tmp_path, "alpha")
    assert plan.testcases == ("small",)
    assert plan.checkpoints == (("example/small", None),)
    assert plan.deferred_testcases == ("large",)
    manifest = json.loads((tmp_path / "families/alpha/tests/manifests/1.json").read_text())
    assert manifest["testcases"][0]["premerge"] is True


@pytest.mark.parametrize("selection", [False, "false", None])
def test_community_scope_cannot_silently_remove_all_e2e_coverage(tmp_path, selection):
    _family(
        tmp_path,
        "alpha",
        [
            {
                "family": "alpha",
                "testcases": [
                    {"name": "one", "premerge": True, "community_gpu": selection},
                ],
            }
        ],
    )
    with pytest.raises(CiError):
        community_gpu_ci.family_plan(tmp_path, "alpha")


def test_summary_rejects_malformed_case_data_and_recomputes_coverage(tmp_path):
    record = {
        "schema_version": 1,
        "complete": True,
        "passed": True,
        "families": [
            {
                "family": "alpha",
                "status": "not_run",
                "phase": "pending",
                "failure_class": None,
                "cases": {"one": "not_run"},
            },
        ],
    }
    checked = community_gpu_ci._checked_summary(record)
    assert not checked["complete"] and not checked["passed"]
    record["families"][0]["cases"]["one"] = {}
    assert community_gpu_ci._checked_summary(record) is None
    path = tmp_path / "log"
    path.write_text(community_gpu_ci.SUMMARY_PREFIX + json.dumps(record))
    assert community_gpu_ci.summarize_log(path, tmp_path / "summary") is None


@pytest.mark.parametrize(
    "failure,expected_code,expected_signal",
    [
        ("ssh", 255, "connection_closed"),
        ("stderr-receipt", 0, None),
        ("nonzero-receipt", 17, None),
        ("failed-phase", 0, None),
        ("unknown-stderr", 1, None),
    ],
)
def test_probe_failure_evidence_preserves_safe_distinctions_without_accepting(
    fake, monkeypatch, capsys, failure, expected_code, expected_signal
):
    snapshots = []
    save = brev_provision._save_lease

    def record(path, lease):
        if "last_probe_evidence" in lease:
            snapshots.append(json.loads(json.dumps(lease["last_probe_evidence"])))
        save(path, lease)

    attempts = 0

    def run(command, deadline, cap=brev_provision.CLI_TIMEOUT, **kwargs):
        nonlocal attempts
        result = fake(command, deadline, cap, **kwargs)
        if command[1] != "exec":
            return result
        attempts += 1
        if attempts > 1:
            return result
        if failure == "ssh":
            return subprocess.CompletedProcess(
                command, 255, "", "Connection closed by 192.0.2.4 private-probe-token\n"
            )
        if failure == "stderr-receipt":
            return subprocess.CompletedProcess(
                command, 0, "", result.stdout + "private-probe-token"
            )
        if failure == "nonzero-receipt":
            return subprocess.CompletedProcess(command, 17, result.stdout, "private-probe-token")
        if failure == "failed-phase":
            return subprocess.CompletedProcess(
                command,
                0,
                result.stdout.replace("TRTMC_PHASE_docker=0", "TRTMC_PHASE_docker=1"),
                "",
            )
        return subprocess.CompletedProcess(
            command, 1, "", "private-probe-token unknown upstream error"
        )

    monkeypatch.setattr(brev_provision, "_save_lease", record)
    monkeypatch.setattr(brev_provision, "_run", run)
    assert fake.provision() == "gpu"
    assert fake.creates == ["gpu"] and attempts == 2
    first = snapshots[0]
    assert first["outcome"] == "completed" and first["returncode"] == expected_code
    assert first["stderr_signals"] == ([expected_signal] if expected_signal else [])
    if failure == "stderr-receipt":
        assert not first["stdout_ready_marker"] and not first["stdout_safe_lines"]
        assert "TRTMC_PHASE_container_gpu=0" in first["stderr_safe_lines"]
    if failure == "failed-phase":
        assert "TRTMC_PHASE_docker=1" in first["stdout_safe_lines"]
    if failure == "unknown-stderr":
        assert first["stderr_present"] and not first["stderr_safe_lines"]
    output = capsys.readouterr().err
    assert '"returncode": ' + str(expected_code) in output
    assert "private-probe-token" not in output and "192.0.2.4" not in output
    assert "private-probe-token" not in fake.lease.read_text()


@pytest.mark.parametrize("eventually_ready", [False, True])
def test_partial_timeout_receipt_is_recorded_but_never_establishes_readiness(
    fake, clock, monkeypatch, capsys, eventually_ready
):
    snapshots = []
    save = brev_provision._save_lease
    attempts = 0

    def record(path, lease):
        if "last_probe_evidence" in lease:
            snapshots.append(json.loads(json.dumps(lease["last_probe_evidence"])))
        save(path, lease)

    def run(command, deadline, cap=brev_provision.CLI_TIMEOUT, **kwargs):
        nonlocal attempts
        result = fake(command, deadline, cap, **kwargs)
        if command[1] == "exec":
            attempts += 1
            if attempts == 1 or not eventually_ready:
                clock.now += min(cap, deadline - clock.now)
                raise brev_provision.CommandTimeout(
                    "bounded wait", result.stdout, "Connection reset private-probe-token"
                )
        return result

    monkeypatch.setattr(brev_provision, "_save_lease", record)
    monkeypatch.setattr(brev_provision, "_run", run)
    if eventually_ready:
        assert fake.provision(timeout=400) == "gpu"
        assert attempts == 2
    else:
        with pytest.raises(brev_provision.ProvisionError, match="deadline"):
            fake.provision(timeout=180)
        assert attempts == 1
        assert json.loads(fake.lease.read_text())["phase"] == "provision_failed"
    first = snapshots[0]
    assert first["outcome"] == "timeout" and first["returncode"] is None
    assert (
        first["stdout_ready_marker"] and "TRTMC_PHASE_container_gpu=0" in first["stdout_safe_lines"]
    )
    assert first["stderr_signals"] == ["connection_reset"]
    assert fake.creates == ["gpu"]
    assert "private-probe-token" not in capsys.readouterr().err + fake.lease.read_text()


@pytest.mark.skipif(os.name != "posix", reason="POSIX process sessions required")
@pytest.mark.parametrize("escaped_pipe", [False, True])
def test_real_probe_timeout_retains_partial_stdout_and_stderr_with_bounded_drain(
    tmp_path, escaped_pipe
):
    child_pid = tmp_path / "escaped-pid"
    cli = tmp_path / "partial_cli.py"
    cli.write_text(
        "import subprocess,sys,time\nfrom pathlib import Path\n"
        "print('TRTMC_PHASE_docker=0', flush=True)\n"
        "print('TRTMC_PHASE_host_gpu=1', file=sys.stderr, flush=True)\n"
        "print('Connection closed by 192.0.2.4 private-probe-token', file=sys.stderr, flush=True)\n"
        + (
            "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'], "
            "start_new_session=True)\n"
            f"Path({str(child_pid)!r}).write_text(str(child.pid))\n"
            if escaped_pipe
            else ""
        )
        + "time.sleep(60)\n",
        encoding="utf-8",
    )
    started = time.monotonic()
    try:
        with pytest.raises(brev_provision.CommandTimeout) as failure:
            brev_provision._run([sys.executable, str(cli)], started + 0.3)
        assert time.monotonic() - started < 2
        error = failure.value
        assert error.stdout == "TRTMC_PHASE_docker=0\n"
        assert "TRTMC_PHASE_host_gpu=1\n" in error.stderr
        evidence = brev_provision._probe_evidence(
            error.stdout, error.stderr, "absent", None, "timeout"
        )
        assert evidence["stdout_safe_lines"] == ["TRTMC_PHASE_docker=0"]
        assert evidence["stderr_safe_lines"] == ["TRTMC_PHASE_host_gpu=1"]
        assert evidence["stderr_signals"] == ["connection_closed"]
        assert "private-probe-token" not in json.dumps(evidence) and "192.0.2.4" not in json.dumps(
            evidence
        )
    finally:
        if child_pid.exists():
            try:
                os.kill(int(child_pid.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass

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


@pytest.mark.parametrize("failed_family", [None, "alpha"])
def test_containers_are_sequential_and_failures_do_not_skip_families(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_family: str | None,
) -> None:
    """Each execution is removed before the next family starts, including failures."""
    events = []
    staged = []
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
            assert "--rm" in command
            assert f"{tmp_path}:/src:ro" in command
            assert (
                f"{Path(community_gpu_ci.__file__).resolve()}:/opt/community_gpu_ci.py:ro"
                in command
            )
            assert "PYTHONPATH=/src" in command
            assert command[-4:] == ["python3.12", "/opt/community_gpu_ci.py", "--family", family]
            assert not any("HF_TOKEN" in value or "docker.sock" in value for value in command)
            events.append(("start", family, command[command.index("--name") + 1]))
            return subprocess.CompletedProcess(command, 17 if family == failed_family else 0)
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

    def docker(command, **kwargs):
        if command[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(command, 0, stdout=image + "\n")
        if "--stage-family" in command or command[:2] == ["docker", "run"]:
            assert not token_file.exists()
            if token_from_file:
                assert "HF_TOKEN" not in os.environ
            runs.append((command, kwargs))
            return subprocess.CompletedProcess(command, 0)
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

    def docker(command, **kwargs):
        if command[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(command, 0, stdout="sha256:" + "b" * 64)
        if "--stage-family" in command:
            return subprocess.CompletedProcess(command, 0)
        if command[:2] == ["docker", "run"]:
            started.append(command[-1])
            return subprocess.CompletedProcess(command, 0)
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
        '#!/bin/sh\nif [ "$1" = image ]; then printf "sha256:%s\\n" ' + "a" * 64 + "; fi\n"
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
        self.events = []
        self.creates = []
        self.states = []
        self.probe_results = []
        self.delete_failures = 0
        self.delete_succeeds = True
        self.inventory_result = None
        self.create_code = 0
        self.conditions = {}
        self.recoveries = 0
        self.recovery_ok = True
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
        action = command[1]
        code, stdout = 0, ""
        if action == "search":
            stdout = json.dumps(self.catalog)
        elif action == "create":
            name = command[2]
            lease = json.loads(self.lease.read_text())
            assert lease["name"] == name and lease["phase"] == "creating"
            assert lease["allocation_pending"] and lease["instance_id"] is None
            assert self.output.read_text().splitlines()[-1] == f"instance_name={name}"
            assert command[3:] == ["--detached", "--mode", "vm"]
            specs = json.loads(input_text)
            assert len(specs) == 1 and specs[0]["target_disk_gb"] == 500
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
                    rows.append(row)
                    self.events.append("inventory:" + row["build_status"])
                else:
                    self.events.append("inventory:absent")
                stdout = json.dumps({"workspaces": rows})
        elif action == "refresh":
            self.events.append("refresh")
        elif action == "exec":
            script = command[-1]
            if "TRTMC_DIAG_" in script:
                self.events.append("diagnostic")
                stdout = "TRTMC_DIAG_CLOUD_STATE=done\nTRTMC_DIAG_HOST_GPU_EXIT=0\n"
            elif "TRTMC_RECOVERY_CONDITION_" in script:
                self.events.append("condition")
                keys = (
                    "cloud_done",
                    "docker_start_limit",
                    "docker_clean_exit",
                    "docker_no_auto_restart",
                    "docker_limit_three",
                    "oneshot_failed",
                    "known_oneshot",
                    "known_cdi_restart",
                )
                stdout = "\n".join(
                    f"TRTMC_RECOVERY_CONDITION_{key}={self.conditions.get(key, 1)}" for key in keys
                )
                stdout += "\nTRTMC_RECOVERY_CONDITION_COMPLETE\nprivate-token\n"
            elif "TRTMC_RECOVERY_ONESHOT_RC" in script:
                self.events.append("recovery")
                self.recoveries += 1
                marker = shlex.split(script.splitlines()[-1])[-1]
                code_value = 0 if self.recovery_ok else 1
                stdout = f"TRTMC_RECOVERY_ONESHOT_RC={code_value}\nTRTMC_RECOVERY_RESTORED_RC=0\n{marker}\n"
                if self.recovery_ok:
                    self.states = [_row(self.name)]
            else:
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


@pytest.fixture
def fake(monkeypatch, tmp_path, clock):
    result = FakeBrev(tmp_path)
    monkeypatch.setenv("GITHUB_OUTPUT", str(result.output))
    monkeypatch.setattr(brev_provision, "_run", result)
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


def test_create_exit_zero_then_terminal_failure_retains_lease_without_replacement(fake):
    fake.states = [_row("gpu", status="FAILURE", build_status="CREATE_FAILED")]
    with pytest.raises(brev_provision.ProvisionError):
        fake.provision()
    assert fake.creates == ["gpu"] and "probe" not in fake.events
    assert fake.name == "gpu" and json.loads(fake.lease.read_text())["phase"] == "provision_failed"
    assert not any(command[1] == "delete" for command in fake.commands)


def test_building_is_gated_before_actual_probe(fake, clock):
    fake.states = [_row("gpu", build_status="BUILDING", shell_status="NOT READY"), _row("gpu")]
    assert fake.provision() == "gpu"
    assert fake.events.index("diagnostic") < fake.events.index("probe")
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
    assert fake.events[-2:] == ["inventory:absent", "inventory:absent"]
    lease = json.loads(fake.lease.read_text())
    assert lease["cleanup_confirmed"] and lease["phase"] == "deleted"
    assert brev_provision.cleanup("gpu", lease_file=fake.lease) == "gpu"


def test_missing_lease_absence_does_not_hide_a_late_allocation(fake, clock):
    with pytest.raises(brev_provision.ProvisionError, match="unconfirmed"):
        brev_provision.cleanup("gpu", lease_file=fake.lease, timeout=130)
    assert clock.now <= 130 and not fake.creates
    lease = json.loads(fake.lease.read_text())
    assert lease["allocation_pending"] and not lease["cleanup_confirmed"]


def test_missing_lease_parent_with_visible_vm_is_created_before_delete(fake, tmp_path):
    fake.name = "gpu"
    path = tmp_path / "missing-artifact" / "nested" / "lease.json"
    assert brev_provision.cleanup("gpu", lease_file=path) == "gpu"
    assert "delete:allocation-gpu" in fake.events
    assert json.loads(path.read_text())["phase"] == "deleted"


def test_ready_lease_initial_absence_still_requests_deletion_by_id(fake, monkeypatch):
    assert fake.provision() == "gpu"
    fake.name = None
    assert brev_provision.cleanup("gpu", lease_file=fake.lease) == "gpu"
    assert "delete:allocation-gpu" in fake.events


def test_other_cleanup_can_delete_the_same_leased_id_idempotently(fake, monkeypatch):
    assert fake.provision() == "gpu"

    def run(command, deadline, cap=brev_provision.CLI_TIMEOUT, **kwargs):
        if command[1] == "delete":
            # A concurrent owning cleanup accepted this ID first. This request
            # fails, but two valid absence reads still establish the final state.
            fake.name = None
            return subprocess.CompletedProcess(command, 1, "", "private-delete-token")
        return fake(command, deadline, cap, **kwargs)

    monkeypatch.setattr(brev_provision, "_run", run)
    assert brev_provision.cleanup("gpu", lease_file=fake.lease) == "gpu"
    assert json.loads(fake.lease.read_text())["cleanup_confirmed"]


@pytest.mark.parametrize("change", [{"name": "other"}, {"instance_id": "other"}, {"sku": "other"}])
def test_cleanup_identity_mismatch_never_deletes(fake, change):
    assert fake.provision() == "gpu"
    lease = json.loads(fake.lease.read_text())
    lease.update(change)
    fake.lease.write_text(json.dumps(lease))
    with pytest.raises(brev_provision.ProvisionError):
        brev_provision.cleanup("gpu", lease_file=fake.lease)
    assert not any(command[1] == "delete" for command in fake.commands)


def test_cleanup_inventory_errors_cannot_count_as_absence(fake, clock, monkeypatch):
    assert fake.provision() == "gpu"

    def unavailable(command, deadline, cap=brev_provision.CLI_TIMEOUT, **kwargs):
        return subprocess.CompletedProcess(command, 1, "", "private-token")

    monkeypatch.setattr(brev_provision, "_run", unavailable)
    with pytest.raises(brev_provision.ProvisionError, match="unconfirmed"):
        brev_provision.cleanup("gpu", lease_file=fake.lease, timeout=120)
    assert clock.now <= 120
    assert not json.loads(fake.lease.read_text())["cleanup_confirmed"]


def test_uncertain_create_absence_never_confirms_cleanup(fake, clock):
    fake.lease.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "gpu",
                "sku": "g6.4xlarge",
                "instance_id": None,
                "allocation_pending": True,
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
    with pytest.raises(brev_provision.ProvisionError):
        brev_provision.cleanup("gpu")
    assert not any(command[1] == "delete" for command in fake.commands)


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


def test_only_known_nebius_signature_allows_one_recovery(fake, capsys):
    fake.states = [_row("gpu", build_status="BUILDING", shell_status="NOT READY")]
    assert fake.provision(provider="nebius", recover_nebius_start_limit=True) == "gpu"
    assert fake.recoveries == 1 and fake.creates == ["gpu"]
    lease = json.loads(fake.lease.read_text())
    assert lease["recovery_completed"] and not lease["stock_bootstrap"]
    assert "private-token" not in capsys.readouterr().err


@pytest.mark.parametrize(
    "condition",
    [
        "cloud_done",
        "docker_start_limit",
        "docker_clean_exit",
        "docker_no_auto_restart",
        "docker_limit_three",
        "oneshot_failed",
        "known_oneshot",
        "known_cdi_restart",
    ],
)
def test_any_missing_recovery_signature_blocks_mutation(fake, condition):
    fake.conditions[condition] = 0
    fake.states = [_row("gpu", build_status="BUILDING", shell_status="NOT READY")]
    with pytest.raises(brev_provision.ProvisionError):
        fake.provision(provider="nebius", recover_nebius_start_limit=True, timeout=180)
    assert fake.recoveries == 0 and fake.creates == ["gpu"]
    assert "state.get('status') == 'done'" in brev_provision._recovery_condition()
    assert "result.returncode == 0" in brev_provision._recovery_condition()


def test_recovery_disabled_or_aws_never_mutates_services(fake):
    assert fake.provision(recover_nebius_start_limit=True) == "gpu"
    assert fake.recoveries == 0 and "condition" not in fake.events


def test_failed_recovery_is_not_repeated_or_reallocated(fake):
    fake.states = [_row("gpu", build_status="BUILDING", shell_status="NOT READY")]
    fake.recovery_ok = False
    with pytest.raises(brev_provision.ProvisionError, match="recovery"):
        fake.provision(provider="nebius", recover_nebius_start_limit=True)
    assert fake.recoveries == 1 and fake.creates == ["gpu"]


def test_actual_recovery_restores_override_on_failed_setup(tmp_path):
    events = tmp_path / "events"
    stubs = r"""
sudo() {
 shift
 printf '%s\n' "$*" >> "$EVENTS"
 case "$1" in
  tee) cat >/dev/null ;;
  timeout) return 22 ;;
  systemctl) if [ "$2" = show ]; then printf '3\n'; fi ;;
 esac
 return 0
}
"""
    result = subprocess.run(
        ["bash", "-c", stubs + brev_provision._recovery_script("ready")],
        env={**os.environ, "EVENTS": str(events)},
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode != 0 and "ready" not in result.stdout.splitlines()
    calls = events.read_text().splitlines()
    assert any(line.startswith("rm -f /run/systemd/system/docker.service.d/") for line in calls)
    assert calls.count("systemctl daemon-reload") >= 2


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


def test_diagnostics_print_only_safe_public_fields(monkeypatch, capsys):
    stdout = "password=private-token\nTRTMC_DIAG_CLOUD_STATE=done\nTRTMC_DIAG_CLOUD_EXIT=1\nTRTMC_DIAG_UNIT_docker_ActiveState=active\nTRTMC_DIAG_UNIT_instance-oneshot_Environment=private-token\n"
    monkeypatch.setattr(
        brev_provision,
        "_run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, stdout, "private-token"),
    )
    brev_provision._diagnose("gpu", 10, 0)
    assert "private-token" not in capsys.readouterr().err


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
    if child_stat.exists():
        assert child_stat.read_text(encoding="utf-8").split()[2] == "Z"


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

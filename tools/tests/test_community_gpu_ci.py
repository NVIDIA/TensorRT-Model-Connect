# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for isolated Community GPU family planning and orchestration."""

from __future__ import annotations

import shlex
import time
from tools import brev_exec as brev_provision

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import community_gpu_ci
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
    }
    row.update(changes)
    return row


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        assert seconds >= 0
        self.now += seconds


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    clock = FakeClock()
    monkeypatch.setattr(brev_provision.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(brev_provision.time, "sleep", clock.sleep)
    return clock


class FakeBrev:
    """Implement CLI responses without launching or contacting a cloud VM."""

    def __init__(self, output: Path) -> None:
        self.output = output
        self.name: str | None = None
        self.creates: list[str] = []
        self.commands: list[list[str]] = []
        self.events: list[str] = []
        self.states: list[dict[str, str]] = []
        self.probe_results: list[str] = []
        self.delete_succeeds = True
        self.inventory_result: str | None = None

    def __call__(
        self,
        command: list[str],
        deadline: float,
        cap: float = brev_provision.CLI_TIMEOUT,
    ) -> subprocess.CompletedProcess[str]:
        assert deadline > brev_provision.time.monotonic()
        assert cap > 0
        self.commands.append(command)
        action = command[1]
        stdout = ""
        code = 0
        if action == "create":
            name = command[2]
            assert (
                self.output.read_text(encoding="utf-8").splitlines()[-1] == f"instance_name={name}"
            )
            assert "--detached" in command
            self.creates.append(name)
            self.name = name
            self.events.append(f"create:{name}")
        elif action == "ls":
            if self.inventory_result is not None and self.name is not None:
                stdout = self.inventory_result
            else:
                rows = []
                if self.name is not None:
                    row = self.states[0] if self.states else _row(self.name)
                    if len(self.states) > 1:
                        self.states.pop(0)
                    row = {**row, "name": self.name}
                    rows.append(row)
                    self.events.append(f"inventory:{row['build_status']}")
                else:
                    self.events.append("inventory:absent")
                stdout = json.dumps({"workspaces": rows})
        elif action == "exec":
            self.events.append("probe")
            outcome = self.probe_results.pop(0) if self.probe_results else "ready"
            if outcome == "ready":
                marker = shlex.split(command[-1].splitlines()[-1])[-1]
                stdout = f"GPU-123\nGPU-123\n{marker}\n"
            elif outcome == "missing-marker":
                stdout = "Brev reports success without the remote receipt\n"
            else:
                code = 1
        elif action == "delete":
            self.events.append(f"delete:{command[2]}")
            if self.delete_succeeds:
                self.name = None
                self.states = []
            else:
                code = 1
        else:
            raise AssertionError(f"unexpected Brev command {action}")
        return subprocess.CompletedProcess(command, code, stdout, "")


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, clock: FakeClock) -> FakeBrev:
    output = tmp_path / "github-output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    fake = FakeBrev(output)
    monkeypatch.setattr(brev_provision, "_run", fake)
    return fake


def test_create_exit_zero_then_failure_is_rejected(fake: FakeBrev) -> None:
    fake.states = [_row("gpu", status="FAILURE", build_status="CREATE_FAILED")]

    with pytest.raises(brev_provision.ProvisionError, match="failed after 1 attempts"):
        brev_provision.provision("gpu", "L40", attempts=1)

    assert "probe" not in fake.events
    assert fake.creates == ["gpu"]
    assert "delete:gpu" in fake.events
    assert fake.name is None


def test_building_is_gated_before_actual_probe(fake: FakeBrev, clock: FakeClock) -> None:
    fake.states = [
        _row("gpu", build_status="BUILDING", shell_status="NOT READY"),
        _row("gpu", build_status="BUILDING", shell_status="NOT READY"),
        _row("gpu"),
    ]

    assert brev_provision.provision("gpu", "L40") == "gpu"

    probe_index = fake.events.index("probe")
    assert fake.events[:probe_index].count("inventory:BUILDING") == 2
    assert fake.events[probe_index - 1] == "inventory:COMPLETED"
    assert clock.now == 10
    assert not any(command[1] == "delete" for command in fake.commands)


@pytest.mark.parametrize("health", ["UNAVAILABLE", "UNHEALTHY"])
def test_initial_health_sync_waits_without_an_extra_vm(fake: FakeBrev, health: str) -> None:
    fake.states = [
        _row("gpu", status="STARTING", health_status=health),
        _row("gpu", health_status=health),
        _row("gpu"),
    ]

    assert brev_provision.provision("gpu", "L40") == "gpu"

    assert fake.creates == ["gpu"]
    assert fake.events.count("probe") == 1


@pytest.mark.parametrize(
    "failure", ["SSH failure", "Docker failure", "GPU failure", "missing-marker"]
)
def test_readiness_failure_then_success_reuses_same_vm(fake: FakeBrev, failure: str) -> None:
    fake.probe_results = [failure, "ready"]

    assert brev_provision.provision("gpu", "L40") == "gpu"

    assert fake.creates == ["gpu"]
    assert fake.events.count("probe") == 2
    script = next(command[-1] for command in fake.commands if command[1] == "exec")
    assert "cloud-init status --wait" in script
    assert "sudo -n docker info" in script
    assert "docker run --rm --gpus all" in script
    assert brev_provision.DEFAULT_PROBE_IMAGE in script


def test_retry_deletes_and_confirms_old_vm_before_fallback(fake: FakeBrev) -> None:
    fake.states = [_row("gpu", status="FAILURE")]

    assert brev_provision.provision("gpu", "L40") == "gpu-r2"

    assert fake.creates == ["gpu", "gpu-r2"]
    delete_index = fake.events.index("delete:gpu")
    replacement_index = fake.events.index("create:gpu-r2")
    assert "inventory:absent" in fake.events[delete_index:replacement_index]
    assert [command for command in fake.commands if command[1] == "create"][1][-2:] == [
        "--provider",
        "aws",
    ]
    assert fake.output.read_text(encoding="utf-8").splitlines() == [
        "instance_name=gpu",
        "instance_name=gpu-r2",
    ]


def test_failed_delete_with_visible_vm_blocks_replacement(fake: FakeBrev, clock: FakeClock) -> None:
    fake.states = [_row("gpu", status="FAILURE")]
    fake.delete_succeeds = False

    with pytest.raises(brev_provision.ProvisionError, match="cleanup.*unconfirmed"):
        brev_provision.provision("gpu", "L40", timeout=80)

    assert fake.creates == ["gpu"]
    assert fake.name == "gpu"
    assert clock.now <= 80


def test_overall_deadline_bounds_all_probe_attempts_and_cleanup(
    fake: FakeBrev, clock: FakeClock
) -> None:
    fake.probe_results = ["GPU failure"] * 30

    with pytest.raises(brev_provision.ProvisionError):
        brev_provision.provision("gpu", "L40", timeout=30)

    assert clock.now <= 30
    assert fake.name is None
    assert len(fake.creates) <= 3


def test_slow_aws_boot_can_use_remaining_budget_after_failed_first_vm(
    fake: FakeBrev, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    replacement_started = 0.0

    def run(command: list[str], deadline: float, cap: float = brev_provision.CLI_TIMEOUT):
        nonlocal replacement_started
        if command[1] == "create" and command[2] == "gpu-r2":
            replacement_started = clock.now
        if command[1] == "ls" and fake.name is not None:
            if fake.name == "gpu" or clock.now < replacement_started + 330:
                fake.states = [_row(fake.name, build_status="BUILDING", shell_status="NOT READY")]
            else:
                fake.states = [_row(fake.name)]
        if command[1] == "delete" and fake.name == "gpu":
            clock.now += 55
        if command[1] == "exec":
            # Image pull and the actual Docker GPU probe follow SSH bring-up.
            if clock.now + 45 >= deadline:
                clock.now = deadline
                raise brev_provision.ProvisionError("probe exhausted its readiness budget")
            clock.now += 45
        return fake(command, deadline, cap)

    monkeypatch.setattr(brev_provision, "_run", run)

    assert brev_provision.provision("gpu", "L40") == "gpu-r2"
    assert fake.creates == ["gpu", "gpu-r2"]
    assert replacement_started == 655
    assert clock.now == 1030


def test_ambiguous_create_with_absent_inventory_never_starts_fallback(
    fake: FakeBrev, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    def run(command: list[str], deadline: float, cap: float = brev_provision.CLI_TIMEOUT):
        if command[1] == "create":
            # The API may allocate after this local client has timed out.
            assert fake.output.read_text(encoding="utf-8").strip() == "instance_name=gpu"
            fake.creates.append(command[2])
            clock.now = deadline
            raise brev_provision.ProvisionError("create timed out")
        if command[1] == "delete":
            return subprocess.CompletedProcess(command, 1, "", "not visible yet")
        return fake(command, deadline, cap)

    monkeypatch.setattr(brev_provision, "_run", run)

    with pytest.raises(brev_provision.ProvisionError, match="cleanup.*unconfirmed"):
        brev_provision.provision("gpu", "L40", timeout=40)

    assert fake.creates == ["gpu"]
    assert clock.now <= 40


@pytest.mark.parametrize(
    "document",
    ["banner\n{}", "[]", "{}", '{"workspaces": {}}', '{"workspaces": [{}]}'],
)
def test_invalid_inventory_cannot_be_ready_or_confirm_deletion(
    fake: FakeBrev, document: str
) -> None:
    fake.name = "gpu"
    fake.inventory_result = document

    with pytest.raises(brev_provision.ProvisionError):
        brev_provision._instance("gpu", 10)


def test_exact_name_matching_and_null_empty_collection(fake: FakeBrev) -> None:
    fake.name = "gpu"
    fake.inventory_result = json.dumps({"workspaces": [_row("gpu-r2")]})
    assert brev_provision._instance("gpu", 10) is None
    fake.inventory_result = '{"workspaces": null}'
    assert brev_provision._instance("gpu", 10) is None


def test_existing_exact_name_is_not_reused_or_deleted(fake: FakeBrev) -> None:
    fake.name = "gpu"

    with pytest.raises(brev_provision.ProvisionError, match="already exists"):
        brev_provision.provision("gpu", "L40")

    assert not fake.creates
    assert not any(command[1] in {"exec", "delete"} for command in fake.commands)


def test_replaced_id_cannot_inherit_previous_readiness_receipt(fake: FakeBrev) -> None:
    fake.states = [_row("gpu"), _row("gpu", id="different-allocation")]

    with pytest.raises(brev_provision.ProvisionError, match="failed after 1 attempts"):
        brev_provision.provision("gpu", "L40", attempts=1)

    assert fake.name is None


def test_interruption_cleans_up_without_new_allocation(
    fake: FakeBrev, monkeypatch: pytest.MonkeyPatch
) -> None:
    def run(command: list[str], deadline: float, cap: float = brev_provision.CLI_TIMEOUT):
        if command[1] == "exec":
            raise KeyboardInterrupt
        return fake(command, deadline, cap)

    monkeypatch.setattr(brev_provision, "_run", run)

    with pytest.raises(KeyboardInterrupt):
        brev_provision.provision("gpu", "L40")

    assert fake.creates == ["gpu"]
    assert fake.name is None


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


def test_cli_failure_and_interruption_have_nonzero_results(monkeypatch: pytest.MonkeyPatch) -> None:
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(brev_provision, "provision", interrupted)
    assert brev_provision.main(["provision", "--instance", "gpu", "--gpu", "L40"]) == 130

    def failed(*args, **kwargs):
        raise brev_provision.ProvisionError("not ready")

    monkeypatch.setattr(brev_provision, "provision", failed)
    assert brev_provision.main(["provision", "--instance", "gpu", "--gpu", "L40"]) == 1

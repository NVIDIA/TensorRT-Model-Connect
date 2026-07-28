# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the conditional real-TensorRT graph-patch CI gate."""

from __future__ import annotations

import json
import signal
import subprocess
from pathlib import Path

import pytest

from tools.ci import graph_patch as graph_patch_ci
from tools.ci.graph_patch import GraphPatchRealTrtGate, GraphPatchRealTrtRunner
from tools.ci.process import CiError


def _write_junit(
    path: Path,
    *,
    tests: int = 1,
    failures: int = 0,
    errors: int = 0,
    skipped: int = 0,
) -> None:
    if failures:
        outcome = "<failure />"
    elif errors:
        outcome = "<error />"
    elif skipped:
        outcome = "<skipped />"
    else:
        outcome = ""
    cases = "".join(f'<testcase name="case-{index}">{outcome}</testcase>' for index in range(tests))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "<testsuites>"
        f'<testsuite tests="{tests}" failures="{failures}" '
        f'errors="{errors}" skipped="{skipped}">'
        f"{cases}</testsuite></testsuites>",
        encoding="utf-8",
    )


def _write_preflight(path: Path, revision: str = "b" * 40) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_revision": revision,
                "tensorrt_version": "11.2.0.113",
                "cuda_status": 0,
                "visible_device_count": 1,
                "builder_created": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _container_metadata(
    labels: dict[str, str],
    *,
    container_id: str,
) -> dict[str, object]:
    return {
        "Id": container_id,
        "Config": {"Labels": labels},
    }


def _orphan_labels(
    *,
    owner_token: str = "b" * 32,
    gpu: str = "2",
    lock_namespace: str = "c" * 64,
    slots: str = "1",
) -> dict[str, str]:
    return {
        "com.nvidia.trtmc.model-proof": "1",
        "com.nvidia.trtmc.graph-patch-real-trt": "1",
        "com.nvidia.trtmc.model-proof.gpu": gpu,
        "com.nvidia.trtmc.model-proof.lock-namespace": lock_namespace,
        "com.nvidia.trtmc.model-proof.slots": slots,
        "com.nvidia.trtmc.graph-patch.owner-token": owner_token,
    }


def test_graph_patch_preflight_requires_one_gpu_and_pinned_revision(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "preflight.json"
    revision = "b" * 40
    _write_preflight(evidence, revision)

    assert (
        GraphPatchRealTrtGate.certify_preflight(
            evidence,
            expected_revision=revision,
        )["visible_device_count"]
        == 1
    )

    payload = json.loads(evidence.read_text(encoding="utf-8"))
    payload["visible_device_count"] = 2
    evidence.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CiError, match="visible_device_count"):
        GraphPatchRealTrtGate.certify_preflight(
            evidence,
            expected_revision=revision,
        )


def test_graph_patch_junit_requires_one_fully_passing_test(tmp_path: Path) -> None:
    junit = tmp_path / "junit.xml"
    _write_junit(junit)

    assert GraphPatchRealTrtGate.certify_junit(junit) == {
        "tests": 1,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
    }


@pytest.mark.parametrize(
    ("tests", "failures", "errors", "skipped"),
    (
        (0, 0, 0, 0),
        (1, 1, 0, 0),
        (1, 0, 1, 0),
        (1, 0, 0, 1),
    ),
)
def test_graph_patch_junit_rejects_empty_or_nonpassing_results(
    tmp_path: Path,
    tests: int,
    failures: int,
    errors: int,
    skipped: int,
) -> None:
    junit = tmp_path / "junit.xml"
    _write_junit(
        junit,
        tests=tests,
        failures=failures,
        errors=errors,
        skipped=skipped,
    )

    with pytest.raises(CiError):
        GraphPatchRealTrtGate.certify_junit(junit)


@pytest.mark.parametrize("content", ("", "<not-closed>", "<other />"))
def test_graph_patch_junit_rejects_missing_or_malformed_evidence(
    tmp_path: Path,
    content: str,
) -> None:
    junit = tmp_path / "junit.xml"
    if content:
        junit.write_text(content, encoding="utf-8")

    with pytest.raises(CiError):
        GraphPatchRealTrtGate.certify_junit(junit)


def test_graph_patch_inner_gate_runs_only_the_exact_real_trt_target(
    tmp_path: Path,
) -> None:
    class RecordingContext:
        repository = tmp_path
        env = {"TRTMC_GRAPH_PATCH_JUNIT": str(tmp_path / "junit.xml")}

        def __init__(self) -> None:
            self.calls: list[tuple[list[str], dict[str, object]]] = []

        def run(
            self,
            command: list[str],
            **kwargs: object,
        ) -> subprocess.CompletedProcess[str]:
            self.calls.append((command, kwargs))
            return subprocess.CompletedProcess(command, 0)

    context = RecordingContext()
    gate = GraphPatchRealTrtGate(context)  # type: ignore[arg-type]
    gate.run_test()

    command, options = context.calls[-1]
    assert command[:3] == ["python3", "-m", "pytest"]
    assert command.count(GraphPatchRealTrtGate.TEST_PATH) == 1
    assert GraphPatchRealTrtGate.TEST_PATH.endswith(
        "::test_multi_instance_rewire_compiles_real_tensorrt_network"
    )
    assert not any(character in GraphPatchRealTrtGate.TEST_PATH for character in "*?[")
    assert f"--junitxml={tmp_path / 'junit.xml'}" in command
    assert options["check"] is False
    assert options["limit"] == "10m"
    assert options["updates"] == {"PYTHONPATH": f"{tmp_path / 'python'}:{tmp_path}"}


def test_graph_patch_host_runner_uses_shared_lease_and_isolated_gpu_container(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    output = tmp_path / "gate-output"

    class RecordingContext:
        env = {
            "RUNNER_TEMP": str(tmp_path),
            "GITHUB_RUN_ID": "42",
            "GITHUB_RUN_ATTEMPT": "3",
            "GITHUB_REPOSITORY": "NVIDIA/TensorRT-Model-Connect",
            "TRTMC_CI_IMAGE": "trtmc:test",
            "TRTMC_MODEL_PROOF_GPU_LEASE_TIMEOUT_SECONDS": "600",
            "TRTMC_GRAPH_PATCH_NVIDIA_SMI_TIMEOUT": "2m",
            "TRTMC_GRAPH_PATCH_TRT_PREFLIGHT_TIMEOUT": "2m",
            "TRTMC_GRAPH_PATCH_TEST_TIMEOUT": "10m",
        }

        def __init__(self) -> None:
            self.repository = repository
            self.commands: list[list[str]] = []

        def executable(self, name: str) -> str:
            return f"/usr/bin/{name}"

        def output(self, command: list[str]) -> str:
            self.commands.append(command)
            if command[:2] == ["git", "rev-parse"]:
                return "b" * 40
            if command[:3] == ["git", "status", "--porcelain=v1"]:
                return ""
            if command[:3] == ["docker", "image", "inspect"]:
                return "sha256:" + "a" * 64
            if command[:3] == ["docker", "ps", "--no-trunc"]:
                return ""
            raise AssertionError(command)

        def run(
            self,
            command: list[str],
            **_kwargs: object,
        ) -> subprocess.CompletedProcess[str]:
            self.commands.append(command)
            if command[:3] == ["docker", "container", "inspect"]:
                return subprocess.CompletedProcess(
                    command,
                    1,
                    stdout="",
                    stderr=f"Error: No such object: {command[-1]}",
                )
            if command[:2] == ["docker", "run"]:
                _write_preflight(output / "preflight.json")
                _write_junit(output / "junit.xml")
            return subprocess.CompletedProcess(command, 0)

    leases: list[object] = []

    class FakeLease:
        def __init__(
            self,
            _context: object,
            model: str,
            resource_class: str,
            artifacts: Path,
        ):
            assert model == "graph-patch-real-trt"
            assert resource_class == "shared"
            assert artifacts == output
            self.gpu_id = 2
            self.slot_ids = [1]
            self.slots_per_gpu = 4
            self.lock_namespace = "c" * 64
            self.released = False
            leases.append(self)

        def acquire(self) -> None:
            return None

        def evidence(self, revision: str) -> dict[str, object]:
            return {
                "schema_version": 1,
                "model": "graph-patch-real-trt",
                "source_revision": revision,
                "gpu_id": "2",
                "gpu_slot_ids": [1],
                "resource_class": "shared",
            }

        def release(self) -> None:
            self.released = True

    monkeypatch.setattr(graph_patch_ci, "GpuLease", FakeLease)
    context = RecordingContext()
    GraphPatchRealTrtRunner(  # type: ignore[arg-type]
        context,
        output_dir=output,
        revision="b" * 40,
        owner_token="d" * 32,
    ).run_host()

    lease = leases[0]
    assert lease.released is True  # type: ignore[attr-defined]
    docker_run = next(command for command in context.commands if command[:2] == ["docker", "run"])
    assert docker_run[docker_run.index("--gpus") + 1] == "device=2"
    assert "com.nvidia.trtmc.model-proof.slots=1" in docker_run
    assert "com.nvidia.trtmc.graph-patch.owner-token=" + "d" * 32 in docker_run
    assert "com.nvidia.trtmc.graph-patch.run-id=42" in docker_run
    assert "com.nvidia.trtmc.graph-patch.run-attempt=3" in docker_run
    assert "com.nvidia.trtmc.graph-patch.repository=NVIDIA/TensorRT-Model-Connect" in docker_run
    assert "--read-only" in docker_run
    assert docker_run[docker_run.index("--network") + 1] == "none"
    assert f"type=bind,src={repository},dst=/src,readonly" in docker_run
    assert f"type=bind,src={output},dst=/artifacts" in docker_run
    assert "TRTMC_GRAPH_PATCH_NVIDIA_SMI_TIMEOUT=2m" in docker_run
    assert "TRTMC_GRAPH_PATCH_TRT_PREFLIGHT_TIMEOUT=2m" in docker_run
    assert "TRTMC_GRAPH_PATCH_TEST_TIMEOUT=10m" in docker_run
    assert docker_run[-5:] == [
        "python3",
        "-m",
        "tools.ci",
        "pipeline",
        "graph-patch-real-trt",
    ]
    assert docker_run[-6] == "sha256:" + "a" * 64
    assert (output / "gpu-lease.json").is_file()
    assert (output / "source-revision.txt").read_text(encoding="utf-8") == "b" * 40 + "\n"
    runner_evidence = json.loads((output / "runner-evidence.json").read_text(encoding="utf-8"))
    assert runner_evidence["source_revision"] == "b" * 40
    assert runner_evidence["requested_image"] == "trtmc:test"
    assert runner_evidence["immutable_image_id"] == "sha256:" + "a" * 64
    assert runner_evidence["gpu_lock_namespace"] == "c" * 64
    assert runner_evidence["owner_labels"]["com.nvidia.trtmc.graph-patch.owner-token"] == "d" * 32
    assert (
        sum(command[:3] == ["docker", "container", "inspect"] for command in context.commands) == 2
    )


def test_graph_patch_host_runner_rejects_dirty_checkout(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()

    class DirtyContext:
        env = {
            "RUNNER_TEMP": str(tmp_path),
            "GITHUB_RUN_ID": "42",
            "GITHUB_RUN_ATTEMPT": "3",
        }

        def __init__(self) -> None:
            self.repository = repository
            self.commands: list[list[str]] = []

        def executable(self, name: str) -> str:
            return f"/usr/bin/{name}"

        def output(self, command: list[str]) -> str:
            self.commands.append(command)
            if command[:2] == ["git", "rev-parse"]:
                return "b" * 40
            if command[:3] == ["git", "status", "--porcelain=v1"]:
                return " M python/tensorrt_model_connect/graph_patch.py"
            raise AssertionError(command)

        def run(
            self,
            command: list[str],
            **_kwargs: object,
        ) -> subprocess.CompletedProcess[str]:
            self.commands.append(command)
            return subprocess.CompletedProcess(command, 0)

    context = DirtyContext()
    with pytest.raises(CiError, match="outside the pinned revision"):
        GraphPatchRealTrtRunner(  # type: ignore[arg-type]
            context,
            revision="b" * 40,
        ).run_host()
    assert not [command for command in context.commands if command[:2] == ["docker", "rm"]]


def test_graph_patch_host_runner_revision_mismatch_makes_no_destructive_call(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()

    class MismatchContext:
        env = {"RUNNER_TEMP": str(tmp_path)}

        def __init__(self) -> None:
            self.repository = repository
            self.commands: list[list[str]] = []
            self.revision_calls = 0

        def executable(self, name: str) -> str:
            return f"/usr/bin/{name}"

        def output(self, command: list[str]) -> str:
            self.commands.append(command)
            if command[:2] == ["git", "rev-parse"]:
                self.revision_calls += 1
                return ("b" if self.revision_calls == 1 else "c") * 40
            raise AssertionError(command)

        def run(
            self,
            command: list[str],
            **_kwargs: object,
        ) -> subprocess.CompletedProcess[str]:
            self.commands.append(command)
            return subprocess.CompletedProcess(command, 0)

    context = MismatchContext()
    with pytest.raises(CiError, match="does not match the pinned tested revision"):
        GraphPatchRealTrtRunner(  # type: ignore[arg-type]
            context,
            revision="b" * 40,
        ).run_host()

    assert not [command for command in context.commands if command[:2] == ["docker", "rm"]]


def test_graph_patch_cleanup_refuses_foreign_same_name_without_removal(
    tmp_path: Path,
) -> None:
    class InventoryContext:
        repository = tmp_path
        env: dict[str, str] = {}

        def __init__(self) -> None:
            self.containers: dict[str, dict[str, object]] = {}
            self.commands: list[list[str]] = []

        def run(
            self,
            command: list[str],
            **_kwargs: object,
        ) -> subprocess.CompletedProcess[str]:
            self.commands.append(command)
            if command[:3] == ["docker", "container", "inspect"]:
                name = command[-1]
                if name not in self.containers:
                    return subprocess.CompletedProcess(
                        command,
                        1,
                        stdout="",
                        stderr=f"Error: No such object: {name}",
                    )
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout=json.dumps(self.containers[name]),
                    stderr="",
                )
            if command[:3] == ["docker", "rm", "-f"]:
                self.containers.pop(command[-1])
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
            raise AssertionError(command)

    context = InventoryContext()
    owner = GraphPatchRealTrtRunner(  # type: ignore[arg-type]
        context,
        owner_token="a" * 32,
    )
    foreign = GraphPatchRealTrtRunner(  # type: ignore[arg-type]
        context,
        owner_token="b" * 32,
    )
    context.containers[owner.container_name] = _container_metadata(
        foreign._ownership_labels(),
        container_id="1" * 64,
    )

    with pytest.raises(CiError, match="refusing to remove foreign"):
        owner._remove_owned_container(reason="before launch")

    assert owner.container_name in context.containers
    assert not [command for command in context.commands if command[:3] == ["docker", "rm", "-f"]]


def test_default_local_runners_use_unique_names_and_scoped_cleanup(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()

    class InventoryContext:
        env = {"RUNNER_TEMP": str(tmp_path)}

        def __init__(self) -> None:
            self.repository = repository
            self.containers: dict[str, dict[str, object]] = {}
            self.removed: list[str] = []

        def run(
            self,
            command: list[str],
            **_kwargs: object,
        ) -> subprocess.CompletedProcess[str]:
            if command[:3] == ["docker", "container", "inspect"]:
                name = command[-1]
                if name not in self.containers:
                    return subprocess.CompletedProcess(
                        command,
                        1,
                        stdout="",
                        stderr=f"Error: No such object: {name}",
                    )
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout=json.dumps(self.containers[name]),
                    stderr="",
                )
            if command[:3] == ["docker", "rm", "-f"]:
                container_id = command[-1]
                name = next(
                    name
                    for name, metadata in self.containers.items()
                    if metadata["Id"] == container_id
                )
                self.removed.append(container_id)
                self.containers.pop(name)
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
            raise AssertionError(command)

    context = InventoryContext()
    first = GraphPatchRealTrtRunner(context)  # type: ignore[arg-type]
    second = GraphPatchRealTrtRunner(context)  # type: ignore[arg-type]

    assert first.owner_token != second.owner_token
    assert first.container_name != second.container_name
    assert first._prepare_output() != second._prepare_output()
    context.containers = {
        first.container_name: _container_metadata(
            first._ownership_labels(),
            container_id="2" * 64,
        ),
        second.container_name: _container_metadata(
            second._ownership_labels(),
            container_id="3" * 64,
        ),
    }
    first.container_launch_attempted = True
    first._cleanup()

    assert context.removed == ["2" * 64]
    assert second.container_name in context.containers


def test_orphan_reclaim_removes_old_owner_on_the_acquired_slot(tmp_path: Path) -> None:
    candidate_id = "5" * 64

    class Context:
        repository = tmp_path
        env: dict[str, str] = {}

        def __init__(self) -> None:
            self.removed: list[str] = []

        def output(self, command: list[str]) -> str:
            assert command[:3] == ["docker", "ps", "--no-trunc"]
            assert "label=com.nvidia.trtmc.model-proof=1" in command
            assert "label=com.nvidia.trtmc.graph-patch-real-trt=1" not in command
            assert "label=com.nvidia.trtmc.model-proof.gpu=2" in command
            assert "label=com.nvidia.trtmc.model-proof.lock-namespace=" + "c" * 64 in command
            return candidate_id

        def run(
            self,
            command: list[str],
            **_kwargs: object,
        ) -> subprocess.CompletedProcess[str]:
            if command[:3] == ["docker", "container", "inspect"]:
                assert command[-1] == candidate_id
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout=json.dumps(
                        _container_metadata(
                            _orphan_labels(owner_token="b" * 32),
                            container_id=candidate_id,
                        )
                    ),
                    stderr="",
                )
            if command[:3] == ["docker", "rm", "-f"]:
                self.removed.append(command[-1])
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
            raise AssertionError(command)

    class Lease:
        gpu_id = 2
        slot_ids = [1]
        slots_per_gpu = 4
        lock_namespace = "c" * 64

    context = Context()
    runner = GraphPatchRealTrtRunner(  # type: ignore[arg-type]
        context,
        owner_token="a" * 32,
    )
    runner.lease = Lease()  # type: ignore[assignment]
    runner._reclaim_orphans()

    assert context.removed == [candidate_id]


def test_orphan_reclaim_removes_ordinary_model_proof_on_acquired_slot(
    tmp_path: Path,
) -> None:
    candidate_id = "a" * 64
    labels = _orphan_labels()
    labels.pop("com.nvidia.trtmc.graph-patch-real-trt")
    labels.pop("com.nvidia.trtmc.graph-patch.owner-token")

    class Context:
        repository = tmp_path
        env: dict[str, str] = {}

        def __init__(self) -> None:
            self.removed: list[str] = []

        def output(self, _command: list[str]) -> str:
            return candidate_id

        def run(
            self,
            command: list[str],
            **_kwargs: object,
        ) -> subprocess.CompletedProcess[str]:
            if command[:3] == ["docker", "container", "inspect"]:
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout=json.dumps(_container_metadata(labels, container_id=candidate_id)),
                    stderr="",
                )
            if command[:3] == ["docker", "rm", "-f"]:
                self.removed.append(command[-1])
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
            raise AssertionError(command)

    class Lease:
        gpu_id = 2
        slot_ids = [1]
        slots_per_gpu = 4
        lock_namespace = "c" * 64

    context = Context()
    runner = GraphPatchRealTrtRunner(context)  # type: ignore[arg-type]
    runner.lease = Lease()  # type: ignore[assignment]
    runner._reclaim_orphans()

    assert context.removed == [candidate_id]


@pytest.mark.parametrize(
    ("gpu", "lock_namespace", "slots"),
    (
        ("2", "c" * 64, "2"),
        ("2", "d" * 64, "1"),
        ("3", "c" * 64, "1"),
    ),
)
def test_orphan_reclaim_preserves_nonoverlapping_or_foreign_leases(
    tmp_path: Path,
    gpu: str,
    lock_namespace: str,
    slots: str,
) -> None:
    candidate_id = "6" * 64

    class Context:
        repository = tmp_path
        env: dict[str, str] = {}

        def __init__(self) -> None:
            self.removed: list[str] = []

        def output(self, _command: list[str]) -> str:
            return candidate_id

        def run(
            self,
            command: list[str],
            **_kwargs: object,
        ) -> subprocess.CompletedProcess[str]:
            if command[:3] == ["docker", "container", "inspect"]:
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout=json.dumps(
                        _container_metadata(
                            _orphan_labels(
                                owner_token="b" * 32,
                                gpu=gpu,
                                lock_namespace=lock_namespace,
                                slots=slots,
                            ),
                            container_id=candidate_id,
                        )
                    ),
                    stderr="",
                )
            if command[:3] == ["docker", "rm", "-f"]:
                self.removed.append(command[-1])
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
            raise AssertionError(command)

    class Lease:
        gpu_id = 2
        slot_ids = [1]
        slots_per_gpu = 4
        lock_namespace = "c" * 64

    context = Context()
    runner = GraphPatchRealTrtRunner(context)  # type: ignore[arg-type]
    runner.lease = Lease()  # type: ignore[assignment]
    runner._reclaim_orphans()

    assert context.removed == []


@pytest.mark.parametrize("slots", ("1,999", "1,1", "4"))
def test_orphan_reclaim_rejects_invalid_slot_sets_without_deletion(
    tmp_path: Path,
    slots: str,
) -> None:
    candidate_id = "b" * 64

    class Context:
        repository = tmp_path
        env: dict[str, str] = {}

        def __init__(self) -> None:
            self.commands: list[list[str]] = []

        def output(self, _command: list[str]) -> str:
            return candidate_id

        def run(
            self,
            command: list[str],
            **_kwargs: object,
        ) -> subprocess.CompletedProcess[str]:
            self.commands.append(command)
            if command[:3] == ["docker", "container", "inspect"]:
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout=json.dumps(
                        _container_metadata(
                            _orphan_labels(slots=slots),
                            container_id=candidate_id,
                        )
                    ),
                    stderr="",
                )
            if command[:3] == ["docker", "rm", "-f"]:
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
            raise AssertionError(command)

    class Lease:
        gpu_id = 2
        slot_ids = [1]
        slots_per_gpu = 4
        lock_namespace = "c" * 64

    context = Context()
    runner = GraphPatchRealTrtRunner(context)  # type: ignore[arg-type]
    runner.lease = Lease()  # type: ignore[assignment]

    with pytest.raises(CiError, match="invalid GPU slot labels"):
        runner._reclaim_orphans()
    assert not [command for command in context.commands if command[:3] == ["docker", "rm", "-f"]]


@pytest.mark.parametrize("inventory", ("unsafe-id", "7" * 64 + "\n" + "7" * 64))
def test_orphan_reclaim_rejects_malformed_inventory_without_deletion(
    tmp_path: Path,
    inventory: str,
) -> None:
    class Context:
        repository = tmp_path
        env: dict[str, str] = {}

        def __init__(self) -> None:
            self.commands: list[list[str]] = []

        def output(self, _command: list[str]) -> str:
            return inventory

        def run(
            self,
            command: list[str],
            **_kwargs: object,
        ) -> subprocess.CompletedProcess[str]:
            self.commands.append(command)
            return subprocess.CompletedProcess(command, 0)

    class Lease:
        gpu_id = 2
        slot_ids = [1]
        slots_per_gpu = 4
        lock_namespace = "c" * 64

    context = Context()
    runner = GraphPatchRealTrtRunner(context)  # type: ignore[arg-type]
    runner.lease = Lease()  # type: ignore[assignment]

    with pytest.raises(CiError, match="orphan inventory"):
        runner._reclaim_orphans()
    assert not [command for command in context.commands if command[:3] == ["docker", "rm", "-f"]]


def test_orphan_reclaim_inventory_error_makes_no_destructive_call(tmp_path: Path) -> None:
    class Context:
        repository = tmp_path
        env: dict[str, str] = {}

        def __init__(self) -> None:
            self.commands: list[list[str]] = []

        def output(self, _command: list[str]) -> str:
            raise CiError("Docker inventory failed")

        def run(
            self,
            command: list[str],
            **_kwargs: object,
        ) -> subprocess.CompletedProcess[str]:
            self.commands.append(command)
            return subprocess.CompletedProcess(command, 0)

    class Lease:
        gpu_id = 2
        slot_ids = [1]
        slots_per_gpu = 4
        lock_namespace = "c" * 64

    context = Context()
    runner = GraphPatchRealTrtRunner(context)  # type: ignore[arg-type]
    runner.lease = Lease()  # type: ignore[assignment]

    with pytest.raises(CiError, match="inventory failed"):
        runner._reclaim_orphans()
    assert not [command for command in context.commands if command[:3] == ["docker", "rm", "-f"]]


@pytest.mark.parametrize("failure", ("error", "malformed"))
def test_orphan_reclaim_inspect_failure_is_transactional(
    tmp_path: Path,
    failure: str,
) -> None:
    first_id = "8" * 64
    second_id = "9" * 64

    class Context:
        repository = tmp_path
        env: dict[str, str] = {}

        def __init__(self) -> None:
            self.commands: list[list[str]] = []

        def output(self, _command: list[str]) -> str:
            return f"{first_id}\n{second_id}"

        def run(
            self,
            command: list[str],
            **_kwargs: object,
        ) -> subprocess.CompletedProcess[str]:
            self.commands.append(command)
            if command[:3] == ["docker", "container", "inspect"]:
                if command[-1] == first_id:
                    return subprocess.CompletedProcess(
                        command,
                        0,
                        stdout=json.dumps(
                            _container_metadata(
                                _orphan_labels(),
                                container_id=first_id,
                            )
                        ),
                        stderr="",
                    )
                if failure == "error":
                    return subprocess.CompletedProcess(
                        command,
                        2,
                        stdout="",
                        stderr="Docker daemon unavailable",
                    )
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout="{}",
                    stderr="",
                )
            if command[:3] == ["docker", "rm", "-f"]:
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
            raise AssertionError(command)

    class Lease:
        gpu_id = 2
        slot_ids = [1]
        slots_per_gpu = 4
        lock_namespace = "c" * 64

    context = Context()
    runner = GraphPatchRealTrtRunner(context)  # type: ignore[arg-type]
    runner.lease = Lease()  # type: ignore[assignment]

    with pytest.raises(CiError, match="inspect|metadata"):
        runner._reclaim_orphans()
    assert not [command for command in context.commands if command[:3] == ["docker", "rm", "-f"]]


def test_signal_path_validates_same_owner_before_finally_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Context:
        repository = tmp_path
        env: dict[str, str] = {}

        def __init__(self) -> None:
            self.metadata: dict[str, object] = {}
            self.removed: list[str] = []

        def run(
            self,
            command: list[str],
            **_kwargs: object,
        ) -> subprocess.CompletedProcess[str]:
            if command[:3] == ["docker", "container", "inspect"]:
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout=json.dumps(self.metadata),
                    stderr="",
                )
            if command[:3] == ["docker", "rm", "-f"]:
                self.removed.append(command[-1])
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
            raise AssertionError(command)

    context = Context()
    runner = GraphPatchRealTrtRunner(  # type: ignore[arg-type]
        context,
        owner_token="e" * 32,
    )
    context.metadata = _container_metadata(
        runner._ownership_labels(),
        container_id="4" * 64,
    )
    runner.container_launch_attempted = True
    monkeypatch.setattr(
        runner,
        "_run_host",
        lambda: runner._signal(signal.SIGTERM, None),
    )

    with pytest.raises(SystemExit) as exit_info:
        runner.run_host()

    assert exit_info.value.code == 143
    assert context.removed == ["4" * 64]


def test_graph_patch_inner_timeout_cannot_exceed_outer_budget(
    tmp_path: Path,
) -> None:
    class Context:
        repository = tmp_path
        env = {
            "TRTMC_GRAPH_PATCH_JUNIT": str(tmp_path / "junit.xml"),
            "TRTMC_GRAPH_PATCH_TEST_TIMEOUT": "11m",
        }

        def run(
            self,
            command: list[str],
            **_kwargs: object,
        ) -> subprocess.CompletedProcess[str]:
            raise AssertionError(command)

    with pytest.raises(CiError, match="must not exceed 600 seconds"):
        GraphPatchRealTrtGate(Context()).run_test()  # type: ignore[arg-type]


def test_graph_patch_host_runner_rejects_non_pinned_revision(tmp_path: Path) -> None:
    class Context:
        repository = tmp_path
        env = {"RUNNER_TEMP": str(tmp_path)}

    with pytest.raises(CiError, match="full Git object ID"):
        GraphPatchRealTrtRunner(  # type: ignore[arg-type]
            Context(),
            revision="feature/graph-patch",
        )

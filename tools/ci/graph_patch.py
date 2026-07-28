# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the graph-patch smoke test on one leased GPU and certify its JUnit.

Boundary: this module owns only the conditional real-TensorRT gate; model
selection, generic container image management, and model proofs stay elsewhere.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import signal
import xml.etree.ElementTree as ET
from pathlib import Path

from .context import CiContext
from .gpu_lease import GpuLease
from .process import CiError


_NVIDIA_SMI_TIMEOUT = "2m"
_TENSORRT_PREFLIGHT_TIMEOUT = "2m"
_PYTEST_TIMEOUT = "10m"
_MAX_GPU_LEASE_SECONDS = 600
_OWNER_LABEL_PREFIX = "com.nvidia.trtmc.graph-patch"


_PREFLIGHT = r"""
import json
import os
from pathlib import Path

import tensorrt as trt

try:
    from cuda.bindings import runtime as cudart
except ImportError:
    from cuda import cudart

status, count = cudart.cudaGetDeviceCount()
if int(status) != 0 or int(count) != 1:
    raise SystemExit(
        f"CUDA preflight failed: cudaGetDeviceCount returned status={status!r}, count={count!r}"
    )
logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
if builder is None:
    raise SystemExit("TensorRT preflight failed: trt.Builder returned None")
evidence = {
    "schema_version": 1,
    "source_revision": os.environ["TRTMC_GRAPH_PATCH_SOURCE_REVISION"],
    "tensorrt_version": trt.__version__,
    "cuda_status": int(status),
    "visible_device_count": int(count),
    "builder_created": True,
}
Path(os.environ["TRTMC_GRAPH_PATCH_PREFLIGHT"]).write_text(
    json.dumps(evidence, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(f"TensorRT={trt.__version__} CUDA_devices={int(count)}")
"""


class GraphPatchRealTrtGate:
    """Run and certify the single real-TensorRT graph-patch test in-container."""

    TEST_PATH = (
        "tests/builder/test_graph_patch_real_trt.py"
        "::test_multi_instance_rewire_compiles_real_tensorrt_network"
    )

    def __init__(self, context: CiContext):
        self.context = context
        configured = context.env.get(
            "TRTMC_GRAPH_PATCH_JUNIT",
            ".ci/graph-patch-real-trt/junit.xml",
        )
        self.junit = Path(configured)
        if not self.junit.is_absolute():
            self.junit = context.repository / self.junit
        configured_preflight = context.env.get(
            "TRTMC_GRAPH_PATCH_PREFLIGHT",
            str(self.junit.with_name("preflight.json")),
        )
        self.preflight_evidence = Path(configured_preflight)
        if not self.preflight_evidence.is_absolute():
            self.preflight_evidence = context.repository / self.preflight_evidence
        self.pytest_returncode: int | None = None

    def preflight(self) -> None:
        """Require a visible NVIDIA GPU, CUDA runtime, and TensorRT builder."""
        self.context.run(
            ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
            limit=self._bounded_timeout(
                "TRTMC_GRAPH_PATCH_NVIDIA_SMI_TIMEOUT",
                _NVIDIA_SMI_TIMEOUT,
                2 * 60,
            ),
        )
        self.preflight_evidence.parent.mkdir(parents=True, exist_ok=True)
        self.preflight_evidence.unlink(missing_ok=True)
        self.context.run(
            ["python3", "-c", _PREFLIGHT],
            limit=self._bounded_timeout(
                "TRTMC_GRAPH_PATCH_TRT_PREFLIGHT_TIMEOUT",
                _TENSORRT_PREFLIGHT_TIMEOUT,
                2 * 60,
            ),
            updates={
                "TRTMC_GRAPH_PATCH_PREFLIGHT": str(self.preflight_evidence),
            },
        )

    def run_test(self) -> None:
        """Run the exact smoke target and always leave certification to JUnit."""
        self.junit.parent.mkdir(parents=True, exist_ok=True)
        self.junit.unlink(missing_ok=True)
        python_path = f"{self.context.repository / 'python'}:{self.context.repository}"
        result = self.context.run(
            [
                "python3",
                "-m",
                "pytest",
                self.TEST_PATH,
                "-q",
                "-x",
                "-rA",
                "--strict-markers",
                "-p",
                "no:cacheprovider",
                "-o",
                "junit_family=xunit2",
                f"--junitxml={self.junit}",
            ],
            limit=self._bounded_timeout(
                "TRTMC_GRAPH_PATCH_TEST_TIMEOUT",
                _PYTEST_TIMEOUT,
                10 * 60,
            ),
            updates={"PYTHONPATH": python_path},
            check=False,
        )
        self.pytest_returncode = result.returncode

    def enforce(self) -> None:
        """Reject empty, failed, errored, skipped, or otherwise nonzero runs."""
        preflight = self.certify_preflight(self.preflight_evidence)
        summary = self.certify_junit(self.junit)
        if self.pytest_returncode is None:
            raise CiError("graph-patch pytest did not run")
        if self.pytest_returncode:
            raise CiError(
                "graph-patch real-TensorRT pytest exited with "
                f"{self.pytest_returncode} despite JUnit summary {summary}"
            )
        print(f"Certified graph-patch real-TensorRT gate: preflight={preflight} junit={summary}")

    @staticmethod
    def certify_preflight(
        path: Path,
        *,
        expected_revision: str | None = None,
    ) -> dict[str, object]:
        """Require explicit evidence for one visible GPU and a TensorRT builder."""
        if not path.is_file():
            raise CiError(f"graph-patch TensorRT preflight evidence is missing: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CiError(f"graph-patch TensorRT preflight evidence is invalid: {path}") from error
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise CiError("graph-patch TensorRT preflight has an unsupported schema")
        required = {
            "cuda_status": 0,
            "visible_device_count": 1,
            "builder_created": True,
        }
        mismatches = [
            f"{name}={payload.get(name)!r}"
            for name, expected in required.items()
            if payload.get(name) != expected
        ]
        revision = payload.get("source_revision")
        version = payload.get("tensorrt_version")
        if not isinstance(revision, str) or not re.fullmatch(
            r"(?:[a-f0-9]{40}|[a-f0-9]{64})", revision
        ):
            mismatches.append(f"source_revision={revision!r}")
        if expected_revision is not None and revision != expected_revision:
            mismatches.append(f"source_revision expected {expected_revision!r}, found {revision!r}")
        if not isinstance(version, str) or not version:
            mismatches.append(f"tensorrt_version={version!r}")
        if mismatches:
            raise CiError("graph-patch TensorRT preflight did not pass: " + ", ".join(mismatches))
        return payload

    @staticmethod
    def certify_junit(path: Path) -> dict[str, int]:
        """Return strict aggregate counts for one pytest xUnit2 document."""
        if not path.is_file():
            raise CiError(f"graph-patch real-TensorRT JUnit is missing: {path}")
        try:
            root = ET.parse(path).getroot()
        except (OSError, ET.ParseError) as error:
            raise CiError(f"graph-patch real-TensorRT JUnit is invalid: {path}") from error
        if root.tag == "testsuite":
            suites = (root,)
        elif root.tag == "testsuites":
            suites = tuple(child for child in root if child.tag == "testsuite")
        else:
            raise CiError(f"graph-patch JUnit has unsupported root element: {root.tag}")
        if not suites:
            raise CiError("graph-patch JUnit contains no test suites")

        summary = {
            name: sum(GraphPatchRealTrtGate._count(suite, name) for suite in suites)
            for name in ("tests", "failures", "errors", "skipped")
        }
        cases = sum(len(suite.findall(".//testcase")) for suite in suites)
        if summary["tests"] < 1 or cases < 1:
            raise CiError("graph-patch real-TensorRT gate collected no tests")
        if cases != summary["tests"]:
            raise CiError(
                "graph-patch JUnit test count is inconsistent: "
                f"tests={summary['tests']} testcases={cases}"
            )
        rejected = {
            name: value for name, value in summary.items() if name != "tests" and value != 0
        }
        if rejected:
            detail = ", ".join(f"{name}={value}" for name, value in rejected.items())
            raise CiError(f"graph-patch real-TensorRT gate did not fully pass: {detail}")
        if root.findall(".//failure") or root.findall(".//error") or root.findall(".//skipped"):
            raise CiError("graph-patch JUnit contains a non-passing testcase element")
        return summary

    @staticmethod
    def _count(suite: ET.Element, name: str) -> int:
        value = suite.get(name, "0")
        if not value.isdigit():
            raise CiError(f"graph-patch JUnit has invalid {name} count: {value!r}")
        return int(value)

    def _bounded_timeout(self, name: str, default: str, maximum_seconds: int) -> str:
        value = self.context.env.get(name, default)
        match = re.fullmatch(r"([1-9][0-9]*)([smh])", value)
        if not match:
            raise CiError(f"{name} must be a positive timeout with an s, m, or h suffix")
        multiplier = {"s": 1, "m": 60, "h": 3600}[match.group(2)]
        seconds = int(match.group(1)) * multiplier
        if seconds > maximum_seconds:
            raise CiError(f"{name} must not exceed {maximum_seconds} seconds")
        return value


class GraphPatchRealTrtRunner:
    """Lease one shared GPU and run the gate in the immutable CI image."""

    def __init__(
        self,
        context: CiContext,
        output_dir: Path | None = None,
        revision: str = "HEAD",
        *,
        owner_token: str | None = None,
    ):
        self.context = context
        self.output_dir = output_dir
        if revision != "HEAD" and not re.fullmatch(
            r"(?:[a-f0-9]{40}|[a-f0-9]{64})",
            revision,
        ):
            raise CiError("graph-patch revision must be HEAD or a full Git object ID")
        self.revision = revision
        self.owner_token = owner_token or secrets.token_hex(16)
        if not re.fullmatch(r"[a-f0-9]{32}", self.owner_token):
            raise CiError("graph-patch owner token must be 32 lowercase hexadecimal characters")
        self.lease: GpuLease | None = None
        self.owner_labels = self._build_ownership_labels()
        self.container_name = self._build_container_name()
        self.container_launch_attempted = False

    def run_host(self) -> None:
        previous = {
            number: signal.signal(number, self._signal)
            for number in (signal.SIGINT, signal.SIGTERM)
        }
        try:
            try:
                self._run_host()
            finally:
                self._cleanup()
        finally:
            for number, handler in previous.items():
                signal.signal(number, handler)

    def _run_host(self) -> None:
        for executable in ("docker", "git", "nvidia-smi"):
            self.context.executable(executable)
        revision = self.context.output(["git", "rev-parse", f"{self.revision}^{{commit}}"])
        checked_out = self.context.output(["git", "rev-parse", "HEAD^{commit}"])
        if checked_out != revision:
            raise CiError(
                "graph-patch checkout does not match the pinned tested revision: "
                f"expected {revision}, found {checked_out}"
            )
        dirty = self.context.output(["git", "status", "--porcelain=v1", "--untracked-files=all"])
        if dirty:
            raise CiError("graph-patch checkout contains changes outside the pinned revision")
        artifacts = self._prepare_output()
        (artifacts / "source-revision.txt").write_text(
            revision + "\n",
            encoding="utf-8",
        )
        image = self.context.env.get("TRTMC_CI_IMAGE", "")
        if not image:
            raise CiError("TRTMC_CI_IMAGE is not set")
        try:
            image_id = self.context.output(
                ["docker", "image", "inspect", "--format", "{{.Id}}", image]
            )
        except CiError as error:
            raise CiError(f"CI image is not present: {image}") from error
        if not re.fullmatch(r"sha256:[a-f0-9]{64}", image_id):
            raise CiError(f"CI image has an invalid immutable image ID: {image_id!r}")

        lease_timeout = self.context.env.get(
            "TRTMC_MODEL_PROOF_GPU_LEASE_TIMEOUT_SECONDS",
            str(_MAX_GPU_LEASE_SECONDS),
        )
        if not lease_timeout.isdigit() or not 1 <= int(lease_timeout) <= _MAX_GPU_LEASE_SECONDS:
            raise CiError(
                "TRTMC_MODEL_PROOF_GPU_LEASE_TIMEOUT_SECONDS must be an integer "
                f"from 1 to {_MAX_GPU_LEASE_SECONDS} for the graph-patch gate"
            )
        self.context.env["TRTMC_MODEL_PROOF_GPU_LEASE_TIMEOUT_SECONDS"] = lease_timeout

        self.lease = GpuLease(
            self.context,
            "graph-patch-real-trt",
            "shared",
            artifacts,
        )
        self.lease.acquire()
        self._reclaim_orphans()
        evidence = self.lease.evidence(revision)
        (artifacts / "gpu-lease.json").write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        runner_evidence = {
            "schema_version": 1,
            "source_revision": revision,
            "requested_image": image,
            "immutable_image_id": image_id,
            "gpu_lock_namespace": self.lease.lock_namespace,
            "container_name": self.container_name,
            "owner_labels": self._ownership_labels(),
        }
        (artifacts / "runner-evidence.json").write_text(
            json.dumps(runner_evidence, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        self._remove_owned_container(reason="before launch")
        self.container_launch_attempted = True
        result = self.context.run(
            self._container_command(image_id, artifacts, revision),
            check=False,
        )
        junit = artifacts / "junit.xml"
        try:
            preflight = GraphPatchRealTrtGate.certify_preflight(
                artifacts / "preflight.json",
                expected_revision=revision,
            )
            summary = GraphPatchRealTrtGate.certify_junit(junit)
        except CiError as error:
            if result.returncode:
                raise CiError(
                    f"graph-patch real-TensorRT container failed with {result.returncode}; {error}"
                ) from error
            raise
        if result.returncode:
            raise CiError(
                "graph-patch real-TensorRT container failed with "
                f"{result.returncode} despite preflight {preflight} and JUnit {summary}"
            )
        print(f"Graph-patch real-TensorRT artifacts: {artifacts}")

    def _prepare_output(self) -> Path:
        scratch_root = Path(self.context.env.get("RUNNER_TEMP", "/tmp")).resolve()
        configured = self.output_dir or (
            scratch_root
            / (
                "trtmc-graph-patch-real-trt-"
                f"{self.context.env.get('GITHUB_RUN_ID', 'local')}-"
                f"{self.context.env.get('GITHUB_RUN_ATTEMPT', '0')}-"
                f"{self.owner_token[:12]}"
            )
        )
        if configured.is_symlink():
            raise CiError(f"graph-patch output must not be a symlink: {configured}")
        output = configured.resolve()
        if output in {
            Path("/"),
            scratch_root,
            self.context.repository,
        } or not output.is_relative_to(scratch_root):
            raise CiError("unsafe graph-patch real-TensorRT output directory")
        if output.exists():
            raise CiError(f"graph-patch output already exists: {output}")
        output.mkdir(parents=True)
        return output

    def _container_command(
        self,
        image: str,
        artifacts: Path,
        revision: str,
    ) -> list[str]:
        assert self.lease and self.lease.gpu_id is not None
        slots = ",".join(map(str, self.lease.slot_ids))
        labels: list[str] = []
        all_labels = {
            "com.nvidia.trtmc.model-proof": "1",
            "com.nvidia.trtmc.graph-patch-real-trt": "1",
            "com.nvidia.trtmc.model-proof.gpu": str(self.lease.gpu_id),
            "com.nvidia.trtmc.model-proof.slots": slots,
            "com.nvidia.trtmc.model-proof.lock-namespace": self.lease.lock_namespace,
            **self._ownership_labels(),
        }
        for name, value in all_labels.items():
            labels.extend(("--label", f"{name}={value}"))
        timeouts = {
            "TRTMC_GRAPH_PATCH_NVIDIA_SMI_TIMEOUT": self.context.env.get(
                "TRTMC_GRAPH_PATCH_NVIDIA_SMI_TIMEOUT",
                _NVIDIA_SMI_TIMEOUT,
            ),
            "TRTMC_GRAPH_PATCH_TRT_PREFLIGHT_TIMEOUT": self.context.env.get(
                "TRTMC_GRAPH_PATCH_TRT_PREFLIGHT_TIMEOUT",
                _TENSORRT_PREFLIGHT_TIMEOUT,
            ),
            "TRTMC_GRAPH_PATCH_TEST_TIMEOUT": self.context.env.get(
                "TRTMC_GRAPH_PATCH_TEST_TIMEOUT",
                _PYTEST_TIMEOUT,
            ),
        }
        timeout_environment: list[str] = []
        for name, value in timeouts.items():
            timeout_environment.extend(("-e", f"{name}={value}"))
        return [
            "docker",
            "run",
            "--rm",
            "--name",
            self.container_name,
            "--read-only",
            "--network",
            "none",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--ipc",
            "private",
            "--gpus",
            f"device={self.lease.gpu_id}",
            *labels,
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--mount",
            f"type=bind,src={self.context.repository},dst=/src,readonly",
            "--mount",
            f"type=bind,src={artifacts},dst=/artifacts",
            "--workdir",
            "/src",
            "--tmpfs",
            "/tmp:rw,exec,nosuid,nodev,size=2g",
            "-e",
            "HOME=/tmp",
            "-e",
            "USER=trtmc-ci",
            "-e",
            "LOGNAME=trtmc-ci",
            "-e",
            "PYTHONHASHSEED=0",
            "-e",
            "PYTHONPATH=/src/python:/src",
            "-e",
            "TRTMC_GRAPH_PATCH_JUNIT=/artifacts/junit.xml",
            "-e",
            "TRTMC_GRAPH_PATCH_PREFLIGHT=/artifacts/preflight.json",
            "-e",
            f"TRTMC_GRAPH_PATCH_SOURCE_REVISION={revision}",
            *timeout_environment,
            image,
            "python3",
            "-m",
            "tools.ci",
            "pipeline",
            "graph-patch-real-trt",
        ]

    def _reclaim_orphans(self) -> None:
        """Remove only stale GPU-proof containers covered by the held flock."""
        assert self.lease and self.lease.gpu_id is not None
        rows = self.context.output(
            [
                "docker",
                "ps",
                "--no-trunc",
                "--filter",
                "label=com.nvidia.trtmc.model-proof=1",
                "--filter",
                f"label=com.nvidia.trtmc.model-proof.gpu={self.lease.gpu_id}",
                "--filter",
                f"label=com.nvidia.trtmc.model-proof.lock-namespace={self.lease.lock_namespace}",
                "--format",
                "{{.ID}}",
            ]
        ).splitlines()
        candidates: list[str] = []
        for row in rows:
            container_id = row.strip()
            if row != container_id or not re.fullmatch(r"[a-f0-9]{64}", container_id):
                raise CiError(f"graph-patch orphan inventory has an unsafe ID: {row!r}")
            if container_id in candidates:
                raise CiError(f"graph-patch orphan inventory has a duplicate ID: {container_id}")
            candidates.append(container_id)

        # Validate the complete inventory before deleting anything.  Acquiring
        # an overlapping flock slot proves a matching leftover container has
        # lost its host owner; non-overlapping runs remain untouched.
        reclaimable: list[str] = []
        for container_id in candidates:
            inspected = self._inspect_container(container_id, allow_absent=False)
            assert inspected is not None
            inspected_id, labels = inspected
            if inspected_id != container_id:
                raise CiError(
                    "graph-patch orphan inspect returned a different immutable ID: "
                    f"expected {container_id}, found {inspected_id}"
                )
            if not self._orphan_matches_lease(labels):
                continue
            reclaimable.append(container_id)

        for container_id in reclaimable:
            removed = self.context.run(
                ["docker", "rm", "-f", container_id],
                check=False,
                capture_output=True,
            )
            if removed.returncode:
                detail = (removed.stderr or removed.stdout or "").strip()
                raise CiError(
                    f"could not remove orphaned graph-patch container {container_id}"
                    + (f": {detail}" if detail else "")
                )

    def _orphan_matches_lease(self, labels: dict[str, str]) -> bool:
        assert self.lease and self.lease.gpu_id is not None
        required = {
            "com.nvidia.trtmc.model-proof": "1",
            "com.nvidia.trtmc.model-proof.gpu": str(self.lease.gpu_id),
            "com.nvidia.trtmc.model-proof.lock-namespace": self.lease.lock_namespace,
        }
        if any(labels.get(name) != value for name, value in required.items()):
            return False
        slots = labels.get("com.nvidia.trtmc.model-proof.slots", "")
        if not re.fullmatch(r"(?:0|[1-9][0-9]*)(?:,(?:0|[1-9][0-9]*))*", slots):
            raise CiError("graph-patch orphan container has invalid GPU slot labels")
        slot_ids = list(map(int, slots.split(",")))
        if len(slot_ids) != len(set(slot_ids)) or any(
            slot < 0 or slot >= self.lease.slots_per_gpu for slot in slot_ids
        ):
            raise CiError("graph-patch orphan container has invalid GPU slot labels")
        graph_marker = labels.get("com.nvidia.trtmc.graph-patch-real-trt")
        if graph_marker not in {None, "1"}:
            raise CiError("graph-patch orphan container has an invalid graph marker label")
        if graph_marker == "1":
            owner_token = labels.get(f"{_OWNER_LABEL_PREFIX}.owner-token", "")
            if not re.fullmatch(r"[a-f0-9]{32}", owner_token):
                raise CiError("graph-patch orphan container has an invalid owner token label")
        return bool(set(slot_ids).intersection(self.lease.slot_ids))

    def _cleanup(self) -> None:
        cleanup_error: CiError | None = None
        try:
            if self.container_launch_attempted:
                self._remove_owned_container(reason="during cleanup")
        except CiError as error:
            cleanup_error = error
        finally:
            if self.lease:
                self.lease.release()
        if cleanup_error:
            raise cleanup_error

    def _ownership_labels(self) -> dict[str, str]:
        return dict(self.owner_labels)

    def _build_ownership_labels(self) -> dict[str, str]:
        return {
            f"{_OWNER_LABEL_PREFIX}.owner-token": self.owner_token,
            f"{_OWNER_LABEL_PREFIX}.run-id": (
                self.context.env.get("GITHUB_RUN_ID", "local") or "local"
            ),
            f"{_OWNER_LABEL_PREFIX}.run-attempt": (
                self.context.env.get("GITHUB_RUN_ATTEMPT", "0") or "0"
            ),
            f"{_OWNER_LABEL_PREFIX}.repository": (
                self.context.env.get("GITHUB_REPOSITORY", "local") or "local"
            ),
        }

    def _inspect_container(
        self,
        target: str,
        *,
        allow_absent: bool = True,
    ) -> tuple[str, dict[str, str]] | None:
        result = self.context.run(
            [
                "docker",
                "container",
                "inspect",
                "--format",
                "{{json .}}",
                target,
            ],
            check=False,
            capture_output=True,
        )
        if result.returncode:
            detail = (result.stderr or result.stdout or "").strip()
            if allow_absent and re.search(
                r"No such (?:object|container)",
                detail,
                flags=re.IGNORECASE,
            ):
                return None
            raise CiError(
                f"could not inspect graph-patch container ownership for {target}"
                + (f": {detail}" if detail else "")
            )
        try:
            payload = json.loads(result.stdout)
        except (TypeError, json.JSONDecodeError) as error:
            raise CiError("graph-patch container ownership metadata is invalid") from error
        if not isinstance(payload, dict):
            raise CiError("graph-patch container ownership metadata is invalid")
        container_id = payload.get("Id")
        config = payload.get("Config")
        labels = config.get("Labels") if isinstance(config, dict) else None
        if (
            not isinstance(container_id, str)
            or not re.fullmatch(r"[a-f0-9]{64}", container_id)
            or not isinstance(labels, dict)
            or not all(
                isinstance(name, str) and isinstance(value, str) for name, value in labels.items()
            )
        ):
            raise CiError("graph-patch container ownership metadata is invalid")
        return container_id, labels

    def _remove_owned_container(self, *, reason: str) -> bool:
        inspected = self._inspect_container(self.container_name)
        if inspected is None:
            return False
        container_id, labels = inspected
        expected = self._ownership_labels()
        mismatches = [
            f"{name} expected {value!r}, found {labels.get(name)!r}"
            for name, value in expected.items()
            if labels.get(name) != value
        ]
        if mismatches:
            raise CiError(
                f"refusing to remove foreign graph-patch container {self.container_name} "
                f"{reason}: " + ", ".join(mismatches)
            )
        removed = self.context.run(
            ["docker", "rm", "-f", container_id],
            check=False,
            capture_output=True,
        )
        if removed.returncode:
            detail = (removed.stderr or removed.stdout or "").strip()
            raise CiError(
                f"could not remove owned graph-patch container {self.container_name}"
                + (f": {detail}" if detail else "")
            )
        return True

    def _signal(self, number: int, _frame: object) -> None:
        raise SystemExit(130 if number == signal.SIGINT else 143)

    def _build_container_name(self) -> str:
        run_id = self.owner_labels[f"{_OWNER_LABEL_PREFIX}.run-id"]
        attempt = self.owner_labels[f"{_OWNER_LABEL_PREFIX}.run-attempt"]
        name = (f"trtmc-graph-patch-real-trt-{run_id}-{attempt}-{self.owner_token[:12]}").replace(
            "_", "-"
        )
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            raise CiError("unsafe graph-patch real-TensorRT container name")
        return name

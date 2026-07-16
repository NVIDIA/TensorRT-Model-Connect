# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host and container halves of the hermetic single-model certification."""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .context import CiContext
from .gpu_lease import GpuLease
from .model_proof_selection import ModelProofSelection, ModelProofSelector
from .process import CiError
from .task_eval import TaskEvalDatasetPreparer


CACHE_COPY_PROGRAM = r"""
import os
import shutil
import subprocess
import sys

source, destination, uid, gid = sys.argv[1:]
try:
    # The destination is a bind mount created by the unprivileged runner.
    # Own its root while cp preserves directory metadata, then return the
    # complete private cache view to the runner below.
    os.chown(destination, 0, 0)
    subprocess.run(
        ["cp", "-a", "--reflink=always", "--no-preserve=ownership", "--", source + "/.", destination + "/"],
        check=True,
    )
    subprocess.run(["chmod", "-R", "u+rwX", "--", destination], check=True)
    subprocess.run(["chown", "-hR", "--", f"{uid}:{gid}", destination], check=True)
except BaseException:
    subprocess.run(["chmod", "-R", "u+rwX", "--", destination], check=False)
    subprocess.run(["chown", "-hR", "--", f"{uid}:{gid}", destination], check=False)
    raise
"""


@dataclass(frozen=True)
class ModelProofRequest:
    """Validated command-line request for one projected model."""

    model: str
    suite: str = "premerge"
    revision: str = "HEAD"
    output_dir: Path | None = None

    def validate(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", self.model):
            raise CiError(f"unsafe model id: {self.model}")
        if self.suite not in {"premerge", "nightly"}:
            raise CiError("--suite must be premerge or nightly")


class ModelReferenceCache:
    """Copy only a declared pinned reference checkout into the proof-private view."""

    def __init__(self, context: CiContext, request: ModelProofRequest):
        self.context = context
        self.request = request

    def prepare(
        self,
        contract: dict[str, str] | None,
        work_dir: Path,
        artifacts_dir: Path,
    ) -> str | None:
        if not contract:
            return None
        configured = self.context.env.get("TRTMC_MODEL_REFERENCE_CACHE_ROOT", "")
        if not configured:
            raise CiError(f"TRTMC_MODEL_REFERENCE_CACHE_ROOT is required for {self.request.model}")
        root = Path(configured).resolve(strict=True)
        if not root.is_dir() or root == Path("/") or root == self.context.repository:
            raise CiError("model reference cache root is invalid")
        raw_source = root / contract["relative_path"]
        if raw_source.is_symlink():
            raise CiError("selected model reference cache must not be a symlink")
        try:
            source = raw_source.resolve(strict=True)
        except OSError as error:
            raise CiError(
                f"selected model reference cache is unavailable: {contract['relative_path']}"
            ) from error
        if not source.is_dir() or not source.is_relative_to(root):
            raise CiError(
                f"selected model reference cache is unavailable: {contract['relative_path']}"
            )
        revision = self.context.output(["git", "-C", source, "rev-parse", "HEAD^{commit}"])
        if revision != contract["revision"]:
            raise CiError(
                f"selected model reference cache revision mismatch: expected {contract['revision']}, "
                f"found {revision}"
            )
        repository = self.context.output(
            ["git", "-C", source, "config", "--get", "remote.origin.url"]
        )
        if repository != contract["repository"]:
            raise CiError(
                f"selected model reference cache repository mismatch: expected "
                f"{contract['repository']}, found {repository}"
            )
        exists = self.context.run(
            ["git", "-C", source, "cat-file", "-e", f"{revision}:{contract['entrypoint']}"],
            check=False,
            capture_output=True,
        )
        if exists.returncode:
            raise CiError(f"pinned model reference entrypoint is absent from commit {revision}")
        tree = self.context.output(["git", "-C", source, "rev-parse", f"{revision}^{{tree}}"])
        private_root = work_dir / "reference-private"
        destination = private_root / contract["relative_path"]
        if private_root.exists():
            raise CiError("proof-private model reference destination already exists")
        destination.mkdir(parents=True)
        archive = subprocess.Popen(
            ["git", "-C", str(source), "archive", "--format=tar", revision],
            stdout=subprocess.PIPE,
            env=self.context.env,
        )
        extract = subprocess.run(
            ["tar", "--no-same-owner", "--no-same-permissions", "-xf", "-", "-C", destination],
            stdin=archive.stdout,
            env=self.context.env,
            check=False,
        )
        if archive.stdout:
            archive.stdout.close()
        archive_rc = archive.wait()
        if archive_rc or extract.returncode:
            raise CiError("pinned model reference cache could not be copied privately")
        entrypoint = destination / contract["entrypoint"]
        if not entrypoint.is_file() or entrypoint.is_symlink():
            raise CiError("private model reference copy is missing its entrypoint")
        for path in destination.rglob("*"):
            if path.is_symlink() and not path.resolve(strict=True).is_relative_to(
                destination.resolve()
            ):
                raise CiError(f"private model reference has an escaping symlink: {path}")
        evidence = {
            "schema_version": 1,
            "model": self.request.model,
            "isolation": "selected-pinned-private",
            "repository": repository,
            "reference_revision": revision,
            "reference_tree": tree,
            "relative_path": contract["relative_path"],
            "entrypoint": contract["entrypoint"],
            "container_storage_root": "/work/reference-private",
            "copy_method": "git-archive",
        }
        (artifacts_dir / "model-reference-cache.json").write_text(
            json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
        )
        return revision


class ModelProofContainerCleaner:
    """Remove only containers carrying the exact workflow job identity labels."""

    def __init__(self, context: CiContext, model: str):
        self.context = context
        self.model = model

    def cleanup(self) -> None:
        run_id = self.context.env.get("GITHUB_RUN_ID", "")
        attempt = self.context.env.get("GITHUB_RUN_ATTEMPT", "")
        if not run_id.isdigit() or int(run_id) < 1:
            raise CiError("GITHUB_RUN_ID must be a positive integer for container cleanup")
        if not attempt.isdigit() or int(attempt) < 1:
            raise CiError("GITHUB_RUN_ATTEMPT must be a positive integer for container cleanup")
        filters = [
            "--filter",
            "label=com.nvidia.trtmc.model-proof.job=1",
            "--filter",
            f"label=com.nvidia.trtmc.model-proof.run-id={run_id}",
            "--filter",
            f"label=com.nvidia.trtmc.model-proof.run-attempt={attempt}",
            "--filter",
            f"label=com.nvidia.trtmc.model-proof.model={self.model}",
        ]
        inventory = [
            "docker",
            "ps",
            "-a",
            "--no-trunc",
            *filters,
            "--format",
            '{{.ID}} {{.Label "com.nvidia.trtmc.model-proof.run-id"}} '
            '{{.Label "com.nvidia.trtmc.model-proof.run-attempt"}} '
            '{{.Label "com.nvidia.trtmc.model-proof.model"}}',
        ]
        for _retry in range(3):
            rows = self.context.output(inventory)
            if not rows:
                return
            ids = []
            for row in rows.splitlines():
                container, labeled_run, labeled_attempt, labeled_model = row.split()
                if not re.fullmatch(r"[a-f0-9]{64}", container):
                    raise CiError(f"model-proof cleanup found an unsafe container ID: {container}")
                if (labeled_run, labeled_attempt, labeled_model) != (run_id, attempt, self.model):
                    raise CiError(
                        "model-proof cleanup found a container with mismatched identity labels"
                    )
                ids.append(container)
            for container in ids:
                self.context.run(["docker", "rm", "-f", container])
        if self.context.output(inventory):
            raise CiError("model-proof containers remain after three cleanup attempts")


class ModelProofRunner:
    """Create a positive source projection, lease a GPU, and run one hermetic proof."""

    def __init__(self, context: CiContext, request: ModelProofRequest):
        request.validate()
        self.context = context
        self.request = request
        self.lease: GpuLease | None = None
        self.container_name = ""
        self.artifacts_dir: Path | None = None
        self.revision = request.revision

    def run_host(self) -> None:
        previous = {
            number: signal.signal(number, self._signal)
            for number in (signal.SIGINT, signal.SIGTERM)
        }
        try:
            self._run_host()
        except BaseException as error:
            self._record_host_error(error)
            self._fallback_report()
            raise
        finally:
            self._cleanup()
            for number, handler in previous.items():
                signal.signal(number, handler)

    def _run_host(self) -> None:
        for executable in ("docker", "git", "tar"):
            self.context.executable(executable)
        self.revision = self.context.output(
            ["git", "rev-parse", f"{self.request.revision}^{{commit}}"]
        )
        output = (
            self.request.output_dir
            or Path(self.context.env.get("RUNNER_TEMP", "/tmp"))
            / f"trtmc-model-proof-{self.request.model}"
        ).resolve()
        if output in {Path("/"), self.context.repository}:
            raise CiError("unsafe model-proof output directory")
        projection = output / "projection"
        self.artifacts_dir = output / "artifacts"
        work = output / "work"
        output.mkdir(parents=True, exist_ok=True)
        for path in (self.artifacts_dir, work):
            if path.exists():
                shutil.rmtree(path)
            path.mkdir(parents=True)
        self._project(projection)
        selection = ModelProofSelector(
            self.request.model, self.request.suite, self.revision, projection
        ).select(self.artifacts_dir / "selection.json")
        ModelReferenceCache(self.context, self.request).prepare(
            selection.reference_cache, work, self.artifacts_dir
        )
        expected = self.context.env.get("TRTMC_MODEL_PROOF_EXPECTED_RESOURCE_CLASS", "")
        if expected and expected not in {"shared", "exclusive_gpu"}:
            raise CiError(
                "TRTMC_MODEL_PROOF_EXPECTED_RESOURCE_CLASS must be shared or exclusive_gpu"
            )
        if expected and expected != selection.resource_class:
            raise CiError(
                f"expected resource class {expected} does not match selected E2E resource class "
                f"{selection.resource_class}"
            )
        models_file = self.artifacts_dir / "cache-check-models.txt"
        models_file.write_text(
            "".join(f"{model}\n" for model in selection.e2e_models), encoding="utf-8"
        )
        image = self.context.env.get("TRTMC_CI_IMAGE", "trtmc-dev-gb300:manylinux_2_39")
        if self.context.run(
            ["docker", "image", "inspect", image], check=False, capture_output=True
        ).returncode:
            raise CiError(f"CI image is not present: {image}")
        runtime_model = str(selection.payload["owners"]["runtime"])
        task_eval_container = self._base_container_name() + "-task-eval-data"
        self.container_name = task_eval_container
        task_eval_dir = TaskEvalDatasetPreparer(
            self.context,
            self.request.suite,
            runtime_model,
            projection,
            work,
            self.artifacts_dir,
            image,
            task_eval_container,
            self._job_labels(),
        ).prepare()
        private_hub = self._prepare_hf_cache(projection, work, image, models_file)
        (self.artifacts_dir / "gpu-lease-requested.txt").write_text(
            selection.resource_class + "\n", encoding="utf-8"
        )
        self.lease = GpuLease(
            self.context, self.request.model, selection.resource_class, self.artifacts_dir
        ).acquire()
        self._reclaim_orphans()
        lease_evidence = self.lease.evidence(self.revision)
        (self.artifacts_dir / "gpu-id.txt").write_text(
            str(lease_evidence["gpu_id"]) + "\n", encoding="utf-8"
        )
        (self.artifacts_dir / "gpu-lease.json").write_text(
            json.dumps(lease_evidence, indent=2) + "\n", encoding="utf-8"
        )
        self._run_proof_container(projection, work, private_hub, image, selection, task_eval_dir)
        for name in ("proof.json", "model-proof-report.html"):
            if not (self.artifacts_dir / name).is_file():
                raise CiError(f"model proof did not emit {name}")
        print(f"Model proof artifacts: {self.artifacts_dir}")

    def _project(self, projection: Path) -> None:
        assert self.artifacts_dir is not None
        command = [
            "python3",
            self.context.repository / "tools/model_ci.py",
            "project",
            "--model",
            self.request.model,
            "--revision",
            self.revision,
            "--output-dir",
            projection,
            "--clean",
        ]
        with (
            (self.artifacts_dir / "projection.json").open("w", encoding="utf-8") as stdout,
            (self.artifacts_dir / "projection.stderr.log").open("w", encoding="utf-8") as stderr,
        ):
            result = subprocess.run(
                [str(item) for item in command],
                cwd=self.context.repository,
                env=self.context.env,
                text=True,
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
        if result.returncode:
            print(
                (self.artifacts_dir / "projection.stderr.log").read_text(encoding="utf-8"),
                file=sys.stderr,
            )
            raise CiError(f"model source projection failed with code {result.returncode}")
        if not (projection / ".trtmc-model-projection.json").is_file():
            raise CiError("model_ci.py did not produce a projection manifest")

    def _job_labels(self) -> list[str]:
        return [
            "--label",
            "com.nvidia.trtmc.model-proof.job=1",
            "--label",
            f"com.nvidia.trtmc.model-proof.run-id={self.context.env.get('GITHUB_RUN_ID', 'local')}",
            "--label",
            f"com.nvidia.trtmc.model-proof.run-attempt={self.context.env.get('GITHUB_RUN_ATTEMPT', '0')}",
            "--label",
            f"com.nvidia.trtmc.model-proof.model={self.request.model}",
        ]

    def _base_container_name(self) -> str:
        value = (
            f"trtmc-model-proof-{self.context.env.get('GITHUB_RUN_ID', 'local')}-"
            f"{self.context.env.get('GITHUB_RUN_ATTEMPT', '0')}-{self.request.model}"
        )
        return value.replace("_", "-")

    def _prepare_hf_cache(
        self, projection: Path, work: Path, image: str, models_file: Path
    ) -> Path:
        assert self.artifacts_dir is not None
        root = self.context.env.get(
            "TRTMC_HF_CACHE",
            self.context.env.get("HF_HOME", str(Path.home() / ".cache/huggingface")),
        )
        hub = Path(self.context.env.get("TRTMC_HF_HUB_CACHE", str(Path(root) / "hub"))).resolve()
        if hub in {Path("/"), self.context.repository}:
            raise CiError("unsafe HF Hub cache path")
        name = self._base_container_name() + "-cache-check"
        self.container_name = name
        self.context.run(["docker", "rm", "-f", name], check=False, capture_output=True)
        command = [
            "docker",
            "run",
            "--rm",
            "--name",
            name,
            *self._job_labels(),
            "--read-only",
            "--network",
            "none",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--mount",
            f"type=bind,src={projection},dst=/src,readonly",
            "--mount",
            f"type=bind,src={self.artifacts_dir},dst=/artifacts",
            "--mount",
            f"type=bind,src={hub},dst=/hf-cache/hub,readonly",
            "--tmpfs",
            "/tmp:rw,exec,nosuid,nodev,size=1g",
            "--workdir",
            "/src",
            "-e",
            "HOME=/tmp",
            "-e",
            "HF_HOME=/tmp/hf-home",
            "-e",
            "HF_HUB_CACHE=/hf-cache/hub",
            "-e",
            "HUGGINGFACE_HUB_CACHE=/hf-cache/hub",
            "-e",
            "HF_MODULES_CACHE=/tmp/hf-modules",
            "-e",
            "PYTHONDONTWRITEBYTECODE=1",
            image,
            "/opt/venv/bin/python",
            "/src/scripts/warm_hf_cache.py",
            "--models-file",
            "/artifacts/cache-check-models.txt",
            "--local-only",
            "--strict",
            "--emit-cache-repos",
            "/artifacts/hf-cache-repos.json",
        ]
        result = self._run_logged(command, self.artifacts_dir / "cache-check.log")
        if result:
            raise CiError(
                f"offline HF cache readiness check failed for {self.request.model} (exit {result})"
            )
        try:
            evidence = self._validated_cache_evidence(hub)
        except (CiError, OSError, json.JSONDecodeError) as error:
            raise CiError(
                "selected Hugging Face cache evidence failed closed validation"
            ) from error
        private_hub = work / "hf-private/hub"
        if private_hub.parent.exists():
            shutil.rmtree(private_hub.parent)
        private_hub.mkdir(parents=True)
        for index, (source, folder) in enumerate(evidence):
            destination = private_hub / folder
            destination.mkdir(mode=0o700)
            copy_name = f"{self._base_container_name()}-hf-cache-copy-{index}"
            self.container_name = copy_name
            self.context.run(["docker", "rm", "-f", copy_name], check=False, capture_output=True)
            copy = [
                "docker",
                "run",
                "--rm",
                "--name",
                copy_name,
                *self._job_labels(),
                "--read-only",
                "--network",
                "none",
                "--cap-drop",
                "ALL",
                "--cap-add",
                "DAC_OVERRIDE",
                "--cap-add",
                "CHOWN",
                "--security-opt",
                "no-new-privileges",
                "--pids-limit",
                "32",
                "--user",
                "0:0",
                "--mount",
                f"type=bind,src={source},dst=/selected-hf-repo,readonly",
                "--mount",
                f"type=bind,src={destination},dst=/private-hf-repo",
                "--entrypoint",
                "/usr/bin/python3",
                image,
                "-c",
                CACHE_COPY_PROGRAM,
                "/selected-hf-repo",
                "/private-hf-repo",
                str(os.getuid()),
                str(os.getgid()),
            ]
            if self.context.run(copy, check=False).returncode:
                raise CiError(
                    f"selected Hugging Face cache repository could not be reflinked: {folder}"
                )
            if not destination.is_dir() or destination.is_symlink():
                raise CiError("selected Hugging Face cache reflink produced an invalid repository")
            if destination.stat().st_uid != os.getuid() or destination.stat().st_gid != os.getgid():
                raise CiError(
                    "selected Hugging Face cache reflink did not return ownership to the runner"
                )
            for path in [destination, *destination.rglob("*")]:
                try:
                    path.chmod(
                        path.stat().st_mode | 0o700
                        if path.is_dir()
                        else path.stat().st_mode | 0o600
                    )
                except OSError:
                    pass
        return private_hub

    def _validated_cache_evidence(self, hub: Path) -> list[tuple[Path, str]]:
        assert self.artifacts_dir is not None
        payload = json.loads(
            (self.artifacts_dir / "hf-cache-repos.json").read_text(encoding="utf-8")
        )
        if payload.get("schema_version") != 1 or payload.get("hub_cache") != "/hf-cache/hub":
            raise CiError("selected Hugging Face cache evidence has an unsupported schema")
        repositories = payload.get("repositories")
        if not isinstance(repositories, list) or not repositories:
            raise CiError("selected HF cache evidence contains no repositories")
        result = []
        seen = set()
        resolved_hub = hub.resolve(strict=True)
        for entry in repositories:
            if not isinstance(entry, dict):
                raise CiError("selected HF cache repository entry must be an object")
            repo_id = entry.get("repo_id")
            if (
                not isinstance(repo_id, str)
                or not repo_id
                or repo_id.startswith("/")
                or any(part in {"", ".", ".."} for part in repo_id.split("/"))
            ):
                raise CiError(f"selected HF cache evidence has an unsafe repo ID: {repo_id!r}")
            folder = "models--" + repo_id.replace("/", "--")
            if entry.get("cache_folder") != folder or entry.get("repo_type") != "model":
                raise CiError(f"selected HF cache evidence is noncanonical for {repo_id!r}")
            if entry.get("cache_path") != f"/hf-cache/hub/{folder}" or folder in seen:
                raise CiError(f"selected HF cache evidence has an invalid path for {repo_id!r}")
            source = resolved_hub / folder
            if (
                source.is_symlink()
                or not source.is_dir()
                or not source.resolve().is_relative_to(resolved_hub)
            ):
                raise CiError(f"selected HF cache repository is unavailable: {repo_id}")
            seen.add(folder)
            result.append((source.resolve(), folder))
        return result

    def _run_proof_container(
        self,
        projection: Path,
        work: Path,
        private_hub: Path,
        image: str,
        selection: ModelProofSelection,
        task_eval_dir: Path | None,
    ) -> None:
        assert self.lease and self.artifacts_dir is not None and self.lease.gpu_id is not None
        name = self._base_container_name()
        self.container_name = name
        slots = ",".join(map(str, self.lease.slot_ids))
        self.context.run(["docker", "rm", "-f", name], check=False, capture_output=True)
        mounts = [
            "--mount",
            f"type=bind,src={projection},dst=/src,readonly",
            "--mount",
            f"type=bind,src={work},dst=/work",
            "--mount",
            f"type=bind,src={self.artifacts_dir},dst=/artifacts",
            "--mount",
            f"type=bind,src={private_hub},dst=/hf-cache/hub",
        ]
        if task_eval_dir is not None:
            mounts.extend(
                [
                    "--mount",
                    f"type=bind,src={task_eval_dir},dst=/task-eval-data,readonly",
                ]
            )
        command = [
            "docker",
            "run",
            "--rm",
            "--name",
            name,
            "--read-only",
            "--network",
            "none",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--ipc",
            "private",
            "--shm-size",
            self.context.env.get("TRTMC_MODEL_PROOF_SHM_SIZE", "16g"),
            "--gpus",
            f"device={self.lease.gpu_id}",
            "--label",
            "com.nvidia.trtmc.model-proof=1",
            *self._job_labels(),
            "--label",
            f"com.nvidia.trtmc.model-proof.gpu={self.lease.gpu_id}",
            "--label",
            f"com.nvidia.trtmc.model-proof.slots={slots}",
            "--label",
            f"com.nvidia.trtmc.model-proof.lock-namespace={self.lease.lock_namespace}",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            *mounts,
            "--workdir",
            "/src",
            "--tmpfs",
            "/tmp:rw,exec,nosuid,nodev,size=4g",
            *self._proof_environment(slots, bool(selection.reference_cache)),
            image,
            "python3",
            "-m",
            "tools.ci",
            "model-proof",
            "--inner",
            "--model",
            self.request.model,
            "--suite",
            self.request.suite,
            "--revision",
            self.revision,
            "--output-dir",
            "/artifacts",
        ]
        rc = self._run_logged(command, self.artifacts_dir / "console.log")
        if rc:
            raise CiError(f"isolated model proof failed for {self.request.model} (exit {rc})")

    def _proof_environment(self, slots: str, has_reference: bool) -> list[str]:
        assert self.lease and self.lease.gpu_id is not None
        values = {
            "HOME": "/tmp",
            "USER": "trtmc-ci",
            "LOGNAME": "trtmc-ci",
            "TMPDIR": "/work/tmp",
            "TEMP": "/work/tmp",
            "TMP": "/work/tmp",
            "XDG_CACHE_HOME": "/work/cache",
            "TORCHINDUCTOR_CACHE_DIR": "/work/torch-cache",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONHASHSEED": "0",
            "TRTMC_MODEL_PLUGIN_STRICT": "1",
            "TRTMC_MODEL_PROOF_GPU_ID": str(self.lease.gpu_id),
            "TRTMC_MODEL_PROOF_GPU_SLOT_IDS": slots,
            "TRTMC_MODEL_PROOF_SLOTS_PER_GPU": str(self.lease.slots_per_gpu),
            "TRTMC_MODEL_PROOF_RESOURCE_CLASS": self.lease.resource_class,
            "TRTMC_MODEL_PROOF_BUILD_JOBS": self.context.env.get(
                "TRTMC_MODEL_PROOF_BUILD_JOBS", "2"
            ),
            "HF_HOME": "/work/hf-home",
            "HF_HUB_CACHE": "/hf-cache/hub",
            "HUGGINGFACE_HUB_CACHE": "/hf-cache/hub",
            "HF_MODULES_CACHE": "/work/hf-modules",
            "TRANSFORMERS_CACHE": "/hf-cache/hub",
        }
        if has_reference:
            values["TRTMC_STORAGE_ROOT"] = "/work/reference-private"
        return [item for name, value in values.items() for item in ("-e", f"{name}={value}")]

    def _reclaim_orphans(self) -> None:
        assert self.lease and self.lease.gpu_id is not None
        try:
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
                    '{{.ID}} {{.Label "com.nvidia.trtmc.model-proof.slots"}}',
                ]
            )
        except CiError as error:
            raise CiError(
                f"could not inspect existing model-proof containers on GPU {self.lease.gpu_id}"
            ) from error
        for row in rows.splitlines():
            container, slot_text = row.split()
            if not re.fullmatch(r"[a-f0-9]{64}", container):
                raise CiError(f"existing model-proof container has an unsafe ID: {container}")
            if not re.fullmatch(r"(?:0|[1-9][0-9]*)(?:,(?:0|[1-9][0-9]*))*", slot_text):
                raise CiError(
                    f"existing model-proof container {container} has invalid GPU slot labels"
                )
            if not set(map(int, slot_text.split(","))).intersection(self.lease.slot_ids):
                continue
            print(f"Removing orphaned model-proof container {container}")
            removed = self.context.run(["docker", "rm", "-f", container], check=False)
            if removed.returncode:
                remaining = self.context.output(
                    [
                        "docker",
                        "ps",
                        "-a",
                        "--no-trunc",
                        "--filter",
                        f"id={container}",
                        "--format",
                        "{{.ID}}",
                    ]
                ).splitlines()
                if container in remaining:
                    raise CiError(f"could not remove orphaned model-proof container {container}")

    def _run_logged(self, command: list[object], path: Path) -> int:
        with path.open("w", encoding="utf-8") as output:
            process = subprocess.Popen(
                [str(item) for item in command],
                cwd=self.context.repository,
                env=self.context.env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                output.write(line)
            return process.wait()

    def _record_host_error(self, error: BaseException) -> None:
        if self.artifacts_dir:
            self.artifacts_dir.mkdir(parents=True, exist_ok=True)
            with (self.artifacts_dir / "host-error.log").open("a", encoding="utf-8") as handle:
                handle.write(f"ERROR: {error}\n")

    def _fallback_report(self) -> None:
        if not self.artifacts_dir:
            return
        self.context.run(
            [
                "python3",
                self.context.repository / ".github/scripts/write-model-proof-fallback-report.py",
                "--artifacts-dir",
                self.artifacts_dir,
                "--model",
                self.request.model,
                "--revision",
                self.revision,
                "--suite",
                self.request.suite,
                "--outcome",
                "failed",
                "--phase",
                "host-setup",
                "--exit-code",
                "1",
                "--preserve-rich-report",
            ],
            check=False,
        )

    def _cleanup(self) -> None:
        if self.container_name:
            self.context.run(
                ["docker", "rm", "-f", self.container_name], check=False, capture_output=True
            )
        if self.lease:
            self.lease.release()

    def _signal(self, number: int, _frame: object) -> None:
        raise SystemExit(130 if number == signal.SIGINT else 143)

    def run_inner(self) -> None:
        ModelProofInnerRunner(self.context, self.request).run()


class ModelProofInnerRunner:
    """Build the projected source from scratch and emit the complete proof evidence."""

    def __init__(self, context: CiContext, request: ModelProofRequest):
        self.context = context
        self.request = request

    def run(self) -> None:
        # Implemented below in deliberately linear stage order: each operation
        # updates its evidence record before the next one starts.
        from .model_proof_inner import ModelProofInnerPipeline

        ModelProofInnerPipeline(self.context, self.request).run()

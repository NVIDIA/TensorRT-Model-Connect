# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve, build, and validate the immutable Docker image used by CI.

Boundary: image fingerprinting and readiness only; containers are started elsewhere.
"""

from __future__ import annotations

import fcntl
import hashlib
import importlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from .process import CiError, CommandRunner, GitHubFiles


FINGERPRINT_LABEL = "org.nvidia.trtmc.ci-input-fingerprint"
IMMUTABLE_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")


@dataclass(frozen=True)
class DockerImageConfig:
    """Environment-controlled inputs for one CI image verification."""

    repository: Path
    base_image: str
    dockerfile: Path
    lock_file: Path
    lock_timeout: int
    verification_dir: Path

    @classmethod
    def from_environment(cls, repository: Path, env: dict[str, str]) -> "DockerImageConfig":
        timeout_text = env.get("TRTMC_CI_IMAGE_LOCK_TIMEOUT", "5400")
        if not timeout_text.isdigit() or int(timeout_text) < 1:
            raise CiError("TRTMC_CI_IMAGE_LOCK_TIMEOUT must be a positive integer")
        return cls(
            repository=repository,
            base_image=env.get("TRTMC_CI_IMAGE", "trtmc-dev-gb300:manylinux_2_39"),
            dockerfile=Path(env.get("TRTMC_CI_DOCKERFILE", "Dockerfile")),
            lock_file=Path(env.get("TRTMC_CI_IMAGE_LOCK_FILE", "/tmp/trtmc-ci-docker-image.lock")),
            lock_timeout=int(timeout_text),
            verification_dir=Path(
                env.get("TRTMC_CI_IMAGE_VERIFICATION_DIR", "/tmp/trtmc-ci-image-verifications")
            ),
        )


@dataclass(frozen=True)
class ImageRequirements:
    """Source-derived contract that the Docker image must satisfy."""

    inputs: tuple[Path, ...]
    fingerprint: str
    tensorrt: str
    modelopt: str
    python_profiles: str


class WorkflowImageLock:
    """Serialize image verification and rebuilds on a self-hosted runner."""

    def __init__(self, path: Path, timeout: int):
        self.path = path
        self.timeout = timeout
        self.handle = None

    def __enter__(self) -> "WorkflowImageLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+", encoding="utf-8")
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise CiError(f"Timed out waiting for CI image lock: {self.path}")
                time.sleep(0.1)

    def __exit__(self, *_: object) -> None:
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()


class DockerImageManager:
    """Keep the host-local CI image synchronized with its source contract."""

    def __init__(self, repository: Path, env: dict[str, str] | None = None):
        self.env = dict(env or os.environ)
        self.config = DockerImageConfig.from_environment(repository.resolve(), self.env)
        self.commands = CommandRunner(cwd=self.config.repository, env=self.env)
        self.github = GitHubFiles(self.env)

    def ensure(self) -> str:
        """Return the verified immutable image ID, rebuilding only when required."""
        with WorkflowImageLock(self.config.lock_file, self.config.lock_timeout):
            requirements = self._read_requirements()
            image = f"{self.config.base_image}-{requirements.fingerprint[:12]}"
            self.github.environment("TRTMC_CI_IMAGE", image)

            stamp = self._verification_stamp(requirements.fingerprint)
            reused_id = self._reuse_verified_image(image, stamp)
            if reused_id:
                return reused_id

            reasons, versions = self._rebuild_reasons(image, requirements)
            if reasons:
                self._build(image, requirements, reasons)
                versions = self._query_versions(image)
            else:
                print(f"CI Docker image '{image}' already matches {self.config.dockerfile}")

            self._validate(image, requirements, versions)
            image_id = self._image_id(image)
            self.github.output("image_ref", image_id)
            self._write_stamp(stamp, image_id)
            print(
                f"CI Docker image '{image}' verified: TensorRT {versions['TENSORRT_VERSION']}, "
                f"modelopt {versions['MODELOPT_VERSION']}, nlohmann/json headers, NeMo prompt "
                f"RNN-T and prebuilt Python profiles ({versions['PYTHON_PROFILES']}) present, "
                f"image {image_id}"
            )
            return image_id

    def _read_requirements(self) -> ImageRequirements:
        registry = self._load_profile_registry()
        default_profile = self._default_profile_name()
        profiles = registry["profiles"]
        expected_profiles = ",".join(sorted(name for name in profiles if name != default_profile))
        if not expected_profiles:
            raise CiError("No family-owned Python execution profiles were declared")

        package_root = Path("python/tensorrt_model_connect")
        assets: set[Path] = set()
        for spec in profiles.values():
            if not isinstance(spec, dict):
                continue
            for field in ("requirements", "verification_script_file"):
                value = str(spec.get(field, "") or "").strip()
                if value:
                    assets.add(package_root / value)

        inputs = {
            self.config.dockerfile,
            Path(".dockerignore"),
            Path(".github/scripts/build-python-profiles.py"),
            package_root / "__init__.py",
            package_root / "python_profiles.py",
            package_root / "families/__init__.py",
            *assets,
        }
        semantic_contract = {"version": registry.get("version"), "profiles": profiles}
        semantic_payload = json.dumps(
            semantic_contract, ensure_ascii=True, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        semantic_fingerprint = hashlib.sha256(semantic_payload).hexdigest()
        fingerprint = self._fingerprint_inputs(tuple(sorted(inputs)), semantic_fingerprint)

        dockerfile_text = (self.config.repository / self.config.dockerfile).read_text(
            encoding="utf-8"
        )
        tensorrt = self._docker_argument(dockerfile_text, "TENSORRT_VERSION")
        modelopt = self._docker_argument(dockerfile_text, "MODELOPT_VERSION")
        return ImageRequirements(
            tuple(sorted(inputs)), fingerprint, tensorrt, modelopt, expected_profiles
        )

    def _load_profile_registry(self) -> dict[str, object]:
        package_path = str(self.config.repository / "python")
        sys.path.insert(0, package_path)
        try:
            module = importlib.import_module("tensorrt_model_connect.python_profiles")
            return module.load_python_profile_registry()
        finally:
            sys.path.remove(package_path)

    def _default_profile_name(self) -> str:
        module = importlib.import_module("tensorrt_model_connect.python_profiles")
        return str(module.DEFAULT_PROFILE)

    def _fingerprint_inputs(self, inputs: tuple[Path, ...], semantic: str) -> str:
        digest = hashlib.sha256()
        digest.update(b"python-profile-registry\0")
        digest.update(semantic.encode("ascii") + b"\n")
        for relative in inputs:
            digest.update(str(relative).encode("utf-8") + b"\0")
            source = self.config.repository / relative
            if source.is_file():
                digest.update(
                    hashlib.sha256(source.read_bytes()).hexdigest().encode("ascii") + b"\n"
                )
            else:
                digest.update(b"missing\n")
        return digest.hexdigest()

    @staticmethod
    def _docker_argument(text: str, name: str) -> str:
        match = re.search(rf"^ARG {re.escape(name)}=(.+)$", text, re.MULTILINE)
        if not match:
            raise CiError(f"Could not find ARG {name} in Dockerfile")
        return match.group(1).strip()

    def _verification_stamp(self, fingerprint: str) -> Path | None:
        run_id = self.env.get("GITHUB_RUN_ID", "")
        attempt = self.env.get("GITHUB_RUN_ATTEMPT", "1")
        if not run_id.isdigit() or not attempt.isdigit() or int(attempt) < 1:
            return None
        self.config.verification_dir.mkdir(parents=True, exist_ok=True)
        return self.config.verification_dir / f"{run_id}-{attempt}-{fingerprint}.verified"

    def _reuse_verified_image(self, image: str, stamp: Path | None) -> str | None:
        if stamp is None or not stamp.is_file():
            return None
        stamped_id = stamp.read_text(encoding="utf-8").strip()
        current_id = self._image_id(image, required=False)
        if IMMUTABLE_IMAGE_ID.fullmatch(stamped_id) and current_id == stamped_id:
            self.github.output("image_ref", current_id)
            print(
                f"CI Docker image '{image}' reused from this workflow run's verified image "
                f"{current_id}"
            )
            return current_id
        stamp.unlink(missing_ok=True)
        return None

    def _rebuild_reasons(
        self, image: str, expected: ImageRequirements
    ) -> tuple[list[str], dict[str, str]]:
        if not self._image_exists(image):
            return [f"CI Docker image '{image}' is missing"], {}

        reasons: list[str] = []
        fingerprint = self._query_fingerprint(image)
        if fingerprint != expected.fingerprint:
            reasons.append(
                "Docker input fingerprint mismatch: image has "
                f"'{fingerprint or 'missing'}', source expects '{expected.fingerprint}'"
            )
        try:
            versions = self._query_versions(image)
        except CiError:
            return reasons + [f"CI Docker image '{image}' could not report dependency versions"], {}
        reasons.extend(self._version_mismatches(versions, expected))
        return reasons, versions

    def _image_exists(self, image: str) -> bool:
        return (
            self.commands.run(
                ["docker", "image", "inspect", image], check=False, capture_output=True
            ).returncode
            == 0
        )

    def _query_fingerprint(self, image: str) -> str:
        result = self.commands.run(
            [
                "docker",
                "image",
                "inspect",
                "--format",
                f'{{{{ index .Config.Labels "{FINGERPRINT_LABEL}" }}}}',
                image,
            ],
            check=False,
            capture_output=True,
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    def _query_versions(self, image: str) -> dict[str, str]:
        probe = r"""
import importlib.metadata as metadata
import json
from pathlib import Path

import tensorrt
from nemo.collections.asr.models.rnnt_bpe_models_prompt import EncDecRNNTBPEModelWithPrompt

print(f"TENSORRT_VERSION={tensorrt.__version__}")
print("MODELOPT_VERSION=" + metadata.version("nvidia-modelopt"))
print("NLOHMANN_JSON_HEADER=" + ("present" if Path("/usr/include/nlohmann/json.hpp").is_file() else "missing"))
print("NEMO_PROMPT_RNNT=available")
manifest_path = Path("/opt/trtmc-python-profiles/.image-ready.json")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if Path("/opt/trtmc-profile-source").exists():
    raise SystemExit("profile builder source leaked into the runtime image")
profiles = manifest.get("profiles")
if not isinstance(profiles, dict) or not profiles:
    raise SystemExit("prebuilt Python profile manifest is empty or invalid")
for name, record in profiles.items():
    if not isinstance(record, dict):
        raise SystemExit(f"invalid prebuilt Python profile record: {name}")
    python = Path(str(record.get("python", "")))
    ready = Path(str(record.get("ready", "")))
    if not python.is_file() or not ready.is_file():
        raise SystemExit(f"prebuilt Python profile is incomplete: {name}")
print("PYTHON_PROFILES=" + ",".join(sorted(profiles)))
"""
        result = self.commands.run(
            [
                "docker",
                "run",
                "--rm",
                "--read-only",
                "--user",
                "65534:65534",
                "--tmpfs",
                "/tmp:rw,exec,nosuid,nodev,size=256m",
                "-e",
                "HOME=/tmp",
                "--entrypoint",
                "python3",
                image,
                "-c",
                probe,
            ],
            capture_output=True,
        )
        return {
            key: value
            for line in result.stdout.splitlines()
            if "=" in line
            for key, value in [line.split("=", 1)]
        }

    @staticmethod
    def _version_mismatches(actual: dict[str, str], expected: ImageRequirements) -> list[str]:
        checks = (
            ("TENSORRT_VERSION", expected.tensorrt, "TensorRT version mismatch"),
            ("MODELOPT_VERSION", expected.modelopt, "modelopt version mismatch"),
        )
        reasons = [
            f"{label}: image has '{actual.get(key, 'unknown')}', Dockerfile expects '{value}'"
            for key, value, label in checks
            if actual.get(key) != value
        ]
        if actual.get("NLOHMANN_JSON_HEADER") != "present":
            reasons.append("nlohmann/json development headers are missing")
        if actual.get("NEMO_PROMPT_RNNT") != "available":
            reasons.append("required NeMo prompt RNN-T capability is missing")
        if actual.get("PYTHON_PROFILES") != expected.python_profiles:
            reasons.append(
                "prebuilt Python profiles differ: image has "
                f"'{actual.get('PYTHON_PROFILES', 'missing')}', source expects "
                f"'{expected.python_profiles}'"
            )
        return reasons

    def _build(self, image: str, expected: ImageRequirements, reasons: list[str]) -> None:
        print(f"Rebuilding CI Docker image '{image}' from {self.config.dockerfile}")
        for reason in reasons:
            print(f"  reason: {reason}")
        self.github.summary(
            f"Rebuilding CI Docker image `{image}` from `{self.config.dockerfile}`."
        )
        for reason in reasons:
            self.github.summary(f"- {reason}")
        changed = self._changed_inputs(expected.inputs)
        if changed:
            self.github.summary()
            self.github.summary("Changed CI Docker image inputs:")
            for path in changed:
                self.github.summary(f"- `{path}`")
        self.commands.run(
            [
                "docker",
                "build",
                "--label",
                f"{FINGERPRINT_LABEL}={expected.fingerprint}",
                "-t",
                image,
                "-f",
                str(self.config.dockerfile),
                ".",
            ]
        )

    def _changed_inputs(self, inputs: tuple[Path, ...]) -> list[str]:
        base = self.env.get("CI_BASE_REF", "")
        if not base:
            return []
        exists = self.commands.run(
            ["git", "cat-file", "-e", f"{base}^{{commit}}"], check=False, capture_output=True
        )
        if exists.returncode != 0:
            return []
        result = self.commands.run(
            ["git", "diff", "--name-only", f"{base}...HEAD", "--", *map(str, inputs)],
            capture_output=True,
        )
        return [line for line in result.stdout.splitlines() if line]

    def _validate(self, image: str, expected: ImageRequirements, versions: dict[str, str]) -> None:
        fingerprint = self._query_fingerprint(image)
        if fingerprint != expected.fingerprint:
            raise CiError(
                f"CI Docker image '{image}' has input fingerprint "
                f"'{fingerprint or 'missing'}'; expected '{expected.fingerprint}'"
            )
        mismatches = self._version_mismatches(versions, expected)
        if mismatches:
            raise CiError("; ".join(mismatches))

    def _image_id(self, image: str, *, required: bool = True) -> str:
        result = self.commands.run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", image],
            check=False,
            capture_output=True,
        )
        image_id = result.stdout.strip() if result.returncode == 0 else ""
        if required and not IMMUTABLE_IMAGE_ID.fullmatch(image_id):
            raise CiError(f"CI Docker image '{image}' returned an invalid immutable ID: {image_id}")
        return image_id

    @staticmethod
    def _write_stamp(stamp: Path | None, image_id: str) -> None:
        if stamp is None:
            return
        temporary = stamp.with_name(f"{stamp.name}.tmp.{os.getpid()}")
        temporary.write_text(f"{image_id}\n", encoding="utf-8")
        temporary.replace(stamp)

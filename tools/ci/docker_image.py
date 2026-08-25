# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve, build, and validate the immutable Docker image used by CI.

Boundary: image fingerprinting and readiness only; containers are started elsewhere.
"""

from __future__ import annotations

import tensorrt_model_connect.utils.fcntl_shim as fcntl
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path

from .process import CiError, CommandRunner, GitHubFiles


FINGERPRINT_LABEL = "org.nvidia.trtmc.ci-input-fingerprint"
ENVIRONMENT_CONTRACT_VERSION = 2
IMMUTABLE_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")
EXACT_TENSORRT_VERSION = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+")
EXACT_APT_VERSION = re.compile(r"[0-9][0-9A-Za-z.+:~_-]*")
TENSORRT_DISTRIBUTIONS = (
    "tensorrt",
    "tensorrt_cu13",
    "tensorrt_cu13_bindings",
    "tensorrt_cu13_libs",
)
TENSORRT_APT_PACKAGES = (
    "libnvinfer-dev",
    "libnvinfer-headers-dev",
    "libnvinfer-headers-plugin-dev",
    "libnvinfer-safe-headers-dev",
    "libnvinfer11",
    "libnvonnxparsers-dev",
    "libnvonnxparsers11",
)
DEFAULT_IMAGE_LOCK_TIMEOUT_SECONDS = 6 * 60 * 60


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
        timeout_text = env.get(
            "TRTMC_CI_IMAGE_LOCK_TIMEOUT",
            str(DEFAULT_IMAGE_LOCK_TIMEOUT_SECONDS),
        )
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
    common_fingerprint: str
    fingerprint: str
    tensorrt: str
    tensorrt_apt: str
    modelopt: str
    python_profiles: str

    @property
    def python_distributions(self) -> dict[str, str]:
        return {name: self.tensorrt for name in TENSORRT_DISTRIBUTIONS}

    @property
    def apt_packages(self) -> dict[str, str]:
        return {name: self.tensorrt_apt for name in TENSORRT_APT_PACKAGES}

    @property
    def tensorrt_major(self) -> str:
        return self.tensorrt.split(".", 1)[0]

    def contract(self) -> dict[str, object]:
        """Return the source-owned runtime contract for an overlay image."""
        return {
            "schema_version": 1,
            "environment_contract_version": ENVIRONMENT_CONTRACT_VERSION,
            "common_input_fingerprint": self.common_fingerprint,
            "input_fingerprint": self.fingerprint,
            "modelopt_version": self.modelopt,
            "python_profiles": self.python_profiles.split(","),
            "tensorrt": {
                "version": self.tensorrt,
                "apt_version": self.tensorrt_apt,
                "python_distributions": self.python_distributions,
                "apt_packages": self.apt_packages,
                "headers": ["NvInferVersion.h", "NvOnnxParser.h"],
                "header_version": self.tensorrt,
                "native_libraries": [
                    "libnvinfer.so",
                    f"libnvinfer.so.{self.tensorrt_major}",
                    "libnvonnxparser.so",
                    f"libnvonnxparser.so.{self.tensorrt_major}",
                    "libnvinfer_builder_resource_sm110.so.*",
                ],
                "native_library_distribution": "tensorrt_cu13_libs",
                "native_runtime_version": self.tensorrt,
            },
        }


class WorkflowImageLock:
    """Serialize image verification and rebuilds on a self-hosted runner."""

    def __init__(self, path: Path, timeout: int):
        self.path = path
        self.timeout = timeout
        self.handle = None

    def _open(self):
        """Open a shared lock without O_CREAT when it already exists.

        Hardened Linux rejects O_CREAT for another user's file directly under
        a sticky directory such as /tmp, even when a shared group can write it.
        """
        try:
            return self.path.open("r+", encoding="utf-8")
        except FileNotFoundError:
            try:
                return self.path.open("x+", encoding="utf-8")
            except FileExistsError:
                # Another runner created the lock between the two opens.
                return self.path.open("r+", encoding="utf-8")

    def __enter__(self) -> "WorkflowImageLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self._open()
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
                f"exact Python, APT header, C++ header, and native runtime contracts, modelopt "
                f"{versions['MODELOPT_VERSION']}, nlohmann/json headers, NeMo prompt RNN-T and "
                f"prebuilt Python profiles ({versions['PYTHON_PROFILES']}) present, "
                f"image {image_id}"
            )
            return image_id

    def source_contract(
        self,
        *,
        tensorrt_version: str | None = None,
        tensorrt_apt_version: str | None = None,
    ) -> dict[str, object]:
        """Return a JSON-compatible contract for one parameterized TRT overlay."""
        return self._read_requirements(
            tensorrt_version=tensorrt_version,
            tensorrt_apt_version=tensorrt_apt_version,
        ).contract()

    def source_contract_json(
        self,
        *,
        tensorrt_version: str | None = None,
        tensorrt_apt_version: str | None = None,
    ) -> str:
        """Serialize ``source_contract`` deterministically for external validation."""
        return json.dumps(
            self.source_contract(
                tensorrt_version=tensorrt_version,
                tensorrt_apt_version=tensorrt_apt_version,
            ),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )

    def validate_image_contract(
        self,
        image: str,
        *,
        tensorrt_version: str | None = None,
        tensorrt_apt_version: str | None = None,
    ) -> dict[str, object]:
        """Validate one immutable overlay image against its Source contract."""
        expected = self._read_requirements(
            tensorrt_version=tensorrt_version,
            tensorrt_apt_version=tensorrt_apt_version,
        )
        self._validate(image, expected, self._query_versions(image))
        return expected.contract()

    def _read_requirements(
        self,
        *,
        tensorrt_version: str | None = None,
        tensorrt_apt_version: str | None = None,
    ) -> ImageRequirements:
        registry, profile_names = self._load_profile_registry()
        profiles = registry["profiles"]
        expected_profiles = ",".join(profile_names)
        if not expected_profiles:
            raise CiError("No prebuilt Python execution profiles were declared")

        package_root = Path("python/tensorrt_model_connect")
        assets: set[Path] = set()
        prebuilt_profiles = {name: profiles[name] for name in profile_names}
        for spec in prebuilt_profiles.values():
            if not isinstance(spec, dict):
                continue
            for field in ("requirements", "verification_script_file"):
                value = str(spec.get(field, "") or "").strip()
                if value:
                    assets.add(self._profile_asset_input(package_root, value, field))

        inputs = {
            self.config.dockerfile,
            Path(".dockerignore"),
            Path(".github/scripts/build-python-profiles.py"),
            package_root / "python_profiles.py",
            *assets,
        }
        dockerfile_text = (self.config.repository / self.config.dockerfile).read_text(
            encoding="utf-8"
        )
        tensorrt = self._exact_tensorrt_version(
            self._docker_argument(dockerfile_text, "TENSORRT_VERSION")
            if tensorrt_version is None
            else tensorrt_version
        )
        tensorrt_apt = self._exact_apt_version(
            self._docker_argument(dockerfile_text, "TENSORRT_APT_VERSION")
            if tensorrt_apt_version is None
            else tensorrt_apt_version
        )
        if not tensorrt_apt.startswith(f"{tensorrt}-"):
            raise CiError(
                "TENSORRT_APT_VERSION must select the same TensorRT version as TENSORRT_VERSION"
            )
        common_semantic_contract = {
            "environment_contract_version": ENVIRONMENT_CONTRACT_VERSION,
            "version": registry.get("version"),
            "profiles": prebuilt_profiles,
        }
        common_semantic_payload = json.dumps(
            common_semantic_contract,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        common_semantic_fingerprint = hashlib.sha256(common_semantic_payload).hexdigest()
        common_fingerprint = self._fingerprint_inputs(
            tuple(sorted(inputs)), common_semantic_fingerprint
        )
        overlay_payload = json.dumps(
            {"apt_version": tensorrt_apt, "version": tensorrt},
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        fingerprint = hashlib.sha256(
            b"tensorrt-overlay\0" + common_fingerprint.encode("ascii") + b"\0" + overlay_payload
        ).hexdigest()

        modelopt = self._docker_argument(dockerfile_text, "MODELOPT_VERSION")
        return ImageRequirements(
            tuple(sorted(inputs)),
            common_fingerprint,
            fingerprint,
            tensorrt,
            tensorrt_apt,
            modelopt,
            expected_profiles,
        )

    def _profile_asset_input(
        self,
        package_root: Path,
        path_spec: str,
        field: str,
    ) -> Path:
        relative = Path(path_spec)
        if relative.is_absolute() or ".." in relative.parts:
            raise CiError(f"Python profile has an unsafe {field} path: {path_spec!r}")
        absolute_root = (self.config.repository / package_root).resolve()
        resolved = (absolute_root / relative).resolve()
        try:
            resolved.relative_to(absolute_root)
        except ValueError as error:
            raise CiError(
                f"Python profile has an unsafe {field} path: {path_spec!r}"
            ) from error
        if not resolved.is_file():
            raise CiError(
                f"Python profile references a missing {field} asset: {path_spec!r}"
            )
        return package_root / relative

    def _load_profile_registry(self) -> tuple[dict[str, object], tuple[str, ...]]:
        package_name = "tensorrt_model_connect"
        package_root = self.config.repository / "python" / package_name
        module_name = f"{package_name}.python_profiles"
        previous_modules = {
            name: module
            for name, module in sys.modules.items()
            if name == package_name or name.startswith(f"{package_name}.")
        }
        for name in previous_modules:
            sys.modules.pop(name, None)
        package = types.ModuleType(package_name)
        package.__package__ = package_name
        package.__path__ = [str(package_root)]
        sys.modules[package_name] = package
        try:
            spec = importlib.util.spec_from_file_location(
                module_name,
                package_root / "python_profiles.py",
            )
            if spec is None or spec.loader is None:
                raise CiError("Could not load the Source Python profile registry")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            registry = module.load_python_profile_registry()
            return registry, module.prebuilt_python_profile_names(registry)
        finally:
            for name in tuple(sys.modules):
                if name == package_name or name.startswith(f"{package_name}."):
                    sys.modules.pop(name, None)
            sys.modules.update(previous_modules)

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

    @staticmethod
    def _exact_tensorrt_version(value: str) -> str:
        if not EXACT_TENSORRT_VERSION.fullmatch(value):
            raise CiError("TENSORRT_VERSION must be an exact four-part version")
        return value

    @staticmethod
    def _exact_apt_version(value: str) -> str:
        if not EXACT_APT_VERSION.fullmatch(value):
            raise CiError("TENSORRT_APT_VERSION must be an exact package version")
        return value

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
import ctypes
import importlib.metadata as metadata
import json
import os
import re
import subprocess
from pathlib import Path

import tensorrt
from nemo.collections.asr.models.rnnt_bpe_models_prompt import EncDecRNNTBPEModelWithPrompt

print(f"TENSORRT_VERSION={tensorrt.__version__}")
distributions = {
    name: metadata.version(name)
    for name in (
        "tensorrt",
        "tensorrt_cu13",
        "tensorrt_cu13_bindings",
        "tensorrt_cu13_libs",
    )
}
print("TENSORRT_PYTHON_DISTRIBUTIONS=" + json.dumps(distributions, separators=(",", ":"), sort_keys=True))
apt_packages = {}
for package in (
    "libnvinfer-dev",
    "libnvinfer-headers-dev",
    "libnvinfer-headers-plugin-dev",
    "libnvinfer-safe-headers-dev",
    "libnvinfer11",
    "libnvonnxparsers-dev",
    "libnvonnxparsers11",
):
    result = subprocess.run(
        ["dpkg-query", "-W", "-f=${Version}", package],
        check=True,
        capture_output=True,
        text=True,
    )
    apt_packages[package] = result.stdout.strip()
print("TENSORRT_APT_PACKAGES=" + json.dumps(apt_packages, separators=(",", ":"), sort_keys=True))

header = Path(os.environ["TRT_INC_DIR"]) / "NvInferVersion.h"
header_text = header.read_text(encoding="utf-8")
header_parts = []
for name in ("MAJOR", "MINOR", "PATCH", "BUILD"):
    values = []
    for macro in (f"TRT_{name}_ENTERPRISE", f"NV_TENSORRT_{name}"):
        match = re.search(rf"^#define\s+{macro}\s+([0-9]+)\b", header_text, re.MULTILINE)
        if match:
            values.append(match.group(1))
    if not values or len(set(values)) != 1:
        raise SystemExit(f"could not resolve one TensorRT {name.lower()} value from {header}")
    header_parts.append(values[0])
print("TENSORRT_HEADER_VERSION=" + ".".join(header_parts))

library_root = Path(metadata.distribution("tensorrt_cu13_libs").locate_file("tensorrt_libs"))
major = header_parts[0]
configured_library_root = Path(os.environ["TRT_LIB_DIR"])
library_search = os.environ.get("LD_LIBRARY_PATH", "").split(":", 1)[0]
if configured_library_root.resolve() != library_root.resolve() or library_search != str(configured_library_root):
    raise SystemExit("TRT_LIB_DIR and LD_LIBRARY_PATH must select tensorrt_cu13_libs")
required_files = (
    Path(os.environ["TRT_INC_DIR"]) / "NvOnnxParser.h",
    library_root / "libnvinfer.so",
    library_root / f"libnvinfer.so.{major}",
    library_root / "libnvonnxparser.so",
    library_root / f"libnvonnxparser.so.{major}",
)
if missing := [str(path) for path in required_files if not path.is_file()]:
    raise SystemExit("missing TensorRT overlay files: " + ", ".join(missing))
for linker_name in ("libnvinfer.so", "libnvonnxparser.so"):
    linker = library_root / linker_name
    versioned = library_root / f"{linker_name}.{major}"
    if not linker.is_symlink() or linker.resolve() != versioned.resolve():
        raise SystemExit(f"invalid TensorRT linker symlink: {linker}")
if next(library_root.glob("libnvinfer_builder_resource_sm110.so.*"), None) is None:
    raise SystemExit("missing TensorRT SM110 builder resource")
print("TENSORRT_OVERLAY_FILES=present")

library = ctypes.CDLL(str(library_root / f"libnvinfer.so.{major}"))
native_parts = []
for symbol in (
    "getInferLibMajorVersion",
    "getInferLibMinorVersion",
    "getInferLibPatchVersion",
    "getInferLibBuildVersion",
):
    function = getattr(library, symbol)
    function.restype = ctypes.c_int32
    native_parts.append(str(function()))
print("TENSORRT_NATIVE_VERSION=" + ".".join(native_parts))
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
            (
                "TENSORRT_PYTHON_DISTRIBUTIONS",
                json.dumps(expected.python_distributions, separators=(",", ":"), sort_keys=True),
                "TensorRT Python distribution versions mismatch",
            ),
            (
                "TENSORRT_APT_PACKAGES",
                json.dumps(expected.apt_packages, separators=(",", ":"), sort_keys=True),
                "TensorRT APT package versions mismatch",
            ),
            (
                "TENSORRT_OVERLAY_FILES",
                "present",
                "TensorRT header or native library is missing",
            ),
            (
                "TENSORRT_HEADER_VERSION",
                expected.tensorrt,
                "TensorRT C++ header version mismatch",
            ),
            (
                "TENSORRT_NATIVE_VERSION",
                expected.tensorrt,
                "TensorRT native runtime version mismatch",
            ),
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

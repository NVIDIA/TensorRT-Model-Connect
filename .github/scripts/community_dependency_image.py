# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build and qualify a family dependency layer on a trusted x86 L4 host."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import platform
import re
import selectors
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2]
PROBE = """
import importlib, importlib.metadata as m, json, os, platform, re, sys, torch, tensorrt
imports = os.environ["TRTMC_DEPENDENCY_IMPORTS"].split(",")
assert all(re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_.]*", name) for name in imports)
for name in imports:
 importlib.import_module(name)
assert platform.machine() == 'x86_64'
assert sys.version_info[:2] == (3, 12)
assert torch.__version__ == '2.12.0+cu130'
assert torch.version.cuda == '13.0'
assert tensorrt.__version__ == '11.1.0.106'
assert torch.cuda.is_available()
assert torch.cuda.get_device_capability(0) == (8, 9)
assert torch.cuda.get_device_name(0) == 'NVIDIA L4'
packages = sorted(f'{d.metadata["Name"]}=={d.version}' for d in m.distributions())
print('TRTMC_IMAGE_PROFILE=' + json.dumps({'platform': 'linux/amd64', 'python_abi': 'cp312',
 'torch': torch.__version__, 'cuda': torch.version.cuda,
 'tensorrt': tensorrt.__version__, 'cxx11abi': torch._C._GLIBCXX_USE_CXX11_ABI,
 'apache_tvm_ffi': m.version('apache-tvm-ffi'),
 'gpu': torch.cuda.get_device_name(0), 'sm': list(torch.cuda.get_device_capability(0)),
 'resolved_dependencies': packages}, sort_keys=True))
"""


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CapturedOutputLimit(subprocess.SubprocessError):
    """Retain only already-bounded public diagnostics after a capture limit."""

    def __init__(self, stream: str, stdout: bytes, stderr: bytes):
        super().__init__(f"Captured {stream} exceeds the output limit")
        self.stdout = stdout.decode("utf-8", errors="replace")
        self.stderr = stderr.decode("utf-8", errors="replace")


def _bounded_capture(
    command: list[str],
    *,
    stdin: str | None = None,
    timeout: float = 900,
    stdout_limit: int = 1024 * 1024,
    stderr_limit: int = 1024 * 1024,
) -> subprocess.CompletedProcess:
    """Reject excess bytes while reading, including pipes held by descendants."""
    if stdin is not None and len(stdin) > 65536:
        raise CapturedOutputLimit("stdin", b"", b"")
    pending = stdin.encode("utf-8") if stdin is not None else b""
    if len(pending) > 65536:
        raise CapturedOutputLimit("stdin", b"", b"")
    deadline = time.monotonic() + timeout
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    limits = {"stdout": stdout_limit, "stderr": stderr_limit}
    written = 0
    failed = True
    try:
        with selectors.DefaultSelector() as selector:
            for label in buffers:
                stream = getattr(process, label)
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_READ, label)
            if process.stdin is not None:
                os.set_blocking(process.stdin.fileno(), False)
                selector.register(process.stdin, selectors.EVENT_WRITE, "stdin")
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(
                        command,
                        timeout,
                        output=bytes(buffers["stdout"]).decode("utf-8", errors="replace"),
                        stderr=bytes(buffers["stderr"]).decode("utf-8", errors="replace"),
                    )
                for key, _ in selector.select(min(remaining, 0.1)):
                    stream, label = key.fileobj, key.data
                    if label == "stdin":
                        try:
                            written += os.write(stream.fileno(), pending[written : written + 16384])
                        except BrokenPipeError:
                            written = len(pending)
                        except BlockingIOError:
                            continue
                        if written == len(pending):
                            selector.unregister(stream)
                            stream.close()
                        continue
                    available = limits[label] - len(buffers[label])
                    try:
                        chunk = os.read(stream.fileno(), min(65536, available + 1))
                    except BlockingIOError:
                        continue
                    if not chunk:
                        selector.unregister(stream)
                        stream.close()
                    elif len(chunk) > available:
                        raise CapturedOutputLimit(
                            label, bytes(buffers["stdout"]), bytes(buffers["stderr"])
                        )
                    else:
                        buffers[label].extend(chunk)
            returncode = process.wait(timeout=max(0, deadline - time.monotonic()))
        failed = False
        return subprocess.CompletedProcess(
            command,
            returncode,
            stdout=bytes(buffers["stdout"]).decode("utf-8", errors="replace"),
            stderr=bytes(buffers["stderr"]).decode("utf-8", errors="replace"),
        )
    finally:
        if failed:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                pass
            try:
                process.wait(timeout=0.25)
            except subprocess.TimeoutExpired:
                pass
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                stream.close()


def run(command: list[str], *, capture: bool = False, stdin: str | None = None) -> str:
    try:
        if capture or stdin is not None:
            result = _bounded_capture(command, stdin=stdin)
            if result.returncode:
                raise subprocess.CalledProcessError(
                    result.returncode, command, output=result.stdout, stderr=result.stderr
                )
        else:
            result = subprocess.run(command, check=True, text=True)
    except (OSError, subprocess.SubprocessError) as error:
        if stdin is not None:
            raise RuntimeError("Credential command failed; captured output is suppressed") from None
        if capture:
            for label in ("stdout", "stderr"):
                stream = getattr(error, label, None)
                if stream:
                    tail = stream.encode("utf-8", errors="replace")[-16384:].decode(
                        "utf-8", errors="ignore"
                    )
                    print(f"Captured {label} tail:\n{tail}", file=sys.stderr, flush=True)
        raise
    return result.stdout if capture else ""


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def require_native_pass(path: Path) -> None:
    cases = ET.parse(path).getroot().findall(".//testcase")
    if (
        len(cases) != 1
        or cases[0].get("name") != "byok_tvm_ffi"
        or any(
            case.find(kind) is not None
            for case in cases
            for kind in ("failure", "error", "skipped")
        )
    ):
        raise RuntimeError("The unchanged native BYOK qualification must execute and pass")


def validated_family(family: str) -> str:
    if not re.fullmatch(r"[a-z][a-z0-9_]*", family):
        raise RuntimeError("The dependency family must be a declared family directory name")
    return family


def _candidate_json(content: bytes | str) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    try:
        value = json.loads(content, object_pairs_hook=unique)
        if not isinstance(value, dict):
            raise ValueError("not an object")
        return value
    except (ValueError, UnicodeError, RecursionError):
        raise RuntimeError("Candidate JSON evidence is invalid") from None


def _private_json(path: Path) -> dict:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        raise RuntimeError("Private candidate credentials are unavailable") from None
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077:
            raise ValueError("not a private regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            content = stream.read(65537)
        if len(content) > 65536:
            raise ValueError("oversized auth")
        return _candidate_json(content)
    except (ValueError, UnicodeError, RecursionError):
        raise RuntimeError("Private candidate credentials are invalid") from None
    finally:
        os.close(descriptor)


def _private_docker(
    command: list[str],
    *,
    stdin: str | None = None,
    timeout: float = 900,
    stdout_limit: int = 1024 * 1024,
    stderr_limit: int = 1024 * 1024,
) -> str:
    try:
        result = _bounded_capture(
            command,
            stdin=stdin,
            timeout=timeout,
            stdout_limit=stdout_limit,
            stderr_limit=stderr_limit,
        )
    except (OSError, subprocess.SubprocessError):
        raise RuntimeError("Private candidate transport failed; output is suppressed") from None
    if result.returncode:
        raise RuntimeError("Private candidate transport failed; output is suppressed")
    return result.stdout


def _require_private_candidate(prefix: str, family: str, token: str) -> None:
    owner, _, package = prefix.removeprefix("ghcr.io/").partition("/")
    package = f"{package}/{family}" if package else family

    def get(route):
        request = urllib.request.Request(
            "https://api.github.com/" + route,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        try:
            with urllib.request.build_opener(NoRedirect()).open(request, timeout=30) as response:
                if response.status != 200:
                    raise ValueError("not HTTP200")
                body = response.read(1024 * 1024 + 1)
            if len(body) > 1024 * 1024:
                raise ValueError("oversized response")
            return _candidate_json(body)
        except (OSError, ValueError, RecursionError, RuntimeError):
            raise RuntimeError("Private candidate visibility could not be verified") from None

    profile = get(f"users/{owner}")
    if (
        profile.get("type") not in ("User", "Organization")
        or not isinstance(profile.get("login"), str)
        or profile["login"].lower() != owner
    ):
        raise RuntimeError("Private candidate owner could not be verified")
    kind = "orgs" if profile["type"] == "Organization" else "users"
    package_info = get(f"{kind}/{owner}/packages/container/" + urllib.parse.quote(package, safe=""))
    if package_info.get("visibility") != "private" or package_info.get("name") != package:
        raise RuntimeError("The candidate must be an existing private package")


def _environment_paths(family: str) -> tuple[str, ...]:
    return (
        "Dockerfile.dev.x86-gpu",
        "requirements/community-ci.txt",
        "requirements/image-environment.py",
        "requirements/community-gpu-linux-amd64.lock",
        "requirements/community-gpu-linux-amd64.json",
        f"families/{family}/requirements.txt",
        f"families/{family}/ci/Dockerfile.dependencies",
        f"families/{family}/ci/build-dependencies.sh",
        f"families/{family}/ci/constraints-linux-amd64.txt",
        f"families/{family}/ci/environment-linux-amd64.lock",
        f"families/{family}/ci/environment-linux-amd64.json",
    )


def _public_git(repository: Path, *arguments: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "-c", f"safe.directory={repository}", "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        raise RuntimeError("Protected public source evidence is unavailable") from None


def _environment_provenance(repository: Path, family: str, manifest: object) -> dict:
    if not isinstance(manifest, dict):
        raise RuntimeError("The embedded public build manifest is invalid")
    environment_sha = manifest.get("source_sha")
    inputs = manifest.get("inputs")
    if (
        type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 1
        or manifest.get("mode") != "locked"
        or manifest.get("native_byok_passed") is not False
        or manifest.get("family_e2e_passed") is not False
        or not isinstance(environment_sha, str)
        or not re.fullmatch(r"[0-9a-f]{40}", environment_sha)
        or not isinstance(inputs, dict)
        or set(inputs) != set(_environment_paths(family))
        or any(
            not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
            for value in inputs.values()
        )
    ):
        raise RuntimeError("The candidate has no complete locked public environment manifest")
    protected = False
    for ref in ("refs/remotes/origin/main", "refs/remotes/origin/ci/developer"):
        try:
            _public_git(SOURCE, "merge-base", "--is-ancestor", environment_sha, ref)
            protected = True
        except RuntimeError:
            pass
    if not protected:
        raise RuntimeError("The environment source is not in protected public history")
    for path, expected in inputs.items():
        object_name = f"{environment_sha}:{path}"
        size = int(_public_git(SOURCE, "cat-file", "-s", object_name))
        if (
            size > 1024 * 1024
            or hashlib.sha256(_public_git(SOURCE, "show", object_name)).hexdigest() != expected
        ):
            raise RuntimeError("The public environment inputs do not match the candidate")
    model_sha = _public_git(repository, "rev-parse", "HEAD").decode().strip()
    if not re.fullmatch(r"[0-9a-f]{40}", model_sha):
        raise RuntimeError("The protected model source is invalid")
    requirement = f"families/{family}/requirements.txt"
    if (
        hashlib.sha256(_public_git(repository, "show", f"{model_sha}:{requirement}")).hexdigest()
        != inputs[requirement]
    ):
        raise RuntimeError("The environment dependencies do not match protected model requirements")
    return {
        "environment_source_sha": environment_sha,
        "model_source_sha": model_sha,
        "inputs": inputs,
    }


def prepare_candidate(
    output: Path, repository: Path, auth_file: Path, *, family: str, candidate_digest: str
) -> None:
    """Pull one private CPU candidate, erase auth, and run the original GPU qualification."""
    try:
        family = validated_family(family)
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", candidate_digest):
            raise RuntimeError("Candidate qualification requires an immutable digest")
        if platform.machine() != "x86_64":
            raise RuntimeError("Candidate qualification requires a real x86_64 host")
        auth = _private_json(auth_file)
        prefix, username, token = (
            auth.get(field) for field in ("registry_prefix", "username", "token")
        )
        if (
            not isinstance(prefix, str)
            or not re.fullmatch(
                r"ghcr\.io/[a-z0-9]+(?:[._-][a-z0-9]+)*(?:/[a-z0-9]+(?:[._-][a-z0-9]+)*)*", prefix
            )
            or not isinstance(username, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]*(?:\[bot\])?", username)
            or not isinstance(token, str)
            or not token.strip()
        ):
            raise RuntimeError("Private candidate credentials are invalid")
        auth_file.unlink(missing_ok=True)
        _require_private_candidate(prefix, family, token)
        reference = f"{prefix}/{family}@{candidate_digest}"
        with tempfile.TemporaryDirectory(prefix="trtmc-candidate-auth-") as config:
            docker = ["docker", "--config", config]
            _private_docker(
                [*docker, "login", "ghcr.io", "--username", username, "--password-stdin"],
                stdin=token,
                timeout=30,
            )
            _private_docker([*docker, "pull", "--platform", "linux/amd64", reference])
            image = _private_docker(
                [*docker, "image", "inspect", "--format", "{{.Id}}", reference], timeout=30
            ).strip()
            if not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
                raise RuntimeError("The private candidate has no immutable local image ID")
        auth.clear()
        token = prefix = reference = ""
        embedded = _private_docker(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--entrypoint",
                "/usr/bin/head",
                image,
                "-c",
                "65537",
                "/opt/trtmc-ci/build-inputs.json",
            ],
            timeout=60,
            stdout_limit=65536,
        )
        if len(embedded.encode()) > 65536:
            raise RuntimeError("The embedded public build manifest exceeds the size limit")
        try:
            manifest = _candidate_json(embedded)
        except RuntimeError:
            raise RuntimeError("The embedded public build manifest is invalid") from None
        provenance = _environment_provenance(repository, family, manifest)
        output.mkdir(parents=True, exist_ok=True)
        run(["docker", "run", "--rm", "--network", "none", image, "python", "-m", "pip", "check"])
        probe = run(
            [
                "docker",
                "run",
                "--rm",
                "--gpus",
                "all",
                "--network",
                "none",
                image,
                "python",
                "-c",
                PROBE,
            ],
            capture=True,
        )
        profiles = [
            line.removeprefix("TRTMC_IMAGE_PROFILE=")
            for line in probe.splitlines()
            if line.startswith("TRTMC_IMAGE_PROFILE=")
        ]
        if len(profiles) != 1:
            raise RuntimeError("The native import probe did not produce exactly one ABI profile")
        abi = json.loads(profiles[0])
        inputs = provenance["inputs"]
        candidate = {
            "schema_version": 1,
            "family": family,
            "digest": candidate_digest,
            "local_image": image,
            "source_sha": provenance["environment_source_sha"],
            **provenance,
            "producer_source_sha": _public_git(SOURCE, "rev-parse", "HEAD").decode().strip(),
            "platform": "linux/amd64",
            "registry_visibility": "private",
            "abi": abi,
            "base_dockerfile_sha256": inputs["Dockerfile.dev.x86-gpu"],
            "base_requirements_sha256": inputs["requirements/community-ci.txt"],
            "family_requirements_sha256": inputs[f"families/{family}/requirements.txt"],
            "dependency_recipe_sha256": inputs[f"families/{family}/ci/Dockerfile.dependencies"],
            "dependency_build_helper_sha256": inputs[f"families/{family}/ci/build-dependencies.sh"],
            "dependency_constraints_sha256": inputs[
                f"families/{family}/ci/constraints-linux-amd64.txt"
            ],
            "base_environment_lock_sha256": inputs["requirements/community-gpu-linux-amd64.lock"],
            "family_environment_lock_sha256": inputs[
                f"families/{family}/ci/environment-linux-amd64.lock"
            ],
            "environment_recorder_sha256": inputs["requirements/image-environment.py"],
            "base_environment_receipt_sha256": inputs[
                "requirements/community-gpu-linux-amd64.json"
            ],
            "family_environment_receipt_sha256": inputs[
                f"families/{family}/ci/environment-linux-amd64.json"
            ],
            "resolved_dependencies_sha256": hashlib.sha256(
                json.dumps(abi["resolved_dependencies"], separators=(",", ":")).encode()
            ).hexdigest(),
            "native_byok_passed": False,
            "family_e2e_passed": False,
            "cleanup_confirmed": False,
        }
        save(output / "candidate.json", candidate)
        _native_qualification(output, repository, image, family)
        candidate["native_byok_passed"] = True
        save(output / "candidate.json", candidate)
    finally:
        auth_file.unlink(missing_ok=True)


def _native_qualification(output: Path, repository: Path, image: str, family: str) -> None:
    command = (
        "set -eu; cmake -S /src -B /proof/native -G Ninja "
        "-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 "
        "-DTRTMC_BUILD_TESTS=ON -DTRTMC_BUILD_EXAMPLES=ON "
        "-DTRTMC_BUILD_SERVER=OFF -DTRTMC_BUILD_BACKEND_RTX=OFF "
        "-DTRTMC_ENABLE_BYOK=ON; cmake --build /proof/native --parallel 8 --target "
        f"trtmc trtmc_runtime trtmc_c trtmc_backend_trt trtmc_model_{family} "
        "test_byok_tvm_ffi; ctest --test-dir /proof/native --output-on-failure "
        "--output-junit /proof/native-byok.xml -R '^byok_tvm_ffi$'"
    )
    run(
        [
            "docker",
            "run",
            "--rm",
            "--gpus",
            "all",
            "--network",
            "none",
            "--volume",
            f"{repository}:/src:ro",
            "--volume",
            f"{output}:/proof",
            image,
            "bash",
            "-c",
            command,
        ]
    )
    require_native_pass(output / "native-byok.xml")


def build(output: Path, repository: Path, *, family: str) -> None:
    family = validated_family(family)
    recipe = SOURCE / "families" / family / "ci" / "Dockerfile.dependencies"
    repository = repository.resolve(strict=True)
    if platform.machine() != "x86_64":
        raise RuntimeError("Dependency production requires a real x86_64 host")
    output.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output).free < 200 * 1024**3:
        raise RuntimeError("Dependency production requires at least200GiB free disk")
    revision = run(
        ["git", "-c", f"safe.directory={repository}", "-C", str(repository), "rev-parse", "HEAD"],
        capture=True,
    ).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeError("The producer source must be an immutable commit")
    base = f"trtmc-community-base:{revision[:16]}"
    image = f"trtmc-community-{family}:{revision[:16]}"
    run(
        [
            "docker",
            "build",
            "--platform",
            "linux/amd64",
            "--file",
            str(repository / "Dockerfile.dev.x86-gpu"),
            "--tag",
            base,
            "--label",
            "org.opencontainers.image.source=https://github.com/NVIDIA/TensorRT-Model-Connect",
            "--label",
            f"org.opencontainers.image.revision={revision}",
            str(repository / "requirements"),
        ]
    )
    base_id = run(["docker", "image", "inspect", "--format", "{{.Id}}", base], capture=True).strip()
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", base_id):
        raise RuntimeError("The base build did not establish an immutable local image ID")
    requirement = repository / "families" / family / "requirements.txt"
    with tempfile.TemporaryDirectory(prefix="trtmc-community-dependencies-") as directory:
        context = Path(directory)
        shutil.copyfile(requirement, context / "requirements.txt")
        shutil.copyfile(recipe, context / "Dockerfile")
        run(
            [
                "docker",
                "build",
                "--platform",
                "linux/amd64",
                "--build-arg",
                f"BASE_IMAGE={base}",
                "--build-arg",
                "MAX_JOBS=8",
                "--tag",
                image,
                "--label",
                "org.opencontainers.image.source=https://github.com/NVIDIA/TensorRT-Model-Connect",
                "--label",
                f"org.opencontainers.image.revision={revision}",
                str(context),
            ]
        )
    run(["docker", "run", "--rm", "--network", "none", image, "python", "-m", "pip", "check"])
    probe_output = run(
        [
            "docker",
            "run",
            "--rm",
            "--gpus",
            "all",
            "--network",
            "none",
            image,
            "python",
            "-c",
            PROBE,
        ],
        capture=True,
    )
    profiles = [
        line.removeprefix("TRTMC_IMAGE_PROFILE=")
        for line in probe_output.splitlines()
        if line.startswith("TRTMC_IMAGE_PROFILE=")
    ]
    if len(profiles) != 1:
        raise RuntimeError("The native import probe did not produce exactly one ABI profile")
    observed = json.loads(profiles[0])
    print("TRTMC_DEPENDENCY_PROFILE=" + json.dumps(observed, sort_keys=True), flush=True)
    closure = hashlib.sha256(
        json.dumps(observed["resolved_dependencies"], separators=(",", ":")).encode()
    ).hexdigest()
    save(
        output / "candidate.json",
        {
            "schema_version": 1,
            "family": family,
            "source_sha": revision,
            "source_tree": run(
                [
                    "git",
                    "-c",
                    f"safe.directory={repository}",
                    "-C",
                    str(repository),
                    "rev-parse",
                    "HEAD^{tree}",
                ],
                capture=True,
            ).strip(),
            "producer_source_sha": run(
                ["git", "-c", f"safe.directory={SOURCE}", "-C", str(SOURCE), "rev-parse", "HEAD"],
                capture=True,
            ).strip(),
            "platform": "linux/amd64",
            "base_local_image": base,
            "base_local_image_id": base_id,
            "local_image": image,
            "base_dockerfile_sha256": digest(repository / "Dockerfile.dev.x86-gpu"),
            "base_requirements_sha256": digest(repository / "requirements" / "community-ci.txt"),
            "family_requirements_sha256": digest(requirement),
            "dependency_recipe_sha256": digest(recipe),
            "resolved_dependencies_sha256": closure,
            "abi": observed,
            "native_byok_passed": False,
            "family_e2e_passed": False,
        },
    )
    _native_qualification(output, repository, image, family)
    candidate = json.loads((output / "candidate.json").read_text())
    candidate["native_byok_passed"] = True
    save(output / "candidate.json", candidate)


def qualify(
    output: Path,
    stage_python: Path,
    token_file: Path | None,
    repository: Path = SOURCE,
    *,
    family: str,
) -> None:
    family = validated_family(family)
    candidate = json.loads((output / "candidate.json").read_text())
    if candidate.get("family") != family:
        raise RuntimeError("The candidate belongs to a different family")
    if candidate.get("native_byok_passed") is not True:
        raise RuntimeError("Native ABI qualification is required before family E2E")
    candidate["family_e2e_passed"] = False
    save(output / "candidate.json", candidate)
    summary_file = output / "family-results" / "summary.json"
    summary_file.unlink(missing_ok=True)
    environment = dict(os.environ)
    environment.update(
        {
            "TRTMC_GPU_SCOPE": "families",
            "TRTMC_GPU_FAMILIES": json.dumps([family]),
            "TRTMC_GPU_DIRECT_FAMILIES": json.dumps([family]),
            "TRTMC_GPU_ADDED_FAMILIES": "[]",
            "CMAKE_CUDA_ARCHITECTURES": "89",
            "TRTMC_GPU_RESULTS_DIR": str(summary_file.parent),
        }
    )
    command = [
        str(stage_python),
        "-I",
        str(SOURCE / "tools" / "community_gpu_ci.py"),
        "--repository",
        str(repository),
        "--containers",
        "--require-family-coverage",
        "--dependencies-prepared",
        "--image",
        candidate["local_image"],
    ]
    if token_file is not None:
        command += ["--checkpoint-token-file", str(token_file)]
    subprocess.run(command, env=environment, check=True)
    if not summary_file.is_file() or summary_file.stat().st_size > 1024 * 1024:
        raise RuntimeError("The original family E2E has no bounded coverage receipt")
    summary = _candidate_json(summary_file.read_bytes())
    rows = summary.get("families")
    if (
        summary.get("complete") is not True
        or summary.get("passed") is not True
        or not isinstance(rows, list)
        or len(rows) != 1
        or not isinstance(rows[0], dict)
        or rows[0].get("family") != family
        or rows[0].get("status") != "passed"
        or not isinstance(rows[0].get("cases"), dict)
        or not rows[0]["cases"]
        or not isinstance(rows[0].get("requested_cases"), list)
        or any(not isinstance(name, str) for name in rows[0]["requested_cases"])
        or any(value != "passed" for value in rows[0]["cases"].values())
        or set(rows[0].get("requested_cases", [])) != set(rows[0]["cases"])
        or rows[0].get("deferred_cases", [])
    ):
        raise RuntimeError("The original family E2E coverage is incomplete")
    candidate["cases"] = rows[0]["cases"]
    candidate["family_e2e_passed"] = True
    save(output / "candidate.json", candidate)


def export_qualification(output: Path, auth_directory: Path) -> None:
    """Copy only nonsecret GPU evidence for the owner/backstop cleanup handoff."""
    candidate = _candidate_json((output / "candidate.json").read_bytes())
    if (
        candidate.get("native_byok_passed") is not True
        or candidate.get("family_e2e_passed") is not True
    ):
        raise RuntimeError("Export requires native ABI and original family E2E qualification")
    fields = {
        "schema_version",
        "family",
        "digest",
        "source_sha",
        "environment_source_sha",
        "model_source_sha",
        "producer_source_sha",
        "inputs",
        "platform",
        "registry_visibility",
        "abi",
        "cases",
        "base_dockerfile_sha256",
        "base_requirements_sha256",
        "family_requirements_sha256",
        "dependency_recipe_sha256",
        "dependency_build_helper_sha256",
        "dependency_constraints_sha256",
        "base_environment_lock_sha256",
        "family_environment_lock_sha256",
        "environment_recorder_sha256",
        "base_environment_receipt_sha256",
        "family_environment_receipt_sha256",
        "resolved_dependencies_sha256",
        "native_byok_passed",
        "family_e2e_passed",
    }
    receipt = {field: value for field, value in candidate.items() if field in fields}
    receipt.update(cleanup_confirmed=False, admitted=False)
    content = json.dumps(receipt)
    if "ghcr.io" in content or "://" in content:
        raise RuntimeError("The public GPU qualification receipt contains nonpublic coordinates")
    path = auth_directory / "qualification.json"
    owner = auth_directory.stat()
    save(path, receipt)
    path.chmod(0o600)
    os.chown(path, owner.st_uid, owner.st_gid)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """Keep the short-lived authorization header on its original API origin."""

    def redirect_request(self, request, response, code, message, headers, new_url):
        return None


def check_registry_access(family: str, digest: str, prefix: str, token: str, username: str) -> dict:
    """Require anonymous and public-repository credentials to be denied, without admission."""
    family = validated_family(family)
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise RuntimeError("Access checking requires an immutable digest")
    if not re.fullmatch(
        r"ghcr\.io/[a-z0-9]+(?:[._-][a-z0-9]+)*(?:/[a-z0-9]+(?:[._-][a-z0-9]+)*)*", prefix
    ):
        raise RuntimeError("Access checking requires a protected GHCR registry prefix")
    if not token:
        raise RuntimeError("The public repository token is unavailable")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]*(?:\[bot\])?", username):
        raise RuntimeError("The public workflow registry username is unavailable")
    path = f"{prefix.removeprefix('ghcr.io/')}/{family}"
    token_url = "https://ghcr.io/token?" + urllib.parse.urlencode(
        {"service": "ghcr.io", "scope": f"repository:{path}:pull"}
    )
    manifest_url = f"https://ghcr.io/v2/{path}/manifests/{digest}"

    def request(url, headers, *, method="GET"):
        try:
            req = urllib.request.Request(url, headers=headers, method=method)
            with urllib.request.build_opener(NoRedirect()).open(req, timeout=15) as response:
                body = response.read(65537) if method == "GET" else b""
                if len(body) > 65536:
                    return 0, None
                return response.status, body
        except urllib.error.HTTPError as error:
            return error.code, None
        except (OSError, ValueError):
            return 0, None

    def probe(credential):
        headers = {}
        if credential:
            encoded = base64.b64encode(f"{username}:{credential}".encode()).decode()
            headers["Authorization"] = f"Basic {encoded}"
        status, body = request(token_url, headers)
        if status == 200:
            try:
                value = json.loads(body)
                bearer = value.get("token", value.get("access_token"))
                if not isinstance(bearer, str) or not bearer or len(bearer) > 16384:
                    raise ValueError("invalid registry token")
            except (ValueError, TypeError, AttributeError, RecursionError):
                return {"status": "unknown", "http_status": 200}
            status, _ = request(
                manifest_url,
                {
                    "Authorization": f"Bearer {bearer}",
                    "Accept": ", ".join(
                        (
                            "application/vnd.oci.image.index.v1+json",
                            "application/vnd.oci.image.manifest.v1+json",
                            "application/vnd.docker.distribution.manifest.v2+json",
                            "application/vnd.docker.distribution.manifest.list.v2+json",
                        )
                    ),
                },
                method="HEAD",
            )
        return {
            "status": "denied"
            if status in {401, 403, 404}
            else "accessible"
            if 200 <= status < 300
            else "unknown",
            "http_status": status or None,
        }

    receipt = {
        "schema_version": 1,
        "family": family,
        "digest": digest,
        "anonymous": probe(None),
        "public_repository_token": probe(token),
    }
    receipt["denied"] = all(
        receipt[actor]["status"] == "denied" for actor in ("anonymous", "public_repository_token")
    )
    return receipt


def require_private_package(reference: str, token: str) -> None:
    package = reference.removeprefix("ghcr.io/nvidia/").split(":", 1)[0]
    request = urllib.request.Request(
        "https://api.github.com/orgs/NVIDIA/packages/container/"
        + urllib.parse.quote(package, safe=""),
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
    )
    try:
        with urllib.request.build_opener(NoRedirect()).open(request, timeout=30) as response:
            if getattr(response, "status", None) != 200:
                raise ValueError("package lookup must return HTTP 200")
            package_info = json.load(response)
    except (OSError, ValueError):
        raise RuntimeError(
            "Private package visibility could not be verified. Bootstrap a known-private "
            "package and grant this workflow authenticated access before publishing full images."
        ) from None
    if not isinstance(package_info, dict):
        raise RuntimeError("Private package visibility could not be verified")
    if package_info.get("visibility") != "private":
        raise RuntimeError("The dependency producer must not publish into a public package")


def audit_package(family: str, digest: str, token: str) -> dict:
    """Read bounded package evidence without publishing or admitting an image."""
    if not isinstance(family, str) or not isinstance(digest, str):
        raise RuntimeError("Package audit requires one valid family and immutable digest")
    family = validated_family(family)
    if len(family) > 63 or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise RuntimeError("Package audit requires one valid family and immutable digest")
    if not isinstance(token, str) or not token:
        raise RuntimeError("Package audit credentials are unavailable")
    package = f"tensorrt-model-connect-community/{family}"
    endpoint = "https://api.github.com/orgs/NVIDIA/packages/container/" + urllib.parse.quote(
        package, safe=""
    )

    def get(suffix=""):
        request = urllib.request.Request(
            endpoint + suffix,
            method="GET",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        try:
            with urllib.request.build_opener(NoRedirect()).open(request, timeout=30) as response:
                payload = response.read(1024 * 1024 + 1)
            if len(payload) > 1024 * 1024:
                raise ValueError("oversized package response")
            return json.loads(payload)
        except (OSError, ValueError):
            raise RuntimeError("Package audit API evidence is unavailable or invalid") from None

    info = get()
    if not isinstance(info, dict):
        raise RuntimeError("Package audit did not receive an object")
    repository = info.get("repository")
    visibility = info.get("visibility")
    if visibility is not None and not isinstance(visibility, (str, bool, int, float)):
        raise RuntimeError("Package audit visibility is not a scalar")

    def identity(value):
        return value if type(value) is int and value > 0 else None

    def text(value):
        return value if isinstance(value, str) and len(value) <= 256 else None

    record = {
        "package_name": package,
        "visibility": visibility,
        "id": identity(info.get("id")),
        "repository": (
            {"id": identity(repository.get("id")), "full_name": text(repository.get("full_name"))}
            if isinstance(repository, dict)
            else None
        ),
        "root_keys": sorted(info),
        "matching_versions": [],
    }
    for page in range(1, 11):
        versions = get(f"/versions?per_page=100&page={page}")
        if not isinstance(versions, list) or any(not isinstance(row, dict) for row in versions):
            raise RuntimeError("Package audit versions are not a complete array")
        for version in versions:
            if version.get("name") != digest:
                continue
            metadata = version.get("metadata", {})
            container = metadata.get("container", {}) if isinstance(metadata, dict) else {}
            tags = container.get("tags", []) if isinstance(container, dict) else []
            if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
                raise RuntimeError("Package audit version tags are invalid")
            record["matching_versions"].append(
                {
                    "id": identity(version.get("id")),
                    "name": digest,
                    "created_at": text(version.get("created_at")),
                    "updated_at": text(version.get("updated_at")),
                    "tags": tags,
                }
            )
        if record["matching_versions"] or len(versions) < 100:
            return record
    raise RuntimeError("Package audit reached its bounded version inventory limit")


def publish(output: Path, registry: str, username: str, token_file: Path, *, family: str) -> None:
    try:
        family = validated_family(family)
        candidate = json.loads((output / "candidate.json").read_text())
        if candidate.get("family") != family:
            raise RuntimeError("The candidate belongs to a different family")
        if (
            candidate.get("native_byok_passed") is not True
            or candidate.get("family_e2e_passed") is not True
        ):
            raise RuntimeError(
                "Publishing requires native ABI and unchanged family GPU qualification"
            )
        if registry != "ghcr.io/nvidia/tensorrt-model-connect-community":
            raise RuntimeError("Publishing is restricted to the project registry namespace")
        with tempfile.TemporaryDirectory(prefix="trtmc-registry-auth-") as directory:
            config = ["docker", "--config", directory]
            token = token_file.read_text(encoding="utf-8").strip()
            if not token:
                raise RuntimeError("The short-lived registry credential is empty")
            require_private_package(f"{registry}/{family}", token)
            run(
                [*config, "login", "ghcr.io", "--username", username, "--password-stdin"],
                capture=True,
                stdin=token,
            )
            token_file.unlink(missing_ok=True)
            key = candidate["resolved_dependencies_sha256"][:24]
            tag = f"{registry}/{family}:{candidate['source_sha'][:16]}-{key}"
            run([*config, "tag", candidate["local_image"], tag])
            run([*config, "push", tag])
            require_private_package(tag, token)
            values = json.loads(
                run(
                    [*config, "image", "inspect", "--format", "{{json .RepoDigests}}", tag],
                    capture=True,
                )
            )
            matches = [
                value for value in values if value.startswith(f"{registry}/{family}@sha256:")
            ]
            if len(matches) != 1 or not re.fullmatch(r".+@sha256:[0-9a-f]{64}", matches[0]):
                raise RuntimeError("Publishing did not establish one immutable registry digest")
            candidate.pop("base_image", None)
            candidate.update(
                image=matches[0],
                registry_visibility="private",
            )
            save(output / "published-candidate.json", candidate)
            # Preserve private root proof while allowing the SSH user to copy
            # only this nonsecret receipt from its existing private auth directory.
            receipt = token_file.parent / "published-candidate.json"
            owner = token_file.parent.stat()
            save(receipt, candidate)
            receipt.chmod(0o600)
            os.chown(receipt, owner.st_uid, owner.st_gid)
    finally:
        token_file.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=(
            "build",
            "qualify",
            "publish",
            "audit",
            "access-check",
            "prepare-candidate",
            "export-qualification",
        ),
    )
    parser.add_argument("--family", type=validated_family, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository", type=Path)
    parser.add_argument("--stage-python", type=Path)
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--auth-file", type=Path)
    parser.add_argument("--auth-directory", type=Path)
    parser.add_argument("--registry", default="ghcr.io/nvidia/tensorrt-model-connect-community")
    parser.add_argument("--username")
    parser.add_argument("--digest")
    args = parser.parse_args()
    if args.phase == "prepare-candidate":
        if args.repository is None or args.auth_file is None or args.digest is None:
            parser.error(
                "Candidate preparation requires protected model source, private auth and digest"
            )
        prepare_candidate(
            args.output,
            args.repository,
            args.auth_file,
            family=args.family,
            candidate_digest=args.digest,
        )
    elif args.phase == "export-qualification":
        if args.auth_directory is None:
            parser.error("Export requires the private SSH-owned handoff directory")
        export_qualification(args.output, args.auth_directory)
    elif args.phase == "access-check":
        if args.digest is None:
            parser.error("Access checking requires an immutable digest")
        record = check_registry_access(
            args.family,
            args.digest,
            os.environ.get("TRTMC_COMMUNITY_REGISTRY", ""),
            os.environ.get("GH_TOKEN", ""),
            os.environ.get("GITHUB_ACTOR", ""),
        )
        save(args.output / "dependency-image-access-check.json", record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if not record["denied"]:
            raise RuntimeError("Private registry denial was not established; no admission")
    elif args.phase == "audit":
        if args.digest is None:
            parser.error("Package audit requires an immutable digest")
        record = audit_package(args.family, args.digest, os.environ.get("GH_TOKEN", ""))
        save(args.output / "dependency-image-audit.json", record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if record["visibility"] != "private":
            raise RuntimeError("Package audit did not establish private visibility; no admission")
    elif args.phase == "build":
        if args.repository is None:
            parser.error("Building requires the immutable protected-main repository")
        build(args.output, args.repository, family=args.family)
    elif args.phase == "qualify":
        if args.stage_python is None or args.repository is None:
            parser.error("Qualification requires the trusted staging interpreter")
        qualify(
            args.output, args.stage_python, args.token_file, args.repository, family=args.family
        )
    else:
        if args.token_file is None or args.username is None:
            parser.error("Publishing requires a private token file and registry username")
        publish(args.output, args.registry, args.username, args.token_file, family=args.family)


if __name__ == "__main__":
    main()

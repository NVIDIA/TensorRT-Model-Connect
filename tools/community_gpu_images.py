#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare a shared Community base and only the requested owner's dependency layer.

This trusted preparation module never imports code from the tested checkout.
Registry authentication is removed before a Dockerfile or image can execute.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import signal
import stat
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request

BASE_INPUTS = (
    "Dockerfile.dev.x86-gpu",
    "requirements/community-ci.txt",
    "requirements/image-environment.py",
    "requirements/community-gpu-linux-amd64.lock",
    "requirements/community-gpu-linux-amd64.json",
)
PLATFORM = "linux/amd64"
IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")
FAMILY = re.compile(r"[a-z][a-z0-9_]*")
MAX_FILE = 1024 * 1024
MAX_CONTEXT = 16 * 1024 * 1024
FAMILY_PREPARATION_SECONDS = 7200
BASE_KEY_LABEL = "org.trtmc.community.base.input-key"
BASE_INPUTS_LABEL = "org.trtmc.community.base.inputs"
BASE_KIND_LABEL = "org.trtmc.community.base.kind"


class ImagePreparationError(RuntimeError):
    """The environment is not ready to hand over to the PR entry point."""


def regular_bytes(path: Path, *, root: Path | None = None, limit: int = MAX_FILE) -> bytes:
    if root is not None:
        try:
            path.resolve().relative_to(root.resolve())
        except ValueError:
            raise ImagePreparationError("Dependency input leaves its declared directory") from None
        relative = path.relative_to(root)
        cursor = root
        for part in relative.parts:
            cursor /= part
            if cursor.is_symlink():
                raise ImagePreparationError("Dependency inputs cannot be symlinks")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ImagePreparationError("Dependency input is not a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            content = stream.read(limit + 1)
        if len(content) > limit:
            raise ImagePreparationError("Dependency input exceeds its size limit")
        return content
    finally:
        os.close(descriptor)


def object_json(raw: bytes | str) -> dict:
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ImagePreparationError("Duplicate environment evidence key")
            value[key] = item
        return value

    try:
        value = json.loads(raw, object_pairs_hook=unique)
    except (ValueError, UnicodeError, RecursionError):
        raise ImagePreparationError("Invalid environment evidence") from None
    if not isinstance(value, dict):
        raise ImagePreparationError("Environment evidence must be an object")
    return value


def base_key(inputs: dict[str, str]) -> str:
    if set(inputs) != set(BASE_INPUTS) or any(
        not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
        for value in inputs.values()
    ):
        raise ImagePreparationError("Shared base has incomplete public inputs")
    return hashlib.sha256(
        json.dumps(
            {"platform": PLATFORM, "inputs": inputs}, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def validate_base(base: dict, prefix: str, expected_inputs: dict[str, str] | None = None) -> dict:
    if not isinstance(prefix, str) or not re.fullmatch(
        r"ghcr\.io/[a-z0-9]+(?:[._-][a-z0-9]+)*(?:/[a-z0-9]+(?:[._-][a-z0-9]+)*)*", prefix
    ):
        raise ImagePreparationError("Invalid protected registry prefix")
    if (
        not isinstance(base, dict)
        or type(base.get("schema_version")) is not int
        or base["schema_version"] != 1
        or base.get("kind") != "community-base"
        or base.get("platform") != PLATFORM
        or base.get("registry_visibility") != "private"
        or base.get("cpu_environment_verified") is not True
        or not isinstance(base.get("image"), str)
        or not re.fullmatch(re.escape(prefix + "/base@") + r"sha256:[0-9a-f]{64}", base["image"])
        or not isinstance(base.get("environment_source_sha"), str)
        or not re.fullmatch(r"[0-9a-f]{40}", base["environment_source_sha"])
        or not isinstance(base.get("inputs"), dict)
    ):
        raise ImagePreparationError("Shared base is not a verified private immutable environment")
    if base.get("input_key") != base_key(base["inputs"]):
        raise ImagePreparationError("Shared base input identity is inconsistent")
    if expected_inputs is not None and base["inputs"] != expected_inputs:
        raise ImagePreparationError("Shared base does not match the trusted CI environment recipe")
    return base


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        return None


def require_private_package(prefix: str, token: str, expected_id: int | None) -> None:
    owner, _, name = prefix.removeprefix("ghcr.io/").partition("/")
    package = (name + "/" if name else "") + "base"
    request = urllib.request.Request(
        "https://api.github.com/orgs/"
        + owner
        + "/packages/container/"
        + urllib.parse.quote(package, safe=""),
        headers={"Authorization": "Bearer " + token, "Accept": "application/vnd.github+json"},
    )
    try:
        with urllib.request.build_opener(NoRedirect()).open(request, timeout=30) as response:
            raw = response.read(MAX_FILE + 1)
            if response.status != 200 or len(raw) > MAX_FILE:
                raise ImagePreparationError("Private base metadata is unavailable")
        metadata = object_json(raw)
    except (OSError, ValueError):
        raise ImagePreparationError("Private base access could not be verified") from None
    if (
        metadata.get("name") != package
        or metadata.get("visibility") != "private"
        or metadata.get("package_type") != "container"
        or (expected_id is not None and metadata.get("id") != expected_id)
    ):
        raise ImagePreparationError("Shared base package identity or visibility changed")


def docker(
    args: list[str],
    *,
    timeout: float = 180,
    capture: bool = False,
    stdin: str | None = None,
    config: str | None = None,
    private: bool = False,
) -> str:
    with tempfile.TemporaryDirectory(prefix="trtmc-image-client-") as empty:
        command = ["docker", "--config", config or empty, *args]
        env = {
            key: os.environ[key] for key in ("PATH", "LANG", "LC_ALL", "TZ") if key in os.environ
        }
        env.update(DOCKER_CONFIG=config or empty, DOCKER_BUILDKIT="1")
        try:
            if not (capture or private or stdin is not None):
                subprocess.run(command, env=env, check=True, timeout=timeout)
                return ""
            return _captured(command, env, stdin, timeout, capture).strip()
        except (OSError, UnicodeError, subprocess.SubprocessError):
            raise ImagePreparationError(
                "Private image operation failed"
                if private or stdin is not None
                else "Dependency image preparation failed; see build output"
            ) from None


def _captured(
    command: list[str], env: dict[str, str], stdin: str | None, timeout: float, capture: bool
) -> str:
    payload = stdin.encode() if stdin is not None else b""
    if len(payload) > MAX_FILE:
        raise ImagePreparationError("Credential input exceeds the size limit")
    output = bytearray()
    deadline = time.monotonic() + timeout
    process = subprocess.Popen(
        command,
        env=env,
        stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        with selectors.DefaultSelector() as selector:
            for stream, kind in ((process.stdout, "out"), (process.stderr, "err")):
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_READ, kind)
            if process.stdin is not None:
                os.set_blocking(process.stdin.fileno(), False)
                selector.register(process.stdin, selectors.EVENT_WRITE, "in")
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout)
                for key, _ in selector.select(remaining):
                    if key.data == "in":
                        try:
                            count = os.write(key.fd, payload[:16384])
                            payload = payload[count:]
                        except BrokenPipeError:
                            payload = b""
                        except BlockingIOError:
                            continue
                        if not payload:
                            selector.unregister(key.fileobj)
                            key.fileobj.close()
                        continue
                    amount = (
                        min(16384, MAX_FILE - len(output) + 1)
                        if capture and key.data == "out"
                        else 16384
                    )
                    try:
                        chunk = os.read(key.fd, amount)
                    except BlockingIOError:
                        continue
                    if not chunk:
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
                    elif capture and key.data == "out":
                        if len(output) + len(chunk) > MAX_FILE:
                            raise ImagePreparationError("Image metadata exceeds its size limit")
                        output.extend(chunk)
            if process.wait(timeout=max(0.01, deadline - time.monotonic())):
                raise subprocess.CalledProcessError(process.returncode, command)
        return output.decode()
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=0.25)
        except subprocess.TimeoutExpired:
            pass
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                stream.close()


def inspect(image: str, *, config: str | None = None, private: bool = False) -> dict:
    try:
        entries = json.loads(
            docker(["image", "inspect", image], capture=True, config=config, private=private)
        )
        value = entries[0]
        if len(entries) != 1 or not IMAGE_ID.fullmatch(value.get("Id", "")):
            raise ValueError
        if value.get("Architecture") != "amd64" or value.get("Os") != "linux":
            raise ValueError
        return value
    except (ValueError, TypeError, KeyError, IndexError):
        raise ImagePreparationError("Prepared image is not one immutable Linux x86 image") from None


def pull_shared_base(catalog: dict, token_path: Path, username: str, tag: str) -> str:
    """Pull with protected credentials, then erase authentication before any image execution."""
    try:
        prefix = catalog.get("registry_prefix", "")
        base = validate_base(catalog.get("base"), prefix)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]*(?:\[bot\])?", username):
            raise ImagePreparationError("Private image reader username is missing")
        token = regular_bytes(token_path).decode().strip()
        if not token:
            raise ImagePreparationError("Private image reader credential is missing")
        require_private_package(prefix, token, base.get("package_id"))
        with tempfile.TemporaryDirectory(prefix="trtmc-base-auth-") as auth:
            docker(
                ["login", "ghcr.io", "--username", username, "--password-stdin"],
                stdin=token,
                config=auth,
                private=True,
                timeout=30,
            )
            docker(
                ["pull", "--platform", PLATFORM, base["image"]],
                config=auth,
                private=True,
                timeout=900,
            )
            image = inspect(base["image"], config=auth, private=True)
            labels = image.get("Config", {}).get("Labels", {}) or {}
            if (
                labels.get(BASE_KEY_LABEL) != base["input_key"]
                or labels.get(BASE_KIND_LABEL) != "community-base"
                or object_json(labels.get(BASE_INPUTS_LABEL, "{}")) != base["inputs"]
                or base["image"] not in image.get("RepoDigests", [])
            ):
                raise ImagePreparationError(
                    "Pulled base does not match the public environment inputs"
                )
        docker(["tag", image["Id"], tag])
        return image["Id"]
    finally:
        token_path.unlink(missing_ok=True)


def _family_inputs(repository: Path, family: str) -> dict[str, bytes]:
    if not FAMILY.fullmatch(family):
        raise ImagePreparationError("Invalid family name")
    owner = repository / "families" / family
    if owner.is_symlink() or not owner.is_dir():
        raise ImagePreparationError("Family dependency owner is unavailable")
    files = {}
    requirements = owner / "requirements.txt"
    if requirements.exists() or requirements.is_symlink():
        files["requirements.txt"] = regular_bytes(requirements, root=repository)
    ci = owner / "ci"
    for name in (
        "constraints-linux-amd64.txt",
        "environment-linux-amd64.lock",
        "environment-linux-amd64.json",
    ):
        path = ci / name
        if path.exists() or path.is_symlink():
            files[name] = regular_bytes(path, root=repository)
    if (ci / "Dockerfile.dependencies").exists() or (ci / "Dockerfile.dependencies").is_symlink():
        for path in sorted(ci.rglob("*")):
            if path.is_symlink():
                raise ImagePreparationError("Family dependency contexts cannot contain symlinks")
            if path.is_file():
                files[str(path.relative_to(ci))] = regular_bytes(path, root=repository)
    if sum(map(len, files.values())) > MAX_CONTEXT:
        raise ImagePreparationError("Family dependency context exceeds the size limit")
    return files


def ensure_family_image(repository: Path, family: str, base_image_id: str, deadline: float) -> str:
    """Build only one selected family's declared layer on the already pulled common base."""
    if not IMAGE_ID.fullmatch(base_image_id):
        raise ImagePreparationError("Family preparation requires an immutable local base ID")
    files = _family_inputs(repository, family)
    if not files:
        return base_image_id
    base = inspect(base_image_id)
    if base["Id"] != base_image_id:
        raise ImagePreparationError("Local shared base identity changed")
    key = hashlib.sha256(
        json.dumps(
            {
                "base_image_id": base_image_id,
                "family": family,
                "inputs": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    tag = "trtmc-family-" + family + ":" + key
    frozen_base = "trtmc-community-base:" + base_image_id.removeprefix("sha256:")
    docker(["tag", base_image_id, frozen_base])
    remaining = min(FAMILY_PREPARATION_SECONDS, deadline - time.monotonic())
    if remaining <= 0:
        raise ImagePreparationError("Family dependency preparation budget exhausted")
    with tempfile.TemporaryDirectory(prefix="trtmc-family-context-") as temporary:
        context = Path(temporary)
        for name, data in files.items():
            path = context / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        recipe = context / "Dockerfile.dependencies"
        if not recipe.is_file():
            arguments = ""
            copies = ""
            for name, option in (
                ("constraints-linux-amd64.txt", "--constraint"),
                ("environment-linux-amd64.lock", "--requirement"),
            ):
                if name in files:
                    copies += f"COPY {name} /opt/trtmc-ci/{name}\n"
                    arguments += f" {option} /opt/trtmc-ci/{name}"
            recipe.write_text(
                "ARG BASE_IMAGE\nFROM ${BASE_IMAGE}\n"
                "COPY requirements.txt /opt/trtmc-ci/family-requirements.txt\n"
                + copies
                + "RUN python -m pip install --disable-pip-version-check --no-cache-dir "
                "--no-build-isolation -r /opt/trtmc-ci/family-requirements.txt "
                + arguments
                + " && python -m pip check\n"
            )
        # The existing public family recipe consumes the public provenance file.
        # It contains no private image reference, token or OCI source override.
        revision = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={repository}",
                "-C",
                str(repository),
                "rev-parse",
                "HEAD",
            ],
            check=True,
            text=True,
            capture_output=True,
            timeout=30,
        ).stdout.strip()
        base_labels = base.get("Config", {}).get("Labels", {}) or {}
        (context / "build-inputs.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "community-family-preparation",
                    "family": family,
                    "model_source_sha": revision,
                    "base_image_id": base_image_id,
                    "base_environment_source_sha": base_labels.get(
                        "org.opencontainers.image.revision"
                    ),
                    "base_input_key": base_labels.get(BASE_KEY_LABEL),
                    "input_key": key,
                    "inputs": {
                        name: hashlib.sha256(data).hexdigest() for name, data in files.items()
                    },
                    "native_byok_passed": False,
                    "family_e2e_passed": False,
                    "qualification": "Prepared dependencies; the PR entry point has not run",
                },
                sort_keys=True,
            )
            + "\n"
        )
        print("Preparing declared dependency layer for " + family, flush=True)
        docker(
            [
                "build",
                "--platform",
                PLATFORM,
                "--file",
                str(recipe),
                "--build-arg",
                "BASE_IMAGE=" + frozen_base,
                "--label",
                "org.trtmc.community.family.input-key=" + key,
                "--tag",
                tag,
                str(context),
            ],
            timeout=remaining,
        )
    prepared = inspect(tag)
    if (prepared.get("Config", {}).get("Labels", {}) or {}).get(
        "org.trtmc.community.family.input-key"
    ) != key:
        raise ImagePreparationError("Family image identity changed during preparation")
    parent_layers = base.get("RootFS", {}).get("Layers")
    child_layers = prepared.get("RootFS", {}).get("Layers")
    if (
        not isinstance(parent_layers, list)
        or not isinstance(child_layers, list)
        or child_layers[: len(parent_layers)] != parent_layers
    ):
        raise ImagePreparationError("Family image does not extend the shared base layers")
    docker(
        [
            "run",
            "--rm",
            "--network",
            "none",
            "--entrypoint",
            "/opt/venv/bin/python",
            prepared["Id"],
            "-m",
            "pip",
            "check",
        ],
        timeout=min(180, max(1, deadline - time.monotonic())),
    )
    return prepared["Id"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pull-base", action="store_true", required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--username", required=True)
    parser.add_argument("--tag", default="trtmc-quickstart-gpu")
    args = parser.parse_args()
    try:
        identity = pull_shared_base(
            object_json(regular_bytes(args.catalog)), args.token_file, args.username, args.tag
        )
    except (ImagePreparationError, OSError, ValueError) as error:
        raise SystemExit("Shared environment preparation failed: " + str(error)) from None
    print(identity)


if __name__ == "__main__":
    main()

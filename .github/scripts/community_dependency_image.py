# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build and qualify a family dependency layer on a trusted x86 L4 host."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
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


def run(command: list[str], *, capture: bool = False, stdin: str | None = None) -> str:
    try:
        result = subprocess.run(
            command, check=True, text=True, capture_output=capture or stdin is not None, input=stdin
        )
    except subprocess.CalledProcessError as error:
        if stdin is not None:
            raise RuntimeError("Credential command failed; captured output is suppressed") from None
        if capture:
            for label, stream in (("stdout", error.stdout), ("stderr", error.stderr)):
                if stream:
                    print(f"Captured {label} tail:\n{stream[-16384:]}", file=sys.stderr, flush=True)
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
    environment = dict(os.environ)
    environment.update(
        {
            "TRTMC_GPU_SCOPE": "families",
            "TRTMC_GPU_FAMILIES": json.dumps([family]),
            "TRTMC_GPU_DIRECT_FAMILIES": json.dumps([family]),
            "TRTMC_GPU_ADDED_FAMILIES": "[]",
            "CMAKE_CUDA_ARCHITECTURES": "89",
        }
    )
    command = [
        str(stage_python),
        "-I",
        str(SOURCE / "tools" / "community_gpu_ci.py"),
        "--repository",
        str(repository),
        "--containers",
        "--image",
        candidate["local_image"],
    ]
    if token_file is not None:
        command += ["--checkpoint-token-file", str(token_file)]
    subprocess.run(command, env=environment, check=True)
    candidate["family_e2e_passed"] = True
    save(output / "candidate.json", candidate)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """Keep the short-lived authorization header on its original API origin."""

    def redirect_request(self, request, response, code, message, headers, new_url):
        return None


def require_private_package(reference: str, token: str, *, allow_missing: bool) -> None:
    package = reference.removeprefix("ghcr.io/nvidia/").split(":", 1)[0]
    request = urllib.request.Request(
        "https://api.github.com/orgs/NVIDIA/packages/container/"
        + urllib.parse.quote(package, safe=""),
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
    )
    try:
        with urllib.request.build_opener(NoRedirect()).open(request, timeout=30) as response:
            package_info = json.load(response)
    except urllib.error.HTTPError as error:
        if error.code == 404 and allow_missing:
            return
        raise RuntimeError("Private package visibility could not be verified") from None
    except (OSError, ValueError):
        raise RuntimeError("Private package visibility could not be verified") from None
    if not isinstance(package_info, dict):
        raise RuntimeError("Private package visibility could not be verified")
    if package_info.get("visibility") != "private":
        raise RuntimeError("The dependency producer must not publish into a public package")


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
            require_private_package(f"{registry}/{family}", token, allow_missing=True)
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
            require_private_package(tag, token, allow_missing=False)
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
    finally:
        token_file.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("build", "qualify", "publish"))
    parser.add_argument("--family", type=validated_family, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository", type=Path)
    parser.add_argument("--stage-python", type=Path)
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--registry", default="ghcr.io/nvidia/tensorrt-model-connect-community")
    parser.add_argument("--username")
    args = parser.parse_args()
    if args.phase == "build":
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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thin family-owned adapter to the pinned Edge direct-builder API."""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
import shutil
import shlex
import subprocess
import sys

from tensorrt_model_connect.build import cmake_prefixes, detect_local_platform

from ..checkpoint_mapper import CheckpointWeights

# One C++ SDK per installation; the manifest is its build-time receipt.
_RELEASES = {
    "0.11.0": (1, "revision", "95515c2f87fba8982db5a519f9022277667b3cc9", "experimental.builder.cli"),
    "0.12.0": (2, "release_revision", "513dcd18291e587c80928911a480761352f3c2d8", "tensorrt_edgellm.builder.cli"),
}


def local_target() -> dict:
    """Return the executing worker identity supplied by generic build mechanics."""
    return detect_local_platform()


def _package_prefixes() -> list[Path]:
    """Use the active wheel's SDK with its native runner, or standalone prefixes."""
    try:
        package = distribution("tensorrt-model-connect")
    except PackageNotFoundError:
        return cmake_prefixes()
    module_dir = Path(package.locate_file("tensorrt_model_connect"))
    receipt = module_dir / "share/trtmc/edge-llm.json"
    if (module_dir / "bin/trtmc").is_file() and (receipt.exists() or receipt.is_symlink()):
        return [module_dir]
    return cmake_prefixes()


def package_present() -> bool:
    """Absence of the optional SDK is a non-match, not a failed Edge build."""
    for prefix in _package_prefixes():
        manifest = prefix / "share/trtmc/edge-llm.json"
        if manifest.exists() or manifest.is_symlink():
            return True
    return False


def installed_package(target: dict) -> dict:
    """Resolve CMake installation via standard prefixes; never install anything.

    Args:
        target: Executing device and SDK identity.

    Returns:
        Validated package metadata with absolute Python and plugin paths.

    Raises:
        FileNotFoundError: No CMake installation or required artifact exists.
        ValueError: Pin, architecture, SDK or contained-path contract differs.
    """
    for prefix in _package_prefixes():
        manifest = prefix / "share/trtmc/edge-llm.json"
        if not manifest.is_file():
            continue
        package = json.loads(manifest.read_text(encoding="utf-8"))
        release = _RELEASES.get(package.get("version"))
        if release is None or package.get("schema_version") != release[0]:
            raise ValueError(f"Unsupported Edge package release or schema: {manifest}")
        revision_key, pinned = release[1:3]
        allowed = (pinned, None) if package["version"] == "0.12.0" else (pinned,)
        if revision_key not in package or package[revision_key] not in allowed:
            raise ValueError(f"Unsupported Edge package source revision: {manifest}")
        if package.get("arch") != target["arch"]:
            raise ValueError("Edge package version/architecture differs from executing worker")
        if target["sm"] not in package.get("architectures", []):
            raise ValueError("Edge package was not built for this local GPU")
        cuda_version = ".".join(str(package.get("cuda_version", "")).split(".")[:2])
        if (
            cuda_version != target["cuda_version"]
            or package.get("tensorrt_version") != target["tensorrt_version"]
        ):
            raise ValueError("Edge package CUDA/TensorRT differs from executing worker")
        for name in ("plugin", *(("python",) if package.get("version") == "0.11.0" else ())):
            relative = Path(package[name])
            path = (prefix / relative).resolve()
            if relative.is_absolute() or not path.is_relative_to(prefix.resolve()):
                raise ValueError(f"Edge package {name} must be contained in its installation")
            if not path.is_file():
                raise FileNotFoundError(f"Edge package {name} is missing: {path}")
            package[name] = str(path)
        return package
    raise FileNotFoundError(
        "Edge-LLM is not installed; enable the optional Edge-LLM CMake dependency "
        "and set CMAKE_PREFIX_PATH to its install prefix"
    )


def builder_python(package: dict, target: dict, requested: Path | None = None) -> str:
    """Select a verified builder interpreter for the chosen native SDK release.

    The native SDK remains required. This only discovers an interpreter; it
    never installs packages or substitutes the wheel's runtime for the C++ SDK.
    Only the legacy 0.11 SDK includes an interpreter to fall back to.
    """
    version = package["version"]
    module = _RELEASES[version][3]
    interpreter = sys.executable if requested is None else str(requested)
    pin_check = ("import transformers; assert transformers.__version__ == '5.17.0'; "
                 if version == "0.12.0" else "")
    try:
        if not Path(interpreter).is_absolute() or not Path(interpreter).is_file():
            raise ValueError("Edge builder Python must be an existing absolute interpreter path")
        probe = subprocess.run(
            [
                interpreter,
                "-I",
                "-c",
                "import json, tensorrt, tensorrt_edgellm; "
                f"{pin_check}from {module} import main; "
                "print(json.dumps([tensorrt_edgellm.__version__, tensorrt.__version__]))",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if json.loads(probe.stdout) != [version, target["tensorrt_version"]]:
            raise ValueError(
                f"Edge builder Python must match Edge {version} and the native TensorRT SDK"
            )
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        if requested is None and version == "0.11.0" and package.get("python"):
            return package["python"]
        raise ValueError(
            "Pass --edge-builder-python with an installed interpreter matching the selected Edge SDK"
        ) from error
    return interpreter


def prepare(request, target: dict, staging: Path, log_path: Path, capacity: int):
    """Build both reasoning components through the official Edge entry point.

    The checkpoint is read in place. Only indexed shards and root metadata are
    streamed into the final bundle, avoiding a second multi-gigabyte staging copy.
    The runtime passes the extracted checkpoint root to Edge's external-weight API.
    """
    package = installed_package(target)
    interpreter = builder_python(package, target, getattr(request, "edge_builder_python", None))
    version = package["version"]
    source = Path(request.model_dir).resolve()
    weights = CheckpointWeights(source)
    checkpoint_files = {}
    for name in sorted(set(weights.mapping.values())):
        relative = Path(name)
        if (relative.is_absolute() or relative.as_posix() != name
                or any(p in {".", ".."} for p in relative.parts)
                or "\\" in name or "\0" in name):
            raise ValueError("Cosmos3-Edge checkpoint shard path is not normalized")
        checkpoint_files[f"edge_llm/checkpoint/{name}"] = source / name
    for path in sorted(source.iterdir()):
        if path.is_file() and path.suffix in {".json", ".jinja", ".model"}:
            checkpoint_files[f"edge_llm/checkpoint/{path.name}"] = path
    engine = staging / "edge_llm/engine"
    command = [
        interpreter, "-I", "-c",
        f"from {_RELEASES[version][3]} import main; main()",
        "--model-dir", str(source), "--engine-dir", str(engine),
        "--components", "llm,visual", "--plugin-path", package["plugin"],
        "--dense", "fp16", "--max-input-len", str(capacity),
        "--max-kv-cache-capacity", str(capacity), "--max-batch-size", "1",
        "--min-image-tokens", "4", "--max-image-tokens", str(capacity),
        "--max-image-tokens-per-image", str(capacity),
        "--externalize-weights", "all",
    ]
    if request.verbose:
        command.append("--verbose")
    with log_path.open("a", encoding="utf-8") as log:
        log.write(shlex.join(command) + "\n")
        log.flush()
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, cwd=staging)
    for name in ("llm.engine", "config.json", "tokenizer.json", "tokenizer_config.json",
                 "chat_template.jinja", "visual/visual.engine", "visual/config.json"):
        path = engine / name
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"Edge builder did not produce required artifact: {name}")
    files = {}
    for path in sorted(engine.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Edge output must not contain symlinks: {path}")
        if path.is_file():
            files[path.relative_to(staging).as_posix()] = path
    files.update(checkpoint_files)
    return files, {
        "version": 1, "edge_release": version,
        "edge_revision": package[_RELEASES[version][1]], "target": target,
        "precision": "fp16", "execution_variant": "autoregressive",
        "max_sequence_length": capacity, "max_input_length": capacity,
        "max_batch_size": 1, "artifacts": list(files),
    }


def publish(request, writer, files: dict, marker: dict) -> None:
    """Publish only a complete offload bundle; publication failure never retries."""
    writer.set_header(family="cosmos3_edge", task=request.task, backend=request.backend)
    for name, path in files.items():
        with path.open("rb") as source, writer.open_section(name) as destination:
            shutil.copyfileobj(source, destination, length=1024 * 1024)
    writer.add_json("edge_llm.json", marker)

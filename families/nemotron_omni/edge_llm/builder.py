# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thin family-owned adapter to the pinned Edge direct-builder API."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import shlex
import subprocess
import sys

from tensorrt_model_connect.build import cmake_prefixes, detect_local_platform


EDGE_REVISION = "95515c2f87fba8982db5a519f9022277667b3cc9"


def local_target() -> dict:
    """Return the executing worker identity supplied by generic build mechanics."""
    return detect_local_platform()


def package_present() -> bool:
    """Absence of the optional SDK is a non-match, not a failed Edge build."""
    for prefix in cmake_prefixes():
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
    for prefix in cmake_prefixes():
        manifest = prefix / "share/trtmc/edge-llm.json"
        if not manifest.is_file():
            continue
        package = json.loads(manifest.read_text(encoding="utf-8"))
        if package.get("schema_version") != 1 or package.get("revision") != EDGE_REVISION:
            raise ValueError(f"Edge package has an unsupported revision/schema: {manifest}")
        if package.get("version") != "0.11.0" or package.get("arch") != target["arch"]:
            raise ValueError("Edge package version/architecture differs from executing worker")
        if target["sm"] not in package.get("architectures", []):
            raise ValueError("Edge package was not built for this local GPU")
        cuda_version = ".".join(str(package.get("cuda_version", "")).split(".")[:2])
        if (
            cuda_version != target["cuda_version"]
            or package.get("tensorrt_version") != target["tensorrt_version"]
        ):
            raise ValueError("Edge package CUDA/TensorRT differs from executing worker")
        for name in ("python", "plugin"):
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


def builder_python(package: dict, target: dict) -> str:
    """Use installed 0.11.0 Python tools when compatible, otherwise the SDK Python.

    The native SDK remains required. This only discovers an interpreter; it
    never installs packages or substitutes the wheel's runtime for the C++ SDK.
    An incompatible ambient install is ignored.
    """
    interpreter = sys.executable
    try:
        if not Path(interpreter).is_absolute() or not Path(interpreter).is_file():
            raise ValueError("Edge builder Python must be an existing absolute interpreter path")
        probe = subprocess.run(
            [
                interpreter,
                "-I",
                "-c",
                "import json, tensorrt, tensorrt_edgellm; "
                "from experimental.builder.cli import main; "
                "print(json.dumps([tensorrt_edgellm.__version__, tensorrt.__version__]))",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if json.loads(probe.stdout) != ["0.11.0", target["tensorrt_version"]]:
            raise ValueError(
                "Edge builder Python must match Edge 0.11.0 and the native TensorRT SDK"
            )
    except (OSError, ValueError, subprocess.SubprocessError):
        return package["python"]
    return interpreter


def prepare(request, target: dict, staging: Path, log_path: Path, capacity: int):
    """Build the text/image components through the official Edge entry point.

    The checkpoint is read in place. Only indexed shards and root metadata are
    streamed into the final bundle, avoiding a second multi-gigabyte staging copy.
    The runtime passes the extracted checkpoint root to Edge's external-weight API.
    """
    package = installed_package(target)
    source = Path(request.model_dir).resolve()
    index = json.loads((source / "model.safetensors.index.json").read_text())
    mapping = index["weight_map"]
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("Omni requires its indexed prepared checkpoint")
    checkpoint_files = {}
    for name in sorted(set(mapping.values())):
        relative = Path(name)
        if (
            relative.is_absolute()
            or relative.as_posix() != name
            or any(p in {".", ".."} for p in relative.parts)
            or "\\" in name
            or "\0" in name
        ):
            raise ValueError("Nemotron Omni checkpoint shard path is not normalized")
        # HF snapshots normally link indexed shards to the adjacent blobs cache.
        # Validate the bundle-relative name, not the source symlink destination.
        path = source / name
        if not path.is_file():
            raise ValueError("Omni checkpoint shard must be an existing file")
        checkpoint_files[f"edge_llm/checkpoint/{name}"] = path
    # Do not publish unrelated user notes or download/provenance sidecars.
    for name in (
        "config.json",
        "generation_config.json",
        "hf_quant_config.json",
        "model.safetensors.index.json",
        "preprocessor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    ):
        path = source / name
        if path.is_file():
            checkpoint_files[f"edge_llm/checkpoint/{name}"] = path
    engine = staging / "edge_llm/engine"
    command = [
        builder_python(package, target),
        "-I",
        "-c",
        "from experimental.builder.cli import main; main()",
        "--model-dir",
        str(source),
        "--engine-dir",
        str(engine),
        "--components",
        "llm,visual",
        "--plugin-path",
        package["plugin"],
        "--dense",
        "auto",
        "--max-input-len",
        str(capacity),
        "--max-kv-cache-capacity",
        str(capacity),
        "--max-batch-size",
        "1",
        "--min-image-tokens",
        "4",
        "--max-image-tokens",
        str(capacity),
        "--max-image-tokens-per-image",
        str(capacity),
    ]
    if request.verbose:
        command.append("--verbose")
    with log_path.open("a", encoding="utf-8") as log:
        log.write(shlex.join(command) + "\n")
        log.flush()
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, cwd=staging)
    for name in (
        "llm.engine",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "visual/visual.engine",
        "visual/config.json",
    ):
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
        "version": 1,
        "edge_revision": EDGE_REVISION,
        "target": target,
        "precision": "fp16",
        "execution_variant": "autoregressive",
        "max_sequence_length": capacity,
        "max_input_length": capacity,
        "max_batch_size": 1,
        "artifacts": list(files),
    }


def publish(request, writer, files: dict, marker: dict) -> None:
    """Publish only a complete offload bundle; publication failure never retries."""
    writer.set_header(family="nemotron_omni", task=request.task, backend=request.backend)
    for name, path in files.items():
        with path.open("rb") as source, writer.open_section(name) as destination:
            shutil.copyfileobj(source, destination, length=1024 * 1024)
    writer.add_json("edge_llm.json", marker)

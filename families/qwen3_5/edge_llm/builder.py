# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thin family-owned adapter to the pinned Edge direct-builder API."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import struct

from tensorrt_model_connect.build import cmake_prefixes, detect_local_platform

EDGE_REVISION = "e8b29522938901f6df19ebeedd4b69bc8edbcd97"


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
        if package.get("version") != "0.10.1" or package.get("arch") != target["arch"]:
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


def copy_draft_safetensors(source: Path, destination: Path) -> None:
    """Align the draft payload for pinned Edge mapped GPU reads, without changing tensors.

    Some official DFlash files omit safetensors JSON-header padding. Only trailing
    JSON whitespace and its length prefix change; descriptors, relative offsets,
    and the complete tensor payload are copied byte-for-byte with bounded memory.
    The original pinned checkpoint is never modified.
    """
    with source.open("rb") as incoming:
        prefix = incoming.read(8)
        if len(prefix) != 8:
            raise ValueError("Truncated DFlash safetensors header")
        length = struct.unpack("<Q", prefix)[0]
        if not 0 < length <= min(100 * 1024 * 1024, source.stat().st_size - 8):
            raise ValueError("Invalid DFlash safetensors header length")
        header = incoming.read(length)
        if len(header) != length or not isinstance(json.loads(header), dict):
            raise ValueError("Invalid DFlash safetensors header object")
        padded = ((length + 8 + 63) // 64) * 64 - 8
        with destination.open("wb") as outgoing:
            outgoing.write(struct.pack("<Q", padded))
            outgoing.write(header)
            outgoing.write(b" " * (padded - length))
            shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)


def prepare(
    request,
    raw: dict,
    target: dict,
    staging: Path,
    log_path: Path,
    *,
    draft_dir: Path | None = None,
) -> tuple[dict, dict]:
    """Map the request to Edge main(argv), returning complete unpublished assets.

    Edge owns model selection, configuration, conversion, graphs and engine
    composition. The family adapter only maps text-generation arguments and
    preserves the checkpoint needed by Edge external-weight APIs.

    Returns:
        (section-name to file mapping, runtime marker).

    Raises:
        Exception: Dependency, upstream build or artifact validation failed.
    """
    package = installed_package(target)
    checkpoint = staging / "edge_llm/checkpoint"
    checkpoint.mkdir(parents=True)
    for source in Path(request.model_dir).iterdir():
        if source.is_file() and (
            source.suffix in {".json", ".safetensors", ".model", ".jinja"}
            or source.name in {"merges.txt", "vocab.txt"}
        ):
            shutil.copy2(source, checkpoint / source.name)
    if not list(checkpoint.glob("*.safetensors")):
        raise ValueError("Edge direct builder requires a safetensors checkpoint")
    if draft_dir is not None:
        draft_checkpoint = checkpoint / "draft"
        draft_checkpoint.mkdir()
        for source in draft_dir.iterdir():
            if source.is_file() and source.suffix in {".json", ".safetensors"}:
                if source.suffix == ".safetensors":
                    copy_draft_safetensors(source, draft_checkpoint / source.name)
                else:
                    shutil.copy2(source, draft_checkpoint / source.name)
        if not list(draft_checkpoint.glob("*.safetensors")):
            raise ValueError("DFlash direct builder requires draft safetensors")
    config = raw.get("text_config", raw)
    limit = request.max_sequence_length or min(int(config["max_position_embeddings"]), 256)
    input_limit = min(limit, 1024) if draft_dir is not None else limit
    engine = staging / "edge_llm/engine"
    # Calling upstream main preserves its complete build/artifact orchestration.
    command = [
        package["python"],
        "-I",
        "-c",
        "from experimental.builder.cli import main; main()",
        "--model-dir",
        str(checkpoint),
        "--engine-dir",
        str(engine),
        "--components",
        "llm",
        "--plugin-path",
        package["plugin"],
        "--dense",
        "fp16",
        "--max-input-len",
        str(input_limit),
        "--max-kv-cache-capacity",
        str(limit),
        "--max-batch-size",
        "1",
        "--externalize-weights",
        "all",
    ]
    if draft_dir is not None:
        command.extend(
            [
                "--spec-type",
                "dflash",
                "--draft-model-dir",
                str(draft_checkpoint),
                "--max-verify-tree-size",
                "16",
                "--max-draft-tree-size",
                "16",
            ]
        )
    if request.verbose:
        command.append("--verbose")
    with log_path.open("a", encoding="utf-8") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, cwd=staging)
    required = ["tokenizer.json", "tokenizer_config.json", "processed_chat_template.json"]
    required += (
        ["spec_base.engine", "spec_draft.engine", "base_config.json", "draft_config.json"]
        if draft_dir is not None
        else ["llm.engine", "config.json"]
    )
    for name in required:
        if not (engine / name).is_file() or (engine / name).stat().st_size == 0:
            raise ValueError(f"Edge builder did not produce required artifact: {name}")
    files = {}
    for directory in (engine, checkpoint):
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"Edge output must not contain symlinks: {path}")
            if path.is_file():
                files[path.relative_to(staging).as_posix()] = path
    return files, {
        "version": 1,
        "edge_revision": EDGE_REVISION,
        "target": target,
        "precision": "fp16",
        "max_sequence_length": limit,
        "max_input_length": input_limit,
        "max_batch_size": 1,
        "artifacts": list(files),
        **(
            {"execution_variant": "dflash", "dflash_block_size": 16}
            if draft_dir is not None
            else {}
        ),
    }


def publish(request, writer, files: dict, marker: dict) -> None:
    """Stream complete Edge sections; publication errors must not retry native."""
    writer.set_header(family=request.family, task=request.task, backend=request.backend)
    for name, path in files.items():
        with path.open("rb") as source, writer.open_section(name) as destination:
            shutil.copyfileobj(source, destination, length=1024 * 1024)
    writer.add_json("edge_llm.json", marker)

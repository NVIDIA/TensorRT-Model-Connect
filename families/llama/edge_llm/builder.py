# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thin family-owned adapter to the pinned Edge direct-builder API."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys

from tensorrt_model_connect.build import cmake_prefixes, detect_local_platform

from ..build_routing import native_kv_architecture_capability
from ..config import ModelConfig

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


def checkpoint_weight_format(model_dir: Path, raw: dict) -> str:
    """Recognize original and ModelOpt FP8/NVFP4 sources; never quantize weights."""
    if any((model_dir / name).exists() for name in ("quantize_config.json", "quant_config.json")):
        raise ValueError("Unsupported Llama Edge quantization metadata")
    declarations = []
    embedded = raw.get("quantization_config")
    if embedded is not None:
        if not isinstance(embedded, dict) or embedded.get("quant_method", "modelopt") != "modelopt":
            raise ValueError("Llama Edge requires ModelOpt packed checkpoints")
        declarations.append(embedded.get("quant_algo"))
    sidecar = model_dir / "hf_quant_config.json"
    if sidecar.exists():
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict) or metadata.get("producer", {}).get("name") != "modelopt":
            raise ValueError("Llama Edge requires ModelOpt packed checkpoints")
        declarations.append(metadata.get("quantization", {}).get("quant_algo"))
    if not declarations:
        return "fp16"
    formats = {"FP8": "fp8", "NVFP4": "nvfp4"}
    if any(value not in formats for value in declarations) or len(set(declarations)) != 1:
        raise ValueError("Unsupported or conflicting Llama Edge quantization metadata")
    return formats[declarations[0]]


def request_weight_format(request, raw: dict) -> str:
    """Require the declared source format without introducing a quantization recipe."""
    source = checkpoint_weight_format(Path(request.model_dir), raw)
    requested = source if request.quantization is None else request.quantization
    if requested == "none":
        requested = "fp16"
    if source != requested:
        raise ValueError(
            f"Llama Edge source weight format {source} does not match requested {requested}"
        )
    return source


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
    weight_format = request_weight_format(request, raw)
    checkpoint = staging / "edge_llm/checkpoint"
    sources = [(Path(request.model_dir), checkpoint)]
    if draft_dir is not None:
        sources.append((draft_dir, checkpoint / "draft"))
    for source_dir, destination in sources:
        destination.mkdir(parents=True)
        for source in source_dir.iterdir():
            if source.is_file() and (
                source.suffix in {".json", ".safetensors", ".bin", ".model", ".jinja"}
                or source.name in {"merges.txt", "vocab.txt"}
            ):
                shutil.copy2(source, destination / source.name)
        if not list(destination.glob("*.safetensors")) and not list(destination.glob("*.bin")):
            raise ValueError("Edge direct builder requires a safetensors or bin checkpoint")
    config = ModelConfig.from_json(json.dumps(raw))
    default_limit = (
        config.max_position_embeddings
        if native_kv_architecture_capability(config).eligible
        else min(config.max_position_embeddings, 256)
    )
    if draft_dir is not None:
        draft_config = json.loads((draft_dir / "config.json").read_text())
        default_limit = min(default_limit, draft_config["max_position_embeddings"])
    limit = request.max_sequence_length or default_limit
    engine = staging / "edge_llm/engine"
    # Calling upstream main preserves its complete build/artifact orchestration.
    command = [
        builder_python(package, target),
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
        "fp16" if weight_format == "fp16" else "auto",
        "--max-input-len",
        str(limit),
        "--max-kv-cache-capacity",
        str(limit),
        "--max-batch-size",
        "1",
        "--externalize-weights",
        "all",
    ]
    if draft_dir is not None:
        command.extend(["--spec-type", "eagle3", "--draft-model-dir", str(checkpoint / "draft")])
    if request.verbose:
        command.append("--verbose")
    with log_path.open("a", encoding="utf-8") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, cwd=staging)
    engine_files = (
        ("spec_base.engine", "spec_draft.engine", "base_config.json", "draft_config.json")
        if draft_dir is not None
        else ("llm.engine", "config.json")
    )
    for name in (
        *engine_files,
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    ):
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
        "weight_format": weight_format,
        "execution_variant": "eagle3" if draft_dir is not None else "autoregressive",
        "max_sequence_length": limit,
        "max_input_length": limit,
        "max_batch_size": 1,
        "artifacts": list(files),
    }


def publish(request, writer, files: dict, marker: dict) -> None:
    """Stream complete Edge sections; publication errors must not retry native."""
    writer.set_header(family=request.family, task=request.task, backend=request.backend)
    for name, path in files.items():
        with path.open("rb") as source, writer.open_section(name) as destination:
            shutil.copyfileobj(source, destination, length=1024 * 1024)
    writer.add_json("edge_llm.json", marker)

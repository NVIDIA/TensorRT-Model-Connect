# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Muse-Glimmer mapping to the pinned Edge 0.11 ONNX builder flow."""

from __future__ import annotations

import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from tensorrt_model_connect.build import (
    cmake_prefixes,
    detect_local_platform,
    subprocess_environment,
)


EDGE_REVISION = "95515c2f87fba8982db5a519f9022277667b3cc9"


def local_target() -> dict:
    return detect_local_platform()


def package_present() -> bool:
    return any((prefix / "share/trtmc/edge-llm.json").is_file() for prefix in cmake_prefixes())


def installed_package(target: dict) -> dict:
    """Resolve one complete native Edge 0.11 package from standard CMake prefixes."""
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
        if package.get("onnx") is not True or package.get("all_native_kernels") is not True:
            raise ValueError(
                "Muse-Glimmer requires the Edge ONNX tools and complete native kernels"
            )
        for name in ("python", "plugin", "onnx_builder"):
            relative = Path(package[name])
            path = (prefix / relative).resolve()
            if relative.is_absolute() or not path.is_relative_to(prefix.resolve()):
                raise ValueError(f"Edge package {name} must be contained in its installation")
            if not path.is_file():
                raise FileNotFoundError(f"Edge package {name} is missing: {path}")
            package[name] = str(path)
        return package
    raise FileNotFoundError(
        "Edge-LLM 0.11.0 is not installed; enable its CMake dependency or install the "
        "0.11.0 wheel alongside a matching native SDK package"
    )


def exporter_python(package: dict, target: dict) -> str:
    """Prefer a compatible 0.11 wheel in the active environment, else SDK Python."""
    interpreter = sys.executable
    try:
        if not Path(interpreter).is_absolute() or not Path(interpreter).is_file():
            raise ValueError("Edge exporter Python must be an existing absolute path")
        probe = subprocess.run(
            [
                interpreter,
                "-I",
                "-c",
                "import json, tensorrt, tensorrt_edgellm; "
                "import tensorrt_edgellm.scripts.export; "
                "print(json.dumps([tensorrt_edgellm.__version__, tensorrt.__version__]))",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if json.loads(probe.stdout) != ["0.11.0", target["tensorrt_version"]]:
            raise ValueError("Edge wheel or TensorRT version differs from the native package")
    except (OSError, ValueError, subprocess.SubprocessError):
        return package["python"]
    return interpreter


def checkpoint_weight_format(model_dir: Path, raw: dict) -> str:
    """Admit only the documented mixed NVFP4/MXFP8 Muse checkpoint recipe."""
    config = raw.get("quantization_config")
    descriptor = model_dir / "hf_quant_config.json"
    if (
        not isinstance(config, dict)
        or config.get("quant_algo") != "MIXED_PRECISION"
        or not descriptor.is_file()
    ):
        raise ValueError("Muse-Glimmer Edge route requires the documented mixed NVFP4 checkpoint")
    document = json.loads(descriptor.read_text(encoding="utf-8"))
    layers = document.get("quantization", {}).get("quantized_layers", {})
    algorithms = {value.get("quant_algo") for value in layers.values() if isinstance(value, dict)}
    if not {"NVFP4", "MXFP8"} <= algorithms:
        raise ValueError("Muse-Glimmer quantization descriptor must contain NVFP4 and MXFP8 layers")
    return "nvfp4"


def request_weight_format(request, raw: dict) -> str:
    source = checkpoint_weight_format(Path(request.model_dir), raw)
    requested = source if request.quantization in {None, "none"} else request.quantization.lower()
    if requested != source:
        raise ValueError(
            f"Muse-Glimmer source format {source} does not match requested {requested}"
        )
    return source


def _run(command: list[str], log, *, cwd: Path, env: dict | None = None) -> None:
    print("$ " + shlex.join(command), file=log, flush=True)
    subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, cwd=cwd, env=env)


def prepare(request, raw: dict, target: dict, staging: Path, log_path: Path) -> tuple[dict, dict]:
    """Export ONNX and build the complete native engine without model reimplementation."""
    if getattr(request, "execution_variant", "autoregressive") == "dflash":
        from .paired import prepare as prepare_paired

        return prepare_paired(request, raw, target, staging, log_path)
    package = installed_package(target)
    weight_format = request_weight_format(request, raw)
    text = raw.get("text_config", raw)
    capacity = int(text["max_position_embeddings"])
    limit = request.max_sequence_length or min(capacity, 1024)
    input_limit = min(limit, 512)
    onnx = staging / "edge_llm/onnx"
    engine = staging / "edge_llm/engine"
    checkpoint = staging / "edge_llm/checkpoint"
    checkpoint.mkdir(parents=True)
    for name in ("config.json", "generation_config.json"):
        source = Path(request.model_dir) / name
        if source.is_file():
            shutil.copy2(source, checkpoint / name)
    export_command = [
        exporter_python(package, target),
        "-I",
        "-m",
        "tensorrt_edgellm.scripts.export",
        str(Path(request.model_dir)),
        str(onnx),
        "--skip-visual",
        "--skip-audio",
    ]
    build_command = [
        package["onnx_builder"],
        f"--onnxDir={onnx / 'llm'}",
        f"--engineDir={engine}",
        f"--maxInputLen={input_limit}",
        f"--maxKVCacheCapacity={limit}",
        "--maxBatchSize=1",
    ]
    environment = subprocess_environment(
        {"EDGELLM_PLUGIN_PATH": package["plugin"]},
        prepend_paths={"LD_LIBRARY_PATH": str(Path(package["plugin"]).parent)},
    )
    with log_path.open("a", encoding="utf-8") as log:
        _run(export_command, log, cwd=staging)
        _run(build_command, log, cwd=staging, env=environment)
    required = (
        "llm.engine",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "embedding.safetensors",
    )
    for name in required:
        if not (engine / name).is_file() or (engine / name).stat().st_size == 0:
            raise ValueError(f"Edge ONNX builder did not produce required artifact: {name}")
    files: dict[str, Path] = {}
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
        "execution_variant": "autoregressive",
        "max_sequence_length": limit,
        "max_input_length": input_limit,
        "max_batch_size": 1,
        "artifacts": list(files),
    }


def publish(request, writer, files: dict, marker: dict) -> None:
    for name, path in files.items():
        with path.open("rb") as source, writer.open_section(name) as destination:
            shutil.copyfileobj(source, destination, length=1024 * 1024)
    writer.add_json("edge_llm.json", marker)

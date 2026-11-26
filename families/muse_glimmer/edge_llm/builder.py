# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Muse-Glimmer mapping to the pinned Edge ONNX builder flow."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, distribution
import json
import logging
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


from ..vision import media_contract
from ..weight_format import request_weight_format


EDGE_REVISION = "95515c2f87fba8982db5a519f9022277667b3cc9"


RELEASE_012 = "513dcd18291e587c80928911a480761352f3c2d8"


def local_target() -> dict:
    return detect_local_platform()


def _sdk_prefixes(*, warn: bool = False) -> list[Path]:
    """Pair a packaged runner with its own SDK; a source installation uses CMake prefixes."""
    prefixes = cmake_prefixes()
    try:
        installed = Path(distribution("tensorrt-model-connect").locate_file("tensorrt_model_connect"))
    except PackageNotFoundError:
        return prefixes
    if not (installed / "bin/trtmc").is_file():
        return prefixes
    relative = Path("share/trtmc/edge-llm.json")
    if warn and any((p / relative).is_file() and (p / relative).resolve() != (installed / relative).resolve()
                    for p in prefixes):
        logging.getLogger(__name__).warning(
            "Ignoring a standalone Edge SDK: the installed Model Connect runner uses its packaged Muse capability.")
    return [installed]


def package_present() -> bool:
    return any((prefix / "share/trtmc/edge-llm.json").is_file() or
               (prefix / "share/trtmc/edge-llm.json").is_symlink() for prefix in _sdk_prefixes(warn=True))


def installed_package(target: dict) -> dict:
    """Resolve one complete, matching native Edge package from standard CMake prefixes."""
    for prefix in _sdk_prefixes():
        manifest = prefix / "share/trtmc/edge-llm.json"
        if not manifest.is_file():
            continue
        package = json.loads(manifest.read_text(encoding="utf-8"))
        release = package.get("version")
        legacy = (
            release == "0.11.0"
            and package.get("schema_version") == 1
            and package.get("revision") == EDGE_REVISION
        )
        current = (
            release == "0.12.0"
            and package.get("schema_version") == 2
            and "release_revision" in package
            and package["release_revision"] in (None, RELEASE_012)
        )
        if not (legacy or current):
            raise ValueError(f"Edge package has an unsupported revision/schema: {manifest}")
        if package.get("arch") != target["arch"]:
            raise ValueError("Edge package architecture differs from executing worker")
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
        for name in (("python", "plugin", "onnx_builder") if legacy else ("plugin", "onnx_builder")):
            relative = Path(package[name])
            path = (prefix / relative).resolve()
            if relative.is_absolute() or not path.is_relative_to(prefix.resolve()):
                raise ValueError(f"Edge package {name} must be contained in its installation")
            if not path.is_file():
                raise FileNotFoundError(f"Edge package {name} is missing: {path}")
            package[name] = str(path)
        return package
    raise FileNotFoundError(
        "No compatible native Edge SDK is installed; enable the CMake dependency. "
        "A Python wheel does not replace the C++ SDK."
    )


def exporter_python(package: dict, target: dict, requested: Path | None = None) -> str:
    """Validate the selected wheel or source exporter against the installed SDK."""
    release = package["version"]
    current = release == "0.12.0"
    interpreter = str(requested) if requested is not None else sys.executable
    try:
        if not Path(interpreter).is_absolute() or not Path(interpreter).is_file():
            raise ValueError("Edge exporter Python must be an existing absolute path")
        module = "tensorrt_edgellm.onnx.cli" if current else "tensorrt_edgellm.scripts.export"
        probe = subprocess.run(
            [interpreter, "-I", "-c", "import json, tensorrt, tensorrt_edgellm; "
             "from importlib.metadata import version; "
             f"import {module}; "
             + ("assert version('transformers') == '5.17.0'; " if current else "")
             + "print(json.dumps([tensorrt_edgellm.__version__, tensorrt.__version__]))"],
            check=True, capture_output=True, text=True, timeout=45,
        )
        if json.loads(probe.stdout) != [release, target["tensorrt_version"]]:
            raise ValueError("Edge exporter or TensorRT version differs from the native package")
    except (OSError, ValueError, subprocess.SubprocessError):
        if not current and requested is None:
            return package["python"]
        raise
    return interpreter


def export_prefix(request, package: dict, target: dict) -> list[str]:
    interpreter = exporter_python(package, target, getattr(request, "edge_builder_python", None))
    module = ("tensorrt_edgellm.onnx.cli" if package["version"] == "0.12.0"
              else "tensorrt_edgellm.scripts.export")
    return [interpreter, "-I", "-m", module]


def release_marker(package: dict) -> dict:
    if package["version"] == "0.12.0":
        return {"edge_release": "0.12.0", "edge_revision": package["release_revision"],
                "builder_flow": "onnx"}
    return {"edge_revision": EDGE_REVISION}


def _run(command: list[str], log, *, cwd: Path, env: dict | None = None) -> None:
    print("$ " + shlex.join(command), file=log, flush=True)
    subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, cwd=cwd, env=env)


def build_visual(onnx: Path, engine: Path, source: Path, package: dict,
                 vision: dict, limit: int, log, staging: Path, env: dict) -> None:
    """Build the released Muse visual encoder next to its matching text engine."""
    if package["version"] != "0.12.0":
        raise ValueError("Muse-Glimmer Edge media requires the 0.12 SDK")
    executable = Path(package["onnx_builder"]).with_name("visual_build")
    _run([str(executable), "--onnxDir", str(onnx / "visual"), "--engineDir", str(engine),
          "--minImageTokens", "4", "--maxImageTokens", str(limit),
          "--maxImageTokensPerImage", str(vision["max_image_tokens"])], log, cwd=staging, env=env)
    visual = engine / "visual"
    for name in ("visual.engine", "config.json", "preprocessor_config.json"):
        if not (visual / name).is_file() or not (visual / name).stat().st_size:
            raise ValueError(f"Muse-Glimmer Edge visual artifact missing: {name}")
    # The released Muse runner reads its frame-token budget from this adjacent file.
    shutil.copy2(source / "processor_config.json", visual / "processor_config.json")


def prepare(request, raw: dict, target: dict, staging: Path, log_path: Path, package=None) -> tuple[dict, dict]:
    """Export ONNX and build the complete native engine without model reimplementation."""
    if package is None:
        package = installed_package(target)
    if getattr(request, "execution_variant", "autoregressive") in {"dflash", "dflash_tree"}:
        from .paired import prepare as prepare_paired

        return prepare_paired(request, raw, target, staging, log_path, package)
    weight_format = request_weight_format(request, raw)
    text = raw.get("text_config", raw)
    capacity = int(text["max_position_embeddings"])
    limit = request.max_sequence_length or min(capacity, 1024)
    vision = media_contract(request, raw, weight_format, limit)
    input_limit = limit if vision else min(limit, 512)
    onnx = staging / "edge_llm/onnx"
    engine = staging / "edge_llm/engine"
    checkpoint = staging / "edge_llm/checkpoint"
    checkpoint.mkdir(parents=True)
    for name in ("config.json", "generation_config.json"):
        source = Path(request.model_dir) / name
        if source.is_file():
            shutil.copy2(source, checkpoint / name)
    export_command = [
        *export_prefix(request, package, target),
        str(Path(request.model_dir)),
        str(onnx),
        *([] if vision else ["--skip-visual"]),
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
        if vision:
            build_visual(onnx, engine, Path(request.model_dir), package, vision, limit, log,
                         staging, environment)
    shutil.rmtree(onnx)  # The bundle contains engines, not the intermediate export.
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
        **release_marker(package),
        **({"vision": vision} if vision else {}),
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

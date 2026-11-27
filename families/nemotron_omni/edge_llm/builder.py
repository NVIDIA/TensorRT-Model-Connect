# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thin family-owned adapter to the pinned Edge direct-builder API."""

from __future__ import annotations

from importlib.util import find_spec
import json
import logging
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys

from tensorrt_model_connect.build import (
    cmake_prefixes,
    detect_local_platform,
    subprocess_environment,
)


EDGE_REVISION = "95515c2f87fba8982db5a519f9022277667b3cc9"


ACOUSTIC_TASKS = frozenset({"audio_text_to_text", "image_audio_text_to_text"})

_RELEASE_012 = "513dcd18291e587c80928911a480761352f3c2d8"


def _prefixes(*, report_conflict: bool = False) -> list[Path]:
    """A wheel runner must load the plugin installed with that runner."""
    spec = find_spec("tensorrt_model_connect")
    if spec is not None and spec.origin:
        root = Path(spec.origin).resolve().parent
        if (root / "bin/trtmc").is_file():
            # A native-only packaged runner must not borrow a standalone Edge SDK.
            if report_conflict and any(
                (p / "share/trtmc/edge-llm.json").is_file() and p.resolve() != root
                for p in cmake_prefixes()
            ):
                logging.getLogger(__name__).warning(
                    "Using the installed Model Connect runner's packaged Edge SDK at %s "
                    "instead of a conflicting standalone prefix",
                    root,
                )
            return [root]
    return cmake_prefixes()


def local_target() -> dict:
    """Return the executing worker identity supplied by generic build mechanics."""
    return detect_local_platform()


def package_present() -> bool:
    """Absence of the optional SDK is a non-match, not a failed Edge build."""
    for prefix in _prefixes():
        manifest = prefix / "share/trtmc/edge-llm.json"
        if manifest.exists() or manifest.is_symlink():
            return True
    return False


def installed_package(target: dict, *, acoustic: bool = False) -> dict:
    """Resolve CMake installation via standard prefixes; never install anything.

    Args:
        target: Executing device and SDK identity.

    Returns:
        Validated package metadata with absolute Python and plugin paths.

    Raises:
        FileNotFoundError: No CMake installation or required artifact exists.
        ValueError: Pin, architecture, SDK or contained-path contract differs.
    """
    for prefix in _prefixes(report_conflict=True):
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
            and package["release_revision"] in (None, _RELEASE_012)
        )
        if not (legacy or current):
            raise ValueError(f"Edge package has an unsupported revision/schema: {manifest}")
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
        names = ["python", "plugin"] if legacy else ["plugin"]
        if acoustic and current and package.get("onnx") is True:
            names.append("audio_builder")
        for name in names:
            relative = Path(package[name])
            path = (prefix / relative).resolve()
            if relative.is_absolute() or not path.is_relative_to(prefix.resolve()):
                raise ValueError(f"Edge package {name} must be contained in its installation")
            if not path.is_file():
                raise FileNotFoundError(f"Edge package {name} is missing: {path}")
            if name == "audio_builder" and not os.access(path, os.X_OK):
                raise ValueError("Edge audio builder is not executable")
            package[name] = str(path)
        return package
    raise FileNotFoundError(
        "Edge-LLM is not installed; enable the optional Edge-LLM CMake dependency "
        "and set CMAKE_PREFIX_PATH to its install prefix"
    )


def builder_python(
    package: dict, target: dict, requested: Path | None = None, *, acoustic: bool = False
) -> str:
    """Use the exact release and TensorRT version in the selected Python environment."""
    release = package["version"]
    legacy = release == "0.11.0"
    interpreter = Path(sys.executable) if requested is None else Path(requested)
    try:
        if not interpreter.is_absolute() or not interpreter.is_file():
            raise ValueError("Edge builder Python must be an existing absolute interpreter")
        module = "experimental.builder.cli" if legacy else "tensorrt_edgellm.builder.cli"
        check = (
            "import json, tensorrt, tensorrt_edgellm; "
            "from importlib.metadata import version; "
            f"from {module} import main; "
            + ("from tensorrt_edgellm.onnx.cli import main as export; " if acoustic else "")
            + ("" if legacy else "assert version('transformers') == '5.17.0'; ")
            + "print(json.dumps([tensorrt_edgellm.__version__, tensorrt.__version__]))"
        )
        probe = subprocess.run(
            [str(interpreter), "-I", "-c", check],
            check=True,
            capture_output=True,
            text=True,
            timeout=45,
        )
        if json.loads(probe.stdout) != [release, target["tensorrt_version"]]:
            raise ValueError(
                "Edge builder Python must match the selected Edge and TensorRT releases"
            )
    except (OSError, ValueError, subprocess.SubprocessError):
        if legacy and requested is None:
            return package["python"]
        raise
    return str(interpreter)


def prepare(request, target: dict, staging: Path, log_path: Path, capacity: int, package=None):
    """Build direct language/vision engines and, when requested, an ONNX acoustic engine.

    The checkpoint is read in place. Only indexed shards and root metadata are
    streamed into the final bundle, avoiding a second multi-gigabyte staging copy.
    The runtime passes the extracted checkpoint root to Edge's external-weight API.
    """
    acoustic = request.task in ACOUSTIC_TASKS
    if package is None:
        package = installed_package(target, acoustic=acoustic)
    current = package["version"] == "0.12.0"
    if acoustic and not (current and package.get("onnx") is True):
        raise ValueError("This Edge SDK does not provide the 0.12 ONNX acoustic builder")
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
    python = builder_python(
        package, target, getattr(request, "edge_builder_python", None), acoustic=acoustic
    )
    command = [
        python,
        "-I",
        "-c",
        (
            "from tensorrt_edgellm.builder.cli import main; main()"
            if package["version"] == "0.12.0"
            else "from experimental.builder.cli import main; main()"
        ),
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
    commands = [command]
    if acoustic:
        # The direct acoustic plan fails in TensorRT enqueue. Export only this
        # component and use the matched SDK's official ONNX audio builder.
        graph = staging / "onnx"
        commands.extend(
            (
                [
                    python,
                    "-I",
                    "-c",
                    "from tensorrt_edgellm.onnx.cli import main; raise SystemExit(main())",
                    str(source),
                    str(graph),
                    "--components",
                    "audio",
                    "--dtype",
                    "float16",
                ],
                [
                    package["audio_builder"],
                    f"--onnxDir={graph / 'audio'}",
                    f"--engineDir={engine}",
                    "--minTimeSteps=8",
                    "--maxTimeSteps=6000",
                ],
            )
        )
    with log_path.open("a", encoding="utf-8") as log:
        for argv in commands:
            log.write(shlex.join(argv) + "\n")
            log.flush()
            env = (
                subprocess_environment({"EDGELLM_PLUGIN_PATH": package["plugin"]})
                if acoustic and argv[0] == package["audio_builder"]
                else None
            )
            subprocess.run(
                argv, check=True, stdout=log, stderr=subprocess.STDOUT, cwd=staging, env=env
            )
    for name in (
        "llm.engine",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "visual/visual.engine",
        "visual/config.json",
        *(("audio/audio_encoder.engine", "audio/config.json") if acoustic else ()),
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
        "edge_revision": package.get("release_revision")
        if package["version"] == "0.12.0"
        else EDGE_REVISION,
        **(
            {
                "edge_release": "0.12.0",
                "builder_flow": "direct_with_onnx_audio" if acoustic else "direct",
            }
            if current
            else {}
        ),
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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thin family-owned adapter to the pinned Edge direct-builder API."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import os
import stat
import sys
import subprocess

from tensorrt_model_connect.build import cmake_prefixes, detect_local_platform


EDGE_REVISION = "95515c2f87fba8982db5a519f9022277667b3cc9"


def local_target() -> dict:
    """Return the executing worker identity supplied by generic build mechanics."""
    return detect_local_platform()


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


def validate_tokenizer(tokenizer: dict, config: dict, engine: dict | None = None) -> list[int]:
    """Preserve source ByteLevel raw policy and complete primary/config EOS union."""
    post = tokenizer.get("post_processor")
    if (
        not isinstance(post, dict)
        or post.get("type") != "ByteLevel"
        or any(
            post.get(key, False) is not False
            for key in ("add_prefix_space", "trim_offsets", "use_regex")
        )
        or config.get("add_bos_token")
        or config.get("add_eos_token")
    ):
        raise ValueError("Unsupported Qwen3.5-MoE raw tokenizer framing")
    eos = config.get("eos_token")
    primary = eos.get("content") if isinstance(eos, dict) else eos
    ids = {
        item.get("id")
        for item in tokenizer.get("added_tokens", [])
        if item.get("content") == primary
    }
    if ids != {248046}:
        raise ValueError("Unexpected source tokenizer primary EOS")
    if engine is not None:
        exported = engine.get("eos_token_id")
        if not isinstance(exported, list) or not exported or set(exported) - {248044, 248046}:
            raise ValueError("Unexpected exported source EOS IDs")
        ids.update(exported)
        if ids != {248044, 248046}:
            raise ValueError("Source EOS union was not preserved")
    return [248044, 248046]


def prepare(request, raw: dict, target: dict, staging: Path, log_path: Path) -> tuple[dict, dict]:
    """Map the request to Edge main(argv), returning complete unpublished assets.

    Edge owns model selection, configuration, conversion, graphs and engine
    composition. The family adapter only maps vision-language arguments and
    preserves the checkpoint needed by Edge external-weight APIs.

    Returns:
        (section-name to file mapping, runtime marker).

    Raises:
        Exception: Dependency, upstream build or artifact validation failed.
    """
    from .precision import source_policy
    from .checkpoint import validate_checkpoint
    from .dispatch import validate_execution

    draft_dir = validate_execution(request, raw)
    execution = getattr(request, 'execution', None)
    package = installed_package(target)
    policy = source_policy(request, raw)
    weight_format = policy["weights"]
    checkpoint = staging / "edge_llm/checkpoint"
    checkpoint.mkdir(parents=True)
    for source in Path(request.model_dir).iterdir():
        if source.is_file() and (
            source.suffix in {".json", ".safetensors", ".model", ".jinja"}
            or source.name in {"merges.txt", "vocab.txt"}
        ):
            destination = checkpoint / source.name
            if source.suffix == ".safetensors":
                try:
                    os.link(source.resolve(), destination)
                except OSError:
                    shutil.copy2(source, destination)
            else:
                # The speculative builder writes tokenizer artifacts twice.
                # Do not propagate immutable HF cache modes to its scratch files.
                shutil.copy2(source, destination)
                destination.chmod(destination.stat().st_mode | stat.S_IWUSR)
    if not list(checkpoint.glob("*.safetensors")):
        raise ValueError("Edge direct builder requires a safetensors checkpoint")
    checkpoint_layout = validate_checkpoint(checkpoint, weight_format)
    if draft_dir is not None:
        draft_checkpoint = checkpoint / "draft"
        draft_checkpoint.mkdir()
        for source in draft_dir.iterdir():
            if source.is_file() and source.suffix in {".json", ".safetensors"}:
                destination = draft_checkpoint / source.name
                if source.suffix == ".safetensors":
                    try:
                        os.link(source.resolve(), destination)
                    except OSError:
                        shutil.copy2(source, destination)
                else:
                    shutil.copyfile(source, destination)
        if not list(draft_checkpoint.glob("*.safetensors")):
            raise ValueError("DFlash requires a complete draft safetensors checkpoint")
    config = raw["text_config"]
    limit = request.max_sequence_length or min(config["max_position_embeddings"], 4096)
    if type(limit) is not int or limit <= 1 or limit > config["max_position_embeddings"]:
        raise ValueError("Invalid Qwen3.5-MoE Edge sequence capacity")
    tokenizer = json.loads((checkpoint / "tokenizer.json").read_text(encoding="utf-8"))
    tokenizer_config = json.loads(
        (checkpoint / "tokenizer_config.json").read_text(encoding="utf-8")
    )
    validate_tokenizer(tokenizer, tokenizer_config)
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
        "llm,visual" if request.task == "vision_language_generation" else "llm",
        "--plugin-path",
        package["plugin"],
        "--dense",
        "auto",
        "--max-input-len",
        str(limit),
        "--max-kv-cache-capacity",
        str(limit),
        "--max-batch-size",
        "1",
        "--min-image-tokens",
        "4",
        "--max-image-tokens",
        "1024",
        "--max-image-tokens-per-image",
        "512",
    ]
    if draft_dir is not None:
        command.extend(["--spec-type", "dflash", "--draft-model-dir", str(draft_checkpoint),
                        "--max-verify-tree-size", "16", "--max-draft-tree-size", "16"])
        # The hybrid base already carries parent/depth metadata for chain replay
        # and accepts both greedy linear and DDTree verification in Edge 0.11.
    if execution is not None and execution.variant == "mtp":
        command.extend(["--spec-type", "mtp", "--max-verify-tree-size", "4",
                        "--max-draft-tree-size", "4"])
    for kind in ("all",):
        command.extend(("--externalize-weights", kind))
    if request.verbose:
        command.append("--verbose")
    with log_path.open("a", encoding="utf-8") as log:
        log.write(json.dumps(command) + "\n")
        log.flush()
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, cwd=staging)
    required = [
        *( ["spec_base.engine", "spec_draft.engine", "base_config.json", "draft_config.json"]
           if execution is not None else ["llm.engine", "config.json"] ),
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    ]
    if request.task == "vision_language_generation":
        required.extend(("visual/visual.engine", "visual/config.json"))
    for name in required:
        if not (engine / name).is_file() or (engine / name).stat().st_size == 0:
            raise ValueError(f"Edge builder did not produce required artifact: {name}")
    validate_tokenizer(
        json.loads((engine / "tokenizer.json").read_text(encoding="utf-8")),
        json.loads((engine / "tokenizer_config.json").read_text(encoding="utf-8")),
        json.loads((engine / ("base_config.json" if execution is not None else "config.json")).read_text(encoding="utf-8")),
    )
    files = {}
    for directory in (engine, checkpoint):
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"Edge output must not contain symlinks: {path}")
            if path.is_file():
                files[path.relative_to(staging).as_posix()] = path
    return files, {
        **({"execution_variant": execution.variant,
            **({"mtp_draft_steps": 3, "mtp_verify_size": 4} if execution.variant == "mtp"
               else {"dflash_block_size": 16})} if execution is not None else {}),
        "version": 1,
        "edge_revision": EDGE_REVISION,
        "target": target,
        "precision": "fp16",
        "weight_format": weight_format,
        "source_policy": policy,
        "checkpoint_layout": checkpoint_layout,
        "task": request.task,
        "eos_token_ids": [248044, 248046],
        "visual_profile": {"min": 4, "max": 1024, "per_image": 512}
        if request.task == "vision_language_generation"
        else None,
        "max_sequence_length": limit,
        "max_input_length": limit,
        "max_batch_size": 1,
        "artifacts": list(files),
    }


def publish(request, writer, files: dict, marker: dict) -> None:
    """Stream complete Edge sections; publication errors must not retry native."""
    for name, path in files.items():
        with path.open("rb") as source, writer.open_section(name) as destination:
            shutil.copyfileobj(source, destination, length=1024 * 1024)
    writer.add_json("edge_llm.json", marker)

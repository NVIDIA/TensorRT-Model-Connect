# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thin family-owned adapter to the pinned Edge direct-builder API."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

from tensorrt_model_connect.build import cmake_prefixes, detect_local_platform, subprocess_environment

EDGE_REVISION = "e8b29522938901f6df19ebeedd4b69bc8edbcd97"
_QUANTIZED_EXTERNAL_WEIGHTS = (
    "int4_ffn", "int4_moe", "nvfp4_moe", "nvfp4_tp", "lm_head", "embedding",
)


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
        if package.get("all_native_kernels") is not True:
            raise ValueError("Nemotron-H requires an Edge package with all native operator groups")
        if package.get("version") != "0.10.1" or package.get("arch") != target["arch"]:
            raise ValueError("Edge package version/architecture differs from executing worker")
        if target["sm"] not in package.get("architectures", []):
            raise ValueError("Edge package was not built for this local GPU")
        cuda_version = ".".join(str(package.get("cuda_version", "")).split(".")[:2])
        if cuda_version != target["cuda_version"] or package.get("tensorrt_version") != target["tensorrt_version"]:
            raise ValueError("Edge package CUDA/TensorRT differs from executing worker")
        for name in ("python", "plugin") + (("onnx_builder",) if package.get("onnx") is True else ()):
            relative = Path(package[name])
            path = (prefix / relative).resolve()
            if relative.is_absolute() or not path.is_relative_to(prefix.resolve()):
                raise ValueError(f"Edge package {name} must be contained in its installation")
            if not path.is_file():
                raise FileNotFoundError(f"Edge package {name} is missing: {path}")
            package[name] = str(path)
        return package
    raise FileNotFoundError("Edge-LLM is not installed; enable the optional Edge-LLM CMake dependency "
                            "and set CMAKE_PREFIX_PATH to its install prefix")


def sequence_length(request, raw: dict) -> int:
    """Resolve the same default capacity as this family's original native builder."""
    value = request.max_sequence_length or min(raw["max_position_embeddings"], 256)
    if isinstance(value, bool):
        raise ValueError("max_sequence_length must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("max_sequence_length must be a positive integer") from error
    if result < 1:
        raise ValueError("max_sequence_length must be a positive integer")
    return result


def prepare(request, raw: dict, target: dict, staging: Path, log_path: Path) -> tuple[dict, dict]:
    """Map the request to Edge main(argv), returning complete unpublished assets.

    Edge owns model selection, configuration, conversion, graphs and engine
    composition. The family adapter only maps text-generation arguments and
    preserves the checkpoint needed by Edge external-weight APIs.

    Returns:
        (section-name to file mapping, runtime marker).

    Raises:
        Exception: Dependency, upstream build or artifact validation failed.
    """
    from .edge_quantization import source_quantization

    source_precision = source_quantization(request, raw)
    if source_precision is None:
        raise ValueError("Unsupported Nemotron-H checkpoint precision")
    package = installed_package(target)
    checkpoint = staging / "edge_llm/checkpoint"
    checkpoint.mkdir(parents=True)
    for source in Path(request.model_dir).iterdir():
        if source.is_file() and (source.suffix in {".json", ".safetensors", ".model", ".jinja"}
                                 or source.name in {"merges.txt", "vocab.txt"}):
            shutil.copy2(source, checkpoint / source.name)
    if not list(checkpoint.glob("*.safetensors")):
        raise ValueError("Edge direct builder requires a safetensors checkpoint")
    limit = sequence_length(request, raw)
    engine = staging / "edge_llm/engine"
    # Calling upstream main preserves its complete build/artifact orchestration.
    command = [package["python"], "-I", "-c",
               "from experimental.builder.cli import main; main()",
               "--model-dir", str(checkpoint), "--engine-dir", str(engine),
               "--components", "llm", "--plugin-path", package["plugin"],
               "--dense", "fp16" if source_precision == "fp16" else "auto", "--max-input-len", str(limit),
               "--max-kv-cache-capacity", str(limit), "--max-batch-size", "1"]
    # The pinned builder cannot externalize FP16 biases on quantized projections:
    # their checkpoint recipes are missing. Bake FP16 parameters (which can enlarge the plan),
    # retaining the original packed quantized weights and every other supported kind.
    external_weights = ("all",) if source_precision == "fp16" else _QUANTIZED_EXTERNAL_WEIGHTS
    for kind in external_weights:
        command.extend(("--externalize-weights", kind))
    if request.verbose:
        command.append("--verbose")
    with log_path.open("a", encoding="utf-8") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, cwd=staging)
    for name in ("llm.engine", "config.json", "tokenizer.json", "tokenizer_config.json", "processed_chat_template.json"):
        if not (engine / name).is_file() or (engine / name).stat().st_size == 0:
            raise ValueError(f"Edge builder did not produce required artifact: {name}")
    from .edge_tokenizer import prepare_tokenizer, source_chat_template

    runtime_tokenizer = staging / "edge_llm/runtime_tokenizer"
    eos = prepare_tokenizer(checkpoint, engine, runtime_tokenizer, raw,
                            chat_template=source_chat_template(checkpoint))
    files = {}
    for directory in (engine, checkpoint, runtime_tokenizer):
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"Edge output must not contain symlinks: {path}")
            if path.is_file():
                files[path.relative_to(staging).as_posix()] = path
    return files, {
        "version": 1, "edge_revision": EDGE_REVISION, "target": target, "precision": "fp16",
        "max_sequence_length": limit, "max_input_length": limit, "max_batch_size": 1,
        "artifacts": list(files), "checkpoint_precision": source_precision,
        "tokenizer_policy": "native_full_eos", "native_eos_token_ids": eos,
        "native_bos_token_id": raw.get("bos_token_id", -1),
    }


def publish(request, writer, files: dict, marker: dict) -> None:
    """Stream complete Edge sections; publication errors must not retry native."""
    writer.set_header(family=request.family, task=request.task, backend=request.backend)
    for name, path in files.items():
        with path.open("rb") as source, writer.open_section(name) as destination:
            shutil.copyfileobj(source, destination, length=1024 * 1024)
    writer.add_json("edge_llm.json", marker)


def prepare_dflash(request, raw: dict, target: dict, staging: Path, log_path: Path,
                   draft: Path) -> tuple[dict, dict]:
    """Build both recurrent-state-aware graphs with the original ONNX toolchain.

    Export consumes the caller's immutable local checkpoints. ONNX plans bake
    projection weights, so the bundle needs only engine assets and tokenizer
    metadata, not a duplicate of either checkpoint's packed weights.
    """
    from .edge_quantization import source_quantization
    from .edge_tokenizer import prepare_tokenizer, source_chat_template

    package = installed_package(target)
    if package.get("onnx") is not True:
        raise ValueError("Nemotron-H DFlash requires an ONNX-enabled Edge SDK")
    source = Path(request.model_dir).resolve()
    draft = draft.resolve()
    draft_config = json.loads((draft / "config.json").read_text(encoding="utf-8"))
    dflash = draft_config.get("dflash_config", {})
    layer_ids = dflash.get("target_layer_ids", [])
    if (draft_config.get("architectures") != ["DFlashDraftModel"]
            or draft_config.get("hidden_size") != raw["hidden_size"]
            or draft_config.get("vocab_size") != raw["vocab_size"]
            or not layer_ids or len(set(layer_ids)) != len(layer_ids)
            or any(type(i) is not int or not 0 <= i < raw["num_hidden_layers"] for i in layer_ids)
            or dflash.get("block_size", 16) != 16):
        raise ValueError("Nemotron-H DFlash companion geometry does not match its target")
    if not list(source.glob("*.safetensors")) or not list(draft.glob("*.safetensors")):
        raise ValueError("Nemotron-H DFlash requires both local safetensors checkpoints")
    limit = sequence_length(request, raw)
    if limit < 16:
        raise ValueError("Nemotron-H DFlash requires capacity for a complete block16")
    checkpoint = staging / "edge_llm/checkpoint"
    checkpoint.mkdir(parents=True)
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json",
                 "chat_template.jinja"):
        if (source / name).is_file():
            shutil.copy2(source / name, checkpoint / name)
    engine = staging / "edge_llm/engine"
    onnx = staging / "onnx"
    # Upstream packed MoE layout is an exporter option, not a model rewrite.
    env = subprocess_environment(
        {"EDGELLM_PLUGIN_PATH": package["plugin"],
         "EDGELLM_NVFP4_MOE_TARGET": "sm12x" if target["sm"] in {120, 121} else f"sm{target['sm']}"},
        prepend_paths={"LD_LIBRARY_PATH": str(Path(package["plugin"]).parent)},
    )
    for role, subdirectory, flag in (("draft", "dflash_draft", "--specDraft"),
                                      ("base", "llm", "--specBase")):
        commands = [
            [package["python"], "-I", "-m", "tensorrt_edgellm.scripts.export",
             str(source), str(onnx), f"--dflash-{role}", "--dflash-draft-dir", str(draft),
             "--skip-visual", "--skip-audio"],
            [package["onnx_builder"], "--onnxDir", str(onnx / subdirectory),
             "--engineDir", str(engine), flag, "--maxInputLen", str(min(limit, 1024)),
             "--maxKVCacheCapacity", str(limit), "--maxBatchSize", "1",
             "--maxVerifyTreeSize", "16", "--maxDraftTreeSize", "16"],
        ]
        with log_path.open("a", encoding="utf-8") as log:
            for command in commands:
                log.write(json.dumps(command) + "\n")
                log.flush()
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT,
                               cwd=staging, env=env)
        # Only intermediates created by this preparation are removed. Source
        # checkpoints and final engine assets remain untouched.
        shutil.rmtree(onnx)
    required = ("spec_base.engine", "spec_draft.engine", "base_config.json", "draft_config.json",
                "embedding.safetensors", "tokenizer.json", "tokenizer_config.json", "processed_chat_template.json")
    for name in required:
        if not (engine / name).is_file() or (engine / name).stat().st_size == 0:
            raise ValueError(f"Edge ONNX builder did not produce required artifact: {name}")
    for role in ("base", "draft"):
        config = json.loads((engine / f"{role}_config.json").read_text(encoding="utf-8"))
        if config.get("spec_decode_type") != "dflash" or config.get("dflash_config", {}).get("block_size") != 16:
            raise ValueError("Edge ONNX builder returned a different speculative execution contract")
    tokenizer = staging / "edge_llm/runtime_tokenizer"
    eos = prepare_tokenizer(checkpoint, engine, tokenizer, raw,
                            chat_template=source_chat_template(checkpoint))
    files = {}
    for directory in (engine, checkpoint, tokenizer):
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"Edge output must not contain symlinks: {path}")
            if path.is_file():
                files[path.relative_to(staging).as_posix()] = path
    return files, {
        "version": 1, "edge_revision": EDGE_REVISION, "target": target, "precision": "fp16",
        "max_sequence_length": limit, "max_input_length": min(limit, 1024), "max_batch_size": 1,
        "artifacts": list(files), "checkpoint_precision": source_quantization(request, raw),
        "tokenizer_policy": "native_full_eos", "native_eos_token_ids": eos,
        "native_bos_token_id": raw.get("bos_token_id", -1), "execution_variant": "dflash",
        "builder_flow": "onnx", "dflash_block_size": 16,
    }

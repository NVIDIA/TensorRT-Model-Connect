# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gemma4 standalone ONNX offload; all component selection belongs to Gemma."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import traceback

from tensorrt_model_connect.build import subprocess_environment

from .builder import EDGE_REVISION, installed_package, local_target

_LOG = logging.getLogger(__name__)


def prepare(request, raw: dict, target: dict, staging: Path, log_path: Path):
    """Export the original checkpoint and build its native Edge components."""
    if (request.backend != "trt" or request.task not in {
            "text_generation", "text_continuation", "images_text_to_text",
            "audio_text_to_text", "image_audio_text_to_text"}
            or request.precision != "fp16" or request.quantization not in {None, "none"}
            or request.max_batch_size != 1 or request.tensor_parallel_size != 1
            or request.context_parallel_size != 1 or request.dynamic_kv_cache
            or request.fp32_layers or request.graph_transform is not None
            or any(value is not None for value in (request.image_height, request.image_width,
                                                   request.video_num_frames))):
        raise ValueError("Gemma4 standalone Edge maps original FP16 text/media TP1/batch1 without build overrides")
    if raw.get("quantization_config") or raw.get("text_config", {}).get("quantization_config"):
        raise ValueError("This Gemma4 profile requires an unquantized checkpoint")
    source = Path(request.model_dir).resolve()
    if any((source / name).exists() for name in
           ("hf_quant_config.json", "quantize_config.json", "quant_config.json")):
        raise ValueError("This Gemma4 profile requires an unquantized checkpoint")
    text = raw.get("text_config")
    if not isinstance(text, dict) or text.get("model_type") != raw["model_type"] + "_text":
        raise ValueError("Gemma4 standalone requires its original nested text configuration")
    limit = request.max_sequence_length or 2048
    if not 2 < limit <= text.get("max_position_embeddings", 0):
        raise ValueError("Gemma4 sequence capacity must fit the original checkpoint")
    has_vision = isinstance(raw.get("vision_config"), dict)
    has_audio = isinstance(raw.get("audio_config"), dict)
    if request.task in {"images_text_to_text", "image_audio_text_to_text"} and not has_vision:
        raise ValueError("Requested Gemma4 task requires a vision tower")
    if request.task in {"audio_text_to_text", "image_audio_text_to_text"} and not has_audio:
        raise ValueError("Requested Gemma4 task requires an audio tower")
    package = installed_package(target, media=has_vision or has_audio)
    if package.get("onnx") is not True or package.get("all_native_kernels") is not True:
        raise ValueError("Gemma4 requires an ONNX-enabled full native Edge SDK")
    engine, onnx = staging / "edge_llm/engine", staging / "onnx"
    checkpoint = staging / "edge_llm/checkpoint"
    checkpoint.mkdir(parents=True)
    shutil.copy2(source / "config.json", checkpoint / "config.json")
    env = subprocess_environment(
        {"EDGELLM_PLUGIN_PATH": package["plugin"]},
        prepend_paths={"LD_LIBRARY_PATH": str(Path(package["plugin"]).parent)},
    )
    input_limit = min(limit - 2, 1024)
    commands = [
        [package["python"], "-I", "-m", "tensorrt_edgellm.scripts.export",
         str(source), str(onnx), "--dtype", "float16"],
        [package["onnx_builder"], "--onnxDir", str(onnx / "llm"),
         "--engineDir", str(engine), "--maxInputLen", str(input_limit),
         "--maxKVCacheCapacity", str(limit), "--maxBatchSize", "1"],
    ]
    if has_vision:
        if raw["model_type"] == "gemma4_unified":
            per_image = raw["vision_config"].get("num_soft_tokens")
            processor = source / "processor_config.json"
            if processor.is_file():
                image_processor = json.loads(processor.read_text()).get("image_processor") or {}
                per_image = image_processor.get("max_soft_tokens", per_image)
        else:
            per_image = raw["vision_config"].get("default_output_length")
        if type(per_image) is not int or per_image < 4:
            raise ValueError("Gemma4 vision requires the checkpoint's image-token capacity")
        commands.append([
            package["visual_builder"], "--onnxDir", str(onnx / "visual"),
            "--engineDir", str(engine), "--minImageTokens", "4",
            "--maxImageTokens", str(max(1024, per_image)),
            "--maxImageTokensPerImage", str(per_image),
        ])
    if has_audio:
        commands.append([
            package["audio_builder"], "--onnxDir", str(onnx / "audio"),
            "--engineDir", str(engine), "--minTimeSteps", "100", "--maxTimeSteps", "6000",
        ])
    with log_path.open("a", encoding="utf-8") as log:
        for command in commands:
            log.write(json.dumps(command) + "\n")
            log.flush()
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT,
                           cwd=staging, env=env)
    required = ["llm.engine", "config.json", "embedding.safetensors",
                "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]
    if text.get("hidden_size_per_layer_input", 0):
        required.append("ple_embedding.safetensors")
    if has_vision:
        required.extend(("visual/visual.engine", "visual/config.json"))
    if has_audio:
        required.extend(("audio/audio_encoder.engine", "audio/config.json"))
    for name in required:
        if not (engine / name).is_file() or (engine / name).stat().st_size == 0:
            raise ValueError(f"Edge Gemma4 builder omitted required artifact: {name}")
    files = {}
    for directory in (engine, checkpoint):
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"Gemma4 Edge output must not contain symlinks: {path}")
            if path.is_file():
                files[path.relative_to(staging).as_posix()] = path
    return files, {
        "version": 1, "edge_revision": EDGE_REVISION, "target": target, "precision": "fp16",
        "max_sequence_length": limit, "max_input_length": input_limit, "max_batch_size": 1,
        "execution_variant": "autoregressive", "builder_flow": "onnx", "task": request.task,
        "vision": has_vision, "audio": has_audio,
        "ple": bool(text.get("hidden_size_per_layer_input", 0)), "artifacts": list(files),
    }


def try_build(request, writer) -> bool:
    """Publish a complete standalone bundle or leave the ordinary native path untouched."""
    source = Path(request.model_dir)
    raw = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if raw.get("model_type") not in {"gemma4", "gemma4_unified"}:
        return False
    descriptor, name = tempfile.mkstemp(
        prefix=f".{request.output_path.name}.edge-onnx-", suffix=".log",
        dir=request.output_path.parent,
    )
    os.close(descriptor)
    log_path = Path(name)
    with tempfile.TemporaryDirectory(prefix="trtmc-gemma4-edge-") as directory:
        try:
            files, marker = prepare(request, raw, local_target(), Path(directory), log_path)
        except Exception as error:
            with log_path.open("a", encoding="utf-8") as log:
                traceback.print_exception(error, file=log)
            _LOG.warning("Gemma4 Edge build failed: %s. Diagnostics: %s. "
                         "Trying the ordinary native family builder.", error, log_path)
            return False
        writer.set_header(family="gemma", task=request.task, backend=request.backend)
        for name, path in files.items():
            with path.open("rb") as handle, writer.open_section(name) as destination:
                shutil.copyfileobj(handle, destination, length=1024 * 1024)
        writer.add_json("edge_llm.json", marker)
    return True

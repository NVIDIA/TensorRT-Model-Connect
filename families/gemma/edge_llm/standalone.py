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

_PUBLISHED_NVFP4_CONFIG_GROUPS = {
    "group_0": {
        "input_activations": {
            "dynamic": False, "num_bits": 4, "type": "float", "group_size": 16,
        },
        "weights": {
            "dynamic": False, "num_bits": 4, "type": "float", "group_size": 16,
        },
        "targets": ["Linear"],
    },
}


def checkpoint_quantization(raw: dict, source: Path, target: dict) -> str | None:
    """Admit original weights or the published Gemma4 dense/MoE NVFP4 layouts.

    This forwards an already quantized checkpoint; it never calibrates weights
    or changes its KV-cache format. Packed companions are not admitted here.
    """
    text = raw.get("text_config") or {}
    quant = raw.get("quantization_config") or text.get("quantization_config")
    sidecar = source / "hf_quant_config.json"
    if any((source / name).exists() for name in ("quantize_config.json", "quant_config.json")):
        raise ValueError("Gemma4 Edge does not map this checkpoint quantization format")
    if not quant and not sidecar.exists():
        return None
    if not isinstance(quant, dict) or not sidecar.is_file():
        raise ValueError("Gemma4 NVFP4 requires matching ModelOpt config and hf_quant_config.json")
    exported = json.loads(sidecar.read_text(encoding="utf-8")).get("quantization", {})
    moe = (text.get("enable_moe_block") is True
           and (text.get("hidden_size"), text.get("num_hidden_layers"),
                text.get("num_experts"), text.get("top_k_experts")) == (2816, 30, 128, 8))
    dense = (text.get("enable_moe_block") is False
             and (text.get("hidden_size"), text.get("num_hidden_layers")) == (5376, 60))
    if (raw.get("model_type") != "gemma4" or text.get("model_type") != "gemma4_text"
            or not (moe or dense)
            or quant.get("quant_method") != "modelopt" or quant.get("quant_algo") != "NVFP4"
            or quant.get("config_groups") != _PUBLISHED_NVFP4_CONFIG_GROUPS
            or exported.get("quant_algo") != "NVFP4" or exported.get("group_size") != 16
            or exported.get("kv_cache_quant_algo") != "FP8"
            or quant.get("kv_cache_scheme") != {"dynamic": False, "num_bits": 8, "type": "float"}
            or text.get("quantization_config") not in (None, quant)):
        raise ValueError("Gemma4 Edge requires a published dense31B or MoE26B-A4B ModelOpt NVFP4 layout")
    if moe and target.get("sm") not in {120, 121}:
        raise ValueError("Gemma4 NVFP4 offload requires a native SM12x Edge installation")
    return "nvfp4"


def prepare(request, raw: dict, target: dict, staging: Path, log_path: Path):
    """Export the checkpoint unchanged and build its native Edge components."""
    if (request.backend != "trt" or request.task not in {
            "text_generation", "text_continuation", "images_text_to_text",
            "audio_text_to_text", "image_audio_text_to_text"}
            or request.precision != "fp16" or request.quantization not in {None, "none"}
            or request.max_batch_size != 1 or request.tensor_parallel_size != 1
            or request.context_parallel_size != 1 or request.dynamic_kv_cache
            or request.fp32_layers or request.graph_transform is not None
            or any(value is not None for value in (request.image_height, request.image_width,
                                                   request.video_num_frames))):
        raise ValueError("Gemma4 standalone Edge maps FP16 compute, checkpoint weights, TP1/batch1 without build overrides")
    source = Path(request.model_dir).resolve()
    checkpoint_quant = checkpoint_quantization(raw, source, target)
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
    package = installed_package(target, vision=has_vision, audio=has_audio)
    if package.get("onnx") is not True or package.get("all_native_kernels") is not True:
        raise ValueError("Gemma4 requires an ONNX-enabled full native Edge SDK")
    engine, onnx = staging / "edge_llm/engine", staging / "onnx"
    checkpoint = staging / "edge_llm/checkpoint"
    checkpoint.mkdir(parents=True)
    shutil.copy2(source / "config.json", checkpoint / "config.json")
    additions = {"EDGELLM_PLUGIN_PATH": package["plugin"]}
    if checkpoint_quant and text.get("enable_moe_block"):
        # The upstream exporter otherwise defaults the MoE plugin to Thor.
        additions["EDGELLM_NVFP4_MOE_TARGET"] = "sm12x"
    env = subprocess_environment(
        additions,
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
    # ModelOpt sidecars can label FP8 cache even without serialized K/V
    # scales. Report what Edge actually built, not the sidecar's declaration.
    cache_dtype = json.loads((engine / "config.json").read_text()).get("kv_cache_dtype")
    if cache_dtype not in {"fp16", "bf16", "fp8"}:
        raise ValueError("Edge Gemma4 engine has an invalid or missing kv_cache_dtype")
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
        "checkpoint_quantization": checkpoint_quant,
        "kv_cache_quantization": "fp8" if cache_dtype == "fp8" else None,
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

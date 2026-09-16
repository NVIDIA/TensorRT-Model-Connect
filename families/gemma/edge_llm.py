# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned Gemma4 paired forwarding to the pinned original ONNX toolchain."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import traceback

from tensorrt_model_connect.build import cmake_prefixes, detect_local_platform, subprocess_environment

EDGE_REVISION = "e8b29522938901f6df19ebeedd4b69bc8edbcd97"


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
        if package.get("version") != "0.10.1" or package.get("arch") != target["arch"]:
            raise ValueError("Edge package version/architecture differs from executing worker")
        if target["sm"] not in package.get("architectures", []):
            raise ValueError("Edge package was not built for this local GPU")
        cuda_version = ".".join(str(package.get("cuda_version", "")).split(".")[:2])
        if cuda_version != target["cuda_version"] or package.get("tensorrt_version") != target["tensorrt_version"]:
            raise ValueError("Edge package CUDA/TensorRT differs from executing worker")
        for name in ("python", "plugin") + (("onnx_builder",) if package.get("onnx") else ()):
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


_LOG = logging.getLogger(__name__)

# Validate the exact source template; the runtime owns its single-user rendering.
_CHAT_ADMISSION = r"""
from transformers import AutoTokenizer
import sys
tokenizer = AutoTokenizer.from_pretrained(sys.argv[1], local_files_only=True, trust_remote_code=False)
for thinking in (False, True):
    for text in ("hello", "  hello\n", "\u2003hello\u3000"):
        expected = "<bos>"
        if thinking:
            expected += "<|turn>system\n<|think|>\n<turn|>\n"
        expected += "<|turn>user\n" + text.strip() + "<turn|>\n<|turn>model\n"
        if not thinking:
            expected += "<|channel>thought\n<channel|>"
        actual = tokenizer.apply_chat_template([{"role": "user", "content": text}],
            tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
        if actual != expected:
            raise ValueError("Gemma4 source chat template differs from the family runtime mapping")
print("Gemma4 single-user chat template admission passed")
"""


def validate_pair(request, execution) -> tuple[dict, Path, int]:
    """Admit only explicit unquantized Gemma4 unified assistant text execution."""
    if execution.variant not in {"mtp", "dspark"} or tuple(c.role for c in execution.checkpoints) != ("draft",):
        raise ValueError("Gemma4 pairs require variant=mtp or dspark and one named draft checkpoint")
    if (request.backend != "trt" or request.task != "text_generation"
            or request.precision != "fp16" or request.quantization not in {None, "none"}
            or request.max_batch_size != 1 or request.tensor_parallel_size != 1
            or request.context_parallel_size != 1 or request.dynamic_kv_cache
            or request.fp32_layers or request.graph_transform is not None
            or any(v is not None for v in (request.image_height, request.image_width,
                                           request.video_num_frames))):
        raise ValueError("Gemma4 paired execution maps FP16 text-only TP1/batch1 without graph transforms")
    source, draft = Path(request.model_dir), execution.checkpoints[0].model_dir
    raw = json.loads((source / "config.json").read_text())
    companion = json.loads((draft / "config.json").read_text())
    if not isinstance(raw, dict) or raw.get("model_type") != "gemma4_unified":
        raise ValueError("Gemma4 MTP expects a Gemma4 unified target")
    if not isinstance(companion, dict):
        raise ValueError("Gemma4 requires a companion configuration object")
    base = raw.get("text_config")
    if not isinstance(base, dict):
        raise ValueError("Gemma4 target requires nested text configuration")
    if execution.variant == "mtp":
        if companion.get("model_type") != "gemma4_unified_assistant":
            raise ValueError("Gemma4 MTP expects a unified assistant")
        assistant = companion.get("text_config")
        if not isinstance(assistant, dict) or companion.get("backbone_hidden_size") != base.get("hidden_size"):
            raise ValueError("Gemma4 MTP target and assistant geometry is incompatible")
    else:
        assistant = companion
        if (companion.get("architectures") != ["Gemma4DSparkModel"]
                or companion.get("target_model_type") != "gemma4_unified"
                or companion.get("hidden_size") != base.get("hidden_size")
                or companion.get("num_target_layers") != base.get("num_hidden_layers")
                or companion.get("block_size") != 7):
            raise ValueError("Gemma4 DSpark requires a matching target and block7 draft")
        layers = companion.get("target_layer_ids")
        if (not isinstance(layers, list) or not layers
                or any(type(i) is not int or not 0 <= i < base["num_hidden_layers"] for i in layers)
                or len(set(layers)) != len(layers)):
            raise ValueError("Invalid Gemma4 DSpark target layer IDs")
        mask = companion.get("mask_token_id")
        if type(mask) is not int or not 0 <= mask < base["vocab_size"]:
            raise ValueError("Invalid Gemma4 DSpark mask token")
    if (assistant.get("vocab_size") != base.get("vocab_size")
            or base.get("enable_moe_block") or assistant.get("enable_moe_block")):
        raise ValueError("Gemma4 pair vocabulary or dense topology is incompatible")
    for directory, config in ((source, raw), (draft, companion)):
        if (config.get("quantization_config") or config.get("text_config", config).get("quantization_config")
                or any((directory / name).exists() for name in
                       ("hf_quant_config.json", "quantize_config.json", "quant_config.json"))):
            raise ValueError("This Gemma4 paired profile requires unquantized checkpoints")
        if not list(directory.glob("*.safetensors")):
            raise ValueError("Gemma4 paired execution requires both local safetensors checkpoints")
    limit = request.max_sequence_length or 1024
    capacities = (base.get("max_position_embeddings"), assistant.get("max_position_embeddings"))
    if any(type(v) is not int or v < limit for v in capacities) or not (4 if execution.variant == "mtp" else 8) < limit <= 1024:
        raise ValueError("Gemma4 pair requires context above its verify size and at most 1024")
    return raw, draft, limit


def prepare(request, raw, draft: Path, limit: int, target: dict, staging: Path, log_path: Path, variant: str):
    """Forward exact checkpoints to the original exporter and native builder."""
    package = installed_package(target)
    if package.get("onnx") is not True:
        raise ValueError("Gemma4 paired execution requires an ONNX-enabled Edge SDK")
    source = Path(request.model_dir).resolve()
    engine, onnx = staging / "edge_llm/engine", staging / "onnx"
    checkpoint = staging / "edge_llm/checkpoint"
    checkpoint.mkdir(parents=True)
    shutil.copy2(source / "config.json", checkpoint / "config.json")
    (checkpoint / "draft").mkdir()
    shutil.copy2(draft / "config.json", checkpoint / "draft/config.json")
    env = subprocess_environment(
        {"EDGELLM_PLUGIN_PATH": package["plugin"]},
        prepend_paths={"LD_LIBRARY_PATH": str(Path(package["plugin"]).parent)},
    )
    exporter = [package["python"], "-I", "-m", "tensorrt_edgellm.scripts.export",
                str(source), str(onnx)]
    if variant == "mtp":
        commands = [exporter + ["--mtp", "--mtp-draft-dir", str(draft.resolve()),
                                "--skip-visual", "--skip-audio"]]
        draft_subdir, verify_size, draft_size, spec_type = "mtp_draft", 4, 4, "gemma4_mtp"
    else:
        commands = [exporter + [flag, "--dspark-draft-dir", str(draft.resolve()),
                                "--skip-visual", "--skip-audio"]
                    for flag in ("--dspark-base", "--dspark-draft")]
        draft_subdir, verify_size, draft_size, spec_type = "dspark_draft", 8, 7, "dspark"
    for subdirectory, flag in (("llm", "--specBase"), (draft_subdir, "--specDraft")):
        commands.append([package["onnx_builder"], "--onnxDir", str(onnx / subdirectory),
                         "--engineDir", str(engine), flag, "--maxInputLen", str(min(limit, 512)),
                         "--maxKVCacheCapacity", str(limit), "--maxBatchSize", "1",
                         "--maxVerifyTreeSize", str(verify_size), "--maxDraftTreeSize", str(draft_size)])
    commands.insert(0, [package["python"], "-I", "-c", _CHAT_ADMISSION, str(source)])
    with log_path.open("a", encoding="utf-8") as log:
        for command in commands:
            log.write(json.dumps(command) + "\n")
            log.flush()
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT,
                           cwd=staging, env=env)
    required = ("spec_base.engine", "spec_draft.engine", "base_config.json", "draft_config.json",
                "embedding.safetensors", "tokenizer.json", "tokenizer_config.json",
                "processed_chat_template.json")
    if variant == "dspark":
        required += ("dspark_heads.safetensors", "dspark_heads_info.json")
    for name in required:
        if not (engine / name).is_file() or (engine / name).stat().st_size == 0:
            raise ValueError(f"Edge Gemma4 MTP builder omitted required artifact: {name}")
    for role in ("base", "draft"):
        config = json.loads((engine / f"{role}_config.json").read_text())
        if config.get("spec_decode_type") != spec_type:
            raise ValueError("Edge returned a different Gemma4 speculative contract")
    files = {}
    for directory in (engine, checkpoint):
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"Gemma4 Edge output must not contain symlinks: {path}")
            if path.is_file():
                files[path.relative_to(staging).as_posix()] = path
    return files, {
        "version": 1, "edge_revision": EDGE_REVISION, "target": target, "precision": "fp16",
        "max_sequence_length": limit, "max_input_length": min(limit, 512), "max_batch_size": 1,
        "execution_variant": variant, "builder_flow": "onnx", "artifacts": list(files),
    }


def build(request, writer, execution) -> None:
    """Prepare the full pair before publication; never fall back to a base-only engine."""
    raw, draft, limit = validate_pair(request, execution)
    descriptor, name = tempfile.mkstemp(prefix=f".{request.output_path.name}.edge-onnx-",
                                        suffix=".log", dir=request.output_path.parent)
    os.close(descriptor)
    log_path = Path(name)
    with tempfile.TemporaryDirectory(prefix="trtmc-gemma4-edge-") as directory:
        try:
            target = local_target()
            if (target["os"], target["arch"], target["sm"]) != ("linux", "x86_64", 80):
                raise ValueError("This Gemma4 MTP Edge profile maps native x86_64 SM80")
            files, marker = prepare(request, raw, draft, limit, target, Path(directory), log_path, execution.variant)
        except Exception as error:
            with log_path.open("a", encoding="utf-8") as log:
                traceback.print_exception(error, file=log)
            _LOG.warning("Gemma4 Edge build failed: %s. Diagnostics: %s. "
                         "Native fallback cannot preserve the requested paired variant.", error, log_path)
            # The native family has no paired Gemma4 topology; do not substitute
            # an autoregressive decoder and misrepresent execution semantics.
            raise NotImplementedError("Native Gemma does not implement the requested paired variant") from error
        writer.set_header(family="gemma", task=request.task, backend=request.backend)
        for name, path in files.items():
            with path.open("rb") as source, writer.open_section(name) as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
        writer.add_json("edge_llm.json", marker)

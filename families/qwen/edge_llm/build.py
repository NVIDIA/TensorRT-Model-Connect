# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a Qwen bundle with the explicitly selected TensorRT Edge-LLM backend."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter


_SECTION_PREFIX = "edge_llm/"
_REQUIRED_FILES = (
    "config.json",
    "llm.engine",
    "embedding.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "processed_chat_template.json",
)


def _require_executable(name: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        raise RuntimeError(
            f"Qwen Edge-LLM requires {name!r} on PATH; install "
            "families/qwen/edge_llm/dependencies.txt and the official native tools"
        )
    return executable


def _validate_request(request: "BuildRequest") -> tuple[int, int]:
    if request.task != "text_generation":
        raise ValueError("Qwen Edge-LLM supports only task=text_generation")
    if request.backend != "edge_llm":
        raise ValueError("Qwen Edge-LLM requires backend=edge_llm")
    if request.precision != "fp16":
        raise ValueError("Qwen Edge-LLM currently supports only precision=fp16")
    if request.tensor_parallel_size != 1 or request.context_parallel_size != 1:
        raise ValueError("Qwen Edge-LLM currently supports only single-device builds")
    if request.quantization not in (None, "none"):
        raise ValueError("Qwen Edge-LLM currently supports only unquantized builds")
    if request.fp32_layers:
        raise ValueError("Qwen Edge-LLM does not support fp32 layer overrides")
    if request.dynamic_kv_cache:
        raise ValueError("Qwen Edge-LLM does not support dynamic_kv_cache")
    if request.graph_transform is not None:
        raise ValueError("Qwen Edge-LLM does not support a graph transform")
    if any(
        value is not None
        for value in (request.image_height, request.image_width, request.video_num_frames)
    ):
        raise ValueError("Qwen Edge-LLM text generation does not accept media dimensions")

    model_config = json.loads((Path(request.model_dir) / "config.json").read_text(encoding="utf-8"))
    if model_config.get("model_type") != "qwen3":
        raise ValueError("Qwen Edge-LLM currently supports only model_type=qwen3")

    max_cache_length = request.max_sequence_length or 4096
    max_input_length = min(max_cache_length, 1024)
    return max_input_length, max_cache_length


def _write_engine_sections(engine_dir: Path, writer: "BundleWriter") -> None:
    for name in _REQUIRED_FILES:
        path = engine_dir / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"TensorRT Edge-LLM engine output is missing {name}")
    files = sorted(path for path in engine_dir.rglob("*") if path.is_file())
    if not files:
        raise RuntimeError("TensorRT Edge-LLM produced an empty engine directory")
    for path in files:
        if path.is_symlink():
            raise RuntimeError(f"TensorRT Edge-LLM engine output contains a symlink: {path}")
        relative = path.relative_to(engine_dir).as_posix()
        with path.open("rb") as source, writer.open_section(_SECTION_PREFIX + relative) as section:
            shutil.copyfileobj(source, section)


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Run the installed official exporter and engine builder once."""

    max_input_length, max_cache_length = _validate_request(request)
    exporter = _require_executable("tensorrt-edgellm-export")
    engine_builder = _require_executable("llm_build")

    output_parent = Path(request.output_path).parent
    with tempfile.TemporaryDirectory(prefix=".trtmc-qwen-edge-", dir=output_parent) as directory:
        workspace = Path(directory)
        export_root = workspace / "onnx"
        engine_dir = workspace / "engine"
        subprocess.run(
            [exporter, str(request.model_dir), str(export_root), "--dtype=float16"],
            check=True,
        )
        onnx_dir = export_root / "llm"
        if not onnx_dir.is_dir():
            raise RuntimeError("TensorRT Edge-LLM exporter did not produce onnx/llm")
        build_command = [
            engine_builder,
            f"--onnxDir={onnx_dir}",
            f"--engineDir={engine_dir}",
            f"--maxInputLen={max_input_length}",
            f"--maxKVCacheCapacity={max_cache_length}",
            f"--maxBatchSize={request.max_batch_size}",
        ]
        if request.verbose:
            build_command.append("--debug")
        subprocess.run(build_command, check=True)

        writer.set_header(family="qwen", task=request.task, backend="edge_llm")
        writer.add_json(
            "edge_llm.json",
            {
                "max_input_length": max_input_length,
                "max_cache_length": max_cache_length,
                "max_batch_size": request.max_batch_size,
            },
        )
        _write_engine_sections(engine_dir, writer)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned Nemotron-H complete-network map and native preparation retry."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import tempfile
import traceback

from . import edge_llm
from .edge_quantization import source_quantization
from .edge_config import topology_matches

_LOG = logging.getLogger(__name__)
# Native platforms with recorded passing family profiles; admission is not qualification.
PLATFORMS = {"x86_64": (80, 120)}
EDGE_DISPATCH = {
    ("linux", arch, sm, precision): edge_llm.prepare
    for arch, sms in PLATFORMS.items()
    for sm in sms
    for precision in ("fp16", "fp8", "nvfp4")
    if precision == "fp16" or sm >= 100
}


def request_matches(request) -> bool:
    """Preserve native early rejection order for unmapped request controls."""
    return (
        request.family == "nemotron_h" and request.backend == "trt" and request.task == "text_generation"
        and request.precision.lower() == "fp16" and request.quantization in {None, "none", "fp8", "nvfp4"}
        and request.max_batch_size == request.tensor_parallel_size == request.context_parallel_size == 1
        and not request.dynamic_kv_cache and not request.fp32_layers and request.graph_transform is None
        and all(value is None for value in (request.image_height, request.image_width, request.video_num_frames))
    )


def candidate(request, raw: dict) -> bool:
    """Preserve native tokenizer contract and require exact source hybrid policy."""
    if not request_matches(request):
        return False
    precision = source_quantization(request, raw)
    if precision is None or not topology_matches(raw, precision):
        return False
    try:
        from .edge_tokenizer import source_chat_template
        source_chat_template(Path(request.model_dir))
        return True
    except (OSError, ValueError):
        return False


def platform_matches(raw: dict, target: dict) -> bool:
    """Admit the native runtime, not just its optional optimized SSD kernel."""
    # MambaPlugin falls back to scalar prefill when SSD cannot implement a shape.
    # Topology, source precision and the installed package are checked separately.
    return target["sm"] >= 80


def build(request, writer, native) -> None:
    """Prepare Edge privately or warn and retry unchanged native once.

    Publication and cancellation errors propagate. Runtime fallback is not part
    of this builder adapter; an Edge bundle is owned entirely by its runtime.
    """
    if not request_matches(request):
        native(request, writer)
        return
    raw = json.loads((Path(request.model_dir) / "config.json").read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("checkpoint config.json must contain an object")
    if not candidate(request, raw):
        native(request, writer)
        return
    if edge_llm.sequence_length(request, raw) > raw["max_position_embeddings"]:
        raise ValueError("Nemotron-H max_sequence_length exceeds checkpoint context capacity")
    failure = None
    descriptor, name = tempfile.mkstemp(prefix=f".{request.output_path.name}.edge-", suffix=".log",
                                        dir=request.output_path.parent)
    os.close(descriptor)
    log_path = Path(name)
    with tempfile.TemporaryDirectory(prefix="trtmc-nemotron-h-edge-") as directory:
        try:
            target = edge_llm.local_target()
            key = (target["os"], target["arch"], target["sm"], source_quantization(request, raw))
            adapter = EDGE_DISPATCH.get(key) if platform_matches(raw, target) else None
            if adapter is not None:
                files, marker = adapter(request, raw, target, Path(directory), log_path)
        except Exception as error:
            failure = error
            with log_path.open("a", encoding="utf-8") as log:
                traceback.print_exception(error, file=log)
            _LOG.warning("nemotron_h Edge build failed: %s. Diagnostics: %s. "
                         "Retrying native once with the unchanged request.", error, log_path, exc_info=True)
        else:
            if adapter is not None:
                edge_llm.publish(request, writer, files, marker)
                return
            log_path.unlink()
    try:
        native(request, writer)
    except Exception as error:
        if failure is not None:
            raise error from failure
        raise


def build_dflash(request, writer, draft: Path) -> None:
    """Preserve a paired request on failure; never replace it with a base-only build."""
    raw = json.loads((Path(request.model_dir) / "config.json").read_text(encoding="utf-8"))
    if not candidate(request, raw) or source_quantization(request, raw) != "nvfp4":
        raise ValueError("Nemotron-H DFlash ONNX requires a compatible NVFP4 target and TP1 text request")
    if edge_llm.sequence_length(request, raw) > raw["max_position_embeddings"]:
        raise ValueError("Nemotron-H max_sequence_length exceeds checkpoint context capacity")
    descriptor, name = tempfile.mkstemp(prefix=f".{request.output_path.name}.edge-onnx-", suffix=".log",
                                        dir=request.output_path.parent)
    os.close(descriptor)
    log_path = Path(name)
    with tempfile.TemporaryDirectory(prefix="trtmc-nemotron-h-dflash-", dir=request.output_path.parent) as directory:
        try:
            target = edge_llm.local_target()
            key = (target["os"], target["arch"], target["sm"], "nvfp4")
            if key not in EDGE_DISPATCH or not platform_matches(raw, target):
                raise ValueError("Nemotron-H DFlash ONNX is unavailable on this native platform")
            files, marker = edge_llm.prepare_dflash(request, raw, target, Path(directory), log_path, draft)
        except Exception as error:
            with log_path.open("a", encoding="utf-8") as log:
                traceback.print_exception(error, file=log)
            _LOG.warning("Nemotron-H DFlash Edge ONNX build failed: %s. Diagnostics: %s. "
                         "Native paired execution is unavailable; refusing base-only fallback.", error, log_path)
            raise NotImplementedError("Native Nemotron-H DFlash fallback is unavailable") from error
        edge_llm.publish(request, writer, files, marker)

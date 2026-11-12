# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Muse-Glimmer family-owned Edge route and unchanged native fallback."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import traceback

from . import builder as edge_llm

_LOG = logging.getLogger(__name__)
EDGE_DISPATCH = {("linux", "x86_64", 120, "nvfp4"): edge_llm.prepare}
_SHAPE = (52, 6656, 19968, 32, 2, 128, 202048)


def candidate(request, raw: dict) -> bool:
    text = raw.get("text_config", raw)
    shape = tuple(
        text.get(name)
        for name in (
            "num_hidden_layers",
            "hidden_size",
            "intermediate_size",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "vocab_size",
        )
    )
    return (
        raw.get("model_type") == "muse_glimmer"
        and text.get("model_type") == "muse_glimmer_text"
        and shape == _SHAPE
        and getattr(request, "execution_variant", "autoregressive") in {"autoregressive", "dflash"}
        and request.backend == "trt"
        and request.task == "text_generation"
        and request.precision.lower() == "fp16"
        and request.quantization in {None, "none", "nvfp4"}
        and request.max_batch_size
        == request.tensor_parallel_size
        == request.context_parallel_size
        == 1
        and not request.dynamic_kv_cache
        and not request.fp32_layers
        and request.graph_transform is None
        and all(
            value is None
            for value in (request.image_height, request.image_width, request.video_num_frames)
        )
    )


def build(request, writer, native) -> None:
    config_path = Path(request.model_dir) / "config.json"
    if not config_path.is_file():
        native(request, writer)
        return
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("text_config", raw), dict):
        raise ValueError(
            "Muse-Glimmer config.json must contain model and text configuration objects"
        )
    if not candidate(request, raw) or sys.platform != "linux" or not edge_llm.package_present():
        native(request, writer)
        return
    capacity = raw.get("text_config", raw).get("max_position_embeddings")
    if type(capacity) is not int or capacity <= 0:
        raise ValueError("Muse-Glimmer max_position_embeddings must be a positive integer")
    if request.max_sequence_length and request.max_sequence_length > capacity:
        raise ValueError("max_sequence_length exceeds checkpoint context capacity")
    failure = None
    descriptor, name = tempfile.mkstemp(
        prefix=f".{request.output_path.name}.edge-", suffix=".log", dir=request.output_path.parent
    )
    os.close(descriptor)
    log_path = Path(name)
    with tempfile.TemporaryDirectory(
        prefix=f".{request.output_path.name}.edge-", dir=request.output_path.parent
    ) as directory:
        try:
            target = edge_llm.local_target()
            weight_format = edge_llm.request_weight_format(request, raw)
            adapter = EDGE_DISPATCH.get((target["os"], target["arch"], target["sm"], weight_format))
            if adapter is not None:
                files, marker = adapter(request, raw, target, Path(directory), log_path)
        except Exception as error:
            failure = error
            with log_path.open("a", encoding="utf-8") as log:
                traceback.print_exception(error, file=log)
            _LOG.warning(
                "Muse-Glimmer Edge build failed: %s. Diagnostics: %s. "
                "Retrying native once with the unchanged request.",
                error,
                log_path,
                exc_info=True,
            )
        except BaseException:
            log_path.unlink(missing_ok=True)
            raise
        else:
            if adapter is not None:
                edge_llm.publish(request, writer, files, marker)
                log_path.unlink()
                return
            log_path.unlink()
    try:
        native(request, writer)
    except Exception as error:
        if failure is not None:
            raise error from failure
        raise

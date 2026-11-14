# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Laguna family builder: poolside ``LagunaForCausalLM`` as one dual-profile engine.

The bundle holds ``engine.plan`` (profile 0 prefills up to 128 rows per launch,
profile 1 decodes one row), ``runtime.json`` and the tokenizer files. Weights
stay at the checkpoint's native BF16 (``precision="bf16"``, the default) or are
widened to FP32 (``precision="fp32"``) for exact small-model parity runs.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import tensorrt as trt

from .checkpoint import BF16, LagunaWeights
from .config import LagunaConfig, load_config
from .graph import add_profiles, define_network


if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter


DEFAULT_MAX_SEQUENCE_LENGTH = 4096
_STORAGE = {"bf16": BF16, "fp32": np.dtype(np.float32)}
_BUNDLE_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "generation_config.json",
)


# Literal fragments of Laguna's chat_template.jinja that the runtime's renderer
# reproduces for one user turn: the leading token, the default system message,
# the system and user framing, and the generation prompt with its thinking
# directive. A template without all of them gets no native chat formatting.
_CHAT_TEMPLATE_FRAGMENTS = (
    '{{- "\u3008|EOS|\u3009" -}}',
    '"You are a helpful, conversationally-fluent assistant made by Poolside. You are here to be '
    'helpful to users through natural language conversations."',
    '{{- "<system>\\n" -}}',
    "{{- system_message.rstrip() -}}",
    '{{- "\\n</system>\\n" -}}',
    '{{- "<user>\\n" + content + "\\n</user>\\n" -}}',
    '{{- "<assistant>\\n" -}}',
    "{{- '</think>' -}}",
    "{{- '<think>' -}}",
)


def _log(message: str) -> None:
    print(f"[trtmc build] laguna: {message}", file=sys.stderr, flush=True)


def _chat_template(model_dir: Path) -> str | None:
    """``"laguna"`` when the checkpoint's chat template is the one the runtime renders."""
    path = model_dir / "chat_template.jinja"
    if not path.is_file():
        return None
    template = path.read_text(encoding="utf-8")
    if all(fragment in template for fragment in _CHAT_TEMPLATE_FRAGMENTS):
        return "laguna"
    _log("chat_template.jinja lacks the Laguna framing the runtime renders; chat is disabled")
    return None


def _build_plan(
    cfg: LagunaConfig, weights: LagunaWeights, max_cache: int, verbose: bool
) -> tuple[trt.IHostMemory, int]:
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.builder_optimization_level = 1
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 16 << 30)
    if weights.storage == np.float32:
        # An FP32 build is an exactness build: keep FP32 GEMMs off TF32.
        config.clear_flag(trt.BuilderFlag.TF32)
    graph = define_network(network, cfg, weights, max_cache)
    chunk = add_profiles(builder, config, max_cache)
    plan = builder.build_serialized_network(network, config)
    del graph, network, config, builder
    if plan is None:
        raise RuntimeError("TensorRT failed to build the Laguna engine")
    return plan, chunk


def _eos_token_ids(model_dir: Path, cfg: LagunaConfig) -> list[int]:
    path = model_dir / "generation_config.json"
    if not path.is_file():
        return list(cfg.eos_token_ids)
    generation = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(generation, dict):
        raise ValueError("generation_config.json must contain one JSON object")
    value = generation.get("eos_token_id", list(cfg.eos_token_ids))
    values = value if isinstance(value, list) else [value]
    if not values or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in values):
        raise ValueError(f"generation_config.json has an invalid eos_token_id: {value!r}")
    return values


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build one Laguna text-continuation bundle."""
    if request.dynamic_kv_cache:
        raise NotImplementedError("laguna does not support dynamic_kv_cache")
    if request.image_height is not None or request.image_width is not None:
        raise NotImplementedError("laguna is a text-only family; image sizes do not apply")
    if request.video_num_frames is not None:
        raise NotImplementedError("laguna does not support video_num_frames")
    if request.max_batch_size != 1:
        raise NotImplementedError("laguna supports only max_batch_size=1")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("laguna builds one single-GPU engine (tensor_parallel_size=1)")
    if request.context_parallel_size != 1:
        raise ValueError("laguna does not support context parallelism")
    if request.task != "text_continuation":
        raise ValueError("laguna supports only task=text_continuation")
    if request.backend != "trt":
        raise ValueError("laguna supports only the trt backend")
    if request.quantization not in (None, "none"):
        raise NotImplementedError(f"laguna does not support quantization={request.quantization!r}")
    if request.fp32_layers:
        raise NotImplementedError("laguna does not expose mixed-precision layer selection")
    if request.mtp_seq_len is not None:
        raise NotImplementedError("laguna does not support MTP speculative decoding")
    precision = str(request.precision).lower()
    if precision not in _STORAGE:
        raise ValueError("laguna precision must be bf16 (native) or fp32")

    model_dir = Path(request.model_dir)
    cfg = load_config(model_dir)
    max_cache = request.max_sequence_length or min(
        cfg.max_position_embeddings, DEFAULT_MAX_SEQUENCE_LENGTH
    )
    if max_cache > cfg.max_position_embeddings:
        raise ValueError("laguna max_sequence_length exceeds the checkpoint's context capacity")
    if not (model_dir / "tokenizer.json").is_file():
        raise FileNotFoundError(f"laguna needs tokenizer.json in {model_dir}")
    eos_token_ids = _eos_token_ids(model_dir, cfg)

    started = time.monotonic()
    weights = LagunaWeights(model_dir, cfg, _STORAGE[precision])
    _log(
        f"loaded {weights.nbytes() / 2**30:.2f} GiB of {precision} weights "
        f"in {time.monotonic() - started:.0f} s"
    )
    started = time.monotonic()
    plan, chunk = _build_plan(cfg, weights, max_cache, bool(request.verbose))
    weights.release()
    _log(f"built a {plan.nbytes / 2**30:.2f} GiB engine in {time.monotonic() - started:.0f} s")

    writer.set_header(family="laguna", task=request.task, backend=request.backend)
    with writer.open_section("engine.plan") as section:
        section.write(plan)
    del plan
    writer.add_json(
        "runtime.json",
        {
            "vocab_size": cfg.vocab_size,
            "hidden_size": cfg.hidden_size,
            "num_hidden_layers": cfg.num_hidden_layers,
            "num_key_value_heads": cfg.num_key_value_heads,
            "head_dim": cfg.head_dim,
            "bos_token_id": cfg.bos_token_id,
            "eos_token_ids": eos_token_ids,
            "pad_token_id": cfg.pad_token_id,
            "max_cache_length": max_cache,
            "prefill_chunk_rows": chunk,
            "precision": precision,
            "chat_template": _chat_template(model_dir),
        },
    )
    for filename in _BUNDLE_FILES:
        path = model_dir / filename
        if path.is_file():
            writer.add_bytes(filename, path.read_bytes())

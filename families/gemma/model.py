# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gemma family plugin — applies +1.0 to RMSNorm gamma and sqrt(hidden) embed scale."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import math
from pathlib import Path

from . import graph_blocks
from .config import ModelConfig
from .checkpoint_mapper import (
    WeightDict,
    _has_tensor,
    _load_tensor,
    _open_safetensors,
    load_standard_weights,
)
from .parallel import ParallelConfig
from .parallel import normalize_parallel_config
from .dual_profile_decoder_tp_builder import build_dual_profile_tp_decoder_engine
from .standard_decoder_builder import build_standard_decoder_engine


if TYPE_CHECKING:
    from .build_request import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter



# Gemma 3 at 4B and above is published as Gemma3ForConditionalGeneration, which
# keeps the decoder under `language_model.model` and adds a vision tower and a
# projector beside it. This family builds the decoder only - image inputs are
# refused in build() - so the vision tensors are simply never read.
_DECODER_PREFIXES = ("model", "language_model.model")


def _decoder_prefix(readers) -> str:
    """Find where the decoder lives in this checkpoint."""
    for prefix in _DECODER_PREFIXES:
        if _has_tensor(readers, f"{prefix}.embed_tokens.weight"):
            return prefix
    raise ValueError(
        "Gemma checkpoint has no decoder embedding under "
        + " or ".join(f"{prefix}.embed_tokens.weight" for prefix in _DECODER_PREFIXES)
    )




# Fields a published Gemma 3 config may leave out because transformers supplies
# them. google/gemma-3-4b-it omits vocab_size and hidden_activation; the unsloth
# mirror of the same weights states both, which is why local runs passed and
# internal CI did not. rms_norm_eps is listed because this family would
# otherwise fall back to 1e-5 where Gemma 3 uses 1e-6 - wrong, and silently so.
# What Gemma3TextConfig fills in when the checkpoint stays silent. Google's own
# configs state geometry and nothing else, so every value here is load-bearing;
# the mirrors that write them out explicitly are why this went unnoticed.
#
# `rope_theta` and `rope_local_base_freq` come from that class's rope_parameters:
#   sliding_attention -> {"rope_type": "default", "rope_theta": 10000.0}
#   full_attention    -> {"rope_type": "linear",  "rope_theta": 1000000.0}
_GEMMA3_CONFIG_DEFAULTS = {
    "hidden_activation": "gelu_pytorch_tanh",
    "rms_norm_eps": 1e-6,
    "max_position_embeddings": 131072,
    "head_dim": 256,
    "query_pre_attn_scalar": 256,
    "rope_theta": 1000000.0,
    "rope_local_base_freq": 10000.0,
    "sliding_window_pattern": 6,
}

# Defaults that belong in config.raw rather than on the config object, because
# graph_blocks reads them back through _gemma_raw.
_GEMMA3_RAW_DEFAULTS = (
    "query_pre_attn_scalar",
    "rope_local_base_freq",
    "sliding_window_pattern",
)


def _apply_gemma3_config_defaults(config: ModelConfig, readers, model_prefix: str) -> None:
    """Fill in what a Gemma 3 config may legitimately omit.

    Shapes are taken from the checkpoint wherever they can be, because the
    tensors are authoritative and a default is only a guess. Only the scalars
    that no tensor carries fall back to the transformers value.
    """
    if str(config.model_type).lower() not in _GEMMA3_MODEL_TYPES:
        return
    raw = graph_blocks._gemma_raw(config)
    if not (raw.get("hidden_activation") or raw.get("hidden_act")):
        config.hidden_act = _GEMMA3_CONFIG_DEFAULTS["hidden_activation"]
    if raw.get("rms_norm_eps") is None:
        config.rms_norm_eps = _GEMMA3_CONFIG_DEFAULTS["rms_norm_eps"]
    if raw.get("max_position_embeddings") is None:
        config.max_position_embeddings = _GEMMA3_CONFIG_DEFAULTS["max_position_embeddings"]
    if raw.get("rope_theta") is None:
        # Gemma 3's global layers use 1e6. The parser's own fallback is 1e4,
        # which is the *local* base, so staying silent here rebuilds every
        # global layer on the wrong rope table.
        config.rope_theta = _GEMMA3_CONFIG_DEFAULTS["rope_theta"]
    for key in _GEMMA3_RAW_DEFAULTS:
        # Written into raw, not onto the config: gemma3_attention_schedule and
        # gemma_attention_scale read these back through _gemma_raw. A nested
        # text_config still wins, because _gemma_raw overlays it last.
        if raw.get(key) is None:
            config.raw[key] = _GEMMA3_CONFIG_DEFAULTS[key]
    if raw.get("head_dim") is None:
        # head_dim is a derived property; _head_dim is the stated override it
        # reads first. Without it the property falls back to
        # hidden_size // num_attention_heads, which is not Gemma 3's head_dim.
        config._head_dim = _GEMMA3_CONFIG_DEFAULTS["head_dim"]
    head_dim = int(config.head_dim)
    # q and k projections state the attention widths outright. Reading them is
    # better than defaulting: google/gemma-3-4b-it omits num_key_value_heads,
    # and the parser then falls back to num_attention_heads, giving a K/V cache
    # twice the width the checkpoint actually has.
    if raw.get("num_attention_heads") is None:
        rows = _projection_rows(readers, model_prefix, "q_proj")
        if rows and head_dim:
            config.num_attention_heads = rows // head_dim
    if raw.get("num_key_value_heads") is None:
        rows = _projection_rows(readers, model_prefix, "k_proj")
        if rows and head_dim:
            config.num_key_value_heads = rows // head_dim


def _projection_rows(readers, model_prefix: str, projection: str) -> int:
    """Output width of a layer-0 attention projection, from tensor metadata."""
    key = f"{model_prefix}.layers.0.self_attn.{projection}.weight"
    reader = readers.tensor_map.get(key)
    if reader is None:
        return 0
    return int(reader.get_slice(key).get_shape()[0])


def _embedding_vocab_size(readers, model_prefix: str) -> int:
    """Read the vocabulary size off the embedding rather than the config.

    google/gemma-3-4b-it does not state vocab_size in its text_config, so the
    config parser defaults it to 0 and the checkpoint mapper then rejects the
    embedding it just loaded. The tensor itself is authoritative; the shape
    comes from safetensors metadata, so nothing is read twice. The unsloth
    mirror of the same weights does state it, which is why this only appeared
    against the official checkpoint.
    """
    key = f"{model_prefix}.embed_tokens.weight"
    reader = readers.tensor_map.get(key)
    if reader is None:
        raise ValueError(f"Gemma checkpoint has no {key}")
    return int(reader.get_slice(key).get_shape()[0])


class _GemmaModel:
    def load_weights(
        self,
        model_dir: str,
        config: ModelConfig,
        *,
        precision: str = "fp32",
    ) -> WeightDict:
        readers = _open_safetensors(Path(model_dir))
        model_prefix = _decoder_prefix(readers)
        _apply_gemma3_config_defaults(config, readers, model_prefix)
        if config.vocab_size <= 0:
            config.vocab_size = _embedding_vocab_size(readers, model_prefix)
        weights = load_standard_weights(
            model_dir, config, precision=precision, model_prefix=model_prefix
        )

        # Fix 1: Gemma uses (1 + gamma) * normalized instead of gamma * normalized.
        for layer_idx in range(config.num_hidden_layers):
            prefix = f"layer.{layer_idx}"
            hf_prefix = f"{model_prefix}.layers.{layer_idx}"
            weights[f"{prefix}.input_norm"] = weights[f"{prefix}.input_norm"] + 1.0
            weights[f"{prefix}.post_attn_norm"] = weights[f"{prefix}.post_attn_norm"] + 1.0
            pre_ffn_key = f"{hf_prefix}.pre_feedforward_layernorm.weight"
            if _has_tensor(readers, pre_ffn_key):
                weights[f"{prefix}.pre_ffn_norm"] = (
                    _load_tensor(readers, pre_ffn_key).astype("float32") + 1.0
                )
            post_ffn_key = f"{hf_prefix}.post_feedforward_layernorm.weight"
            if _has_tensor(readers, post_ffn_key):
                weights[f"{prefix}.post_ffn_norm"] = (
                    _load_tensor(readers, post_ffn_key).astype("float32") + 1.0
                )
            # Gemma 3's per-head query/key norms are RMSNorm too, so they take
            # the same (1 + gamma). The mapper loads them whenever they are
            # present; without this they are applied as gamma alone, which is
            # near zero and destroys the attention scores.
            for norm in ("q_norm", "k_norm"):
                norm_key = f"{prefix}.{norm}"
                if norm_key in weights:
                    weights[norm_key] = weights[norm_key] + 1.0
        weights["final_norm"] = weights["final_norm"] + 1.0

        # Fix 2: Gemma scales embedding by sqrt(hidden_size).
        scale = math.sqrt(config.hidden_size)
        weights["embedding"] = weights["embedding"] * scale

        return weights

    def build_engine(
        self,
        config: ModelConfig,
        weights: WeightDict,
        max_cache_length: int,
        *,
        precision: str = "fp32",
        quant_ctx=None,
        verbose: bool = False,
        parallel_config=None,
    ) -> bytes:
        activation = _checkpoint_gated_activation(config)
        parallel = normalize_parallel_config(parallel_config)
        if parallel.enabled:
            return build_dual_profile_tp_decoder_engine(
                config,
                weights,
                max_cache_length,
                precision=precision,
                quant_ctx=quant_ctx,
                verbose=verbose,
                activation=activation,
                parallel_config=parallel,
            )

        return build_standard_decoder_engine(
            config,
            weights,
            max_cache_length,
            precision=precision,
            quant_ctx=quant_ctx,
            verbose=verbose,
            activation=activation,
        )


def _checkpoint_gated_activation(config: ModelConfig) -> str:
    """Return the checkpoint-declared activation for Gemma's gated MLP."""
    # Gemma 3 at 4B and above states this under text_config beside a vision
    # config, so the nested fields have to be consulted too.
    raw = graph_blocks._gemma_raw(config)
    activation = str(
        config.hidden_act
        or raw.get("hidden_activation")
        or raw.get("hidden_act")
        or ""
    ).strip()
    supported = {"gelu_pytorch_tanh", "gelu_new", "gelu", "silu"}
    if activation not in supported:
        raise ValueError(
            "Gemma requires a supported checkpoint gated activation; "
            f"got {activation or '<missing>'!r}"
        )
    return activation


_BUNDLE_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.model",
)


# The model types this family builds, matching families/gemma/support.py. The
# check used to be a "gemma" prefix, which also accepted gemma3 and gemma4:
# those add sliding-window attention and a second rope table, neither of which
# this family has, so the prefix let them build a full-attention graph and
# generate quietly wrong text rather than being refused.
_GEMMA3_MODEL_TYPES = frozenset({"gemma3", "gemma3_text"})
# Gemma 3 activations do not fit fp16. Running the reference in fp32 and taking
# the largest absolute value leaving each decoder layer, against the fp16
# maximum of 65504:
#
#   gemma-3-270m   peak 102956   11 of 18 layers over
#   gemma-3-1b     peak  61040    0 of 26 layers over (a 7% margin)
#   gemma-3-4b     peak 298680   29 of 34 layers over
#
# The two that overflow emit token 0 repeatedly. The 1B stays inside the range
# on one prompt by 7%, which is luck rather than headroom, so fp16 is refused
# for the generation rather than per width. Gemma 2 peaks at 4060 and keeps it.
_FP16_UNSAFE_MODEL_TYPES = _GEMMA3_MODEL_TYPES


_SUPPORTED_MODEL_TYPES = frozenset({"gemma", "gemma2"}) | _GEMMA3_MODEL_TYPES


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result



def _eos_token_ids(value: object) -> list[int]:
    """Normalise the stop tokens into a list.

    Gemma 2 ships ``[1, 107]`` and Gemma 3 ``[1, 106]``. Keeping only the first
    drops ``<end_of_turn>``, so a chat turn never terminates. Booleans are
    rejected because ``bool`` is an ``int`` subclass and would pass as an id.
    """
    values = value if isinstance(value, list) else [value]
    ids: list[int] = []
    for item in values:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError("gemma eos_token_id must be an integer or a list of integers")
        ids.append(int(item))
    if not ids:
        raise ValueError("gemma eos_token_id must name at least one token")
    return ids


def _runtime_config(model_dir: Path, config: ModelConfig, **updates) -> dict:
    runtime = {
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
        "bos_token_id": config.bos_token_id,
        "eos_token_id": config.eos_token_id,
        "pad_token_id": config.pad_token_id,
    }
    eos = config.eos_token_id
    generation_path = model_dir / "generation_config.json"
    if generation_path.is_file():
        generation = json.loads(generation_path.read_text(encoding="utf-8"))
        if not isinstance(generation, dict):
            raise ValueError("generation_config.json must contain one JSON object")
        if "eos_token_id" in generation:
            eos = generation["eos_token_id"]
    eos_token_ids = _eos_token_ids(eos)
    # The scalar stays the first id so a bundle keeps working on a runtime that
    # predates the list; the list is written only when it adds something. Gemma
    # 2 names [1, 107] and Gemma 3 names [1, 106], and in both the second id is
    # <end_of_turn> - the token a chat turn actually ends on.
    runtime["eos_token_id"] = eos_token_ids[0]
    if len(eos_token_ids) > 1:
        runtime["eos_token_ids"] = eos_token_ids
    runtime.update(updates)
    return runtime


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build one Gemma bundle through family-owned code only."""
    from .build_request import coerce_request

    request = coerce_request(request)
    from .edge_llm.config import GemmaBuildRequest

    if isinstance(request, GemmaBuildRequest) and request.execution is not None:
        from .edge_llm.builder import build as build_pair

        request.execution.validate_local()
        build_pair(request, writer, request.execution)
        return

    if request.dynamic_kv_cache:
        raise NotImplementedError("gemma does not support dynamic_kv_cache")

    if request.image_height is not None:
        raise NotImplementedError("gemma does not support image_height")

    if request.image_width is not None:
        raise NotImplementedError("gemma does not support image_width")

    if request.video_num_frames is not None:
        raise NotImplementedError("gemma does not support video_num_frames")

    if request.max_batch_size != 1:
        raise NotImplementedError("gemma does not support max_batch_size")

    if request.context_parallel_size != 1:
        raise ValueError("this family does not support context parallelism")

    if request.task != "text_generation":
        raise ValueError("gemma supports only task=text_generation")

    model_dir = Path(request.model_dir)
    config = ModelConfig.from_dir(model_dir)
    if str(config.model_type).lower() not in _SUPPORTED_MODEL_TYPES:
        raise ValueError(f"Gemma does not support model_type={config.model_type!r}")
    precision = str(request.precision).lower()
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("Gemma precision must be fp32, fp16, or bf16")
    if precision == "fp16" and str(config.model_type).lower() in _FP16_UNSAFE_MODEL_TYPES:
        raise NotImplementedError(
            f"gemma does not support fp16 for {config.model_type!r}: its activations "
            "exceed the fp16 range and the engine returns a single repeated token; "
            "use bf16 or fp32"
        )
    max_sequence_length = _positive_int(
        request.max_sequence_length or min(config.max_position_embeddings, 256),
        "max_sequence_length",
    )
    # Gemma 2 and Gemma 3 interleave sliding-window and global attention. This
    # family builds full attention for every layer, which is the same thing
    # only while the sequence stays inside the window. Refuse anything longer
    # rather than return quietly wrong text.
    window = config.raw.get("sliding_window")
    if window and max_sequence_length > int(window):
        raise NotImplementedError(
            f"gemma builds full attention for every layer, so it supports "
            f"max_sequence_length up to the checkpoint's sliding_window "
            f"({int(window)}); {max_sequence_length} was requested"
        )
    if max_sequence_length > config.max_position_embeddings:
        raise ValueError("Gemma max_sequence_length exceeds checkpoint context capacity")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("Gemma has no qualified family-owned quantized build")
    if request.fp32_layers:
        raise NotImplementedError("Gemma does not expose mixed-precision layer selection")

    parallel = ParallelConfig(
        tp_size=_positive_int(request.tensor_parallel_size, "tensor_parallel_size")
    )
    parallel.validate()
    model = _GemmaModel()
    config.raw["_model_dir"] = str(model_dir)
    config.raw["_resolved_build_precision"] = precision
    config.raw["_parallel_build_enabled"] = parallel.enabled
    weights = model.load_weights(str(model_dir), config, precision=precision)

    writer.set_header(family="gemma", task=request.task, backend=request.backend)
    if parallel.enabled:
        for rank in range(parallel.tp_size):
            plan = model.build_engine(
                config,
                weights,
                max_sequence_length,
                precision=precision,
                quant_ctx=None,
                verbose=bool(request.verbose),
                parallel_config=parallel.for_rank(rank),
            )
            writer.add_bytes(f"engine.rank{rank}.plan", plan)
        layout = "dual_profile"
    else:
        config.raw["_decoder_engine_role"] = "prefill"
        prefill = model.build_engine(
            config,
            weights,
            max_sequence_length,
            precision=precision,
            quant_ctx=None,
            verbose=bool(request.verbose),
            parallel_config=parallel,
        )
        config.raw["_decoder_engine_role"] = "decode"
        decode = model.build_engine(
            config,
            weights,
            max_sequence_length,
            precision=precision,
            quant_ctx=None,
            verbose=bool(request.verbose),
            parallel_config=parallel,
        )
        config.raw.pop("_decoder_engine_role", None)
        writer.add_bytes("engine.plan", decode)
        writer.add_bytes("prefill.plan", prefill)
        layout = "split"

    writer.add_json(
        "runtime.json",
        _runtime_config(
            model_dir,
            config,
            precision=precision,
            max_cache_length=max_sequence_length,
            decoder_engine_layout=layout,
            tensor_parallel_size=parallel.tp_size,
            tensor_parallel_mode="tensor_parallel" if parallel.enabled else "single",
        ),
    )
    for filename in _BUNDLE_FILES:
        path = model_dir / filename
        if path.is_file():
            writer.add_bytes(filename, path.read_bytes())

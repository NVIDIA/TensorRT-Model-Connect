# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen family plugin — Qwen, Qwen2, Qwen3, QwQ (text-only, not VL).

Unquantized Qwen3 uses only the family-owned TensorRT native KV path. Qualified
FP8 builds retain the family-owned quantized graph; all other unsupported
Qwen3 modes fail closed. Other Qwen variants retain their explicit graph routes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .config import ModelConfig
from .parallel import ParallelConfig
from .checkpoint_mapper import WeightDict, load_standard_weights
from .parallel import normalize_parallel_config
from .build_routing import (
    native_kv_architecture_capability,
    native_kv_build_capability,
    native_kv_cache_geometry,
)
from . import checkpoint_mapper
from . import weight_stripping
from .native_kv_contract import validate_native_kv_weights
from .dual_profile_decoder_builder import build_dual_profile_decoder_engine
from .standard_decoder_builder import build_standard_decoder_engine
from .dual_profile_decoder_tp_builder import build_dual_profile_tp_decoder_engine


def _quant_format_name(quant_ctx) -> str | None:
    quant_format = getattr(getattr(quant_ctx, "profile", None), "format", None)
    return getattr(quant_format, "name", None)


def _uses_qualified_qwen3_fp8_path(
    config: ModelConfig,
    max_cache_length: int,
    *,
    precision: str,
    quant_ctx,
    parallel: ParallelConfig,
    debug_layer_outputs: bool,
) -> bool:
    """Recognize the explicit, previously qualified Qwen3 FP8 graph route."""

    raw = config.raw
    precision_name = str(precision).lower()
    qualified_route = (
        not parallel.enabled
        and precision_name == "fp16"
        and raw.get("_decoder_engine_layout") == "dual_profile"
    ) or (parallel.enabled and parallel.tp_size == 4 and precision_name == "bf16")
    if (
        not native_kv_architecture_capability(config).eligible
        or _quant_format_name(quant_ctx) != "fp8"
        or not qualified_route
        or debug_layer_outputs
        or raw.get("_fp32_layers")
    ):
        return False
    try:
        native_kv_cache_geometry(config, max_cache_length)
    except ValueError:
        return False
    return True


if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter



def _strip_weights_requested(config: ModelConfig) -> bool:
    """Read the family-owned weight-stripping build flag.

    Combinations that cannot strip are rejected in build() before any weight
    is loaded, so this is a plain read.
    """
    return bool(config.raw.get("_strip_weights"))


class _QwenModel:
    def default_build_precision(self, config: ModelConfig) -> str:
        capability = native_kv_architecture_capability(config)
        return "bf16" if capability.eligible else "fp32"

    def default_max_cache_length(self, config: ModelConfig) -> int:
        """Use the model's complete context for native Qwen3."""
        capability = native_kv_architecture_capability(config)
        return int(config.max_position_embeddings) if capability.eligible else 256

    def load_weights(
        self,
        model_dir: str,
        config: ModelConfig,
        *,
        precision: str = "fp32",
    ) -> WeightDict:
        return load_standard_weights(model_dir, config, precision=precision)

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
        debug_layer_outputs: bool = False,
    ) -> bytes:
        parallel = normalize_parallel_config(parallel_config)
        config.raw.pop("_native_kv_cache_metadata", None)
        capability = native_kv_build_capability(
            config,
            precision=precision,
            max_cache_length=max_cache_length,
            parallel_enabled=parallel.enabled,
            quantized=quant_ctx is not None,
            debug_layer_outputs=debug_layer_outputs,
        )
        qualified_fp8 = _uses_qualified_qwen3_fp8_path(
            config,
            max_cache_length,
            precision=precision,
            quant_ctx=quant_ctx,
            parallel=parallel,
            debug_layer_outputs=debug_layer_outputs,
        )
        if capability.eligible:
            validate_native_kv_weights(config, weights)
            config.raw["_native_kv_cache_metadata"] = {
                "native_kv_contract_version": 1,
                "native_kv_cache": True,
            }
            role = str(config.raw.get("_decoder_engine_role", ""))
            if role not in ("prefill", "decode"):
                raise ValueError(
                    "native Qwen3 requires explicit split engine role "
                    f"'prefill' or 'decode', got {role!r}"
                )
            def _build_native() -> bytes:
                return build_dual_profile_decoder_engine(
                    config,
                    weights,
                    max_cache_length,
                    precision=precision,
                    quant_ctx=None,
                    verbose=verbose,
                    profile_mode=role,
                    native_kv_cache=True,
                )

            if not _strip_weights_requested(config):
                return _build_native()
            # Stripped build: GEMM operands become null placeholders and their
            # values are collected for a refit sidecar instead of being baked.
            session = weight_stripping.StripSession()
            with weight_stripping.stripping(session):
                plan = _build_native()
            config.raw.setdefault("_strip_sessions", {})[role] = session
            print(f"[trtmc build] Stripped {len(session.entries)} weights "
                  f"({session.total_bytes / (1 << 20):.1f} MiB) from the "
                  f"{role} engine", file=sys.stderr)
            return plan

        if capability.applicable and not qualified_fp8:
            raise ValueError(
                f"Qwen3 requires the fixed-capacity native KV path: {capability.reason}"
            )

        if parallel.enabled:
            if debug_layer_outputs:
                raise NotImplementedError(
                    "Qwen tensor-parallel debug layer outputs are not supported"
                )
            return build_dual_profile_tp_decoder_engine(
                config,
                weights,
                max_cache_length,
                precision=precision,
                quant_ctx=quant_ctx,
                verbose=verbose,
                parallel_config=parallel,
            )
        return build_standard_decoder_engine(
            config,
            weights,
            max_cache_length,
            precision=precision,
            quant_ctx=quant_ctx,
            verbose=verbose,
            debug_layer_outputs=debug_layer_outputs,
        )

    def get_bundle_config_overrides(
        self,
        config: ModelConfig,
    ) -> dict | None:
        """Mark bundles that use the native KV runtime contract."""
        metadata = config.raw.get("_native_kv_cache_metadata")
        return dict(metadata) if isinstance(metadata, dict) else None


_BUNDLE_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.model",
)


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


def _runtime_config(model_dir: Path, config: ModelConfig, model: _QwenModel, **updates) -> dict:
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
    runtime.update(model.get_bundle_config_overrides(config) or {})
    generation_path = model_dir / "generation_config.json"
    if generation_path.is_file():
        generation = json.loads(generation_path.read_text(encoding="utf-8"))
        if not isinstance(generation, dict):
            raise ValueError("generation_config.json must contain one JSON object")
        if "eos_token_id" in generation:
            runtime["eos_token_id"] = generation["eos_token_id"]
    runtime.update(updates)
    return runtime



def _add_refit_weights_section(config: ModelConfig, writer: "BundleWriter") -> None:
    """Carry the refit weights for a stripped plan inside the bundle.

    A stripped engine holds no weight values, so the bundle ships them in a
    safetensors section and the runtime feeds them to ``IRefitter`` before
    creating any execution context. Both the prefill and decode engines want
    the identical set, so one section serves both and the weights are stored
    once rather than baked twice.
    """
    sessions = config.raw.get("_strip_sessions") or {}
    if not sessions:
        return
    from safetensors.numpy import save

    merged: dict = {}
    for role, session in sorted(sessions.items()):
        for name, array in session.entries.items():
            existing = merged.get(name)
            if existing is None:
                merged[name] = array
            elif existing.shape != array.shape or existing.dtype != array.dtype:
                raise ValueError(
                    f"engine role {role!r} disagrees on weight {name!r}: "
                    f"{existing.shape}/{existing.dtype} vs {array.shape}/{array.dtype}")
    # Under native layout every stripped weight is byte-identical to a tensor
    # already in the checkpoint, so the bundle can point at it instead of
    # carrying a second copy. Verified byte-for-byte below; any mismatch falls
    # back to embedding the weights.
    manifest = _refit_manifest(config, merged)
    if manifest is not None:
        # Section names are the contract with the C++ runtime; they must match
        # kQwenRefitManifestSection / kQwenRefitWeightsSection in
        # runtime/refit_weights.h.
        writer.add_json("refit_manifest.json", manifest)
        total = sum(e["nbytes"] for e in manifest["weights"].values())
        print(f"[trtmc build] Refit manifest: {len(manifest['weights'])} weights "
              f"({total / (1 << 20):.1f} MiB) referenced in place from the checkpoint "
              f"-- no weights copied into the bundle", file=sys.stderr)
        return

    payload = save(merged)
    writer.add_bytes("refit_weights", payload)
    print(f"[trtmc build] Refit weights section: {len(merged)} weights "
          f"({len(payload) / (1 << 20):.1f} MiB)", file=sys.stderr)


def _refit_manifest(config: ModelConfig, merged: dict) -> dict | None:
    """Describe where each stripped weight already lives in the checkpoint.

    Every value must be a TensorRef -- the loader hands one out only when the
    weight needs no transform, so its location is the weight. Nothing has to be
    mapped back to an HF tensor name or re-read to be checked: the offset came
    from the file the builder read, rather than being reconstructed from a
    parallel table that could go stale.

    Returns None when any weight was materialized (an fp16 build converts, so
    its bytes are not the checkpoint's), and the caller embeds them instead.
    """
    model_dir = Path(str(config.raw.get("_model_dir") or ""))
    if not model_dir.is_dir():
        return None

    entries: dict[str, dict] = {}
    files: dict[str, dict] = {}
    for name, value in merged.items():
        if not isinstance(value, checkpoint_mapper.TensorRef):
            return None
        dtype = checkpoint_mapper.ST_DTYPE_NAMES.get(np.dtype(value.dtype))
        if dtype is None:
            return None
        try:
            # The runtime resolves entries against checkpoint_dir, which an env
            # override may repoint, so the manifest must name files relatively.
            relative = value.path.relative_to(model_dir)
        except ValueError:
            return None
        entries[name] = {
            "file": str(relative), "offset": value.offset, "nbytes": value.nbytes,
            "dtype": dtype, "shape": [int(d) for d in value.shape],
        }
        files.setdefault(str(relative), {"nbytes": value.path.stat().st_size})

    return {
        "schema_version": 1,
        "checkpoint_dir": str(model_dir),
        "files": files,
        "weights": entries,
    }


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build one Qwen bundle through family-owned code."""
    if request.dynamic_kv_cache:
        raise NotImplementedError("qwen does not support dynamic_kv_cache")

    if request.image_height is not None:
        raise NotImplementedError("qwen does not support image_height")

    if request.image_width is not None:
        raise NotImplementedError("qwen does not support image_width")

    if request.video_num_frames is not None:
        raise NotImplementedError("qwen does not support video_num_frames")

    if request.max_batch_size != 1:
        raise NotImplementedError("qwen does not support max_batch_size")

    if request.context_parallel_size != 1:
        raise ValueError("this family does not support context parallelism")

    if request.task != "text_generation":
        raise ValueError("qwen supports only task=text_generation")
    model_dir = Path(request.model_dir)
    config = ModelConfig.from_dir(model_dir)
    model_type = str(config.model_type).lower()
    unsupported_variant = (
        "vl" in model_type
        or "moe" in model_type
        or "omni" in model_type
        or "image" in model_type
        or model_type in {"qwen3_5", "qwen3.5"}
    )
    if unsupported_variant or not (model_type.startswith("qwen") or model_type.startswith("qwq")):
        raise ValueError(f"Qwen does not support model_type={config.model_type!r}")
    precision = str(request.precision).lower()
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("Qwen precision must be fp32, fp16, or bf16")
    model = _QwenModel()
    default_length = model.default_max_cache_length(config)
    max_sequence_length = _positive_int(
        request.max_sequence_length or default_length, "max_sequence_length"
    )
    if max_sequence_length > config.max_position_embeddings:
        raise ValueError("Qwen max_sequence_length exceeds checkpoint context capacity")
    quantized = request.quantization == "fp8"
    if request.quantization not in {None, "none", "fp8"}:
        raise NotImplementedError(f"Qwen does not support quantization={request.quantization!r}")
    if request.fp32_layers:
        raise NotImplementedError("Qwen does not expose mixed-precision layer selection")
    parallel = ParallelConfig(
        tp_size=_positive_int(request.tensor_parallel_size, "tensor_parallel_size")
    )
    parallel.validate()
    if quantized:
        qualified_route = model_type == "qwen3" and (
            (not parallel.enabled and precision == "fp16")
            or (parallel.tp_size == 4 and precision == "bf16")
        )
        if not qualified_route:
            raise ValueError("Qwen FP8 supports only Qwen3 FP16 single-device or BF16 TP4 builds")
    config.raw["_model_dir"] = str(model_dir)
    config.raw["_resolved_build_precision"] = precision
    config.raw["_parallel_build_enabled"] = parallel.enabled
    config.raw["_quantized_build_requested"] = quantized
    strip_weights = bool(getattr(request, "strip_weights", False))
    if strip_weights:
        # Validated here rather than at engine-build time: only the native KV
        # path consults the flag, so an unsupported build would otherwise drop
        # it silently and emit a fully baked bundle.
        if quantized:
            raise ValueError("strip_weights is not supported for quantized builds")
        if parallel.enabled:
            raise ValueError("strip_weights is not supported for tensor-parallel builds")
        if not native_kv_architecture_capability(config).eligible:
            # Only the native KV split path strips. Rejecting here keeps the
            # layout switch below confined to the builder that understands it.
            raise ValueError(
                "strip_weights requires the native Qwen3 KV path "
                f"({native_kv_architecture_capability(config).reason})")
    config.raw["_strip_weights"] = strip_weights
    quant_ctx = None
    if quantized:
        from . import graph_ops
        from .quantization import calibrate_qwen_fp8

        config.raw["_decoder_engine_layout"] = "dual_profile"
        quant_ctx = calibrate_qwen_fp8(model_dir, config, graph_ops)
    # Keeping the checkpoint's own layout is not a user choice: it is how a
    # stripped build avoids transforming weights at all, which is what lets the
    # bundle reference the checkpoint in place instead of carrying a copy. It
    # needs the engine dtype to equal the checkpoint dtype, so an fp16 build
    # keeps the transformed layout and ships an embedded refit section instead.
    # Quantized, tensor-parallel and non-native-KV builds are rejected above,
    # so stripping here always means the single-device split path -- the only
    # builder that reads the layout flag.
    native_layout = strip_weights and precision == "bf16"
    checkpoint_mapper.set_native_layout(native_layout)
    try:
        weights = model.load_weights(str(model_dir), config, precision=precision)
        writer.set_header(family="qwen", task=request.task, backend=request.backend)
        if quantized and parallel.enabled:
            for rank in range(parallel.tp_size):
                plan = model.build_engine(
                    config,
                    weights,
                    max_sequence_length,
                    precision=precision,
                    quant_ctx=quant_ctx,
                    verbose=bool(request.verbose),
                    debug_layer_outputs=False,
                    parallel_config=parallel.for_rank(rank),
                )
                writer.add_bytes(f"engine.rank{rank}.plan", plan)
            layout = "dual_profile"
        elif quantized:
            plan = model.build_engine(
                config,
                weights,
                max_sequence_length,
                precision=precision,
                quant_ctx=quant_ctx,
                verbose=bool(request.verbose),
                debug_layer_outputs=False,
                parallel_config=parallel,
            )
            writer.add_bytes("engine.plan", plan)
            layout = "dual_profile"
        elif parallel.enabled:
            for rank in range(parallel.tp_size):
                plan = model.build_engine(
                    config,
                    weights,
                    max_sequence_length,
                    precision=precision,
                    quant_ctx=None,
                    verbose=bool(request.verbose),
                    debug_layer_outputs=False,
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
                debug_layer_outputs=False,
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
                debug_layer_outputs=False,
                parallel_config=parallel,
            )
            config.raw.pop("_decoder_engine_role", None)
            writer.add_bytes("engine.plan", decode)
            writer.add_bytes("prefill.plan", prefill)
            _add_refit_weights_section(config, writer)
            layout = "split"
    finally:
        checkpoint_mapper.set_native_layout(False)
    writer.add_json(
        "runtime.json",
        _runtime_config(
            model_dir,
            config,
            model,
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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SAM2 family registration and model-owned TensorRT Python build hooks."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

from pathlib import Path
from typing import Any, Callable

import tensorrt as trt


from .checkpoint_mapper import Checkpoint, load_checkpoint, load_public_core_checkpoint
from .model_config import (
    config_from_dir,
    CHECKPOINT_RELATIVE_PATH,
    PUBLIC_CHECKPOINT_RELATIVE_PATH,
    resolve_package_root,
    resolve_public_file,
    resolve_public_package_root,
)


_PRECISION = "mixed_bf16_fp32"
_PUBLIC_CORE_VARIANT = "public_sam2_1_small_with_synthetic_bbox_v1"
_DEFAULT_WORKSPACE_BYTES = 8 << 30
_GENERIC_EMBED_CANDIDATES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.model",
    "preprocessor_config.json",
    "processor_config.json",
)
_EXTRA_PLAN_SECTIONS = (
    "sam2_prompt_engine_plan",
    "sam2_recurrent_h1_engine_plan",
    "sam2_recurrent_h2_engine_plan",
    "sam2_recurrent_h3_engine_plan",
    "sam2_recurrent_h4_engine_plan",
)


def _checkpoint(weights: dict[str, Any]) -> Checkpoint:
    value = weights.get("_sam2_checkpoint")
    if not isinstance(value, Checkpoint):
        raise ValueError("SAM2 weights do not contain an authenticated checkpoint")
    return value


def _validate_supported_build(model_dir: str, package_root: Path) -> None:
    checked_directories = {Path(model_dir).resolve(), package_root.resolve()}
    unexpected = sorted(
        str(directory / name)
        for directory in checked_directories
        for name in _GENERIC_EMBED_CANDIDATES
        if (directory / name).exists()
    )
    if unexpected:
        raise ValueError(
            "SAM2 package contains files that the generic bundle writer would embed as "
            f"unsupported sections: {', '.join(unexpected)}"
        )


def _serialize(
    checkpoint: Checkpoint,
    populate: Callable[[Any, Any, Checkpoint], Any],
    *,
    expected_inputs: int,
    expected_outputs: int,
    section: str,
    verbose: bool,
) -> bytes:
    severity = trt.Logger.VERBOSE if verbose else trt.Logger.WARNING
    logger = trt.Logger(severity)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    if network is None:
        raise RuntimeError("TensorRT failed to create the SAM2 network")
    # TensorRT holds host weight pointers until serialization completes.
    # Retain the graph owner (and its generated NumPy buffers) for that lifetime.
    graph_owner = populate(trt, network, checkpoint)
    if (
        network.num_inputs != expected_inputs
        or network.num_outputs != expected_outputs
        or network.num_layers <= 0
    ):
        raise RuntimeError(f"SAM2 graph did not satisfy its exact contract: {section}")

    build_config = builder.create_builder_config()
    if build_config is None:
        raise RuntimeError("TensorRT failed to create the SAM2 builder configuration")
    build_config.builder_optimization_level = 3
    build_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, _DEFAULT_WORKSPACE_BYTES)
    build_config.clear_flag(trt.BuilderFlag.TF32)
    build_config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

    plan = builder.build_serialized_network(network, build_config)
    if plan is None:
        raise RuntimeError(f"TensorRT produced no SAM2 plan: {section}")
    result = bytes(plan)
    del graph_owner
    return result


if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter


class _Sam2Model:
    """Build the fixed SAM2 image and video-tracker graphs in Python."""

    default_build_precision = _PRECISION

    @staticmethod
    def _require_precision(precision: str) -> None:
        if precision != _PRECISION:
            raise ValueError(f"SAM2 supports only {_PRECISION!r} precision, got {precision!r}")

    def load_weights(
        self, model_dir: str, _config: Any, *, precision: str = _PRECISION
    ) -> dict[str, Any]:
        self._require_precision(precision)
        root = resolve_package_root(model_dir)
        public_root = resolve_public_package_root(model_dir)
        if root is None and public_root is None:
            raise ValueError(f"unsupported SAM2 package: {model_dir}")
        root = root or public_root
        assert root is not None
        _validate_supported_build(model_dir, root)
        if public_root is not None:
            _config.raw["_sam2_checkpoint_variant"] = _PUBLIC_CORE_VARIANT
            return {
                "_sam2_checkpoint": load_public_core_checkpoint(
                    resolve_public_file(public_root, PUBLIC_CHECKPOINT_RELATIVE_PATH)
                )
            }
        return {"_sam2_checkpoint": load_checkpoint(root / CHECKPOINT_RELATIVE_PATH)}

    def get_bundle_config_overrides(self, config: Any) -> dict[str, Any] | None:
        variant = config.raw.get("_sam2_checkpoint_variant")
        return {"sam2_checkpoint_variant": variant} if variant else None

    def build_engine(
        self,
        _config: Any,
        weights: dict[str, Any],
        _max_cache_length: int,
        *,
        precision: str = _PRECISION,
        verbose: bool = False,
    ) -> bytes:
        self._require_precision(precision)
        from .image_builder import populate_image_network

        return _serialize(
            _checkpoint(weights),
            populate_image_network,
            expected_inputs=1,
            expected_outputs=9,
            section="engine_plan",
            verbose=verbose,
        )

    def build_extra_engines(
        self,
        _config: Any,
        weights: dict[str, Any],
        _max_cache_length: int,
        *,
        precision: str = _PRECISION,
        verbose: bool = False,
    ) -> dict[str, bytes]:
        self._require_precision(precision)
        from .tracker_builder import populate_tracker_network

        checkpoint = _checkpoint(weights)
        plans: dict[str, bytes] = {}
        for history_frames, section in enumerate(_EXTRA_PLAN_SECTIONS):

            def populate(
                trt: Any, network: Any, source: Checkpoint, h: int = history_frames
            ) -> Any:
                return populate_tracker_network(trt, network, source, h)

            plans[section] = _serialize(
                checkpoint,
                populate,
                expected_inputs=4 if history_frames == 0 else 5,
                expected_outputs=3,
                section=section,
                verbose=verbose,
            )
        return plans


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build one SAM2 video-segmentation bundle."""
    if request.dynamic_kv_cache:
        raise NotImplementedError("sam2 does not support dynamic_kv_cache")

    if request.max_sequence_length is not None:
        raise NotImplementedError("sam2 does not support max_sequence_length")

    if request.image_height not in {None, 1280}:
        raise NotImplementedError("sam2 supports only image_height=1280")

    if request.image_width not in {None, 1088}:
        raise NotImplementedError("sam2 supports only image_width=1088")

    if request.video_num_frames not in {None, 5}:
        raise NotImplementedError("sam2 supports only video_num_frames=5")

    if request.max_batch_size != 1:
        raise NotImplementedError("sam2 does not support max_batch_size")

    if request.context_parallel_size != 1:
        raise ValueError("this family does not support context parallelism")

    if request.task != "video_segmentation":
        raise ValueError("sam2 supports only task=video_segmentation")
    if (
        request.tensor_parallel_size != 1
        or request.quantization not in {None, "none"}
        or request.fp32_layers
    ):
        raise NotImplementedError("SAM2 supports only its fixed single-device build")
    model_dir = Path(request.model_dir)
    raw = config_from_dir(model_dir)
    if raw is None:
        raise ValueError("SAM2 model directory does not match the supported package layout")
    config = SimpleNamespace(raw=raw, model_type="sam2")
    model = _Sam2Model()
    weights = model.load_weights(str(model_dir), config)
    plan = model.build_engine(
        config, weights, 1, precision=request.precision, verbose=request.verbose
    )
    extra = model.build_extra_engines(
        config, weights, 1, precision=request.precision, verbose=request.verbose
    )
    names = (
        "prompt.plan",
        "recurrent.1.plan",
        "recurrent.2.plan",
        "recurrent.3.plan",
        "recurrent.4.plan",
    )
    writer.set_header(family="sam2", task=request.task, backend=request.backend)
    writer.add_bytes("engine.plan", plan)
    for source, target in zip(_EXTRA_PLAN_SECTIONS, names, strict=True):
        writer.add_bytes(target, extra[source])
    writer.add_json("runtime.json", model.get_bundle_config_overrides(config) or {})

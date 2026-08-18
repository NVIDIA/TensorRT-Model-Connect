# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct TensorRT builders for the reviewed SAM2.1-HOI source package.

The source package supplies immutable configuration, checkpoint tensors, and
the PyTorch accuracy oracle.  Production plans are reconstructed from those
tensors with TensorRT's Network Definition API.  There is no framework graph
export or parser in this build path.  Data-dependent detection postprocessing,
interaction pairing, and video-memory policy remain in the native runtime.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import hashlib
from pathlib import Path
import threading


IMAGE_FEATURE_SECTION = "engine_plan"
HOI_DETECTOR_SECTION = "sam2_hoi_detector_engine_plan"
INTERACTION_SECTION = "sam2_hoi_interaction_engine_plan"
PROMPT_TRACKER_SECTION = "sam2_hoi_prompt_tracker_engine_plan"
RECURRENT_TRACKER_SECTION = "sam2_hoi_recurrent_tracker_engine_plan"
MEMORY_ENCODER_SECTION = "sam2_hoi_memory_encoder_engine_plan"
NATIVE_PLUGIN_SECTION = "sam2_hoi_native_plugin_so"
ENGINE_PLAN_SECTIONS = (
    IMAGE_FEATURE_SECTION,
    HOI_DETECTOR_SECTION,
    INTERACTION_SECTION,
    PROMPT_TRACKER_SECTION,
    RECURRENT_TRACKER_SECTION,
    MEMORY_ENCODER_SECTION,
)
BUNDLE_SECTIONS = (*ENGINE_PLAN_SECTIONS, NATIVE_PLUGIN_SECTION)

_SUPPORTED_PRECISIONS = frozenset({"fp32", "bf16"})
_LOADED_NATIVE_PLUGIN_HANDLES: dict[Path, object] = {}
_LOADED_NATIVE_DEPENDENCY_HANDLES: dict[Path, object] = {}
_LOADED_NATIVE_PLUGIN_SHA256: str | None = None
_LOADED_NATIVE_PLUGIN_LOCK = threading.Lock()


@dataclass(frozen=True)
class TensorSpec:
    """One stable engine input and its optional TensorRT shape profile."""

    name: str
    dtype: str
    shape: tuple[int | None, ...]
    min_shape: tuple[int, ...] | None = None
    opt_shape: tuple[int, ...] | None = None
    max_shape: tuple[int, ...] | None = None
    dynamic_axes: tuple[tuple[int, str], ...] = ()

    def example_shape(self) -> tuple[int, ...]:
        if self.opt_shape is not None:
            return self.opt_shape
        if any(dimension is None for dimension in self.shape):
            raise RuntimeError(f"Dynamic SAM2 HOI input {self.name!r} has no opt shape")
        return tuple(int(dimension) for dimension in self.shape)

    def validate(self) -> None:
        dynamic = any(dimension is None for dimension in self.shape)
        profiles = (self.min_shape, self.opt_shape, self.max_shape)
        if dynamic != all(profile is not None for profile in profiles):
            raise RuntimeError(
                f"SAM2 HOI input {self.name!r} must define all profile shapes "
                "exactly when it has dynamic dimensions"
            )
        if dynamic and not self.dynamic_axes:
            raise RuntimeError(f"SAM2 HOI input {self.name!r} has unnamed dynamic axes")
        if not dynamic and (any(profile is not None for profile in profiles) or self.dynamic_axes):
            raise RuntimeError(f"Fixed SAM2 HOI input {self.name!r} has a dynamic profile")


@dataclass(frozen=True)
class EngineContract:
    """Stable runtime binding contract for one TensorRT plan."""

    section: str
    inputs: tuple[TensorSpec, ...]
    outputs: tuple[str, ...]

    def validate(self) -> None:
        if self.section != IMAGE_FEATURE_SECTION and not self.section.startswith("sam2_hoi_"):
            raise RuntimeError(f"Invalid SAM2 HOI bundle section: {self.section!r}")
        input_names = [item.name for item in self.inputs]
        if len(input_names) != len(set(input_names)) or len(self.outputs) != len(set(self.outputs)):
            raise RuntimeError(f"Duplicate binding in SAM2 HOI contract {self.section}")
        for item in self.inputs:
            item.validate()


def _normalize_precision(precision: str) -> str:
    normalized = str(precision).strip().lower()
    if normalized not in _SUPPORTED_PRECISIONS:
        supported = ", ".join(sorted(_SUPPORTED_PRECISIONS))
        raise ValueError(
            f"SAM2 HOI native build supports precision {{{supported}}}, got {precision!r}"
        )
    return normalized


def _work_dtype(precision: str) -> str:
    return "bfloat16" if _normalize_precision(precision) == "bf16" else "float32"


def image_feature_contract(precision: str = "fp32") -> EngineContract:
    _normalize_precision(precision)
    contract = EngineContract(
        section=IMAGE_FEATURE_SECTION,
        inputs=(TensorSpec("pixel_values", "float32", (1, 3, 1024, 1024)),),
        outputs=(
            "tracker_feature_0",
            "tracker_feature_1",
            "tracker_feature_2",
            "tracker_position_2",
            "detector_feature_0",
            "detector_feature_1",
            "detector_feature_2",
        ),
    )
    contract.validate()
    return contract


def hoi_detector_contract(precision: str = "fp32") -> EngineContract:
    work_dtype = _work_dtype(precision)
    contract = EngineContract(
        section=HOI_DETECTOR_SECTION,
        inputs=(
            TensorSpec("detector_feature_0", work_dtype, (1, 256, 128, 128)),
            TensorSpec("detector_feature_1", work_dtype, (1, 256, 64, 64)),
            TensorSpec("detector_feature_2", work_dtype, (1, 256, 32, 32)),
        ),
        outputs=("class_scores", "boxes_cxcywh", "query_embeddings"),
    )
    contract.validate()
    return contract


def interaction_contract(precision: str = "fp32") -> EngineContract:
    _normalize_precision(precision)
    contract = EngineContract(
        section=INTERACTION_SECTION,
        inputs=(
            TensorSpec(
                "pair_features",
                "float32",
                (None, 512),
                min_shape=(1, 512),
                opt_shape=(8, 512),
                max_shape=(22_500, 512),
                dynamic_axes=((0, "pair_count"),),
            ),
        ),
        outputs=("interaction_probabilities",),
    )
    contract.validate()
    return contract


def prompt_tracker_contract(precision: str = "fp32") -> EngineContract:
    work_dtype = _work_dtype(precision)
    contract = EngineContract(
        section=PROMPT_TRACKER_SECTION,
        inputs=(
            TensorSpec("tracker_feature_0", work_dtype, (1, 32, 256, 256)),
            TensorSpec("tracker_feature_1", work_dtype, (1, 64, 128, 128)),
            # The source FPN explicitly promotes the 64x64 top-down sum.
            TensorSpec("tracker_feature_2", "float32", (1, 256, 64, 64)),
            TensorSpec("point_coords", "float32", (2, 3, 2)),
            TensorSpec("point_labels", "int32", (2, 3)),
        ),
        outputs=("pred_masks", "object_pointer", "object_score_logits", "selected_iou"),
    )
    contract.validate()
    return contract


def recurrent_tracker_contract(precision: str = "fp32") -> EngineContract:
    work_dtype = _work_dtype(precision)
    memory_profile = (
        (2, 1, 64, 64, 64),
        (2, 3, 64, 64, 64),
        (2, 7, 64, 64, 64),
    )
    memory_offset_profile = ((2, 1), (2, 3), (2, 7))
    pointer_profile = ((2, 1, 256), (2, 2, 256), (2, 16, 256))
    pointer_offset_profile = ((2, 1), (2, 2), (2, 16))
    contract = EngineContract(
        section=RECURRENT_TRACKER_SECTION,
        inputs=(
            TensorSpec("tracker_feature_0", work_dtype, (1, 32, 256, 256)),
            TensorSpec("tracker_feature_1", work_dtype, (1, 64, 128, 128)),
            TensorSpec("tracker_feature_2", "float32", (1, 256, 64, 64)),
            TensorSpec("tracker_position_2", "float32", (1, 256, 64, 64)),
            TensorSpec(
                "memory_features",
                work_dtype,
                (2, None, 64, 64, 64),
                *memory_profile,
                dynamic_axes=((1, "memory_frames"),),
            ),
            TensorSpec(
                "memory_position",
                work_dtype,
                (2, None, 64, 64, 64),
                *memory_profile,
                dynamic_axes=((1, "memory_frames"),),
            ),
            TensorSpec(
                "memory_temporal_offsets",
                "int32",
                (2, None),
                *memory_offset_profile,
                dynamic_axes=((1, "memory_frames"),),
            ),
            TensorSpec(
                "object_pointers",
                "float32",
                (2, None, 256),
                *pointer_profile,
                dynamic_axes=((1, "pointer_count"),),
            ),
            TensorSpec(
                "object_pointer_temporal_offsets",
                "float32",
                (2, None),
                *pointer_offset_profile,
                dynamic_axes=((1, "pointer_count"),),
            ),
            TensorSpec("object_pointer_time_denominator", "float32", (1,)),
        ),
        outputs=("pred_masks", "object_pointer", "object_score_logits", "selected_iou"),
    )
    contract.validate()
    return contract


def memory_encoder_contract(precision: str = "fp32") -> EngineContract:
    _normalize_precision(precision)
    contract = EngineContract(
        section=MEMORY_ENCODER_SECTION,
        inputs=(
            TensorSpec("tracker_feature_2", "float32", (1, 256, 64, 64)),
            TensorSpec("pred_masks", "float32", (2, 1, 256, 256)),
            TensorSpec("object_score_logits", "float32", (2, 1)),
            TensorSpec("is_mask_from_points", "int32", (2, 1)),
        ),
        outputs=("new_memory_features", "new_memory_position"),
    )
    contract.validate()
    return contract


def ensure_native_plugin_loaded(*, verbose: bool = False) -> Path:
    """Build and load the model-owned DSO with its build-process CUDA closure.

    This check protects engine construction and serialization. It does not
    attest the separately deployed C++ inference process, whose CUDA runtime
    image remains an external qualification gate.
    """

    from . import native_plugin_builder

    global _LOADED_NATIVE_PLUGIN_SHA256
    path = native_plugin_builder.ensure_native_plugin(verbose=verbose).resolve()
    plugin_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    with _LOADED_NATIVE_PLUGIN_LOCK:
        if _LOADED_NATIVE_PLUGIN_HANDLES:
            if plugin_sha256 != _LOADED_NATIVE_PLUGIN_SHA256:
                raise RuntimeError(
                    "A different SAM2 HOI native plugin is already loaded in this process"
                )
            native_plugin_builder._verify_loaded_cublaslt(path)
            return next(iter(_LOADED_NATIVE_PLUGIN_HANDLES))

        # A DT_NEEDED entry records only the cuBLASLt SONAME. Reject a process
        # that already mapped a different implementation, then preload the
        # exact receipt-pinned DSO by absolute path before loading the plugin.
        native_plugin_builder._verify_loaded_cublaslt(path, allow_unloaded=True)
        expected = native_plugin_builder._expected_runtime_cublaslt(path)
        dependency_path = Path(str(expected["path"]))
        dependency_handle = ctypes.CDLL(str(dependency_path), mode=ctypes.RTLD_GLOBAL)
        _LOADED_NATIVE_DEPENDENCY_HANDLES[dependency_path] = dependency_handle
        native_plugin_builder._verify_loaded_cublaslt(path)
        handle = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        native_plugin_builder._verify_loaded_cublaslt(path)
        _LOADED_NATIVE_PLUGIN_HANDLES[path] = handle
        _LOADED_NATIVE_PLUGIN_SHA256 = plugin_sha256
    return path


def build_image_feature_engine(
    model_dir: str,
    *,
    precision: str = "fp32",
    verbose: bool = False,
) -> bytes:
    """Build Hiera, tracker FPN projections, and detector PAFPN directly."""

    from .checkpoint import load_checkpoint
    from .native_image_builder import build_image_feature_engine as build_native_image

    precision = _normalize_precision(precision)
    # Exact Hiera LayerNorm is a model-owned plugin at both FP32 and BF16
    # precision boundaries, so its creator must be registered for either path.
    ensure_native_plugin_loaded(verbose=verbose)
    return build_native_image(
        load_checkpoint(model_dir),
        precision=precision,
        verbose=verbose,
    )


def build_hoi_detector_engine(
    model_dir: str,
    *,
    precision: str = "fp32",
    verbose: bool = False,
) -> bytes:
    """Build the raw Co-DINO detector with direct TensorRT layers/plugins."""

    from .checkpoint import load_checkpoint
    from .native_detector_builder import build_hoi_detector_engine as build_native_detector

    precision = _normalize_precision(precision)
    ensure_native_plugin_loaded(verbose=verbose)
    return build_native_detector(
        load_checkpoint(model_dir),
        precision=precision,
        verbose=verbose,
    )


def build_tracker_engines(
    model_dir: str,
    *,
    precision: str = "fp32",
    verbose: bool = False,
) -> dict[str, bytes]:
    """Build the B2 prompt, recurrent, and memory plans directly."""

    from .checkpoint import load_checkpoint
    from .native_tracker_builder import build_tracker_engines as build_native_trackers

    precision = _normalize_precision(precision)
    ensure_native_plugin_loaded(verbose=verbose)
    return build_native_trackers(
        load_checkpoint(model_dir),
        precision=precision,
        verbose=verbose,
    )


__all__ = [
    "BUNDLE_SECTIONS",
    "ENGINE_PLAN_SECTIONS",
    "HOI_DETECTOR_SECTION",
    "IMAGE_FEATURE_SECTION",
    "INTERACTION_SECTION",
    "MEMORY_ENCODER_SECTION",
    "NATIVE_PLUGIN_SECTION",
    "PROMPT_TRACKER_SECTION",
    "RECURRENT_TRACKER_SECTION",
    "build_hoi_detector_engine",
    "build_image_feature_engine",
    "build_tracker_engines",
    "ensure_native_plugin_loaded",
    "hoi_detector_contract",
    "image_feature_contract",
    "interaction_contract",
    "memory_encoder_contract",
    "prompt_tracker_contract",
    "recurrent_tracker_contract",
]

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamic Qwen3-VL language conditioner for MiniMax-H3 presentations.

Unlike the legacy fixed-length T2VA text engine, this engine consumes the
complete processor presentation at its real runtime length.  Vision rows are
hard-selected over token embeddings, the three Qwen3-VL DeepStack tensors are
hard-selected and injected after complete language layers 0, 1, and 2, and
the unnormalized hidden state after language layer 49 is published as FP32.

Engine ABI (``L`` is runtime-dynamic through the workflow profile maximum):

* ``input_ids``: int32 ``[L]``;
* ``mrope_position_ids``: int32 ``[3, L]``;
* ``vision_embeddings``: BF16 ``[L, 5120]``;
* ``vision_selector``: int32 ``[L, 1]``, exactly zero or one;
* ``deepstack_embeddings_0..2``: BF16 ``[L, 5120]``; and
* ``encoder_hidden_states``: FP32 ``[L, 5120]``.

There is no attention-mask input.  Native causal ``IAttention`` sees exactly
the runtime ``L`` rows, so optimization-profile capacity is never visible as
padding.  The graph contains no layers 50..63, final norm, or LM head.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
import math
import sys
from typing import Any, Mapping, Sequence

import ml_dtypes
import numpy as np

from .config import (
    FL2VA_KEYFRAME_COUNTS,
    FL2VA_KEYFRAME_ROWS_1344X768,
    MiniMaxH3Config,
    REF2VA_MAX_IMAGE_CONDITION_VIDEO_ROWS,
    REF2VA_MAX_VIDEO_ROWS_PER_LATENT_FRAME,
    TEXT_ENCODER_DEFAULT_WORKSPACE_BYTES,
)


_LANGUAGE_PREFIX = "model.language_model"
_PRESENTATION_DIMENSION = "presentation_rows"
_H3_PROFILE = MiniMaxH3Config()
_WORKFLOWS = ("fl2va", "ref2va")
_REF2VA_MAX_IMAGES = 9
_REF2VA_MAX_VIDEOS = 3
_REF2VA_MAX_REFERENCES = 12
# Qwen sees a reference video at 2 fps and temporally merges pairs, producing
# ``ceil(duration_seconds)`` blocks per file. Under the 15-second aggregate
# duration cap, splitting duration across three files can contribute two extra
# partial final blocks, hence at most 17 ordered video vision runs.
_REF2VA_MAX_VIDEO_RUNS = 17
_REF2VA_MAX_IMAGE_RUN_ROWS = REF2VA_MAX_IMAGE_CONDITION_VIDEO_ROWS // _REF2VA_MAX_IMAGES
_REF2VA_MAX_VIDEO_RUN_ROWS = REF2VA_MAX_VIDEO_ROWS_PER_LATENT_FRAME


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"MiniMax-H3 {label} must be a mapping")
    return value


def _require_exact_field(config: Mapping[str, Any], name: str, expected: object) -> None:
    if name not in config:
        raise ValueError(f"MiniMax-H3 text config is missing {name!r}")
    actual = config[name]
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"MiniMax-H3 text config {name!r} must be {expected!r}, got {actual!r}")


@dataclass(frozen=True)
class MiniMaxH3LanguageConditionerSpec:
    """Validated dynamic language-stack and presentation ABI."""

    workflow: str = "fl2va"
    hidden_size: int = 5120
    intermediate_size: int = 25600
    vocab_size: int = 151936
    available_layers: int = 64
    output_layers: int = 50
    num_heads: int = 64
    num_kv_heads: int = 8
    head_dim: int = 128
    rope_theta: float = 5_000_000.0
    norm_eps: float = 1.0e-6
    mrope_section: tuple[int, int, int] = (24, 20, 20)
    mrope_interleaved: bool = True
    min_rows: int = _H3_PROFILE.min_text_rows
    opt_rows: int = _H3_PROFILE.text_rows
    max_rows: int = _H3_PROFILE.max_text_rows
    vision_rows_per_keyframe: int = FL2VA_KEYFRAME_ROWS_1344X768
    max_keyframes: int = max(FL2VA_KEYFRAME_COUNTS)
    deepstack_levels: int = 3
    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    max_reference_images: int = _REF2VA_MAX_IMAGES
    max_reference_videos: int = _REF2VA_MAX_VIDEOS
    max_references: int = _REF2VA_MAX_REFERENCES
    max_video_runs: int = _REF2VA_MAX_VIDEO_RUNS
    max_image_run_rows: int = _REF2VA_MAX_IMAGE_RUN_ROWS
    max_video_run_rows: int = _REF2VA_MAX_VIDEO_RUN_ROWS

    def __post_init__(self) -> None:
        integer_fields = (
            "hidden_size",
            "intermediate_size",
            "vocab_size",
            "available_layers",
            "output_layers",
            "num_heads",
            "num_kv_heads",
            "head_dim",
            "min_rows",
            "opt_rows",
            "max_rows",
            "vision_rows_per_keyframe",
            "max_keyframes",
            "deepstack_levels",
            "image_token_id",
            "video_token_id",
            "vision_start_token_id",
            "vision_end_token_id",
            "max_reference_images",
            "max_reference_videos",
            "max_references",
            "max_video_runs",
            "max_image_run_rows",
            "max_video_run_rows",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"MiniMax-H3 language {name} must be a positive integer")
        if self.workflow not in _WORKFLOWS:
            raise ValueError(
                f"MiniMax-H3 language workflow must be one of {_WORKFLOWS}, got {self.workflow!r}"
            )
        if not self.min_rows <= self.opt_rows <= self.max_rows:
            raise ValueError("MiniMax-H3 presentation rows must satisfy min <= opt <= max")
        if self.output_layers > self.available_layers:
            raise ValueError("MiniMax-H3 output layer count exceeds the available Qwen layers")
        if self.num_heads % self.num_kv_heads:
            raise ValueError("MiniMax-H3 num_heads must be divisible by num_kv_heads")
        if self.head_dim % 2:
            raise ValueError("MiniMax-H3 Qwen head_dim must be even")
        if not isinstance(self.mrope_interleaved, bool) or not self.mrope_interleaved:
            raise ValueError("MiniMax-H3 requires interleaved Qwen3-VL MRoPE frequencies")
        _mrope_frequency_axis_map(self.mrope_section, self.head_dim, interleaved=True)
        if self.deepstack_levels != 3:
            raise ValueError("MiniMax-H3 Qwen3-VL requires exactly three DeepStack levels")
        if self.output_layers < self.deepstack_levels:
            raise ValueError("MiniMax-H3 language stack ends before all DeepStack injections")
        if not math.isclose(self.rope_theta, 5_000_000.0, rel_tol=0.0, abs_tol=0.0):
            raise ValueError("MiniMax-H3 Qwen3-VL rope_theta must be 5000000")
        if not math.isclose(self.norm_eps, 1.0e-6, rel_tol=0.0, abs_tol=0.0):
            raise ValueError("MiniMax-H3 Qwen3-VL RMSNorm epsilon must be 1e-6")
        token_ids = {
            self.image_token_id,
            self.video_token_id,
            self.vision_start_token_id,
            self.vision_end_token_id,
        }
        if len(token_ids) != 4 or max(token_ids) >= self.vocab_size:
            raise ValueError(
                "MiniMax-H3 vision token IDs must be distinct and inside the vocabulary"
            )

    @property
    def attention_size(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def kv_attention_size(self) -> int:
        return self.num_kv_heads * self.head_dim

    @property
    def allowed_vision_rows(self) -> tuple[int, ...]:
        if self.workflow != "fl2va":
            return ()
        return tuple(
            count * self.vision_rows_per_keyframe for count in range(self.max_keyframes + 1)
        )

    @classmethod
    def for_workflow(cls, workflow: str, **overrides: Any) -> "MiniMaxH3LanguageConditionerSpec":
        """Create the workflow-specific padless optimization envelope."""

        if workflow not in _WORKFLOWS:
            raise ValueError(
                f"MiniMax-H3 language workflow must be one of {_WORKFLOWS}, got {workflow!r}"
            )
        profile = {
            "workflow": workflow,
            "min_rows": (
                _H3_PROFILE.min_text_rows
                if workflow == "fl2va"
                else _H3_PROFILE.ref2va_min_text_rows
            ),
            "opt_rows": (
                _H3_PROFILE.text_rows if workflow == "fl2va" else _H3_PROFILE.ref2va_opt_text_rows
            ),
            "max_rows": (
                _H3_PROFILE.max_text_rows
                if workflow == "fl2va"
                else _H3_PROFILE.ref2va_max_text_rows
            ),
        }
        profile.update(overrides)
        return cls(**profile)

    @classmethod
    def from_checkpoint_config(
        cls,
        checkpoint_config: Mapping[str, Any],
        *,
        workflow: str = "fl2va",
    ) -> "MiniMaxH3LanguageConditionerSpec":
        root = _require_mapping(checkpoint_config, "text-encoder config")
        if root.get("model_type") != "qwen3_vl":
            raise ValueError("MiniMax-H3 text encoder must have model_type='qwen3_vl'")
        if root.get("architectures") != ["Qwen3VLForConditionalGeneration"]:
            raise ValueError("MiniMax-H3 text encoder must use Qwen3VLForConditionalGeneration")
        if root.get("image_token_id") != 151655:
            raise ValueError("MiniMax-H3 image_token_id must be 151655")
        if root.get("video_token_id") != 151656:
            raise ValueError("MiniMax-H3 video_token_id must be 151656")
        if root.get("vision_start_token_id") != 151652:
            raise ValueError("MiniMax-H3 vision_start_token_id must be 151652")
        if root.get("vision_end_token_id") != 151653:
            raise ValueError("MiniMax-H3 vision_end_token_id must be 151653")
        text = _require_mapping(root.get("text_config"), "text_config")
        expected = {
            "model_type": "qwen3_vl_text",
            "dtype": "bfloat16",
            "hidden_size": 5120,
            "intermediate_size": 25600,
            "vocab_size": 151936,
            "num_hidden_layers": 64,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "rope_theta": 5_000_000,
            "rms_norm_eps": 1.0e-6,
            "attention_bias": False,
            "attention_dropout": 0.0,
            "hidden_act": "silu",
        }
        for name, value in expected.items():
            _require_exact_field(text, name, value)
        if workflow not in _WORKFLOWS:
            raise ValueError(
                f"MiniMax-H3 language workflow must be one of {_WORKFLOWS}, got {workflow!r}"
            )
        required_positions = (
            _H3_PROFILE.max_text_rows if workflow == "fl2va" else _H3_PROFILE.ref2va_max_text_rows
        )
        max_positions = text.get("max_position_embeddings")
        if (
            not isinstance(max_positions, int)
            or isinstance(max_positions, bool)
            or max_positions < required_positions
        ):
            raise ValueError(
                "MiniMax-H3 text max_position_embeddings must cover the "
                f"{workflow} profile maximum {required_positions}"
            )
        rope_scaling = _require_mapping(text.get("rope_scaling"), "text rope_scaling")
        if rope_scaling.get("rope_type") != "default":
            raise ValueError("MiniMax-H3 text rope_scaling.rope_type must be 'default'")
        if rope_scaling.get("mrope_interleaved") is not True:
            raise ValueError("MiniMax-H3 text MRoPE must be interleaved")
        if rope_scaling.get("mrope_section") != [24, 20, 20]:
            raise ValueError("MiniMax-H3 text mrope_section must be [24, 20, 20]")
        return cls.for_workflow(
            workflow,
            hidden_size=text["hidden_size"],
            intermediate_size=text["intermediate_size"],
            vocab_size=text["vocab_size"],
            available_layers=text["num_hidden_layers"],
            num_heads=text["num_attention_heads"],
            num_kv_heads=text["num_key_value_heads"],
            head_dim=text["head_dim"],
            rope_theta=float(text["rope_theta"]),
            norm_eps=float(text["rms_norm_eps"]),
            mrope_section=tuple(rope_scaling["mrope_section"]),
            mrope_interleaved=rope_scaling["mrope_interleaved"],
        )


def expected_weight_shapes(
    spec: MiniMaxH3LanguageConditionerSpec | None = None,
) -> dict[str, tuple[int, ...]]:
    """Return exactly the embedding plus language layers 0 through 49."""

    spec = spec or MiniMaxH3LanguageConditionerSpec()
    shapes: dict[str, tuple[int, ...]] = {
        f"{_LANGUAGE_PREFIX}.embed_tokens.weight": (spec.vocab_size, spec.hidden_size)
    }
    for index in range(spec.output_layers):
        prefix = f"{_LANGUAGE_PREFIX}.layers.{index}"
        shapes.update(
            {
                f"{prefix}.input_layernorm.weight": (spec.hidden_size,),
                f"{prefix}.post_attention_layernorm.weight": (spec.hidden_size,),
                f"{prefix}.self_attn.q_proj.weight": (
                    spec.attention_size,
                    spec.hidden_size,
                ),
                f"{prefix}.self_attn.k_proj.weight": (
                    spec.kv_attention_size,
                    spec.hidden_size,
                ),
                f"{prefix}.self_attn.v_proj.weight": (
                    spec.kv_attention_size,
                    spec.hidden_size,
                ),
                f"{prefix}.self_attn.o_proj.weight": (
                    spec.hidden_size,
                    spec.attention_size,
                ),
                f"{prefix}.self_attn.q_norm.weight": (spec.head_dim,),
                f"{prefix}.self_attn.k_norm.weight": (spec.head_dim,),
                f"{prefix}.mlp.gate_proj.weight": (
                    spec.intermediate_size,
                    spec.hidden_size,
                ),
                f"{prefix}.mlp.up_proj.weight": (
                    spec.intermediate_size,
                    spec.hidden_size,
                ),
                f"{prefix}.mlp.down_proj.weight": (
                    spec.hidden_size,
                    spec.intermediate_size,
                ),
            }
        )
    return shapes


def checkpoint_keys() -> tuple[str, ...]:
    return tuple(expected_weight_shapes())


def validate_conditioner_weights(
    weights: Mapping[str, Any], spec: MiniMaxH3LanguageConditionerSpec
) -> None:
    """Require exact first-50-layer ownership and checkpoint-native dtypes."""

    if not isinstance(weights, Mapping):
        raise ValueError("MiniMax-H3 language weights must be a mapping")
    expected = expected_weight_shapes(spec)
    owned = {name for name in weights if name.startswith(f"{_LANGUAGE_PREFIX}.")}
    missing = sorted(set(expected) - set(weights))
    unexpected = sorted(owned - set(expected))
    if missing or unexpected:
        raise ValueError(
            "MiniMax-H3 language checkpoint contract mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    allowed_dtypes = {np.dtype(np.float32), np.dtype(ml_dtypes.bfloat16)}
    for name, shape in expected.items():
        value = weights[name]
        actual_shape = tuple(int(dimension) for dimension in getattr(value, "shape", ()))
        if actual_shape != shape:
            raise ValueError(
                f"MiniMax-H3 language tensor {name!r} must have shape {shape}, got {actual_shape}"
            )
        try:
            dtype = np.dtype(value.dtype)
        except (AttributeError, TypeError) as error:
            raise ValueError(
                f"MiniMax-H3 language tensor {name!r} must expose a NumPy dtype"
            ) from error
        if dtype not in allowed_dtypes:
            raise ValueError(
                f"MiniMax-H3 language tensor {name!r} must be BF16 or FP32, got {dtype}"
            )


def _require_array(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: np.dtype,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ValueError(f"MiniMax-H3 binding {name!r} must be a NumPy array")
    if tuple(value.shape) != shape:
        raise ValueError(f"MiniMax-H3 binding {name!r} must have shape {shape}, got {value.shape}")
    if value.dtype != dtype:
        raise ValueError(f"MiniMax-H3 binding {name!r} must have dtype {dtype}, got {value.dtype}")
    if not value.flags.c_contiguous:
        raise ValueError(f"MiniMax-H3 binding {name!r} must be C-contiguous")
    return value


def _runtime_int_tuple(value: Sequence[int] | None, *, name: str) -> tuple[int, ...]:
    if value is None or isinstance(value, (str, bytes)):
        raise ValueError(f"MiniMax-H3 Ref2VA requires runtime-supplied {name}")
    result = tuple(value)
    if any(not isinstance(item, int) or isinstance(item, bool) for item in result):
        raise ValueError(f"MiniMax-H3 Ref2VA {name} must contain integers")
    return result


def _validate_ref2va_runs(
    *,
    input_ids: np.ndarray,
    run_starts: np.ndarray,
    run_ends: np.ndarray,
    run_lengths: np.ndarray,
    supplied_lengths: Sequence[int] | None,
    supplied_reference_ids: Sequence[int] | None,
    spec: MiniMaxH3LanguageConditionerSpec,
) -> None:
    declared_lengths = _runtime_int_tuple(supplied_lengths, name="vision_run_lengths")
    reference_ids = _runtime_int_tuple(supplied_reference_ids, name="vision_run_reference_ids")
    if any(length <= 0 for length in declared_lengths):
        raise ValueError("MiniMax-H3 Ref2VA vision_run_lengths must be positive")
    if declared_lengths != tuple(int(length) for length in run_lengths):
        raise ValueError(
            "MiniMax-H3 Ref2VA runtime vision_run_lengths do not match the presentation"
        )
    if len(reference_ids) != len(declared_lengths):
        raise ValueError(
            "MiniMax-H3 Ref2VA vision_run_reference_ids must align with vision_run_lengths"
        )
    if any(reference < 0 or reference >= spec.max_references for reference in reference_ids):
        raise ValueError("MiniMax-H3 Ref2VA vision reference IDs exceed the file cap")
    if any(left > right for left, right in zip(reference_ids, reference_ids[1:])):
        raise ValueError(
            "MiniMax-H3 Ref2VA vision runs must preserve nondecreasing reference order"
        )

    run_kinds = []
    for start, end in zip(run_starts.tolist(), run_ends.tolist()):
        token_id = int(input_ids[start])
        if np.any(input_ids[start : end + 1] != token_id):
            raise ValueError("MiniMax-H3 Ref2VA vision runs cannot mix image and video pads")
        if token_id == spec.image_token_id:
            run_kinds.append("image")
        elif token_id == spec.video_token_id:
            run_kinds.append("video")
        else:
            raise ValueError("MiniMax-H3 Ref2VA vision run has an unknown pad token")

    grouped: dict[int, list[int]] = {}
    for index, reference in enumerate(reference_ids):
        grouped.setdefault(reference, []).append(index)
    image_references = 0
    video_references = 0
    video_runs = 0
    for indexes in grouped.values():
        kinds = {run_kinds[index] for index in indexes}
        if len(kinds) != 1:
            raise ValueError("MiniMax-H3 Ref2VA reference cannot mix image and video runs")
        kind = kinds.pop()
        if kind == "image":
            image_references += 1
            if len(indexes) != 1:
                raise ValueError("MiniMax-H3 Ref2VA image references require exactly one run")
            if declared_lengths[indexes[0]] > spec.max_image_run_rows:
                raise ValueError("MiniMax-H3 Ref2VA image run exceeds the model-card cap")
        else:
            video_references += 1
            video_runs += len(indexes)
            if any(declared_lengths[index] > spec.max_video_run_rows for index in indexes):
                raise ValueError("MiniMax-H3 Ref2VA video run exceeds the model-card cap")
    if image_references > spec.max_reference_images:
        raise ValueError("MiniMax-H3 Ref2VA exceeds the nine-image file cap")
    if video_references > spec.max_reference_videos:
        raise ValueError("MiniMax-H3 Ref2VA exceeds the three-video file cap")
    if len(grouped) > spec.max_references:
        raise ValueError("MiniMax-H3 Ref2VA exceeds the total reference file cap")
    if video_runs > spec.max_video_runs:
        raise ValueError("MiniMax-H3 Ref2VA exceeds the aggregate video-duration cap")


def validate_presentation_bindings(
    *,
    input_ids: np.ndarray,
    mrope_position_ids: np.ndarray,
    vision_embeddings: np.ndarray,
    vision_selector: np.ndarray,
    deepstack_embeddings: Sequence[np.ndarray],
    vision_run_lengths: Sequence[int] | None = None,
    vision_run_reference_ids: Sequence[int] | None = None,
    spec: MiniMaxH3LanguageConditionerSpec | None = None,
) -> int:
    """Validate one native runtime presentation before binding the engine."""

    spec = spec or MiniMaxH3LanguageConditionerSpec()
    if not isinstance(input_ids, np.ndarray) or input_ids.ndim != 1:
        raise ValueError("MiniMax-H3 input_ids must be a rank-one NumPy array")
    rows = int(input_ids.shape[0])
    if rows < spec.min_rows or rows > spec.max_rows:
        raise ValueError(
            f"MiniMax-H3 presentation rows must be in [{spec.min_rows}, {spec.max_rows}]"
        )
    _require_array(input_ids, name="input_ids", shape=(rows,), dtype=np.dtype(np.int32))
    if np.any(input_ids < 0) or np.any(input_ids >= spec.vocab_size):
        raise ValueError("MiniMax-H3 input_ids must be inside the checkpoint vocabulary")
    positions = _require_array(
        mrope_position_ids,
        name="mrope_position_ids",
        shape=(3, rows),
        dtype=np.dtype(np.int32),
    )
    if np.any(positions < 0) or np.any(positions >= spec.max_rows):
        raise ValueError(f"MiniMax-H3 mrope_position_ids must be in [0, {spec.max_rows})")
    bf16 = np.dtype(ml_dtypes.bfloat16)
    _require_array(
        vision_embeddings,
        name="vision_embeddings",
        shape=(rows, spec.hidden_size),
        dtype=bf16,
    )
    selector = _require_array(
        vision_selector,
        name="vision_selector",
        shape=(rows, 1),
        dtype=np.dtype(np.int32),
    )
    if np.any((selector != 0) & (selector != 1)):
        raise ValueError("MiniMax-H3 vision_selector must contain only zero or one")
    pad_token_ids = (
        (spec.image_token_id,)
        if spec.workflow == "fl2va"
        else (spec.image_token_id, spec.video_token_id)
    )
    if not np.array_equal(selector[:, 0] == 1, np.isin(input_ids, pad_token_ids)):
        pad_label = "image-pad" if spec.workflow == "fl2va" else "image/video-pad"
        raise ValueError(
            f"MiniMax-H3 vision_selector must select exactly the {pad_label} token rows"
        )
    selected_rows = int(selector.sum(dtype=np.int64))
    if spec.workflow == "fl2va" and selected_rows not in spec.allowed_vision_rows:
        raise ValueError("MiniMax-H3 vision_selector must select exactly 0, 1008, or 2016 rows")
    selected = selector[:, 0] == 1
    selected_indices = np.flatnonzero(selected)
    run_starts = np.empty((0,), dtype=np.int64)
    run_ends = np.empty((0,), dtype=np.int64)
    run_lengths = np.empty((0,), dtype=np.int64)
    if selected_rows:
        boundaries = np.flatnonzero(np.diff(selected_indices) != 1)
        run_starts = np.concatenate((selected_indices[:1], selected_indices[boundaries + 1]))
        run_ends = np.concatenate((selected_indices[boundaries], selected_indices[-1:]))
        run_lengths = run_ends - run_starts + 1
        for start, end in zip(run_starts.tolist(), run_ends.tolist()):
            if (
                start == 0
                or input_ids[start - 1] != spec.vision_start_token_id
                or end + 1 >= rows
                or input_ids[end + 1] != spec.vision_end_token_id
            ):
                raise ValueError(
                    "MiniMax-H3 image-pad runs must be bounded by vision-start/end tokens"
                )
        if not np.all(np.isfinite(vision_embeddings[selected])):
            raise ValueError("MiniMax-H3 active vision embeddings must be finite")
    if spec.workflow == "fl2va":
        if vision_run_lengths is not None or vision_run_reference_ids is not None:
            raise ValueError("MiniMax-H3 FL2VA does not accept Ref2VA vision-run metadata")
        if len(run_starts) > spec.max_keyframes or np.any(
            run_lengths != spec.vision_rows_per_keyframe
        ):
            raise ValueError(
                "MiniMax-H3 image-pad rows must form one or two complete keyframe runs"
            )
    else:
        _validate_ref2va_runs(
            input_ids=input_ids,
            run_starts=run_starts,
            run_ends=run_ends,
            run_lengths=run_lengths,
            supplied_lengths=vision_run_lengths,
            supplied_reference_ids=vision_run_reference_ids,
            spec=spec,
        )
    if len(deepstack_embeddings) != spec.deepstack_levels:
        raise ValueError("MiniMax-H3 requires exactly three DeepStack presentation tensors")
    for index, value in enumerate(deepstack_embeddings):
        deepstack = _require_array(
            value,
            name=f"deepstack_embeddings_{index}",
            shape=(rows, spec.hidden_size),
            dtype=bf16,
        )
        if selected_rows and not np.all(np.isfinite(deepstack[selected])):
            raise ValueError(f"MiniMax-H3 active DeepStack embeddings {index} must be finite")
    return selected_rows


def _mrope_frequency_axis_map(
    sections: tuple[int, int, int], rotary_dim: int, *, interleaved: bool
) -> np.ndarray:
    if (
        not isinstance(sections, tuple)
        or len(sections) != 3
        or any(
            not isinstance(width, int) or isinstance(width, bool) or width <= 0
            for width in sections
        )
    ):
        raise ValueError("MiniMax-H3 mrope_section must contain three positive integers")
    half = rotary_dim // 2
    if sum(sections) != half:
        raise ValueError("MiniMax-H3 mrope_section must sum to half the rotary dimension")
    if not interleaved:
        return np.repeat(np.arange(3, dtype=np.int32), sections)
    axes = np.zeros(half, dtype=np.int32)
    for axis in (1, 2):
        columns = np.arange(axis, sections[axis] * 3, 3, dtype=np.int32)
        if len(columns) != sections[axis] or (len(columns) and int(columns[-1]) >= half):
            raise ValueError("MiniMax-H3 interleaved mrope_section is not representable")
        axes[columns] = axis
    if int(np.count_nonzero(axes == 0)) != sections[0]:
        raise ValueError("MiniMax-H3 interleaved MRoPE has the wrong temporal tail")
    return axes


def _make_rope_tables(
    spec: MiniMaxH3LanguageConditionerSpec,
) -> tuple[np.ndarray, np.ndarray]:
    inverse = np.float32(1.0) / np.power(
        np.float32(spec.rope_theta),
        np.arange(0, spec.head_dim, 2, dtype=np.float32) / np.float32(spec.head_dim),
    )
    frequency = np.outer(np.arange(spec.max_rows, dtype=np.float32), inverse)
    return (
        np.ascontiguousarray(np.cos(frequency).astype(ml_dtypes.bfloat16)),
        np.ascontiguousarray(np.sin(frequency).astype(ml_dtypes.bfloat16)),
    )


def _set_row_dimension(tensor, axis: int) -> None:
    setter = getattr(tensor, "set_dimension_name", None)
    if not callable(setter):
        raise RuntimeError("TensorRT does not support named dynamic dimensions")
    setter(axis, _PRESENTATION_DIMENSION)


def _declare_inputs(network, spec: MiniMaxH3LanguageConditionerSpec, trt) -> dict[str, Any]:
    inputs = {
        "input_ids": network.add_input("input_ids", trt.int32, (-1,)),
        "mrope_position_ids": network.add_input("mrope_position_ids", trt.int32, (3, -1)),
        "vision_embeddings": network.add_input(
            "vision_embeddings", trt.bfloat16, (-1, spec.hidden_size)
        ),
        "vision_selector": network.add_input("vision_selector", trt.int32, (-1, 1)),
    }
    for index in range(spec.deepstack_levels):
        inputs[f"deepstack_embeddings_{index}"] = network.add_input(
            f"deepstack_embeddings_{index}",
            trt.bfloat16,
            (-1, spec.hidden_size),
        )
    rejected = sorted(name for name, tensor in inputs.items() if tensor is None)
    if rejected:
        raise RuntimeError(f"TensorRT rejected MiniMax-H3 conditioner inputs: {rejected}")
    _set_row_dimension(inputs["input_ids"], 0)
    _set_row_dimension(inputs["mrope_position_ids"], 1)
    _set_row_dimension(inputs["vision_embeddings"], 0)
    _set_row_dimension(inputs["vision_selector"], 0)
    for index in range(spec.deepstack_levels):
        _set_row_dimension(inputs[f"deepstack_embeddings_{index}"], 0)
    return inputs


def _add_optimization_profile(builder, config, spec: MiniMaxH3LanguageConditionerSpec) -> None:
    profile = builder.create_optimization_profile()
    shapes = {
        "input_ids": ((spec.min_rows,), (spec.opt_rows,), (spec.max_rows,)),
        "mrope_position_ids": (
            (3, spec.min_rows),
            (3, spec.opt_rows),
            (3, spec.max_rows),
        ),
        "vision_embeddings": (
            (spec.min_rows, spec.hidden_size),
            (spec.opt_rows, spec.hidden_size),
            (spec.max_rows, spec.hidden_size),
        ),
        "vision_selector": (
            (spec.min_rows, 1),
            (spec.opt_rows, 1),
            (spec.max_rows, 1),
        ),
    }
    for index in range(spec.deepstack_levels):
        shapes[f"deepstack_embeddings_{index}"] = shapes["vision_embeddings"]
    for name, (minimum, optimum, maximum) in shapes.items():
        if profile.set_shape(name, minimum, optimum, maximum) is False:
            raise RuntimeError(f"TensorRT rejected MiniMax-H3 profile for {name}")
    profile_index = config.add_optimization_profile(profile)
    if profile_index < 0:
        raise RuntimeError("TensorRT rejected the MiniMax-H3 language profile")


def _selector_condition(network, selector, trt, op):
    zero = op.constant(network, np.zeros((1, 1), dtype=np.int32), dtype=np.int32)
    return network.add_elementwise(selector, zero, trt.ElementWiseOperation.GREATER).get_output(0)


def _hard_select(network, condition, when_true, when_false):
    return network.add_select(condition, when_true, when_false).get_output(0)


def _hard_gate(network, condition, value, trt, op):
    zero = op.constant(network, np.zeros((1, 1), dtype=np.float32))
    zero = op.cast(network, zero, trt.bfloat16)
    return _hard_select(network, condition, value, zero)


def _rows_to_heads(network, tensor, heads: int, head_dim: int, trt):
    reshape = network.add_shuffle(tensor)
    reshape.reshape_dims = (-1, heads, head_dim)
    reshape.second_transpose = trt.Permutation([1, 0, 2])
    batch = network.add_shuffle(reshape.get_output(0))
    batch.reshape_dims = (1, heads, -1, head_dim)
    return batch.get_output(0)


def _heads_to_rows(network, tensor, width: int, trt):
    reshape = network.add_shuffle(tensor)
    reshape.first_transpose = trt.Permutation([0, 2, 1, 3])
    reshape.reshape_dims = (-1, width)
    return reshape.get_output(0)


def _per_head_norm(network, tensor, weight, heads: int, spec, trt, op):
    reshape = network.add_shuffle(tensor)
    reshape.reshape_dims = (-1, heads, spec.head_dim)
    normalized = op.qwen_rms_norm(
        network, reshape.get_output(0), weight, spec.head_dim, spec.norm_eps
    )
    flatten = network.add_shuffle(normalized)
    flatten.reshape_dims = (-1, heads * spec.head_dim)
    return flatten.get_output(0)


def _select_mrope_cache(network, cache, position_ids, spec, trt, op):
    selected = network.add_gather(cache, position_ids, 0).get_output(0)
    axes = _mrope_frequency_axis_map(
        spec.mrope_section, spec.head_dim, interleaved=spec.mrope_interleaved
    )
    axis_values = []
    for axis in range(3):
        axis_index = op.constant(network, np.asarray([axis], dtype=np.int32), dtype=np.int32)
        axis_values.append(network.add_gather(selected, axis_index, 0).get_output(0))
    result = axis_values[0]
    for axis in (1, 2):
        mask = op.constant(
            network,
            np.asarray(axes == axis, dtype=np.bool_).reshape(1, 1, -1),
            dtype=np.bool_,
        )
        result = network.add_select(mask, axis_values[axis], result).get_output(0)
    return result


def _apply_mrope(network, tensor, cache, position_ids, heads: int, spec, trt, op):
    source_dtype = tensor.dtype
    value = _rows_to_heads(network, tensor, heads, spec.head_dim, trt)
    cosine, sine = cache
    cosine = _select_mrope_cache(network, cosine, position_ids, spec, trt, op)
    sine = _select_mrope_cache(network, sine, position_ids, spec, trt, op)
    cosine = op.cast(network, cosine, source_dtype)
    sine = op.cast(network, sine, source_dtype)
    rotary = network.add_rotary_embedding(value, cosine, sine, False, spec.head_dim)
    if rotary is None:
        raise RuntimeError("TensorRT rejected MiniMax-H3 interleaved MRoPE")
    return _heads_to_rows(network, rotary.get_output(0), heads * spec.head_dim, trt)


def _language_layer(network, hidden, weights, index, rope_cache, position_ids, spec, trt, op):
    prefix = f"{_LANGUAGE_PREFIX}.layers.{index}"
    normalized = op.qwen_rms_norm(
        network,
        hidden,
        weights[f"{prefix}.input_layernorm.weight"],
        spec.hidden_size,
        spec.norm_eps,
    )
    q = op.linear(network, normalized, weights[f"{prefix}.self_attn.q_proj.weight"])
    k = op.linear(network, normalized, weights[f"{prefix}.self_attn.k_proj.weight"])
    v = op.linear(network, normalized, weights[f"{prefix}.self_attn.v_proj.weight"])
    q = _per_head_norm(
        network,
        q,
        weights[f"{prefix}.self_attn.q_norm.weight"],
        spec.num_heads,
        spec,
        trt,
        op,
    )
    k = _per_head_norm(
        network,
        k,
        weights[f"{prefix}.self_attn.k_norm.weight"],
        spec.num_kv_heads,
        spec,
        trt,
        op,
    )
    q = _apply_mrope(network, q, rope_cache, position_ids, spec.num_heads, spec, trt, op)
    k = _apply_mrope(network, k, rope_cache, position_ids, spec.num_kv_heads, spec, trt, op)
    q = _rows_to_heads(network, q, spec.num_heads, spec.head_dim, trt)
    k = _rows_to_heads(network, k, spec.num_kv_heads, spec.head_dim, trt)
    v = _rows_to_heads(network, v, spec.num_kv_heads, spec.head_dim, trt)
    scale = op.constant(
        network,
        np.full((1, 1, 1, 1), 1.0 / math.sqrt(spec.head_dim), dtype=np.float32),
    )
    scale = op.cast(network, scale, q.dtype)
    q = network.add_elementwise(q, scale, trt.ElementWiseOperation.PROD).get_output(0)
    attention = network.add_attention(q, k, v, trt.AttentionNormalizationOp.SOFTMAX, True)
    if attention is None:
        raise RuntimeError(f"TensorRT failed to add MiniMax-H3 attention layer {index}")
    attention.name = f"{prefix}.self_attn.native_causal_attention"
    attention.metadata = f"trtmc.native_op=IAttention;source={attention.name}"
    attention.get_output(0).name = f"{attention.name}.output"
    attention.decomposable = False
    update = _heads_to_rows(network, attention.get_output(0), spec.attention_size, trt)
    update = op.linear(network, update, weights[f"{prefix}.self_attn.o_proj.weight"])
    hidden = network.add_elementwise(hidden, update, trt.ElementWiseOperation.SUM).get_output(0)

    normalized = op.qwen_rms_norm(
        network,
        hidden,
        weights[f"{prefix}.post_attention_layernorm.weight"],
        spec.hidden_size,
        spec.norm_eps,
    )
    gate = op.linear(network, normalized, weights[f"{prefix}.mlp.gate_proj.weight"])
    up = op.linear(network, normalized, weights[f"{prefix}.mlp.up_proj.weight"])
    gate = op.silu(network, gate)
    gated = network.add_elementwise(gate, up, trt.ElementWiseOperation.PROD).get_output(0)
    update = op.linear(network, gated, weights[f"{prefix}.mlp.down_proj.weight"])
    return network.add_elementwise(hidden, update, trt.ElementWiseOperation.SUM).get_output(0)


def _assemble_language_conditioner_graph(network, weights, spec, inputs, trt, op):
    table = op.weight_constant(network, weights[f"{_LANGUAGE_PREFIX}.embed_tokens.weight"])
    table = op.cast(network, table, trt.bfloat16)
    token_embeddings = network.add_gather(table, inputs["input_ids"], 0).get_output(0)
    selector = _selector_condition(network, inputs["vision_selector"], trt, op)
    hidden = _hard_select(network, selector, inputs["vision_embeddings"], token_embeddings)
    cosine, sine = _make_rope_tables(spec)
    rope_cache = (
        op.weight_constant(network, cosine),
        op.weight_constant(network, sine),
    )
    for index in range(spec.output_layers):
        hidden = _language_layer(
            network,
            hidden,
            weights,
            index,
            rope_cache,
            inputs["mrope_position_ids"],
            spec,
            trt,
            op,
        )
        if index < spec.deepstack_levels:
            deepstack = _hard_gate(
                network,
                selector,
                inputs[f"deepstack_embeddings_{index}"],
                trt,
                op,
            )
            hidden = network.add_elementwise(
                hidden, deepstack, trt.ElementWiseOperation.SUM
            ).get_output(0)
    output = op.cast(network, hidden, trt.float32)
    output.name = "encoder_hidden_states"
    network.mark_output(output)
    return output


def _build_language_conditioner_engine(
    weights: Mapping[str, Any],
    spec: MiniMaxH3LanguageConditionerSpec,
    *,
    verbose: bool,
    consume_weights: bool,
    workspace_bytes: int | None,
) -> bytes:
    from tensorrt_model_connect import trt_compat

    from . import graph_ops as op

    trt = trt_compat.get_trt()
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    op.configure_builder(config)
    op.configure_workspace(
        config,
        workspace_bytes,
        default_bytes=TEXT_ENCODER_DEFAULT_WORKSPACE_BYTES,
    )
    inputs = _declare_inputs(network, spec, trt)
    _add_optimization_profile(builder, config, spec)
    try:
        _assemble_language_conditioner_graph(network, weights, spec, inputs, trt, op)
        op.validate_native_network(
            network,
            expected_attentions=spec.output_layers,
            label="dynamic language conditioner",
        )
        print(
            "[minimax-h3] building dynamic Qwen3-VL language conditioner: "
            f"workflow={spec.workflow}, rows={spec.min_rows}..{spec.max_rows}, "
            f"layers={spec.output_layers}, "
            f"mrope={spec.mrope_section}, deepstack={spec.deepstack_levels}",
            file=sys.stderr,
        )
        plan = builder.build_serialized_network(network, config)
    finally:
        op.release_weight_buffers(network)
        if consume_weights:
            weights.clear()
    if plan is None:
        raise RuntimeError("TensorRT failed to build MiniMax-H3 language conditioner")
    del network, config, builder
    gc.collect()
    return bytes(plan)


def build_language_conditioner_engine(
    checkpoint_config: Mapping[str, Any],
    weights: Mapping[str, Any],
    *,
    workflow: str = "fl2va",
    verbose: bool = False,
    consume_weights: bool = False,
    workspace_bytes: int | None = None,
) -> bytes:
    """Build the exact dynamic first-50-layer H3 language conditioner."""

    spec = MiniMaxH3LanguageConditionerSpec.from_checkpoint_config(
        checkpoint_config, workflow=workflow
    )
    validate_conditioner_weights(weights, spec)
    return _build_language_conditioner_engine(
        weights,
        spec,
        verbose=verbose,
        consume_weights=consume_weights,
        workspace_bytes=workspace_bytes,
    )


__all__ = [
    "MiniMaxH3LanguageConditionerSpec",
    "build_language_conditioner_engine",
    "checkpoint_keys",
    "expected_weight_shapes",
    "validate_conditioner_weights",
    "validate_presentation_bindings",
]

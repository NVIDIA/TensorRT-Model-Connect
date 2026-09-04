# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark semantics for public Task API operations.

This module deliberately knows nothing about model families or E2E manifests.  It
describes what one public pipeline call produces and how those observations are
reduced.  ``task_adapters`` owns the separate manifest-task-to-operation seam.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .types import BenchmarkError


@dataclass(frozen=True)
class RateMetric:
    """Sum an observation field and divide it by total measured seconds."""

    observation_field: str
    result_name: str
    inverse_result_name: str | None = None


@dataclass(frozen=True)
class PerItemLatencyMetric:
    """Normalize each request latency by the number of produced items."""

    count_field: str
    result_name: str


MetricFactory = Callable[
    [Mapping[str, Any]],
    tuple[tuple[RateMetric, ...], PerItemLatencyMetric | None],
]


@dataclass(frozen=True)
class OperationSpec:
    """Performance semantics for one native worker Task API operation."""

    name: str
    supports_batch: bool = False
    rate_metrics: tuple[RateMetric, ...] = ()
    stage_timings: tuple[str, ...] = ()
    per_item_latency: PerItemLatencyMetric | None = None
    metric_factory: MetricFactory | None = None

    def metrics_for_request(
        self, request: Mapping[str, Any]
    ) -> tuple[tuple[RateMetric, ...], PerItemLatencyMetric | None]:
        if self.metric_factory is not None:
            return self.metric_factory(request)
        return self.rate_metrics, self.per_item_latency


def _generated_media_metrics(
    request: Mapping[str, Any],
) -> tuple[tuple[RateMetric, ...], PerItemLatencyMetric]:
    media_type = str(request.get("media_type", "image") or "image")
    if media_type == "image":
        return (
            (RateMetric("generated_images", "images_per_s"),),
            PerItemLatencyMetric("generated_images", "seconds_per_image_p50"),
        )
    if media_type == "video":
        return (
            (
                RateMetric("generated_images", "videos_per_s"),
                RateMetric("generated_frames", "frames_per_s"),
            ),
            PerItemLatencyMetric("generated_images", "seconds_per_video_p50"),
        )
    raise BenchmarkError(f"unsupported generated media type {media_type!r}")


_OPERATIONS = (
    OperationSpec(
        name="generate",
        rate_metrics=(RateMetric("output_tokens", "output_tokens_per_s"),),
        stage_timings=("prefill_ms", "decode_ms"),
    ),
    OperationSpec(
        name="generate_image",
        supports_batch=True,
        rate_metrics=(RateMetric("generated_images", "images_per_s"),),
        per_item_latency=PerItemLatencyMetric("generated_images", "seconds_per_image_p50"),
        metric_factory=_generated_media_metrics,
    ),
    OperationSpec(
        name="generate_audio",
        rate_metrics=(
            RateMetric(
                "output_audio_seconds",
                "audio_seconds_per_s",
                inverse_result_name="realtime_factor",
            ),
            RateMetric("output_samples", "audio_samples_per_s"),
        ),
    ),
    OperationSpec(
        name="speak",
        rate_metrics=(
            RateMetric(
                "input_audio_seconds",
                "input_audio_seconds_per_s",
                inverse_result_name="input_realtime_factor",
            ),
            RateMetric("output_audio_seconds", "output_audio_seconds_per_s"),
        ),
    ),
    OperationSpec(
        name="segment",
        rate_metrics=(
            RateMetric("segmented_images", "images_per_s"),
            RateMetric("mask_pixels", "mask_pixels_per_s"),
        ),
    ),
    OperationSpec(
        name="segment_prompted",
        rate_metrics=(
            RateMetric("segmented_images", "images_per_s"),
            RateMetric("generated_masks", "masks_per_s"),
            RateMetric("mask_pixels", "mask_pixels_per_s"),
        ),
    ),
    OperationSpec(
        name="classify",
        rate_metrics=(RateMetric("classified_images", "images_per_s"),),
    ),
    OperationSpec(
        name="extract_features",
        rate_metrics=(RateMetric("processed_images", "images_per_s"),),
    ),
    OperationSpec(
        name="disparity",
        rate_metrics=(
            RateMetric("stereo_pairs", "stereo_pairs_per_s"),
            RateMetric("disparity_pixels", "disparity_pixels_per_s"),
        ),
    ),
    OperationSpec(
        name="rerank",
        rate_metrics=(RateMetric("documents", "documents_per_s"),),
    ),
    OperationSpec(
        name="encode",
        rate_metrics=(
            RateMetric("embedding_vectors", "embedding_vectors_per_s"),
            RateMetric("embedding_elements", "embedding_elements_per_s"),
        ),
    ),
    OperationSpec(
        name="embed",
        rate_metrics=(
            RateMetric("embedding_vectors", "embedding_vectors_per_s"),
            RateMetric("embedding_elements", "embedding_elements_per_s"),
        ),
    ),
    OperationSpec(
        name="solve",
        rate_metrics=(
            RateMetric("windows", "windows_per_s"),
            RateMetric("forecast_elements", "forecast_elements_per_s"),
        ),
    ),
    OperationSpec(
        name="control",
        rate_metrics=(RateMetric("action_steps", "action_steps_per_s"),),
    ),
    OperationSpec(
        name="transcribe",
        rate_metrics=(
            RateMetric(
                "input_audio_seconds",
                "audio_seconds_per_s",
                inverse_result_name="realtime_factor",
            ),
            RateMetric("output_tokens", "output_tokens_per_s"),
        ),
        stage_timings=("first_partial_ms",),
    ),
)


def _index_operations() -> dict[str, OperationSpec]:
    indexed: dict[str, OperationSpec] = {}
    for operation in _OPERATIONS:
        if operation.name in indexed:
            raise RuntimeError(f"duplicate benchmark operation {operation.name!r}")
        indexed[operation.name] = operation
    return indexed


_BY_NAME = _index_operations()


def operation_for_name(name: str) -> OperationSpec:
    try:
        return _BY_NAME[name]
    except KeyError as exc:
        available = ", ".join(sorted(_BY_NAME))
        raise BenchmarkError(
            f"benchmark operation {name!r} is not registered; available: {available}"
        ) from exc


def registered_operations() -> tuple[OperationSpec, ...]:
    return _OPERATIONS

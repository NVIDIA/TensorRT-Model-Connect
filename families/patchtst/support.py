# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint-owned PatchTST identity and semantic head selection."""

from tensorrt_model_connect.model_support import FamilySupport, ModelMetadata, family_support


def _config_value(config, key: str, fallback=None):
    if isinstance(config, dict):
        return config.get(key, fallback)
    return getattr(config, key, fallback)


def _normalize_task_type(config) -> str:
    explicit = str(
        _config_value(config, "patchtst_task", _config_value(config, "task_type", ""))
    ).lower()
    if explicit:
        if "class" in explicit:
            return "classification"
        if "regress" in explicit:
            return "regression"
        if "forecast" in explicit or "predict" in explicit:
            return "forecast"

    problem_type = str(_config_value(config, "problem_type", "")).lower()
    if "class" in problem_type:
        return "classification"
    if "regress" in problem_type:
        return "regression"

    architectures = _config_value(config, "architectures", [])
    if isinstance(architectures, str):
        architectures = [architectures]
    for arch in architectures or []:
        arch_l = str(arch).lower()
        if "class" in arch_l:
            return "classification"
        if "regress" in arch_l:
            return "regression"
        if "forecast" in arch_l or "predict" in arch_l:
            return "forecast"
    return "forecast"


def semantic_task(config: dict[str, object]) -> str:
    mode = _normalize_task_type(config)
    if mode == "forecast":
        return "series_to_point_forecast"
    if mode == "regression":
        if config.get("loss") == "mse":
            return "series_to_regression_values"
        if str(config.get("distribution_output", "")).lower() in {
            "normal",
            "student_t",
            "negative_binomial",
        }:
            return "series_to_regression_distribution"
        raise NotImplementedError("PatchTST regression distribution is unsupported")
    raise NotImplementedError("PatchTST supports forecast and regression heads only")


_matches = family_support(
    model_types=("patchtst",),
    tasks=(
        "series_to_point_forecast",
        "series_to_regression_values",
        "series_to_regression_distribution",
    ),
    default_task="series_to_point_forecast",
)


def describe(metadata: ModelMetadata) -> FamilySupport | None:
    if _matches(metadata) is None:
        return None
    try:
        task = semantic_task(metadata.config)
    except NotImplementedError:
        return None
    return FamilySupport(tasks=(task,), default_task=task)

"""PatchTSMixer build strategy — numeric time-series models.

This strategy wraps HuggingFace PatchTSMixer checkpoints so torch.export
sees a raw tensor interface:
  - past_values:   float32 [1, context_length, num_input_channels]
  - observed_mask: float32 [1, context_length, num_input_channels]

The wrapper keeps a single float32 output tensor alive for TensorRT export
and runtime extraction. The concrete output field depends on the task kind:
  - prediction -> prediction_outputs
  - classification -> prediction_outputs
  - regression -> regression_outputs
  - pretraining -> prediction_outputs
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _raw_config_value(config, key: str, default=None):
    raw = getattr(config, "raw", None)
    if isinstance(raw, dict) and key in raw:
        return raw[key]
    return getattr(config, key, default)


def _normalize_task_kind(task: str) -> str:
    task = task.lower().strip()
    if "regress" in task:
        return "regression"
    if "class" in task:
        return "classification"
    if "pretrain" in task:
        return "pretraining"
    if "forecast" in task or "predict" in task or "prediction" in task:
        return "prediction"
    return task


def infer_patchtsmixer_task_kind(config) -> str:
    """Infer the PatchTSMixer task kind from HF config metadata."""
    task = _raw_config_value(config, "task_type", "")
    if isinstance(task, str) and task.strip():
        return _normalize_task_kind(task)

    architectures = _raw_config_value(config, "architectures", [])
    if isinstance(architectures, str):
        architectures = [architectures]
    for arch in architectures or []:
        arch_l = str(arch).lower()
        if "pretrain" in arch_l:
            return "pretraining"
        if "regress" in arch_l:
            return "regression"
        if "class" in arch_l:
            return "classification"
        if "predict" in arch_l:
            return "prediction"

    if _raw_config_value(config, "prediction_length", None) is not None:
        return "prediction"
    if _raw_config_value(config, "num_targets", None) is not None:
        return "regression"
    return "prediction"


def get_patchtsmixer_context_length(config, fallback: int) -> int:
    value = _raw_config_value(config, "context_length", fallback)
    return int(value) if int(value) > 0 else int(fallback)


def get_patchtsmixer_num_input_channels(config) -> int:
    for key in ("num_input_channels", "input_size", "feature_size"):
        value = _raw_config_value(config, key, 0)
        if isinstance(value, int) and value > 0:
            return value
    return 1


def get_patchtsmixer_prediction_length(config) -> int:
    value = _raw_config_value(config, "prediction_length", 1)
    return int(value) if int(value) > 0 else 1


def get_patchtsmixer_num_targets(config) -> int:
    for key in ("num_targets", "num_labels"):
        value = _raw_config_value(config, key, 0)
        if isinstance(value, int) and value > 0:
            return value
    return get_patchtsmixer_prediction_length(config)


def resolve_patchtsmixer_model_class(task_kind: str):
    from transformers import (
        PatchTSMixerForPrediction,
        PatchTSMixerForPretraining,
        PatchTSMixerForRegression,
        PatchTSMixerForTimeSeriesClassification,
    )

    task_kind = _normalize_task_kind(task_kind)
    if task_kind == "pretraining":
        return PatchTSMixerForPretraining
    if task_kind == "classification":
        return PatchTSMixerForTimeSeriesClassification
    if task_kind == "regression":
        return PatchTSMixerForRegression
    return PatchTSMixerForPrediction


class PatchTSMixerWrapper(nn.Module):
    """Wrap PatchTSMixer for raw TRT compilation."""

    def __init__(
        self,
        model: nn.Module,
        config,
        context_length: int,
        num_input_channels: int,
        *,
        compute_dtype: torch.dtype = torch.float16,
        task_kind: str = "prediction",
    ):
        super().__init__()
        self.model = model
        self.context_length = int(context_length)
        self.num_input_channels = int(num_input_channels)
        self.compute_dtype = compute_dtype
        self.task_kind = _normalize_task_kind(task_kind)
        self.prediction_length = get_patchtsmixer_prediction_length(config)
        self.num_targets = get_patchtsmixer_num_targets(config)

    def _select_output(self, outputs):
        for key in (
            "prediction_outputs",
            "regression_outputs",
            "classification_outputs",
            "logits",
            "score",
            "output",
            "last_hidden_state",
        ):
            if hasattr(outputs, key):
                value = getattr(outputs, key)
                if value is not None:
                    return value
        if isinstance(outputs, (tuple, list)) and outputs:
            return outputs[0]
        if isinstance(outputs, dict):
            for key in (
                "prediction_outputs",
                "regression_outputs",
                "classification_outputs",
                "logits",
                "score",
                "output",
                "last_hidden_state",
            ):
                if key in outputs and outputs[key] is not None:
                    return outputs[key]
        raise RuntimeError("PatchTSMixerWrapper: could not locate an output tensor")

    def forward(self, past_values, observed_mask=None):
        past_values = past_values.to(self.compute_dtype)

        if observed_mask is None:
            observed_mask = torch.ones_like(past_values, dtype=self.compute_dtype)
        else:
            observed_mask = observed_mask.to(self.compute_dtype)

        masked_values = past_values * observed_mask

        if self.task_kind in ("prediction", "pretraining"):
            outputs = self.model(
                past_values=masked_values,
                observed_mask=observed_mask,
                return_loss=False,
                return_dict=True,
            )
        else:
            outputs = self.model(
                past_values=masked_values,
                return_loss=False,
                return_dict=True,
            )

        tensor = self._select_output(outputs)
        return (tensor.to(torch.float32),)


class PatchTSMixerBuildStrategy:
    """Build strategy for PatchTSMixer numeric time-series models."""

    name = "patchtsmixer"
    runtime_strategy = "patchtsmixer_torchtrt"

    def wrap_model(
        self,
        model: nn.Module,
        config,
        max_cache_length: int,
        *,
        compute_dtype: torch.dtype | None = None,
    ) -> nn.Module:
        if compute_dtype is None:
            compute_dtype = torch.float16
        task_kind = infer_patchtsmixer_task_kind(config)
        context_length = get_patchtsmixer_context_length(config, max_cache_length)
        num_input_channels = get_patchtsmixer_num_input_channels(config)
        return PatchTSMixerWrapper(
            model,
            config,
            context_length,
            num_input_channels,
            compute_dtype=compute_dtype,
            task_kind=task_kind,
        )

    def make_export_args(
        self,
        config,
        max_cache_length: int,
        *,
        precision: str = "fp16",
    ) -> tuple:
        context_length = get_patchtsmixer_context_length(config, max_cache_length)
        num_input_channels = get_patchtsmixer_num_input_channels(config)
        past_values = torch.zeros(
            (1, context_length, num_input_channels),
            dtype=torch.float32,
            device="cuda",
        )
        observed_mask = torch.ones_like(past_values)
        return (past_values, observed_mask)

    def pre_export_setup(self) -> None:
        pass

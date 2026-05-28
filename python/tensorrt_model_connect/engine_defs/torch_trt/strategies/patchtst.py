"""PatchTST build strategy — numeric sequence models.

This strategy keeps the exported TRT interface numeric-only:
  - past_values: float32 [1, context_length, num_input_channels]
  - past_observed_mask: float32 [1, context_length, num_input_channels]

The wrapper casts inputs to the model compute dtype internally and always
returns a single float32 tensor so the runtime can extract it uniformly.
"""

from __future__ import annotations

import sys
from typing import Any

import torch
import torch.nn as nn


def _config_value(config: Any, key: str, fallback: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, fallback)
    return getattr(config, key, fallback)


def _normalize_task_type(config: Any) -> str:
    explicit = str(
        _config_value(config, "patchtst_task",
                      _config_value(config, "task_type", ""))).lower()
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

    # PatchTST is primarily a forecasting model when no task metadata is set.
    return "forecast"


def _precision_to_dtype(precision: str) -> torch.dtype:
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp32":
        return torch.float32
    return torch.float16


def _context_length(config: Any, fallback: int) -> int:
    value = _config_value(config, "context_length", fallback)
    try:
        return max(int(value), 1)
    except (TypeError, ValueError):
        return max(int(fallback), 1)


def _num_input_channels(config: Any) -> int:
    value = _config_value(config, "num_input_channels", 1)
    try:
        return max(int(value), 1)
    except (TypeError, ValueError):
        return 1


def _select_model_output(outputs: Any, task_type: str) -> torch.Tensor:
    def _coerce_tensor(value: Any) -> torch.Tensor | None:
        if torch.is_tensor(value):
            return value
        if isinstance(value, (tuple, list)) and value:
            tensors = [item for item in value if torch.is_tensor(item)]
            if len(tensors) == len(value):
                return torch.stack(tensors, dim=-1)
        return None

    if isinstance(outputs, torch.Tensor):
        return outputs

    if isinstance(outputs, (tuple, list)):
        for value in outputs:
            tensor = _coerce_tensor(value)
            if tensor is not None:
                return tensor
        raise RuntimeError("PatchTST wrapper produced no tensor outputs")

    preferred_names = {
        "classification": ("prediction_logits", "logits", "output", "prediction"),
        "regression": ("regression_outputs", "prediction_outputs", "logits",
                        "output", "prediction"),
        "forecast": ("prediction_outputs", "prediction", "output", "logits"),
    }.get(task_type, ("prediction_outputs", "output", "logits"))

    for name in preferred_names:
        value = getattr(outputs, name, None)
        tensor = _coerce_tensor(value)
        if tensor is not None:
            return tensor

    if hasattr(outputs, "__dict__"):
        for value in outputs.__dict__.values():
            tensor = _coerce_tensor(value)
            if tensor is not None:
                return tensor

    raise RuntimeError("PatchTST wrapper could not find a tensor output")


class PatchTSTWrapper(nn.Module):
    """Wrap a PatchTST HF model with TRT-friendly numeric I/O."""

    def __init__(self, model: nn.Module, task_type: str,
                 compute_dtype: torch.dtype):
        super().__init__()
        self.model = model
        self.task_type = task_type
        self.compute_dtype = compute_dtype

    def forward(self, past_values, past_observed_mask=None):
        values = past_values.to(self.compute_dtype)

        kwargs = {"past_values": values}
        if past_observed_mask is not None:
            kwargs["past_observed_mask"] = past_observed_mask.to(torch.float32).gt(0.5)

        outputs = self.model(**kwargs)
        tensor = _select_model_output(outputs, self.task_type)
        return (tensor.to(torch.float32),)


class PatchTSTBuildStrategy:
    """Build strategy for PatchTST numeric sequence models."""

    name = "patchtst"
    runtime_strategy = "patchtst_torchtrt"

    def wrap_model(
        self,
        model: nn.Module,
        config,
        max_cache_length: int,
        *,
        compute_dtype: torch.dtype | None = None,
    ) -> nn.Module:
        task_type = _normalize_task_type(config)
        dtype = compute_dtype or torch.float16
        return PatchTSTWrapper(model, task_type, dtype)

    def make_export_args(
        self,
        config,
        max_cache_length: int,
        *,
        precision: str = "fp16",
    ) -> tuple:
        # Keep the exported engine input float32 so the C++ runtime can feed
        # it directly from its float* solve() interface.
        _ = _precision_to_dtype(precision)
        context_length = _context_length(config, max_cache_length)
        num_input_channels = _num_input_channels(config)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        past_values = torch.zeros(
            (1, context_length, num_input_channels),
            dtype=torch.float32,
            device=device,
        )
        past_observed_mask = torch.ones(
            (1, context_length, num_input_channels),
            dtype=torch.float32,
            device=device,
        )
        return (past_values, past_observed_mask)

    def pre_export_setup(self) -> None:
        pass


def _register_strategy() -> None:
    pkg = sys.modules.get(__package__)
    if pkg is None:
        return

    registry = getattr(pkg, "_strategies", None)
    if registry is None:
        init_registry = getattr(pkg, "_init_registry", None)
        registry = init_registry() if callable(init_registry) else {}

    registry["patchtst"] = PatchTSTBuildStrategy()
    setattr(pkg, "_strategies", registry)


_register_strategy()

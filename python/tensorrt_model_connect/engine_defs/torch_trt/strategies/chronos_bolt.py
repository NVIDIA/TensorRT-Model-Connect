"""Chronos-Bolt build strategy.

This strategy wraps the official Chronos-Bolt forecasting model with a
numeric I/O contract:
  - input: context tensor [B, T], left-padded with NaNs
  - output: quantile forecast tensor [B, Q, H]
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def _chronos_raw_config(config: Any) -> dict[str, Any]:
    raw = getattr(config, "raw", {}) or {}
    chronos_cfg = raw.get("chronos_config")
    if isinstance(chronos_cfg, dict):
        return chronos_cfg
    return raw


def _first_positive_int(raw: dict[str, Any], keys: tuple[str, ...], fallback: int) -> int:
    for key in keys:
        value = raw.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return fallback


def _num_quantiles(raw: dict[str, Any]) -> int:
    quantiles = raw.get("quantiles")
    if isinstance(quantiles, (list, tuple)) and quantiles:
        return max(1, len(quantiles))
    value = raw.get("num_quantiles")
    if isinstance(value, int) and value > 0:
        return value
    return 3


class ChronosBoltForecastWrapper(nn.Module):
    """Wrap the official Chronos-Bolt model with numeric tensor I/O."""

    def __init__(
        self,
        model: nn.Module,
        max_context_length: int,
    ) -> None:
        super().__init__()
        self.model = model
        self.max_context_length = max_context_length
        param = next(self.model.parameters(), None)
        if param is not None:
            self.to(device=param.device)

    def forward(self, context):
        values = context.to(torch.float32)
        if values.dim() == 1:
            values = values.unsqueeze(0)
        if values.dim() != 2:
            raise ValueError(
                f"ChronosBoltForecastWrapper expects context [B, T], got {tuple(values.shape)}"
            )
        if values.size(1) > self.max_context_length:
            values = values[:, -self.max_context_length:]
        outputs = self.model(context=values)
        return (outputs.quantile_preds.to(torch.float32),)


class ChronosBoltBuildStrategy:
    name = "chronos_bolt"
    runtime_strategy = "chronos_bolt_torchtrt"

    def wrap_model(
        self,
        model: nn.Module,
        config,
        max_cache_length: int,
        *,
        compute_dtype: torch.dtype | None = None,
    ) -> nn.Module:
        raw = _chronos_raw_config(config)
        context_length = _first_positive_int(
            raw,
            ("context_length", "input_length", "max_context_length"),
            fallback=max_cache_length,
        )

        wrapper = ChronosBoltForecastWrapper(
            model=model,
            max_context_length=context_length,
        )
        if compute_dtype is not None:
            wrapper = wrapper.to(dtype=compute_dtype)
        return wrapper

    def make_export_args(
        self,
        config,
        max_cache_length: int,
        *,
        precision: str = "fp16",
    ) -> tuple:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        raw = _chronos_raw_config(config)
        context_length = _first_positive_int(
            raw,
            ("context_length", "input_length", "max_context_length"),
            fallback=max_cache_length,
        )
        context = torch.full(
            (1, context_length), float("nan"), dtype=torch.float32, device=device)
        valid_len = min(context_length, 16)
        if valid_len > 0:
            context[:, -valid_len:] = torch.linspace(
                0.0,
                1.0,
                steps=valid_len,
                dtype=torch.float32,
                device=device,
            )
        return (context,)

    def pre_export_setup(self) -> None:
        pass


plugin = ChronosBoltBuildStrategy()

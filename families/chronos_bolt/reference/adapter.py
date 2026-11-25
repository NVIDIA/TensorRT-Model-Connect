# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chronos-Bolt's native pipeline for the trtmc-perf-serve reference backend (``solve``): the official
``chronos`` package's ChronosBoltPipeline, its quantile forecast over the model's prediction length, laid out as
the ``series_to_quantile_forecast`` Task returns it: [quantile, horizon, channel] with the quantile levels."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping



class Adapter:
    def __init__(self, spec: Any, host: Any) -> None:
        self.host = host
        from chronos import ChronosBoltPipeline

        self.spec = spec
        options = {"revision": spec.revision} if spec.revision else {}
        self.pipeline = ChronosBoltPipeline.from_pretrained(spec.model, device_map=spec.device, dtype=spec.dtype,
                                                            **options)
        host.maybe_compile(self.pipeline.model, spec)

    def invoke(self, request: Mapping[str, Any], artifact_base: Path) -> Any:
        import torch

        past = [float(value) for value in self.host.required(request, "past_values")]
        context = torch.tensor(past, dtype=torch.float32, device=self.spec.device)
        forecast, model_ms = self.host.timed(lambda: self.pipeline.predict(
            context, prediction_length=self.pipeline.model_prediction_length, limit_prediction_length=True))
        # The pipeline's [series, quantile, horizon] for one univariate series is the Task's [quantile, horizon, 1].
        quantiles = forecast[0].unsqueeze(-1)
        observation = self.host.tensor_observation(quantiles, artifact_base, inline=True)
        return self.host.invocation({**observation, "axes": ["quantile", "horizon", "channel"],
                                     "quantile_levels": [float(level) for level in self.pipeline.quantiles]}, model_ms)

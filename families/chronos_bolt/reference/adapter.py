# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chronos-Bolt's native pipeline for the trtmc-perf-serve reference backend (``solve``): the official
``chronos`` package's ChronosBoltPipeline, its quantile forecast over the model's prediction length."""

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

    def invoke(self, request: Mapping[str, Any], artifact_base: Path) -> Any:
        import torch

        past = [float(value) for value in self.host.required(request, "past_values")]
        context = torch.tensor(past, dtype=torch.float32, device=self.spec.device)
        forecast, model_ms = self.host.timed(lambda: self.pipeline.predict(
            context, prediction_length=self.pipeline.model_prediction_length, limit_prediction_length=True))
        return self.host.invocation(self.host.tensor_observation(forecast, artifact_base, inline=True), model_ms)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TimesFM's native pipeline for the trtmc-perf-serve reference backend (``solve``): Transformers'
TimesFmModelForPrediction decoder on the context left-padded to the model's context length (padding mask,
the request's frequency), the point forecast over the model's horizon."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping



class Adapter:
    def __init__(self, spec: Any, host: Any) -> None:
        self.host = host
        import transformers

        self.spec = spec
        self.model = transformers.TimesFmModelForPrediction.from_pretrained(
            spec.model, dtype=spec.dtype, **spec.pretrained_kwargs()).eval().to(spec.device)

    def invoke(self, request: Mapping[str, Any], artifact_base: Path) -> Any:
        import torch

        context = int(self.model.config.context_length)
        raw = [float(value) for value in self.host.required(request, "past_values")]
        count = min(len(raw), context)
        values, padding = [0.0] * context, [1] * context
        values[-count:], padding[-count:] = raw[-count:], [0] * count
        device, dtype = self.spec.device, self.spec.dtype

        def run() -> Any:
            decoded = self.model.decoder(
                past_values=torch.tensor(values, dtype=dtype, device=device).reshape(1, context),
                past_values_padding=torch.tensor(padding, dtype=torch.int32, device=device).reshape(1, context),
                freq=torch.tensor([[int(request.get("frequency", 0))]], dtype=torch.long, device=device),
                output_attentions=False, output_hidden_states=False)
            output = self.model._postprocess_output(decoded.last_hidden_state, (decoded.loc, decoded.scale))
            return output[:, -1, : self.model.config.horizon_length, 0]

        forecast, model_ms = self.host.timed(run)
        return self.host.invocation(self.host.tensor_observation(forecast, artifact_base, inline=True), model_ms)

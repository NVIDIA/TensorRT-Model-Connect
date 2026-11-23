# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PatchTST's Transformers regression model for ``solve``, as tests/test_e2e.py runs it: the flat row-major
``past_values`` as [context, channel] with its observed mask, and the regression head's distribution
parameters stacked on the last axis (the bundle's output layout)."""

from pathlib import Path
from typing import Any, Mapping


class Adapter:
    def __init__(self, spec: Any, host: Any) -> None:
        from transformers import PatchTSTForRegression

        self.spec, self.host = spec, host
        self.model = host.maybe_compile(PatchTSTForRegression.from_pretrained(
            spec.model, **spec.model_kwargs()).to(spec.device, spec.dtype).eval(), spec)
        self.channels = int(self.model.config.num_input_channels)
        self.context = int(self.model.config.context_length)

    def invoke(self, request: Mapping[str, Any], artifact_base: Path) -> Any:
        import torch

        values = torch.tensor([float(value) for value in self.host.required(request, "past_values")])
        mask = torch.tensor([float(value) for value in request.get("observed_mask") or [1.0] * values.numel()])
        if values.numel() != self.context * self.channels or mask.numel() != values.numel():
            raise self.host.Error(f"PatchTST regression takes {self.context} x {self.channels} past values "
                                  f"and as many mask values, got {values.numel()} and {mask.numel()}")
        past = values.reshape(1, self.context, self.channels).to(self.spec.device, self.spec.dtype)
        observed = mask.reshape(1, self.context, self.channels).gt(0.5).to(self.spec.device)

        def call() -> Any:
            output = self.model(past_values=past, past_observed_mask=observed).regression_outputs
            return torch.stack(output, dim=-1) if isinstance(output, tuple) else output

        output, model_ms = self.host.timed(call)
        return self.host.invocation(self.host.tensor_observation(output, artifact_base, inline=True), model_ms)

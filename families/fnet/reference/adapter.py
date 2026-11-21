# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FNet's reference uses the same padded sequence length as its Fourier engine."""

from pathlib import Path
from typing import Any, Mapping


class Adapter:
    def __init__(self, spec: Any, host: Any) -> None:
        import transformers

        self.spec, self.host = spec, host
        self.max_length = int(spec.options["max_sequence_length"])
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(spec.model, **spec.pretrained_kwargs())
        self.model = transformers.FNetModel.from_pretrained(
            spec.model, **spec.model_kwargs()).to(spec.device).eval()
        if spec.mode == "compile":
            import torch

            self.model.forward = torch.compile(self.model.forward)

    def invoke(self, request: Mapping[str, Any], artifact_base: Path) -> Any:
        batch = self.tokenizer(
            str(self.host.required(request, "prompt")), return_tensors="pt",
            padding="max_length", truncation=True, max_length=self.max_length).to(self.spec.device)
        output, ms = self.host.timed(lambda: self.model(**batch).last_hidden_state[:, 0])
        observation = self.host.tensor_observation(output, artifact_base, inline=True)
        observation.update(dim=int(output.shape[-1]), feature_kind="token")
        return self.host.invocation(observation, ms)

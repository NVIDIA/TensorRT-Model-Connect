# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned Transformers DINOv3 features, with the same complete output as the public Task."""

from pathlib import Path
from typing import Any, Mapping


class Adapter:
    def __init__(self, spec: Any, host: Any) -> None:
        from transformers import AutoImageProcessor, AutoModel

        if spec.mode != "eager":
            raise host.Error("DINOv3's native reference supports eager mode")
        self.spec, self.host = spec, host
        self.processor = AutoImageProcessor.from_pretrained(spec.model, **spec.pretrained_kwargs())
        self.model = (
            AutoModel.from_pretrained(spec.model, **spec.model_kwargs()).to(spec.device).eval()
        )

    def invoke(self, request: Mapping[str, Any], artifact_base: Path) -> Any:
        image = self.host.load_image(str(self.host.required(request, "image_path")))
        encoded = self.processor(images=image, return_tensors="pt")
        encoded = {
            name: value.to(self.spec.device, self.spec.dtype)
            if value.is_floating_point()
            else value.to(self.spec.device)
            for name, value in encoded.items()
        }
        output, model_ms = self.host.timed(lambda: self.model(**encoded))
        hidden = output.last_hidden_state.detach().float().cpu().numpy()
        pooled = output.pooler_output.detach().float().cpu().numpy()
        observation = {
            "processed_images": 1,
            "feature_elements": int(hidden.size + pooled.size),
            "last_hidden_state_shape": list(hidden.shape),
            "pooler_output_shape": list(pooled.shape),
            "num_register_tokens": int(getattr(self.model.config, "num_register_tokens", 0)),
        }
        return self.host.invocation(
            self.host.deferred_values(
                observation, {"last_hidden_state": hidden, "pooler_output": pooled}
            ),
            model_ms,
        )

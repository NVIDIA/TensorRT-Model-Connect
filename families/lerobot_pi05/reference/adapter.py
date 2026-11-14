# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""pi0.5's native pipeline for ``control_chunk``: the pinned LeRobot policy and its processors, as
tests/official_reference.py runs them (eager, no torch.compile), on one recorded observation with the request's
flow ``noise``, reported as the TRTMC worker reports the chunk (``actions`` row-major [step, component])."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

from families.lerobot_pi05.tests.official_reference import CHUNK_SIZE, MAX_ACTION_DIM

TOKENIZER = Path(sys.prefix) / "trtmc-reference" / "paligemma-tokenizer"  # reference/prepare.py
DTYPES = {"fp32": "float32", "bf16": "bfloat16"}


class Adapter:
    def __init__(self, spec: Any, host: Any) -> None:
        import torch
        from huggingface_hub import snapshot_download
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi05.configuration_pi05 import PI05Config
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy

        if spec.precision not in DTYPES:
            raise host.Error("the pi0.5 reference runs at fp32 or bf16")
        self.spec, self.host, self.torch = spec, host, torch
        torch.set_grad_enabled(False)
        model = Path(snapshot_download(spec.model, revision=spec.revision))
        config = PI05Config.from_pretrained(str(model))
        config.dtype, config.device, config.compile_model = DTYPES[spec.precision], spec.device, False
        self.policy = PI05Policy.from_pretrained(str(model), config=config).to(spec.device).eval()
        self.pre, self.post = make_pre_post_processors(
            self.policy.config, pretrained_path=str(model),
            preprocessor_overrides={"tokenizer_processor": {"tokenizer_name": str(TOKENIZER)},
                                    "device_processor": {"device": spec.device}})

    def invoke(self, request: Mapping[str, Any], artifact_base: Path) -> Any:
        torch = self.torch
        noise = np.asarray(self.host.required(request, "noise"), dtype=np.float32)
        if noise.size != CHUNK_SIZE * MAX_ACTION_DIM:
            raise self.host.Error("the pi0.5 noise must hold chunk_size x max_action_dim values")

        def image(path: str):
            pixels = np.asarray(self.host.load_image(path).convert("RGB"), dtype=np.float32) / 255.0
            return torch.from_numpy(pixels.transpose(2, 0, 1).copy())

        frame = {camera["name"]: image(camera["image_path"]) for camera in self.host.required(request, "images")}
        frame.update({"observation.state": torch.tensor(self.host.required(request, "state"), dtype=torch.float32),
                      "task": str(request.get("prompt", ""))})
        served = torch.from_numpy(noise.reshape(1, CHUNK_SIZE, MAX_ACTION_DIM)).to(self.spec.device)

        def run():
            chunk = self.policy.predict_action_chunk(self.pre(frame), noise=served)
            return self.post(chunk.float()).cpu().numpy()[0]

        actions, model_ms = self.host.timed(run)
        return self.host.invocation({"action_steps": int(actions.shape[0]), "action_dim": int(actions.shape[1]),
                                     "action_values": int(actions.size), "actions": actions.reshape(-1).tolist(),
                                     "axes": ["step", "action_component"]}, model_ms)

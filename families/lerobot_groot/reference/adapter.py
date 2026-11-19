# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GR00T N1.7's native pipeline for ``control_chunk``: the pinned LeRobot policy and its processors
(tests/official_reference.py) on one recorded observation, its flow noise the request's ``noise``, reported as
the TRTMC worker reports the chunk (``actions`` row-major [step, component], robot units)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from families.lerobot_groot.tests.official_reference import (ACTION_HORIZON, CAMERAS, MAX_ACTION_DIM, _served_noise,
                                                            load_policy)

SOURCE = Path(__file__).resolve().parents[1] / "tests/fixture-source.json"


class Adapter:
    def __init__(self, spec: Any, host: Any) -> None:
        import json

        from huggingface_hub import snapshot_download

        if spec.precision not in ("fp32", "bf16"):
            raise host.Error("the GR00T reference runs at fp32 or bf16 (LeRobot's bfloat16 autocast)")
        self.spec, self.host = spec, host
        source = json.loads(SOURCE.read_text(encoding="utf-8"))
        model = Path(snapshot_download(spec.model, revision=spec.revision))
        base = Path(snapshot_download(source["base_model"]["hf_id"], revision=source["base_model"]["hf_revision"]))
        public, revision = source["tokenizer"]["public_copy"].split("@")
        tokenizer = Path(snapshot_download(public, revision=revision, allow_patterns=["*.json", "*.txt", "*.jinja"]))
        self.policy, self.pre, self.post = load_policy(model, base, tokenizer, precision=spec.precision,
                                                       device=spec.device)

    def invoke(self, request: Mapping[str, Any], artifact_base: Path) -> Any:
        import torch

        cameras = {camera["name"]: camera["image_path"] for camera in self.host.required(request, "images")}
        if set(cameras) != set(CAMERAS):
            raise self.host.Error(f"the GR00T observation needs the cameras {CAMERAS}")
        noise = np.asarray(self.host.required(request, "noise"), dtype=np.float32)
        if noise.size != ACTION_HORIZON * MAX_ACTION_DIM:
            raise self.host.Error("the GR00T noise must hold action_horizon x max_action_dim values")

        def image(path: str):
            pixels = np.asarray(self.host.load_image(path).convert("RGB"), dtype=np.float32) / 255.0
            return torch.from_numpy(pixels.transpose(2, 0, 1).copy())

        frame = {CAMERAS[0]: image(cameras[CAMERAS[0]]), CAMERAS[1]: image(cameras[CAMERAS[1]]),
                 "observation.state": torch.tensor(self.host.required(request, "state"), dtype=torch.float32),
                 "task": str(request.get("prompt", ""))}
        served = torch.from_numpy(noise.reshape(1, ACTION_HORIZON, MAX_ACTION_DIM))

        def run():
            with _served_noise(served) as calls:
                chunk = self.policy.predict_action_chunk(self.pre(frame))
            if len(calls) != 1:
                raise self.host.Error(f"the action head drew its noise {len(calls)} times, not once")
            return self.post(chunk.float()).cpu().numpy()[0]

        actions, model_ms = self.host.timed(run)
        return self.host.invocation({"action_steps": int(actions.shape[0]), "action_dim": int(actions.shape[1]),
                                     "action_values": int(actions.size), "actions": actions.reshape(-1).tolist(),
                                     "axes": ["step", "action_component"]}, model_ms)

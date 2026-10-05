# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ACT's native pipeline for the trtmc-perf-serve reference backend (``control``): the pinned LeRobot source
(reference/prepare.py) through the family's official loader, one action chunk per recorded observation, reported
as the TRTMC worker reports it (``actions`` row-major [step, component])."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from families.lerobot_act.tests.official_reference import load_policy, predict_actions


class Adapter:
    def __init__(self, spec: Any, host: Any) -> None:
        self.host = host
        from huggingface_hub import snapshot_download

        if spec.precision != "fp32":
            raise self.host.Error("the ACT reference runs at fp32")
        self.spec = spec
        checkpoint = Path(snapshot_download(spec.model, revision=spec.revision))
        self.torch, self.policy, self.config, self.device = load_policy(Path(sys.prefix) / "trtmc-reference/lerobot",
                                                                         checkpoint)

    def invoke(self, request: Mapping[str, Any], artifact_base: Path) -> Any:
        # The recorded observation, decoded before the timed call (preloaded), as the official load_observation does.
        pixels = np.asarray(self.host.load_image(str(self.host.required(request, "image_path"))), dtype=np.float32)
        state = np.frombuffer(self.host.load_bytes(str(self.host.required(request, "state_path"))), dtype="<f4")
        if pixels.shape != (480, 640, 3) or state.shape != (14,):
            raise self.host.Error(
                "the ACT observation does not have the qualified shape (480x640 RGB, 14 state values)")
        actions, model_ms = self.host.timed(
            lambda: predict_actions(self.torch, self.policy, self.config, self.device, pixels, state))
        return self.host.invocation({"action_steps": int(actions.shape[0]), "action_dim": int(actions.shape[1]),
                                     "action_values": int(actions.size), "actions": actions.reshape(-1).tolist(),
                                     "axes": ["step", "action_component"]}, model_ms)

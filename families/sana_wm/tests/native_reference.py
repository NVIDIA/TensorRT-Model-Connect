# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SANA-WM's native pipeline for the trtmc-perf-serve reference backend (``world_model``): the pinned official
SANA source and model (native_prepare.py) with its refiner and local stage-1 text encoder, loaded once; per request
the image is resized and center-cropped, the action string becomes the camera trajectory and the request's
intrinsics follow the crop, and the official pipeline renders the video (THWC frames in 0-1)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from families.sana_wm.tests.official_reference import install_local_stage1_text_encoder


class Adapter:
    def __init__(self, spec: Any, host: Any) -> None:
        self.host = host
        import pyrallis

        root = Path(sys.prefix) / "trtmc-reference"
        repository, model = root / "Sana", root / "SANA-model"
        if not (repository / "inference_video_scripts/wm/inference_sana_wm.py").is_file() or not model.is_dir():
            raise self.host.Error(f"the SANA-WM reference is not prepared: {root}")
        sys.path.insert(0, str(repository))
        from inference_video_scripts.wm import inference_sana_wm as official

        self.spec, self.official = spec, official
        config = pyrallis.parse(config_class=official.InferenceConfig, config_path=model / "config.yaml", args=[])
        refiner = official.RefinerSettings(root=model / "refiner", gemma_root=model / "refiner/text_encoder", seed=42)
        install_local_stage1_text_encoder(official, model / "stage1_text_encoder")
        self.pipeline = official.SanaWMPipeline(config=config, model_path=model / "dit/sana_wm_1600m_720p.safetensors",
                                                device=spec.device, refiner=refiner)

    def invoke(self, request: Mapping[str, Any], artifact_base: Path) -> Any:
        import tempfile

        official = self.official
        frames = int(request.get("num_frames", 321))
        trajectory = official.action_string_to_c2w(str(self.host.required(request, "action")),
                                                   translation_speed=float(request.get("translation_speed", 0.055)),
                                                   rotation_speed_deg=float(request.get("rotation_speed_deg", 1.2)))
        if official._snap_num_frames(frames, stride=8, upper_bound=int(trajectory.shape[0])) != frames:
            raise self.host.Error(f"the official pipeline cannot render exactly {frames} frames")
        trajectory = trajectory[:frames]
        cropped, source_size, resized_size, offset = official.resize_and_center_crop(
            self.host.load_image(str(self.host.required(request, "image_path"))).convert("RGB"))
        fx, fy, cx, cy = (float(value) for value in self.host.required(request, "camera_intrinsics"))
        matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix=".npy") as handle:  # the official loader reads a file
            np.save(handle.name, np.repeat(matrix[None], frames, axis=0))
            intrinsics = official.load_intrinsics(Path(handle.name), frames)
        intrinsics = official.transform_intrinsics_for_crop(intrinsics, source_size, resized_size, offset)
        params = official.GenerationParams(num_frames=frames, fps=int(request.get("fps", 16)),
                                           step=int(request.get("num_steps", 60)), cfg_scale=float(request.get("cfg_scale", 5.0)),
                                           flow_shift=float(request.get("flow_shift", 9.8)), seed=int(request.get("seed", 42)))
        prompt = str(self.host.required(request, "prompt"))
        result, model_ms = self.host.timed(
            lambda: self.pipeline.generate(cropped, prompt, trajectory, intrinsics, params))
        if not isinstance(result, dict) or "video" not in result:
            raise self.host.Error("the official SANA-WM pipeline returned no video")
        video = result["video"] if request.get("no_action_overlay", True) else official.apply_overlay(result["video"], result["c2w"])
        video = np.asarray(video)
        # The video is written (output.npy) after the timed call returned.
        return self.host.invocation({**self.host.tensor_observation(video, artifact_base), "media_type": "video",
                                     "num_frames": int(video.shape[0]), "height": int(video.shape[1]),
                                     "width": int(video.shape[2])}, model_ms)

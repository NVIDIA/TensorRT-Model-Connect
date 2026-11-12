# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config resolution and build-request checks that need no GPU and no weights."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from families.stable_video_diffusion import config as config_module
from families.stable_video_diffusion import model as model_module

# The component configs of stabilityai/stable-video-diffusion-img2vid-xt
# @9e43909513c6714f1bc78bcb44d96e733cd242aa, without their local paths.
MODEL_INDEX = {
    "_class_name": "StableVideoDiffusionPipeline",
    "_diffusers_version": "0.24.0.dev0",
    "feature_extractor": ["transformers", "CLIPImageProcessor"],
    "image_encoder": ["transformers", "CLIPVisionModelWithProjection"],
    "scheduler": ["diffusers", "EulerDiscreteScheduler"],
    "unet": ["diffusers", "UNetSpatioTemporalConditionModel"],
    "vae": ["diffusers", "AutoencoderKLTemporalDecoder"],
}
UNET = {
    "_class_name": "UNetSpatioTemporalConditionModel",
    "addition_time_embed_dim": 256,
    "block_out_channels": [320, 640, 1280, 1280],
    "cross_attention_dim": 1024,
    "down_block_types": ["CrossAttnDownBlockSpatioTemporal"] * 3 + ["DownBlockSpatioTemporal"],
    "in_channels": 8,
    "layers_per_block": 2,
    "num_attention_heads": [5, 10, 20, 20],
    "num_frames": 25,
    "out_channels": 4,
    "projection_class_embeddings_input_dim": 768,
    "sample_size": 96,
    "transformer_layers_per_block": 1,
    "up_block_types": ["UpBlockSpatioTemporal"] + ["CrossAttnUpBlockSpatioTemporal"] * 3,
}
VAE = {
    "_class_name": "AutoencoderKLTemporalDecoder",
    "block_out_channels": [128, 256, 512, 512],
    "down_block_types": ["DownEncoderBlock2D"] * 4,
    "force_upcast": True,
    "in_channels": 3,
    "latent_channels": 4,
    "layers_per_block": 2,
    "out_channels": 3,
    "sample_size": 768,
    "scaling_factor": 0.18215,
}
IMAGE_ENCODER = {
    "architectures": ["CLIPVisionModelWithProjection"],
    "hidden_act": "gelu",
    "hidden_size": 1280,
    "image_size": 224,
    "intermediate_size": 5120,
    "layer_norm_eps": 1e-05,
    "model_type": "clip_vision_model",
    "num_attention_heads": 16,
    "num_channels": 3,
    "num_hidden_layers": 32,
    "patch_size": 14,
    "projection_dim": 1024,
}
SCHEDULER = {
    "_class_name": "EulerDiscreteScheduler",
    "beta_end": 0.012,
    "beta_schedule": "scaled_linear",
    "beta_start": 0.00085,
    "interpolation_type": "linear",
    "num_train_timesteps": 1000,
    "prediction_type": "v_prediction",
    "sigma_max": 700.0,
    "sigma_min": 0.002,
    "steps_offset": 1,
    "timestep_spacing": "leading",
    "timestep_type": "continuous",
    "use_karras_sigmas": True,
}
FEATURE_EXTRACTOR = {
    "image_mean": [0.48145466, 0.4578275, 0.40821073],
    "image_std": [0.26862954, 0.26130258, 0.27577711],
    "image_processor_type": "CLIPImageProcessor",
}


def _checkpoint(tmp_path, **overrides):
    files = {
        "model_index.json": MODEL_INDEX,
        "unet/config.json": UNET,
        "vae/config.json": VAE,
        "image_encoder/config.json": IMAGE_ENCODER,
        "scheduler/scheduler_config.json": SCHEDULER,
        "feature_extractor/preprocessor_config.json": FEATURE_EXTRACTOR,
    }
    for name, payload in files.items():
        payload = {**payload, **overrides.get(name, {})}
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
    return tmp_path


def test_the_published_configs_resolve(tmp_path):
    cfg = config_module.resolve(_checkpoint(tmp_path))
    assert cfg["unet"]["added_time_ids"] == 3
    assert cfg["unet"]["num_attention_heads"] == [5, 10, 20, 20]
    assert cfg["unet"]["num_frames"] == 25
    assert cfg["vae"]["force_upcast"] is True
    assert cfg["scheduler"] == {"sigma_min": 0.002, "sigma_max": 700.0,
                                "init_noise_sigma_includes_unit": True}
    assert cfg["image_encoder"]["projection_dim"] == cfg["unet"]["cross_attention_dim"]


def test_runtime_config_carries_the_pipeline_defaults(tmp_path):
    cfg = config_module.resolve(_checkpoint(tmp_path))
    runtime = model_module._runtime_config(cfg, height=576, width=1024, frames=25, precision="fp16")
    assert (runtime["latent_height"], runtime["latent_width"], runtime["num_frames"]) == (72, 128, 25)
    assert runtime["defaults"] == {"num_steps": 25, "min_guidance_scale": 1.0,
                                   "max_guidance_scale": 3.0, "fps": 7, "motion_bucket_id": 127,
                                   "noise_aug_strength": 0.02, "seed": 0}
    assert runtime["clip_image_size"] == 224
    assert runtime["scaling_factor"] == pytest.approx(0.18215)


@pytest.mark.parametrize(("name", "change", "message"), [
    ("scheduler/scheduler_config.json", {"prediction_type": "epsilon"}, "predict v"),
    ("scheduler/scheduler_config.json", {"use_karras_sigmas": False}, "Karras"),
    ("scheduler/scheduler_config.json", {"sigma_min": None}, "Karras range"),
    ("unet/config.json", {"transformer_layers_per_block": 2}, "one transformer layer"),
    ("image_encoder/config.json", {"hidden_act": "quick_gelu"}, "exact GELU"),
    ("image_encoder/config.json", {"image_size": 336}, "224"),
    ("model_index.json", {"_class_name": "StableDiffusionPipeline"}, "model_index.json"),
])
def test_unbuilt_variants_are_refused(tmp_path, name, change, message):
    with pytest.raises(NotImplementedError, match=message):
        config_module.resolve(_checkpoint(tmp_path, **{name: change}))


class _Writer:
    def __init__(self):
        self.sections = {}

    def set_header(self, **header):
        self.header = header

    def add_bytes(self, name, data):
        self.sections[name] = data

    def add_json(self, name, payload):
        self.sections[name] = payload


def _request(model_dir, **changes):
    request = {
        "model_dir": model_dir, "task": "initial_image_text_to_video", "precision": "fp16",
        "backend": "trt", "max_sequence_length": None, "image_height": None, "image_width": None,
        "video_num_frames": None, "max_batch_size": 1, "tensor_parallel_size": 1,
        "context_parallel_size": 1, "quantization": None, "fp32_layers": (),
        "dynamic_kv_cache": False, "verbose": False, "mtp_seq_len": None,
    }
    request.update(changes)
    return SimpleNamespace(**request)


@pytest.mark.parametrize(("change", "error"), [
    ({"task": "text_to_video"}, ValueError),
    ({"backend": "trt_rtx"}, ValueError),
    ({"precision": "bf16"}, ValueError),
    ({"max_sequence_length": 77}, NotImplementedError),
    ({"max_batch_size": 2}, NotImplementedError),
    ({"tensor_parallel_size": 2}, NotImplementedError),
    ({"context_parallel_size": 2}, NotImplementedError),
    ({"quantization": "fp8"}, NotImplementedError),
    ({"fp32_layers": (1,)}, NotImplementedError),
    ({"dynamic_kv_cache": True}, NotImplementedError),
    ({"mtp_seq_len": 2}, NotImplementedError),
    ({"image_height": 600}, ValueError),
    ({"image_width": 1000}, ValueError),
])
def test_unsupported_requests_fail_before_weights_are_read(tmp_path, change, error):
    writer = _Writer()
    with pytest.raises(error):
        model_module.build(_request(_checkpoint(tmp_path), **change), writer)
    assert writer.sections == {}

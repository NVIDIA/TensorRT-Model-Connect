# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from families.cosmos3 import model
from families.cosmos3.vae_step_builder import Cosmos3VaeStepProfile, require_vae_build_profile
from tensorrt_model_connect import BuildRequest


SHARDS = tuple(f"diffusion_pytorch_model-{index:05d}-of-00007.safetensors" for index in range(1, 8))


class RecordingWriter:
    def __init__(self) -> None:
        self.header = None
        self.sections: dict[str, bytes] = {}
        self.json_sections: dict[str, object] = {}

    def set_header(self, **header) -> None:
        self.header = header

    def add_bytes(self, name: str, value: bytes) -> None:
        assert name not in self.sections
        self.sections[name] = value

    def add_json(self, name: str, value: object) -> None:
        assert name not in self.json_sections
        self.json_sections[name] = value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _checkpoint(root: Path) -> Path:
    transformer = root / "transformer"
    vae = root / "vae"
    tokenizer = root / "text_tokenizer"
    transformer.mkdir(parents=True)
    vae.mkdir()
    tokenizer.mkdir()
    _write_json(root / "model_index.json", {"_class_name": "Cosmos3OmniDiffusersPipeline"})
    _write_json(
        transformer / "config.json",
        {
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "num_hidden_layers": 36,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": 151936,
            "latent_channel": 48,
            "latent_patch_size": 2,
            "patch_latent_dim": 192,
            "rms_norm_eps": 1.0e-6,
            "rope_theta": 5_000_000.0,
            "timestep_scale": 0.001,
            "hidden_act": "silu",
            "attention_bias": False,
            "qk_norm_for_text": True,
            "qk_norm_for_diffusion": True,
            "rope_scaling": {"mrope_section": [24, 20, 20]},
        },
    )
    _write_json(
        transformer / "diffusion_pytorch_model.safetensors.index.json",
        {"weight_map": {f"tensor.{index}": name for index, name in enumerate(SHARDS)}},
    )
    for name in SHARDS:
        (transformer / name).write_bytes(b"shard")
    _write_json(
        vae / "config.json",
        {
            "z_dim": 48,
            "scale_factor_spatial": 16,
            "scale_factor_temporal": 4,
            "patch_size": 2,
            "latents_mean": [0.0] * 48,
            "latents_std": [1.0] * 48,
        },
    )
    (vae / "diffusion_pytorch_model.safetensors").write_bytes(b"vae")
    (tokenizer / "tokenizer.json").write_bytes(b"tokenizer")
    (tokenizer / "tokenizer_config.json").write_bytes(b"tokenizer-config")
    return root


def _request(model_dir: Path, **changes) -> BuildRequest:
    request = BuildRequest(
        model_dir=model_dir,
        output_path=model_dir.parent / "cosmos3.bundle",
        family="cosmos3",
        task="image_generation",
        precision="bf16",
        max_sequence_length=4096,
        image_height=720,
        image_width=1280,
        video_num_frames=189,
        context_parallel_size=2,
    )
    return replace(request, **changes)


def test_build_emits_only_the_exact_family_bundle_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_dir = _checkpoint(tmp_path / "checkpoint")
    calls = []
    monkeypatch.setattr(
        model,
        "_build_denoiser",
        lambda path, **options: calls.append(("denoiser", path, options)) or b"denoiser",
    )
    monkeypatch.setattr(
        model,
        "_load_vae_weights",
        lambda path: calls.append(("load-vae", path)) or {"weight": object()},
    )
    monkeypatch.setattr(
        model,
        "_build_vae",
        lambda weights, **options: (
            calls.append(("vae", weights, options))
            or (b"first" if options["first_frame_only"] else b"recurrent")
        ),
    )
    writer = RecordingWriter()

    model.build(_request(model_dir), writer)

    assert writer.header == {
        "family": "cosmos3",
        "task": "image_generation",
        "backend": "trt",
    }
    assert writer.sections == {
        "denoiser.plan": b"denoiser",
        "vae.plan": b"recurrent",
        "vae.first_frame.plan": b"first",
        "tokenizer.json": b"tokenizer",
        "tokenizer_config.json": b"tokenizer-config",
    }
    assert writer.json_sections == {
        "runtime.json": {
            "negative_prompt": "blurry, distorted, low quality, jittery, deformed",
            "num_inference_steps": 35,
            "guidance_scale": 6.0,
            "flow_shift": 10.0,
            "seed": 42,
            "video_height": 720,
            "video_width": 1280,
            "video_num_frames": 189,
            "frame_rate": 24,
            "text_seq_len": 4096,
            "context_parallel_size": 2,
        }
    }
    assert calls[0] == (
        "denoiser",
        model_dir / "transformer",
        {"context_parallel_size": 2, "verbose": False},
    )
    assert calls[1] == ("load-vae", model_dir / "vae")
    assert calls[2][0:2] == ("vae", calls[2][1])
    assert calls[2][2] == {"first_frame_only": False, "verbose": False}
    assert calls[3][0:2] == ("vae", calls[3][1])
    assert calls[3][2] == {"first_frame_only": True, "verbose": False}
    assert calls[2][1] is calls[3][1]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"task": "text_generation"}, "task=image_generation"),
        ({"precision": "fp16"}, "precision=bf16"),
        ({"max_batch_size": 2}, "max_batch_size=1"),
        ({"tensor_parallel_size": 2}, "does not use tensor parallelism"),
        ({"context_parallel_size": 3}, "must be 1 or 2"),
        ({"quantization": "fp8"}, "does not support quantization"),
        ({"fp32_layers": (0,)}, "does not support fp32_layers"),
        ({"max_sequence_length": 2048}, "max_sequence_length=4096"),
        ({"image_height": 704}, "fixed full-quality"),
        ({"image_width": 1216}, "fixed full-quality"),
        ({"video_num_frames": 185}, "fixed full-quality"),
    ],
)
def test_build_rejects_every_unsupported_profile(
    tmp_path: Path, changes: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        model.build(_request(tmp_path / "missing", **changes), RecordingWriter())


def test_checkpoint_has_no_alternate_tokenizer_or_weight_names(tmp_path: Path) -> None:
    model_dir = _checkpoint(tmp_path / "checkpoint")
    tokenizer = model_dir / "text_tokenizer"
    tokenizer.rename(model_dir / "tokenizer")
    with pytest.raises(FileNotFoundError, match="text_tokenizer"):
        model._require_checkpoint(model_dir)

    model_dir = _checkpoint(tmp_path / "second-checkpoint")
    exact_index = model_dir / "transformer/diffusion_pytorch_model.safetensors.index.json"
    exact_index.unlink()
    (model_dir / "transformer/model.safetensors.index.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="diffusion_pytorch_model"):
        model._require_checkpoint(model_dir)


def test_vae_builder_has_one_exact_bf16_profile() -> None:
    profile = Cosmos3VaeStepProfile(45, 80)
    assert require_vae_build_profile(profile, (8, 0)) is None
    with pytest.raises(ValueError, match="45x80"):
        require_vae_build_profile(Cosmos3VaeStepProfile(44, 80), (8, 0))
    with pytest.raises(RuntimeError, match="8.0 or newer"):
        require_vae_build_profile(profile, (7, 5))

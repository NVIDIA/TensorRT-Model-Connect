# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only checks of support resolution, the checkpoint contract and request validation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file
from tensorrt_model_connect import BuildRequest
from tensorrt_model_connect.model_support import ModelMetadata, resolve_family

from .. import config as config_module
from ..checkpoint import load_checkpoint, require
from ..model import TASK, _normalization_json, _validate, prefix_length
from ..support import describe
from ..tokenizer import TOKENIZER_BYTES, _verified, load_tokenizer_model

NORMALIZER = "policy_preprocessor_step_2_normalizer_processor.safetensors"
UNNORMALIZER = "policy_postprocessor_step_0_unnormalizer_processor.safetensors"


def _config() -> dict:
    return {
        "type": "pi05",
        "n_obs_steps": 1,
        "empty_cameras": 1,
        "paligemma_variant": "gemma_2b",
        "action_expert_variant": "gemma_300m",
        "chunk_size": 50,
        "n_action_steps": 10,
        "max_action_dim": 32,
        "max_state_dim": 32,
        "num_inference_steps": 10,
        "min_period": 0.004,
        "max_period": 4.0,
        "image_resolution": [224, 224],
        "tokenizer_max_length": 200,
        "input_features": {
            "observation.images.image": {"type": "VISUAL", "shape": [3, 256, 256]},
            "observation.images.image2": {"type": "VISUAL", "shape": [3, 256, 256]},
            "observation.state": {"type": "STATE", "shape": [8]},
        },
        "output_features": {"action": {"type": "ACTION", "shape": [7]}},
    }


def _processors(*, stats: bool, mode: str = "MEAN_STD", padding_side: str = "right") -> tuple[dict, dict]:
    norm_map = {"VISUAL": "IDENTITY", "STATE": mode, "ACTION": mode}
    pre_norm = {"registry_name": "normalizer_processor", "config": {"eps": 1e-8, "norm_map": norm_map}}
    post_norm = {"registry_name": "unnormalizer_processor", "config": {"eps": 1e-8, "norm_map": norm_map}}
    if stats:
        pre_norm["state_file"] = NORMALIZER
        post_norm["state_file"] = UNNORMALIZER
    tokenizer = {
        "registry_name": "tokenizer_processor",
        "config": {"max_length": 200, "padding_side": padding_side, "padding": "max_length",
                   "tokenizer_name": "google/paligemma-3b-pt-224"},
    }
    return {"steps": [pre_norm, tokenizer]}, {"steps": [post_norm]}


def _model_dir(root: Path, *, stats: bool = False, config: dict | None = None, **processor_args) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(config or _config()), encoding="utf-8")
    pre, post = _processors(stats=stats, **processor_args)
    (root / "policy_preprocessor.json").write_text(json.dumps(pre), encoding="utf-8")
    (root / "policy_postprocessor.json").write_text(json.dumps(post), encoding="utf-8")
    if stats:
        save_file({"observation.state.mean": np.arange(8, dtype=np.float32),
                   "observation.state.std": np.full(8, 2.0, np.float32)}, str(root / NORMALIZER))
        save_file({"action.mean": np.arange(7, dtype=np.float32),
                   "action.std": np.full(7, 3.0, np.float32)}, str(root / UNNORMALIZER))
    return root


def test_support_claims_only_pi05_policies() -> None:
    support = describe(ModelMetadata(config={"type": "pi05"}, model_index={}))
    assert support is not None
    assert support.tasks == (TASK,) and support.default_task == TASK
    assert support.default_precision == "bf16"
    family, resolved = resolve_family(ModelMetadata(config={"type": "pi05"}, model_index={}))
    assert family == "lerobot_pi05" and resolved == support
    for other in ({"type": "pi0"}, {"type": "pi0_fast"}, {"type": "act"}, {}):
        assert describe(ModelMetadata(config=other, model_index={})) is None


def test_config_reads_the_checkpoint_contract(tmp_path) -> None:
    cfg = config_module.load_config(_model_dir(tmp_path / "ckpt"))
    assert cfg.cameras == ("observation.images.image", "observation.images.image2")
    assert (cfg.empty_cameras, cfg.camera_slots) == (1, 3)
    assert (cfg.camera_height, cfg.camera_width) == (256, 256)
    assert (cfg.state_dim, cfg.action_dim, cfg.chunk_size) == (8, 7, 50)
    assert cfg.tokenizer_name == "google/paligemma-3b-pt-224"
    # No statistics in the directory: LeRobot passes the value through.
    assert cfg.state_normalization.is_identity and cfg.action_normalization.is_identity
    assert prefix_length(cfg) == 3 * 256 + 200


def test_config_ignores_a_placeholder_camera_listed_as_a_feature(tmp_path) -> None:
    raw = _config()
    raw["input_features"]["observation.images.empty_camera_0"] = {"type": "VISUAL", "shape": [3, 224, 224]}
    cfg = config_module.load_config(_model_dir(tmp_path / "ckpt", config=raw))
    assert len(cfg.cameras) == 2 and cfg.camera_slots == 3


def test_config_loads_statistics_when_the_checkpoint_has_them(tmp_path) -> None:
    cfg = config_module.load_config(_model_dir(tmp_path / "ckpt", stats=True))
    assert not cfg.state_normalization.is_identity
    assert cfg.state_normalization.mean == tuple(float(v) for v in range(8))
    assert cfg.action_normalization.std == (3.0,) * 7
    assert _normalization_json(cfg.action_normalization)["mode"] == "MEAN_STD"
    assert _normalization_json(config_module.Normalization("MEAN_STD", 1e-8)) == {"mode": "IDENTITY"}


def test_config_rejects_unsupported_normalization_only_when_statistics_exist(tmp_path) -> None:
    with pytest.raises(NotImplementedError, match="QUANTILES"):
        config_module.load_config(_model_dir(tmp_path / "with-stats", stats=True, mode="QUANTILES"))
    # Without statistics the mode is moot, as in LeRobot.
    cfg = config_module.load_config(_model_dir(tmp_path / "no-stats", mode="QUANTILES"))
    assert cfg.state_normalization.is_identity


def test_config_requires_the_statistics_it_names(tmp_path) -> None:
    root = _model_dir(tmp_path / "ckpt", stats=True)
    (root / NORMALIZER).unlink()
    with pytest.raises(FileNotFoundError, match="normalizer_processor"):
        config_module.load_config(root)
    save_file({"observation.state.mean": np.zeros(8, np.float32)}, str(root / NORMALIZER))
    with pytest.raises(KeyError, match="std"):
        config_module.load_config(root)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("n_obs_steps", 2),
        ("paligemma_variant", "gemma_300m"),
        ("action_expert_variant", "gemma_2b"),
        ("image_resolution", [256, 256]),
        ("type", "pi0"),
        ("use_relative_actions", True),
        ("rtc_config", {"enabled": True}),
    ],
)
def test_config_rejects_semantic_drift(tmp_path, field: str, value: object) -> None:
    raw = _config()
    raw[field] = value
    with pytest.raises(ValueError, match=field):
        config_module.load_config(_model_dir(tmp_path / "ckpt", config=raw))


def test_config_rejects_unsupported_camera_and_tokenizer_layouts(tmp_path) -> None:
    raw = _config()
    raw["input_features"]["observation.images.image2"]["shape"] = [3, 224, 224]
    with pytest.raises(NotImplementedError, match="share one image shape"):
        config_module.load_config(_model_dir(tmp_path / "mixed", config=raw))
    raw = _config()
    for spec in raw["input_features"].values():
        if spec["type"] == "VISUAL":
            spec["shape"] = [3, 224, 256]
    with pytest.raises(NotImplementedError, match="square"):
        config_module.load_config(_model_dir(tmp_path / "wide", config=raw))
    with pytest.raises(NotImplementedError, match="right-padded"):
        config_module.load_config(_model_dir(tmp_path / "left", padding_side="left"))


def _request(**overrides) -> BuildRequest:
    values = {"model_dir": Path("."), "output_path": Path("out.bundle"), "family": "lerobot_pi05",
              "task": TASK, "precision": "bf16"}
    values.update(overrides)
    return BuildRequest(**values)


def test_request_validation_accepts_the_supported_shape(tmp_path) -> None:
    cfg = config_module.load_config(_model_dir(tmp_path / "ckpt"))
    assert _validate(_request(), cfg) == "bf16"
    assert _validate(_request(precision="FP32", max_sequence_length=200, image_height=256), cfg) == "fp32"


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"task": "robot_control"}, ValueError),
        ({"precision": "fp16"}, ValueError),
        ({"max_batch_size": 2}, NotImplementedError),
        ({"tensor_parallel_size": 2}, NotImplementedError),
        ({"dynamic_kv_cache": True}, NotImplementedError),
        ({"quantization": "fp8"}, NotImplementedError),
        ({"max_sequence_length": 100}, NotImplementedError),
        ({"image_height": 224}, NotImplementedError),
        ({"image_width": 224}, NotImplementedError),
    ],
)
def test_request_validation_rejects_unsupported_requests(tmp_path, overrides: dict, error: type) -> None:
    cfg = config_module.load_config(_model_dir(tmp_path / "ckpt"))
    with pytest.raises(error):
        _validate(_request(**overrides), cfg)


def test_checkpoint_loading_and_shape_checks(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="model.safetensors"):
        load_checkpoint(tmp_path)
    save_file(
        {"paligemma_with_expert.paligemma.lm_head.weight": np.zeros((4, 2), np.float32),
         "paligemma_with_expert.gemma_expert.lm_head.weight": np.zeros((4, 2), np.float32),
         "paligemma_with_expert.time_mlp_in.bias": np.zeros(3, np.float32)},
        str(tmp_path / "model.safetensors"),
    )
    weights = load_checkpoint(tmp_path)
    # The shared prefix is dropped and the training-only expert head is not loaded.
    assert set(weights) == {"paligemma.lm_head.weight", "time_mlp_in.bias"}
    assert weights["time_mlp_in.bias"].dtype == np.float32
    assert require(weights, "time_mlp_in.bias", (3,)).shape == (3,)
    with pytest.raises(ValueError, match="expected"):
        require(weights, "time_mlp_in.bias", (4,))
    with pytest.raises(KeyError, match="missing tensor"):
        require(weights, "absent", (1,))
    save_file({"time_mlp_in.bias": np.zeros(3, np.float32)}, str(tmp_path / "model.safetensors"))
    with pytest.raises(KeyError, match="lm_head"):
        load_checkpoint(tmp_path)


def test_tokenizer_asset_is_pinned_by_content(tmp_path) -> None:
    with pytest.raises(ValueError, match="does not match"):
        _verified(b"\x00" * TOKENIZER_BYTES, "test")
    (tmp_path / "tokenizer.model").write_bytes(b"not the paligemma tokenizer")
    with pytest.raises(ValueError, match="does not match"):
        load_tokenizer_model(tmp_path)

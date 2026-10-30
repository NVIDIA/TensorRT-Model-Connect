# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only checks of the checkpoint contract, request validation and the graph's constant tables."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file
from tensorrt_model_connect import BuildRequest

from .. import action_head, backbone, graph, vision
from .. import config as config_module
from .. import model as model_module
from ..checkpoint import canonical_name, category, embedding_key, load_checkpoint, require
from ..model import TASK, build, crop_size, sequence_bounds
from ..tokenizer import (
    PUBLIC_COPY,
    PUBLIC_REVISION,
    TEMPLATE_TOKENS,
    TOKENIZER_BYTES,
    TOKENIZER_FILE,
    TOKENIZER_REPO,
    TOKENIZER_REVISION,
    _checked,
    load_tokenizer_json,
)

PACK_STATS = "policy_preprocessor_step_2_groot_n1_7_pack_inputs_v1.safetensors"
UNPACK_STATS = "policy_postprocessor_step_0_groot_action_unpack_unnormalize_v2.safetensors"
CAMERAS = ("observation.images.image", "observation.images.wrist_image")


def _config(**overrides) -> dict:
    raw = {
        "type": "groot",
        "n_obs_steps": 1,
        "input_features": {
            "observation.images.wrist_image": {"type": "VISUAL", "shape": [256, 256, 3]},
            "observation.images.image": {"type": "VISUAL", "shape": [256, 256, 3]},
            "observation.state": {"type": "STATE", "shape": [8]},
        },
        "output_features": {"action": {"type": "ACTION", "shape": [7]}},
        "chunk_size": 16,
        "n_action_steps": 16,
        "max_state_dim": 132,
        "max_action_dim": 132,
        "action_decode_transform": "libero",
        "embodiment_tag": "libero_sim",
        "num_inference_timesteps": 4,
        "use_relative_actions": False,
        "use_peft": False,
    }
    raw.update(overrides)
    return raw


def _pack(**overrides) -> dict:
    config = {
        "state_horizon": 1, "action_horizon": 40, "valid_action_horizon": 40, "video_horizon": None,
        "max_state_dim": 132, "max_action_dim": 132, "language_key": "task", "formalize_language": True,
        "embodiment_tag": "libero_sim", "embodiment_mapping": {"libero_sim": 2, "droid_sim": 3},
        "normalize_min_max": True, "clip_outliers": True, "use_percentiles": True,
        "video_modality_keys": None, "raw_stats": {}, "modality_config": {},
    }
    config.update(overrides)
    return {"registry_name": "groot_n1_7_pack_inputs_v1", "config": config, "state_file": PACK_STATS}


def _vlm(**overrides) -> dict:
    config = {"model_name": "nvidia/Cosmos-Reason2-2B", "image_crop_size": [230, 230],
              "image_target_size": [256, 256], "shortest_image_edge": 256, "crop_fraction": 0.95,
              "use_albumentations": True, "letter_box_transform": False, "device": "cuda"}
    config.update(overrides)
    return {"registry_name": "groot_n1_7_vlm_encode_v1", "config": config}


def _unpack(**overrides) -> dict:
    config = {"env_action_dim": 7, "normalize_min_max": True, "clip_normalized_action": True,
              "libero_gripper_action": True, "libero_gripper_binarize": True}
    config.update(overrides)
    return {"registry_name": "groot_action_unpack_unnormalize_v2", "config": config, "state_file": UNPACK_STATS}


def _model_dir(root: Path, *, config: dict | None = None, pre: list | None = None, post: list | None = None,
               stats: bool = True) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(config or _config()), encoding="utf-8")
    pre = pre or [{"registry_name": "rename_observations_processor", "config": {"rename_map": {}}},
                  {"registry_name": "to_batch_processor", "config": {}}, _pack(), _vlm(),
                  {"registry_name": "device_processor", "config": {"device": "cuda"}}]
    post = post or [_unpack(), {"registry_name": "device_processor", "config": {"device": "cpu"}}]
    (root / "policy_preprocessor.json").write_text(json.dumps({"steps": pre}), encoding="utf-8")
    (root / "policy_postprocessor.json").write_text(json.dumps({"steps": post}), encoding="utf-8")
    if stats:
        # LeRobot stores these as float64; the state has an empty range on dimension 2.
        state_min = np.arange(8, dtype=np.float64)
        state_max = state_min + 2.0
        state_max[2] = state_min[2]
        save_file({"observation.state.min": state_min, "observation.state.max": state_max,
                   "action.min": np.full(7, -0.5), "action.max": np.full(7, 1.5)}, str(root / PACK_STATS))
        save_file({"action.min": np.full(7, -0.5), "action.max": np.full(7, 1.5)}, str(root / UNPACK_STATS))
    return root


def test_config_reads_the_checkpoint_contract(tmp_path) -> None:
    cfg = config_module.load_config(_model_dir(tmp_path / "ckpt"))
    # The pack step feeds the cameras in sorted key order when the checkpoint names none.
    assert cfg.cameras == CAMERAS
    assert (cfg.camera_height, cfg.camera_width, cfg.state_dim, cfg.action_dim) == (256, 256, 8, 7)
    assert (cfg.chunk_size, cfg.action_horizon, cfg.num_inference_steps) == (16, 40, 4)
    assert (cfg.embodiment_tag, cfg.embodiment_id) == ("libero_sim", 2)
    assert cfg.formalize_language and cfg.clip_state and cfg.clip_action and cfg.libero_gripper
    assert cfg.state_min[2] == cfg.state_max[2] == 2.0
    assert cfg.action_min == (-0.5,) * 7 and cfg.action_max == (1.5,) * 7
    assert cfg.image == config_module.ImageTransform(shortest_edge=256, crop_fraction=0.95,
                                                     target_height=256, target_width=256)
    assert crop_size(cfg) == 243  # int(256 * 0.95), as LeRobot's crop computes it
    # <|im_start|> user \n, two cameras of 64 image tokens between vision markers, <|im_end|> \n.
    assert sequence_bounds(cfg) == (137, 161, 393)


def test_camera_order_follows_the_checkpoint_video_keys(tmp_path) -> None:
    pre = [{"registry_name": "rename_observations_processor", "config": {"rename_map": {}}},
           {"registry_name": "to_batch_processor", "config": {}},
           _pack(video_modality_keys=["wrist_image", "image"]), _vlm(),
           {"registry_name": "device_processor", "config": {}}]
    cfg = config_module.load_config(_model_dir(tmp_path / "ckpt", pre=pre))
    assert cfg.cameras == tuple(reversed(CAMERAS))
    pre[2] = _pack(video_modality_keys=["image"])
    with pytest.raises(NotImplementedError, match="video_modality_keys"):
        config_module.load_config(_model_dir(tmp_path / "missing", pre=pre))


def test_chunk_is_the_shortest_requested_horizon(tmp_path) -> None:
    cfg = config_module.load_config(_model_dir(tmp_path / "a", config=_config(chunk_size=50, n_action_steps=8)))
    assert cfg.chunk_size == 8
    cfg = config_module.load_config(_model_dir(tmp_path / "b", config=_config(chunk_size=60, n_action_steps=60)))
    assert cfg.chunk_size == 40  # the model's native horizon caps it


def test_gripper_transform_follows_action_decode_transform(tmp_path) -> None:
    # LeRobot overwrites the saved libero_gripper_action flag from the policy config.
    cfg = config_module.load_config(_model_dir(tmp_path / "none", config=_config(action_decode_transform=None)))
    assert not cfg.libero_gripper
    post = [_unpack(libero_gripper_action=False), {"registry_name": "device_processor", "config": {}}]
    cfg = config_module.load_config(_model_dir(tmp_path / "libero", post=post))
    assert cfg.libero_gripper


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("type", "pi05"),
        ("n_obs_steps", 2),
        ("max_state_dim", 64),
        ("max_action_dim", 32),
        ("use_relative_actions", True),
        ("use_peft", True),
        ("action_decode_transform", "droid"),
    ],
)
def test_config_rejects_semantic_drift(tmp_path, field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        config_module.load_config(_model_dir(tmp_path / "ckpt", config=_config(**{field: value})))


@pytest.mark.parametrize(
    ("pack", "message"),
    [
        ({"action_horizon": 16}, "action_horizon"),
        ({"video_horizon": 2}, "video_horizon"),
        ({"language_key": "instruction"}, "language_key"),
        ({"normalize_min_max": False}, "normalize_min_max"),
        ({"raw_stats": {"state": {}}}, "raw_stats"),
        ({"modality_config": {"action": {}}}, "modality_config"),
        ({"embodiment_tag": "new_embodiment"}, "embodiment_mapping"),
    ],
)
def test_config_refuses_pack_settings_it_does_not_reproduce(tmp_path, pack: dict, message: str) -> None:
    pre = [{"registry_name": "rename_observations_processor", "config": {"rename_map": {}}},
           {"registry_name": "to_batch_processor", "config": {}}, _pack(**pack), _vlm(),
           {"registry_name": "device_processor", "config": {}}]
    with pytest.raises(ValueError, match=message):
        config_module.load_config(_model_dir(tmp_path / "ckpt", pre=pre))


@pytest.mark.parametrize(
    ("vlm", "message"),
    [
        ({"model_name": "Qwen/Qwen3-VL-8B-Instruct"}, "backbone"),
        ({"use_albumentations": False}, "use_albumentations"),
        ({"letter_box_transform": True}, "letter boxing"),
        ({"image_target_size": [224, 224]}, "image_target_size"),
    ],
)
def test_config_refuses_image_processing_it_does_not_reproduce(tmp_path, vlm: dict, message: str) -> None:
    pre = [{"registry_name": "rename_observations_processor", "config": {"rename_map": {}}},
           {"registry_name": "to_batch_processor", "config": {}}, _pack(), _vlm(**vlm),
           {"registry_name": "device_processor", "config": {}}]
    with pytest.raises(NotImplementedError, match=message):
        config_module.load_config(_model_dir(tmp_path / "ckpt", pre=pre))


def test_config_refuses_cameras_off_the_image_token_grid(tmp_path) -> None:
    # 240 pixels are 15 patches a side, which the 2x2 patch merger cannot pair up.
    assert config_module.MERGED_PATCH == vision.PATCH * vision.MERGE
    cameras = {camera: {"type": "VISUAL", "shape": [240, 240, 3]} for camera in CAMERAS}
    config = _config(input_features={**_config()["input_features"], **cameras})
    pre = [{"registry_name": "rename_observations_processor", "config": {"rename_map": {}}},
           {"registry_name": "to_batch_processor", "config": {}}, _pack(),
           _vlm(image_target_size=[240, 240], shortest_image_edge=240),
           {"registry_name": "device_processor", "config": {}}]
    with pytest.raises(NotImplementedError, match="multiple of 32"):
        config_module.load_config(_model_dir(tmp_path / "ckpt", config=config, pre=pre))


def test_config_refuses_other_processor_pipelines(tmp_path) -> None:
    pre = [{"registry_name": "rename_observations_processor", "config": {"rename_map": {}}},
           {"registry_name": "to_batch_processor", "config": {}},
           {"registry_name": "relative_actions_processor", "config": {}}, _pack(), _vlm(),
           {"registry_name": "device_processor", "config": {}}]
    with pytest.raises(NotImplementedError, match="relative_actions_processor"):
        config_module.load_config(_model_dir(tmp_path / "extra", pre=pre))
    post = [{"registry_name": "groot_n1_7_action_decode_v1", "config": {}},
            {"registry_name": "device_processor", "config": {}}]
    with pytest.raises(NotImplementedError, match="groot_n1_7_action_decode_v1"):
        config_module.load_config(_model_dir(tmp_path / "decode", post=post))
    pre = [{"registry_name": "rename_observations_processor", "config": {"rename_map": {"a": "b"}}},
           {"registry_name": "to_batch_processor", "config": {}}, _pack(), _vlm(),
           {"registry_name": "device_processor", "config": {}}]
    with pytest.raises(NotImplementedError, match="rename_map"):
        config_module.load_config(_model_dir(tmp_path / "rename", pre=pre))
    post = [_unpack(env_action_dim=6), {"registry_name": "device_processor", "config": {}}]
    with pytest.raises(NotImplementedError, match="action size"):
        config_module.load_config(_model_dir(tmp_path / "unpack", post=post))


def test_config_requires_its_statistics_and_lerobot_keys(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match=PACK_STATS):
        config_module.load_config(_model_dir(tmp_path / "missing", stats=False))
    root = _model_dir(tmp_path / "partial")
    save_file({"observation.state.mean": np.zeros(8)}, str(root / PACK_STATS))
    with pytest.raises(ValueError, match="min/max"):
        config_module.load_config(root)
    raw = _config()
    raw["input_features"]["observation.joint_state"] = raw["input_features"].pop("observation.state")
    with pytest.raises(ValueError, match="observation.state"):
        config_module.load_config(_model_dir(tmp_path / "keys", config=raw))
    raw = _config()
    raw["input_features"]["observation.images.image"]["shape"] = [224, 224, 3]
    with pytest.raises(NotImplementedError, match="share one image shape"):
        config_module.load_config(_model_dir(tmp_path / "mixed", config=raw))


def _request(model_dir: Path, **overrides) -> BuildRequest:
    values = {"model_dir": model_dir, "output_path": Path("out.bundle"), "family": "lerobot_groot",
              "task": TASK, "precision": "bf16"}
    values.update(overrides)
    return BuildRequest(**values)


class _Writer:
    def __init__(self) -> None:
        self.header: dict = {}
        self.sections: dict = {}

    def set_header(self, **fields) -> None:
        self.header = fields

    def add_bytes(self, name: str, data: bytes) -> None:
        self.sections[name] = data

    def add_json(self, name: str, value: dict) -> None:
        self.sections[name] = value


@pytest.fixture
def stubbed_build(monkeypatch):
    """Replace only the heavy steps (weights, tokenizer, engine builds) of ``build``."""
    import families.lerobot_groot.checkpoint as checkpoint_module

    calls: list[str] = []
    monkeypatch.setattr(checkpoint_module, "load_checkpoint", lambda _: calls.append("weights") or {})
    monkeypatch.setattr(model_module, "load_tokenizer_json", lambda _: calls.append("tokenizer") or b"{}")
    monkeypatch.setattr(model_module, "_build_condition_plan", lambda *a, **k: calls.append("condition") or b"C")
    monkeypatch.setattr(model_module, "_build_denoise_plan", lambda *a, **k: calls.append("denoise") or b"D")
    return calls


def test_build_writes_the_bundle_contract(tmp_path, stubbed_build) -> None:
    writer = _Writer()
    build(_request(_model_dir(tmp_path / "ckpt"), precision="FP32", image_height=256), writer)
    assert writer.header == {"family": "lerobot_groot", "task": TASK, "backend": "trt"}
    assert writer.sections["condition.plan"] == b"C" and writer.sections["denoise.plan"] == b"D"
    assert writer.sections["tokenizer.json"] == b"{}"
    runtime = writer.sections["runtime.json"]
    assert runtime["cameras"] == list(CAMERAS)
    assert (runtime["camera_size"], runtime["crop"], runtime["precision"]) == (256, 243, "fp32")
    assert (runtime["chunk_size"], runtime["action_horizon"], runtime["num_inference_steps"]) == (16, 40, 4)
    assert (runtime["state_dim"], runtime["action_dim"], runtime["max_state_dim"]) == (8, 7, 132)
    assert runtime["max_instruction_tokens"] == 256 and runtime["libero_gripper"] is True
    assert runtime["state_min"][2] == runtime["state_max"][2] == 2.0
    assert stubbed_build == ["tokenizer", "weights", "condition", "denoise"]


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"task": "robot_control"}, ValueError, "task"),
        ({"precision": "fp16"}, ValueError, "precision"),
        ({"backend": "trt_rtx", "max_batch_size": 2}, NotImplementedError, "batch size one"),
        ({"tensor_parallel_size": 2}, NotImplementedError, "one device"),
        ({"context_parallel_size": 2}, NotImplementedError, "one device"),
        ({"dynamic_kv_cache": True}, NotImplementedError, "does not support dynamic_kv_cache"),
        ({"video_num_frames": 4}, NotImplementedError, "video_num_frames"),
        ({"mtp_seq_len": 2}, NotImplementedError, "MTP"),
        ({"quantization": "fp8"}, NotImplementedError, "quantization"),
        ({"fp32_layers": (3,)}, NotImplementedError, "mixed-precision"),
        ({"max_sequence_length": 100}, NotImplementedError, "instruction tokens"),
        ({"image_height": 224}, NotImplementedError, "height"),
        ({"image_width": 224}, NotImplementedError, "width"),
    ],
)
def test_build_rejects_unsupported_requests_before_any_heavy_work(
    tmp_path, stubbed_build, overrides: dict, error: type, message: str
) -> None:
    with pytest.raises(error, match=message):
        build(_request(_model_dir(tmp_path / "ckpt"), **overrides), _Writer())
    assert stubbed_build == []


def test_checkpoint_loading_and_shape_checks(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="model.safetensors"):
        load_checkpoint(tmp_path)
    save_file(
        {"_groot_model.backbone.model.lm_head.weight": np.zeros((4, 2), np.float32),
         "_groot_model.backbone.model.model.language_model.norm.weight": np.zeros(2, np.float32),
         "_groot_model.action_head.vlln.bias": np.zeros(3, np.float32)},
        str(tmp_path / "model.safetensors"),
    )
    weights = load_checkpoint(tmp_path)
    # The policy prefix is dropped and the unused final norm is not loaded.
    assert set(weights) == {"backbone.model.lm_head.weight", "action_head.vlln.bias"}
    assert embedding_key(weights) == "backbone.model.lm_head.weight"
    assert require(weights, "action_head.vlln.bias", (3,)).dtype == np.float32
    with pytest.raises(ValueError, match="expected"):
        require(weights, "action_head.vlln.bias", (4,))
    with pytest.raises(KeyError, match="missing tensor"):
        require(weights, "absent", (1,))
    save_file({"_groot_model.action_head.vlln.bias": np.zeros(3, np.float32)}, str(tmp_path / "model.safetensors"))
    with pytest.raises(KeyError, match="lm_head"):
        load_checkpoint(tmp_path)


def test_untied_embedding_table_wins_and_names_are_unique(tmp_path) -> None:
    save_file(
        {"_groot_model.backbone.model.lm_head.weight": np.zeros((4, 2), np.float32),
         "_groot_model.backbone.model.model.language_model.embed_tokens.weight": np.ones((4, 2), np.float32)},
        str(tmp_path / "model.safetensors"),
    )
    weights = load_checkpoint(tmp_path)
    assert embedding_key(weights) == "backbone.model.model.language_model.embed_tokens.weight"
    assert canonical_name("_groot_model.action_head.vlln.bias") == "action_head.vlln.bias"
    save_file({"_groot_model.action_head.vlln.bias": np.zeros(3, np.float32),
               "action_head.vlln.bias": np.zeros(3, np.float32)}, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="twice"):
        load_checkpoint(tmp_path)


def test_category_layers_keep_one_embodiment() -> None:
    weights = {"x.W": np.arange(3 * 2 * 4, dtype=np.float32).reshape(3, 2, 4),
               "x.b": np.arange(3 * 4, dtype=np.float32).reshape(3, 4)}
    weight, bias = category(weights, "x.", 2, (2, 4))
    assert np.array_equal(weight, weights["x.W"][2]) and np.array_equal(bias, weights["x.b"][2])
    with pytest.raises(ValueError, match="outside"):
        category(weights, "x.", 3, (2, 4))
    with pytest.raises(ValueError, match="expected"):
        category(weights, "x.", 0, (4, 2))


def test_checkpoint_stored_in_bfloat16_loads_as_float32(tmp_path) -> None:
    import subprocess
    import sys

    import ml_dtypes

    values = np.array([0.5, -1.25, 3.0, 1000.0], dtype=ml_dtypes.bfloat16)
    save_file({"_groot_model.backbone.model.lm_head.weight": values.reshape(2, 2),
               "_groot_model.action_head.vlln.bias": values}, str(tmp_path / "model.safetensors"))
    # A fresh interpreter: this test registers bfloat16 itself, which would hide a loader
    # that forgot to. Checkpoints trained in bf16 store it, and NumPy has no such dtype alone.
    repository = Path(__file__).resolve().parents[3]
    script = (
        "import sys; sys.path[:0] = [sys.argv[1], sys.argv[1] + '/core/builder']\n"
        "from families.lerobot_groot.checkpoint import load_checkpoint\n"
        "w = load_checkpoint(sys.argv[2])['action_head.vlln.bias']\n"
        "print(w.dtype, w.tolist())\n"
    )
    result = subprocess.run([sys.executable, "-I", "-c", script, str(repository), str(tmp_path)],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "float32 [0.5, -1.25, 3.0, 1000.0]"


def _tokenizer_json(tokens: dict[str, int]) -> bytes:
    body = json.dumps({"added_tokens": [{"content": k, "id": v} for k, v in tokens.items()]}).encode()
    return body + b" " * (TOKENIZER_BYTES - len(body))


def test_tokenizer_asset_is_pinned_by_size_and_template_tokens(tmp_path) -> None:
    assert _checked(_tokenizer_json(TEMPLATE_TOKENS), "test")
    with pytest.raises(ValueError, match="not the supported vocabulary"):
        _checked(b"{}", "test")
    with pytest.raises(ValueError, match="template tokens"):
        _checked(_tokenizer_json({**TEMPLATE_TOKENS, "<|image_pad|>": 151654}), "test")
    (tmp_path / TOKENIZER_FILE).write_bytes(b"{}")
    with pytest.raises(ValueError, match="not the supported vocabulary"):
        load_tokenizer_json(tmp_path)


def test_tokenizer_falls_back_to_the_pinned_hub_revision(tmp_path, monkeypatch) -> None:
    import huggingface_hub

    cached = tmp_path / "cache" / TOKENIZER_FILE
    cached.parent.mkdir()
    cached.write_bytes(_tokenizer_json(TEMPLATE_TOKENS))
    calls = []

    def download(repo_id, filename, *, revision):
        calls.append((repo_id, filename, revision))
        return str(cached)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    assert load_tokenizer_json(checkpoint) == cached.read_bytes()
    assert calls == [(TOKENIZER_REPO, TOKENIZER_FILE, TOKENIZER_REVISION)]

    def unavailable(*_args, **_kwargs):
        raise OSError("offline and not cached")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", unavailable)
    with pytest.raises(FileNotFoundError, match="gated"):
        load_tokenizer_json(checkpoint)


def test_manifests_declare_the_public_tokenizer_copy() -> None:
    # CI cannot authorize a new family's gated dependency from an untrusted PR manifest.
    expected = [{"path": TOKENIZER_FILE,
                 "url": f"https://huggingface.co/{PUBLIC_COPY}/resolve/{PUBLIC_REVISION}/{TOKENIZER_FILE}"}]
    manifests = sorted((Path(__file__).parent / "manifests").glob("*.json"))
    assert manifests
    for path in manifests:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["external_files"] == expected, path.name
        assert "hf_dependencies" not in manifest, path.name


def test_rotate_half_matrix_is_hf_rotate_half() -> None:
    x = np.random.default_rng(0).standard_normal((5, 8)).astype(np.float32)
    expected = np.concatenate([-x[:, 4:], x[:, :4]], axis=1)
    assert np.array_equal(x @ graph.rotate_half_matrix(8), expected)


def test_mrope_frequencies_interleave_temporal_height_and_width() -> None:
    matrix = backbone.mrope_frequencies()
    half = backbone.HEAD_DIM // 2
    inverse = 1.0 / backbone.ROPE_THETA ** (np.arange(0, backbone.HEAD_DIM, 2) / backbone.HEAD_DIM)
    rows = matrix.argmax(axis=0)
    # apply_interleaved_mrope: H at 1, 4, ..., W at 2, 5, ... up to 3 * section, T elsewhere.
    expected = [1 if (i % 3 == 1 and i < 60) else 2 if (i % 3 == 2 and i < 60) else 0 for i in range(half)]
    assert rows.tolist() == expected
    assert np.count_nonzero(matrix) == half
    np.testing.assert_allclose(matrix.sum(axis=0), inverse, rtol=1e-6)


def test_vision_position_embeddings_are_bilinear_and_block_ordered() -> None:
    side = vision.POSITION_GRID
    table = np.random.default_rng(1).standard_normal((side * side, 3)).astype(np.float32)
    got = vision.position_embeddings(table, 4, 4)
    grid = table.reshape(side, side, 3).astype(np.float64)
    coords = np.linspace(0, side - 1, 4)
    expected = np.zeros((4, 4, 3))
    for r, y in enumerate(coords):
        for c, x in enumerate(coords):
            y0, x0 = int(y), int(x)
            y1, x1 = min(y0 + 1, side - 1), min(x0 + 1, side - 1)
            dy, dx = y - y0, x - x0
            expected[r, c] = ((1 - dy) * (1 - dx) * grid[y0, x0] + (1 - dy) * dx * grid[y0, x1]
                              + dy * (1 - dx) * grid[y1, x0] + dy * dx * grid[y1, x1])
    # Merge-block order: the 2x2 blocks row-major, each block row-major.
    order = [(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (0, 3), (1, 2), (1, 3),
             (2, 0), (2, 1), (3, 0), (3, 1), (2, 2), (2, 3), (3, 2), (3, 3)]
    np.testing.assert_allclose(got, np.array([expected[r, c] for r, c in order]), rtol=1e-5, atol=1e-5)


def test_vision_rotary_tables_rotate_rows_then_columns() -> None:
    cos, sin = vision.rotary_tables(4, 4)
    dim = vision.HEAD_DIM // 2
    inverse = 1.0 / 10000.0 ** (np.arange(0, dim, 2) / dim)
    # Patch 5 in merge-block order is row 0, column 3.
    angles = np.concatenate([0 * inverse, 3 * inverse])
    np.testing.assert_allclose(cos[5], np.cos(np.concatenate([angles, angles])), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(sin[5], np.sin(np.concatenate([angles, angles])), rtol=1e-6, atol=1e-6)


def test_time_embedding_frequencies_match_the_reference_formulas() -> None:
    half = action_head.WIDTH // 2
    np.testing.assert_allclose(action_head.action_time_frequencies(),
                               np.exp(-np.arange(half) * math.log(10000.0) / half), rtol=1e-6)
    half = action_head.TIME_CHANNELS // 2
    np.testing.assert_allclose(action_head.dit_time_frequencies(),
                               np.exp(-math.log(10000.0) * np.arange(half) / (half - 1)), rtol=1e-6)

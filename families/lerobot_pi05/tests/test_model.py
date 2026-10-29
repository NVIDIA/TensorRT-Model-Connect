# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only checks of support resolution, the checkpoint contract and request validation."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from safetensors.numpy import save_file
from tensorrt_model_connect import BuildRequest

from .. import config as config_module
from .. import model as model_module
from ..checkpoint import canonical_name, embedding_key, load_checkpoint, require
from ..model import TASK, _normalization_json, _prefix_length, build
from ..tokenizer import (
    PUBLIC_COPY,
    SPECIAL_IDS,
    TOKENIZER_BYTES,
    TOKENIZER_FILE,
    TOKENIZER_REPO,
    TOKENIZER_REVISION,
    VOCABULARY_PIECES,
    _checked,
    check_vocabulary,
    load_tokenizer_model,
)

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


_STATS = {"MEAN_STD": ("mean", "std"), "MIN_MAX": ("min", "max"),
          "QUANTILES": ("q01", "q99"), "QUANTILE10": ("q10", "q90")}
_FEATURES = {
    "observation.images.image": {"type": "VISUAL", "shape": [3, 256, 256]},
    "observation.state": {"type": "STATE", "shape": [8]},
    "action": {"type": "ACTION", "shape": [7]},
}


def _processors(*, stats: bool, mode: str = "MEAN_STD", padding_side: str = "right",
                features: dict | None = None, observation_keys: list | None = None,
                extra_steps: tuple = (), normalize_after_prompt: bool = False,
                max_length: int = 200) -> tuple[dict, dict]:
    """Processor documents shaped like the ones LeRobot saves for pi05."""
    norm_map = {"VISUAL": "IDENTITY", "STATE": mode, "ACTION": mode}
    settings = {"eps": 1e-8, "norm_map": norm_map,
                "features": _FEATURES if features is None else features}
    if observation_keys is not None:
        settings["normalize_observation_keys"] = observation_keys
    pre_norm = {"registry_name": "normalizer_processor", "config": settings}
    post_norm = {"registry_name": "unnormalizer_processor", "config": dict(settings)}
    if stats:
        pre_norm["state_file"] = NORMALIZER
        post_norm["state_file"] = UNNORMALIZER
    prompt = {"registry_name": "pi05_prepare_state_tokenizer_processor_step", "config": {}}
    tokenizer = {
        "registry_name": "tokenizer_processor",
        "config": {"max_length": max_length, "padding_side": padding_side, "padding": "max_length",
                   "tokenizer_name": "google/paligemma-3b-pt-224"},
    }
    middle = [prompt, pre_norm] if normalize_after_prompt else [pre_norm, prompt]
    pre = [{"registry_name": "rename_observations_processor", "config": {"rename_map": {}}},
           {"registry_name": "to_batch_processor", "config": {}}, *extra_steps, *middle, tokenizer,
           {"registry_name": "device_processor", "config": {"device": "cuda"}}]
    post = [post_norm, {"registry_name": "device_processor", "config": {"device": "cpu"}}]
    return {"steps": pre}, {"steps": post}


def _write_stats(root: Path, mode: str, *, state: bool = True) -> None:
    low, high = _STATS[mode]
    state_low = np.arange(8, dtype=np.float32)
    state_high = state_low + 4.0
    state_high[2] = state_low[2]  # a zero spread on this dimension
    state_stats = {f"observation.state.{low}": state_low, f"observation.state.{high}": state_high}
    save_file(state_stats if state else {"observation.images.image.mean": np.zeros(3, np.float32)},
              str(root / NORMALIZER))
    # The dataset range first: under MIN_MAX the mode's own statistics are that range.
    action = {"action.min": np.full(7, -2.0, np.float32), "action.max": np.full(7, 2.5, np.float32),
              f"action.{low}": -np.ones(7, np.float32), f"action.{high}": np.full(7, 3.0, np.float32)}
    save_file(action, str(root / UNNORMALIZER))


def _model_dir(root: Path, *, stats: bool = False, config: dict | None = None,
               state_stats: bool = True, **processor_args) -> Path:
    mode = processor_args.get("mode", "MEAN_STD")
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(config or _config()), encoding="utf-8")
    pre, post = _processors(stats=stats, **processor_args)
    (root / "policy_preprocessor.json").write_text(json.dumps(pre), encoding="utf-8")
    (root / "policy_postprocessor.json").write_text(json.dumps(post), encoding="utf-8")
    if stats and mode in _STATS:
        _write_stats(root, mode, state=state_stats)
    return root


def test_config_reads_the_checkpoint_contract(tmp_path) -> None:
    cfg = config_module.load_config(_model_dir(tmp_path / "ckpt"))
    assert cfg.cameras == ("observation.images.image", "observation.images.image2")
    assert (cfg.empty_cameras, cfg.camera_slots) == (1, 3)
    assert (cfg.camera_height, cfg.camera_width) == (256, 256)
    assert (cfg.state_dim, cfg.action_dim, cfg.chunk_size) == (8, 7, 50)
    assert cfg.tokenizer_name == "google/paligemma-3b-pt-224"
    # No statistics in the directory: LeRobot passes the value through.
    assert cfg.state_normalization.is_identity and cfg.action_normalization.is_identity
    assert _prefix_length(cfg) == 3 * 256 + 200


def test_config_ignores_a_placeholder_camera_listed_as_a_feature(tmp_path) -> None:
    raw = _config()
    raw["input_features"]["observation.images.empty_camera_0"] = {"type": "VISUAL", "shape": [3, 224, 224]}
    cfg = config_module.load_config(_model_dir(tmp_path / "ckpt", config=raw))
    assert len(cfg.cameras) == 2 and cfg.camera_slots == 3


def test_config_loads_mean_std_statistics_and_the_action_range(tmp_path) -> None:
    cfg = config_module.load_config(_model_dir(tmp_path / "ckpt", stats=True))
    state = cfg.state_normalization
    assert state.mode == "MEAN_STD" and not state.is_identity
    assert state.offset == tuple(float(v) for v in range(8))  # mean
    assert state.scale[0] == 4.0  # std, used with eps at run time
    assert cfg.action_normalization.offset == (-1.0,) * 7 and cfg.action_normalization.scale == (3.0,) * 7
    assert cfg.action_range == ((-2.0,) * 7, (2.5,) * 7)
    assert _normalization_json(cfg.action_normalization)["mode"] == "MEAN_STD"
    assert _normalization_json(config_module.Normalization("MEAN_STD", 1e-8)) == {"mode": "IDENTITY"}


@pytest.mark.parametrize("mode", ["QUANTILES", "QUANTILE10", "MIN_MAX"])
def test_config_reads_every_lerobot_range_mode(tmp_path, mode: str) -> None:
    # LeRobot's pi05 default is QUANTILES, so a plain lerobot-train output uses it.
    cfg = config_module.load_config(_model_dir(tmp_path / "ckpt", stats=True, mode=mode))
    state = cfg.state_normalization
    assert state.mode == mode and not state.is_identity
    assert state.offset == tuple(float(v) for v in range(8))  # the lower statistic
    # scale is the spread to the upper statistic, eps where it is exactly zero (dimension 2).
    assert state.scale[0] == 4.0 and state.scale[2] == np.float32(1e-8)
    assert cfg.action_normalization.offset == (-1.0,) * 7 and cfg.action_normalization.scale == (4.0,) * 7
    assert _normalization_json(state)["scale"][2] == pytest.approx(1e-8)
    expected_range = ((-1.0,) * 7, (3.0,) * 7) if mode == "MIN_MAX" else ((-2.0,) * 7, (2.5,) * 7)
    assert cfg.action_range == expected_range


def test_state_is_normalized_only_when_the_step_selects_it(tmp_path) -> None:
    # LeRobot normalizes an observation only if the step lists it as a non-action feature and
    # normalize_observation_keys keeps it; the action is unnormalized regardless.
    unlisted = {k: v for k, v in _FEATURES.items() if k != "observation.state"}
    for name, kwargs in (("unlisted", {"features": unlisted}),
                         ("filtered", {"observation_keys": ["observation.images.image"]}),
                         ("as-action", {"features": {**_FEATURES, "observation.state": {"type": "ACTION"}}})):
        cfg = config_module.load_config(_model_dir(tmp_path / name, stats=True, **kwargs))
        assert cfg.state_normalization.is_identity, name
        assert not cfg.action_normalization.is_identity, name
    kept = config_module.load_config(
        _model_dir(tmp_path / "kept", stats=True, observation_keys=["observation.state"]))
    assert not kept.state_normalization.is_identity


def test_statistics_absent_for_a_feature_pass_it_through_like_lerobot(tmp_path) -> None:
    cfg = config_module.load_config(_model_dir(tmp_path / "ckpt", stats=True, state_stats=False))
    assert cfg.state_normalization.is_identity
    assert not cfg.action_normalization.is_identity


def test_config_rejects_an_unknown_normalization_mode_only_when_statistics_exist(tmp_path) -> None:
    with pytest.raises(NotImplementedError, match="LOG_SCALE"):
        root = _model_dir(tmp_path / "with-stats", stats=True, mode="LOG_SCALE")
        _write_stats(root, "MEAN_STD")
        config_module.load_config(root)
    # Without statistics the mode is moot, as in LeRobot.
    cfg = config_module.load_config(_model_dir(tmp_path / "no-stats", mode="LOG_SCALE"))
    assert cfg.state_normalization.is_identity


def test_config_requires_the_statistics_it_names(tmp_path) -> None:
    root = _model_dir(tmp_path / "ckpt", stats=True)
    (root / NORMALIZER).unlink()
    with pytest.raises(FileNotFoundError, match="normalizer_processor"):
        config_module.load_config(root)
    # Present for the feature but missing what the mode needs: LeRobot raises too.
    save_file({"observation.state.mean": np.zeros(8, np.float32)}, str(root / NORMALIZER))
    with pytest.raises(ValueError, match="std"):
        config_module.load_config(root)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"extra_steps": ({"registry_name": "image_crop_processor", "config": {}},)}, "image_crop_processor"),
        ({"extra_steps": ({"registry_name": "relative_actions_processor", "config": {"enabled": True}},)},
         "enabled"),
        ({"normalize_after_prompt": True}, "before the prompt step"),
        ({"max_length": 48}, "max_length"),
    ],
)
def test_config_refuses_processor_steps_it_does_not_reproduce(tmp_path, kwargs: dict, message: str) -> None:
    with pytest.raises((NotImplementedError, ValueError), match=message):
        config_module.load_config(_model_dir(tmp_path / "ckpt", stats=True, **kwargs))


def test_config_accepts_disabled_relative_action_steps(tmp_path) -> None:
    disabled = ({"registry_name": "relative_actions_processor", "config": {"enabled": False}},)
    cfg = config_module.load_config(_model_dir(tmp_path / "ckpt", stats=True, extra_steps=disabled))
    assert not cfg.state_normalization.is_identity


def test_config_requires_lerobots_state_and_action_keys(tmp_path) -> None:
    raw = _config()
    raw["input_features"]["observation.joint_state"] = raw["input_features"].pop("observation.state")
    with pytest.raises(ValueError, match="observation.state"):
        config_module.load_config(_model_dir(tmp_path / "ckpt", config=raw))


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


def _request(model_dir: Path, **overrides) -> BuildRequest:
    values = {"model_dir": model_dir, "output_path": Path("out.bundle"), "family": "lerobot_pi05",
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
    calls: list[str] = []
    metadata = SimpleNamespace(piece_count=VOCABULARY_PIECES, **SPECIAL_IDS)
    monkeypatch.setattr(model_module, "load_checkpoint", lambda _: calls.append("weights") or {})
    monkeypatch.setattr(model_module, "load_tokenizer_model", lambda _: calls.append("tokenizer") or b"tokenizer")
    monkeypatch.setattr(model_module, "convert_paligemma_tokenizer_model", lambda _: (b"bpe", metadata))
    monkeypatch.setattr(model_module, "_build_prefix_plan", lambda *a, **k: calls.append("prefix") or b"P")
    monkeypatch.setattr(model_module, "_build_denoise_plan", lambda *a, **k: calls.append("denoise") or b"D")
    return calls


def test_build_writes_the_bundle_contract(tmp_path, stubbed_build) -> None:
    model_dir = _model_dir(tmp_path / "ckpt", stats=True)
    writer = _Writer()
    build(_request(model_dir, precision="FP32", max_sequence_length=200, image_height=256), writer)
    assert writer.header == {"family": "lerobot_pi05", "task": TASK, "backend": "trt"}
    assert writer.sections["prefix.plan"] == b"P" and writer.sections["denoise.plan"] == b"D"
    assert writer.sections["tokenizer.bpe"] == b"bpe"
    runtime = writer.sections["runtime.json"]
    assert runtime["cameras"] == ["observation.images.image", "observation.images.image2"]
    assert (runtime["empty_cameras"], runtime["prefix_tokens"], runtime["precision"]) == (1, 968, "fp32")
    assert (runtime["state_dim"], runtime["action_dim"], runtime["max_action_dim"]) == (8, 7, 32)
    assert runtime["state_normalization"]["mode"] == "MEAN_STD"
    assert runtime["action_normalization"]["scale"] == [3.0] * 7
    assert runtime["action_range"] == {"min": [-2.0] * 7, "max": [2.5] * 7}
    assert stubbed_build == ["tokenizer", "weights", "prefix", "denoise"]


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
        ({"max_sequence_length": 100}, NotImplementedError, "prompt length"),
        ({"image_height": 224}, NotImplementedError, "height"),
        ({"image_width": 224}, NotImplementedError, "width"),
    ],
)
def test_build_rejects_unsupported_requests_before_any_heavy_work(
    tmp_path, stubbed_build, overrides: dict, error: type, message: str
) -> None:
    model_dir = _model_dir(tmp_path / "ckpt")
    with pytest.raises(error, match=message):
        build(_request(model_dir, **overrides), _Writer())
    assert stubbed_build == []


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


def test_checkpoint_stored_in_bfloat16_loads_as_float32(tmp_path) -> None:
    import subprocess
    import sys

    import ml_dtypes

    values = np.array([0.5, -1.25, 3.0, 1000.0], dtype=ml_dtypes.bfloat16)
    save_file(
        {"paligemma_with_expert.paligemma.lm_head.weight": values.reshape(2, 2),
         "paligemma_with_expert.time_mlp_in.bias": values},
        str(tmp_path / "model.safetensors"),
    )
    # A fresh interpreter: this test registers bfloat16 itself, which would hide a loader
    # that forgot to. Checkpoints trained in bf16 store it, and NumPy has no such dtype alone.
    repository = Path(__file__).resolve().parents[3]
    script = (
        "import sys; sys.path[:0] = [sys.argv[1], sys.argv[1] + '/core/builder']\n"
        "from families.lerobot_pi05.checkpoint import load_checkpoint\n"
        "w = load_checkpoint(sys.argv[2])['time_mlp_in.bias']\n"
        "print(w.dtype, w.tolist())\n"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(repository), str(tmp_path)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "float32 [0.5, -1.25, 3.0, 1000.0]"


@pytest.mark.parametrize(
    ("stored", "canonical"),
    [
        ("paligemma_with_expert.action_in_proj.weight", "action_in_proj.weight"),
        ("model.paligemma_with_expert.action_in_proj.weight", "action_in_proj.weight"),
        ("model.paligemma_with_expert.gemma_expert.model.layers.0.mlp.up_proj.weight",
         "gemma_expert.model.layers.0.mlp.up_proj.weight"),
        ("time_mlp_in.bias", "time_mlp_in.bias"),
    ],
)
def test_both_checkpoint_layouts_share_canonical_names(stored: str, canonical: str) -> None:
    # A converted checkpoint and a lerobot-train output differ only by the leading prefixes.
    assert canonical_name(stored) == canonical


def test_train_output_layout_loads_and_prefers_the_embedding_table(tmp_path) -> None:
    low, high = np.zeros((4, 2), np.float32), np.ones((4, 2), np.float32)
    save_file(
        {"model.paligemma_with_expert.paligemma.lm_head.weight": low,
         "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight": high,
         "model.paligemma_with_expert.gemma_expert.lm_head.weight": low,
         "model.time_mlp_in.bias": np.zeros(3, np.float32)},
        str(tmp_path / "model.safetensors"),
    )
    weights = load_checkpoint(tmp_path)
    assert "time_mlp_in.bias" in weights and "gemma_expert.lm_head.weight" not in weights
    # Untied tables: LeRobot looks tokens up in embed_tokens, not lm_head.
    assert embedding_key(weights) == "paligemma.model.language_model.embed_tokens.weight"
    assert weights[embedding_key(weights)].sum() == 8.0
    with pytest.raises(KeyError, match="none of"):
        embedding_key({"unrelated": low})
    # Two spellings of one tensor are an error, not a silent overwrite.
    save_file({"paligemma_with_expert.time_mlp_in.bias": low, "model.paligemma_with_expert.time_mlp_in.bias": low},
              str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="twice"):
        load_checkpoint(tmp_path)


def test_tokenizer_asset_is_pinned_by_size_and_vocabulary(tmp_path) -> None:
    with pytest.raises(ValueError, match="not the supported vocabulary"):
        _checked(b"\x00" * (TOKENIZER_BYTES - 1), "test")
    assert _checked(b"\x00" * TOKENIZER_BYTES, "test")
    (tmp_path / "tokenizer.model").write_bytes(b"not the paligemma tokenizer")
    with pytest.raises(ValueError, match="not the supported vocabulary"):
        load_tokenizer_model(tmp_path)
    check_vocabulary(SimpleNamespace(piece_count=VOCABULARY_PIECES, **SPECIAL_IDS))
    with pytest.raises(ValueError, match="special ids"):
        check_vocabulary(SimpleNamespace(piece_count=VOCABULARY_PIECES, **{**SPECIAL_IDS, "bos_id": 1}))
    with pytest.raises(ValueError, match="pieces"):
        check_vocabulary(SimpleNamespace(piece_count=1000, **SPECIAL_IDS))


def test_tokenizer_falls_back_to_the_pinned_hub_revision(tmp_path, monkeypatch) -> None:
    import huggingface_hub

    cached = tmp_path / "cache" / TOKENIZER_FILE
    cached.parent.mkdir()
    cached.write_bytes(b"\x01" * TOKENIZER_BYTES)
    calls = []

    def download(repo_id, filename, *, revision):
        calls.append((repo_id, filename, revision))
        return str(cached)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    assert load_tokenizer_model(checkpoint) == cached.read_bytes()
    assert calls == [(TOKENIZER_REPO, TOKENIZER_FILE, TOKENIZER_REVISION)]

    def unavailable(*_args, **_kwargs):
        raise OSError("offline and not cached")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", unavailable)
    with pytest.raises(FileNotFoundError, match="gated"):
        load_tokenizer_model(checkpoint)


def test_manifests_declare_the_public_tokenizer_copy() -> None:
    # CI cannot authorize a new family's gated dependency from an untrusted PR manifest.
    expected = [{"path": TOKENIZER_FILE, "url": PUBLIC_COPY + "?generation=1711547605575873"}]
    manifests = sorted((Path(__file__).parent / "manifests").glob("*.json"))
    assert manifests
    for path in manifests:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["external_files"] == expected, path.name
        assert "hf_dependencies" not in manifest, path.name

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Builder contract and graph parity of the videomae family.

The CPU tests cover the configuration and request checks. The GPU tests build
the engine of small random VideoMAE checkpoints and compare its logits with
transformers' VideoMAEForPreTraining in float32 for several tube selections.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from families.videomae import config as config_module, graph, model

# config.json and preprocessor_config.json of MCG-NJU/videomae-base at the pinned revision.
BASE_CONFIG = {
    "architectures": ["VideoMAEForPreTraining"],
    "attention_probs_dropout_prob": 0.0,
    "decoder_hidden_size": 384,
    "decoder_intermediate_size": 1536,
    "decoder_num_attention_heads": 6,
    "decoder_num_hidden_layers": 4,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.0,
    "hidden_size": 768,
    "image_size": 224,
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "layer_norm_eps": 1e-12,
    "model_type": "videomae",
    "norm_pix_loss": True,
    "num_attention_heads": 12,
    "num_channels": 3,
    "num_frames": 16,
    "num_hidden_layers": 12,
    "patch_size": 16,
    "qkv_bias": True,
    "torch_dtype": "float32",
    "transformers_version": "4.22.0.dev0",
    "tubelet_size": 2,
    "use_mean_pooling": False,
}
BASE_PROCESSOR = {
    "do_center_crop": True,
    "do_normalize": True,
    "do_resize": True,
    "feature_extractor_type": "VideoMAEFeatureExtractor",
    "image_mean": [0.485, 0.456, 0.406],
    "image_std": [0.229, 0.224, 0.225],
    "resample": 2,
    "size": 224,
}


def _write(directory: Path, config: dict, processor: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (directory / "preprocessor_config.json").write_text(json.dumps(processor), encoding="utf-8")
    return directory


def test_base_checkpoint_configuration(tmp_path):
    cfg = config_module.resolve(_write(tmp_path, BASE_CONFIG, BASE_PROCESSOR))
    assert (cfg["num_frames"], cfg["image_size"], cfg["patch_size"], cfg["tubelet_size"]) == (
        16, 224, 16, 2)
    assert (cfg["grid"], cfg["tubes"], cfg["tube_values"]) == (14, 1568, 1536)
    assert cfg["encoder"] == {"width": 768, "heads": 12, "layers": 12, "intermediate": 3072}
    assert cfg["decoder"] == {"width": 384, "heads": 6, "layers": 4, "intermediate": 1536}
    assert cfg["layer_norm_eps"] == 1e-12 and cfg["final_layernorm"] is True
    assert cfg["image_mean"] == [0.485, 0.456, 0.406]
    assert cfg["image_std"] == [0.229, 0.224, 0.225]


def test_omitted_keys_take_the_transformers_defaults(tmp_path):
    config = {"architectures": ["VideoMAEForPreTraining"], "model_type": "videomae"}
    cfg = config_module.resolve(_write(tmp_path, config, {}))
    # VideoMAEConfig defaults use_mean_pooling to true: no final encoder norm.
    assert cfg["final_layernorm"] is False
    assert cfg["tubes"] == 1568
    # VideoMAEImageProcessor defaults its statistics to 0.5.
    assert cfg["image_mean"] == [0.5, 0.5, 0.5] and cfg["image_std"] == [0.5, 0.5, 0.5]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"norm_pix_loss": False}, "norm_pix_loss"),
        ({"hidden_act": "gelu_new"}, "hidden_act"),
        ({"qkv_bias": False}, "qkv_bias"),
        ({"num_channels": 1}, "num_channels"),
        ({"image_size": [224, 160]}, "square image_size"),
    ],
)
def test_unreproduced_model_flags_are_refused(tmp_path, change, message):
    with pytest.raises(NotImplementedError, match=message):
        config_module.resolve(_write(tmp_path, {**BASE_CONFIG, **change}, BASE_PROCESSOR))


@pytest.mark.parametrize(
    "change",
    [
        {"architectures": ["VideoMAEForVideoClassification"]},
        {"model_type": "vit_mae"},
        {"num_frames": 15},
        {"patch_size": 15},
        {"hidden_size": 770},
        {"layer_norm_eps": 0},
        {"layer_norm_eps": None},
        {"use_mean_pooling": None},
        {"architectures": "VideoMAEForPreTraining"},
        {"image_size": [224, 224.0]},
        {"num_frames": 16.0},
        # 0 and infinity once the graph stores them in float32.
        {"layer_norm_eps": 1e-100},
        {"layer_norm_eps": 1e100},
    ],
)
def test_invalid_model_configurations_are_refused(tmp_path, change):
    with pytest.raises(ValueError):
        config_module.resolve(_write(tmp_path, {**BASE_CONFIG, **change}, BASE_PROCESSOR))


@pytest.mark.parametrize(
    "change",
    [
        {"do_resize": None},
        {"do_normalize": None},
        {"rescale_factor": True},
        {"image_mean": [0.485, True, 0.406]},
        {"image_std": [0.229, 0.224, None]},
        {"size": {"shortest_edge": 224.0}},
        {"crop_size": {"height": 224, "width": True}},
        # Finite and positive, yet (pixel - mean) / std overflows float32.
        {"image_std": [0.229, 0.224, 1e-40]},
        {"image_mean": [0.485, 0.456, 1e300]},
        # Infinite in float32, which the runtime refuses to load.
        {"image_std": [0.229, 0.224, 1e300]},
    ],
)
def test_mistyped_processor_values_are_refused(tmp_path, change):
    with pytest.raises(ValueError):
        config_module.resolve(_write(tmp_path, BASE_CONFIG, {**BASE_PROCESSOR, **change}))


@pytest.mark.parametrize("name", ["config.json", "preprocessor_config.json"])
def test_repeated_json_keys_are_refused(tmp_path, name):
    _write(tmp_path, BASE_CONFIG, BASE_PROCESSOR)
    path = tmp_path / name
    text = path.read_text(encoding="utf-8")
    path.write_text(text[:-1] + ', "do_resize": true, "model_type": "videomae"}', encoding="utf-8")
    with pytest.raises(ValueError, match="repeated JSON keys"):
        config_module.resolve(tmp_path)


@pytest.mark.parametrize(
    "change",
    [
        {"size": 256},
        {"size": {"height": 256, "width": 224}},
        {"crop_size": {"height": 200, "width": 200}},
        {"do_rescale": False},
        {"rescale_factor": 1 / 127.5},
        {"do_normalize": False},
    ],
)
def test_processors_that_change_a_model_size_clip_are_refused(tmp_path, change):
    with pytest.raises(NotImplementedError):
        config_module.resolve(_write(tmp_path, BASE_CONFIG, {**BASE_PROCESSOR, **change}))


def test_processor_without_resize_or_crop_keeps_model_size_clips(tmp_path):
    processor = {**BASE_PROCESSOR, "do_resize": False, "do_center_crop": False, "size": 512}
    assert config_module.resolve(_write(tmp_path, BASE_CONFIG, processor))["image_size"] == 224


def test_sinusoid_table_matches_the_reference_bit_for_bit():
    from transformers.models.videomae.modeling_videomae import get_sinusoid_encoding_table

    for positions, width in ((1568, 768), (1568, 384), (32, 64)):
        expected = get_sinusoid_encoding_table(positions, width)[0].numpy()
        actual = graph.sinusoid_table(positions, width)
        assert actual.dtype == np.float32 and np.array_equal(actual, expected)


class _Options:
    """The E2E selector options of a pytest configuration."""

    def __init__(self, **values):
        self.values = values

    def getoption(self, name):
        return self.values.get(name)


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ({}, (["videomae-base-box-mask", "videomae-base-tube-mask"], False)),
        ({"--e2e-model": ["videomae"]}, (["videomae-base-box-mask", "videomae-base-tube-mask"], True)),
        ({"--e2e-model": ["videomae"], "--e2e-testcase": ["videomae-base-box-mask"]},
         (["videomae-base-box-mask"], True)),
        ({"--e2e-testcase": ["videomae-base-tube-mask"]}, (["videomae-base-tube-mask"], True)),
        # A selection of another family skips this file.
        ({"--e2e-model": ["bert"]}, ([], True)),
        ({"--e2e-testcase": ["bert-base-uncased"]}, ([], True)),
    ],
)
def test_e2e_selection(options, expected):
    from families.videomae.tests import test_e2e

    assert test_e2e._selected_cases(_Options(**options)) == expected


@pytest.mark.parametrize(
    "options",
    [
        {"--e2e-model": ["videomae"], "--e2e-testcase": ["videomae-base-typo"]},
        {"--e2e-model": ["videomae-base"], "--e2e-testcase": ["videomae-base-typo"]},
        {"--e2e-model": ["videomae-base-tube-mask"], "--e2e-testcase": ["videomae-base-box-mask"]},
    ],
)
def test_e2e_selection_that_names_the_family_but_matches_nothing_fails(options):
    from families.videomae.tests import test_e2e

    with pytest.raises(pytest.UsageError, match="matches none of its cases"):
        test_e2e._selected_cases(_Options(**options))


def test_e2e_selection_from_a_models_file_that_matches_nothing_fails(tmp_path):
    from families.videomae.tests import test_e2e

    models = tmp_path / "models.txt"
    models.write_text("# selected families\nvideomae\n", encoding="utf-8")
    with pytest.raises(pytest.UsageError, match="matches none of its cases"):
        test_e2e._selected_cases(_Options(**{"--e2e-models-file": str(models),
                                             "--e2e-testcase": ["videomae-base-typo"]}))


class _Writer:
    """Records what a build would write, so request checks can run without a GPU."""

    def __init__(self):
        self.sections = {}

    def set_header(self, **header):
        self.header = header

    def add_bytes(self, name, data):
        self.sections[name] = data

    def add_json(self, name, value):
        self.sections[name] = value


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ({"task": "text_to_video"}, ValueError),
        ({"backend": "trt_rtx"}, ValueError),
        ({"precision": "fp16"}, NotImplementedError),
        ({"quantization": "fp8"}, NotImplementedError),
        ({"fp32_layers": (1,)}, NotImplementedError),
        ({"max_batch_size": 2}, NotImplementedError),
        ({"tensor_parallel_size": 2}, NotImplementedError),
        ({"context_parallel_size": 2}, NotImplementedError),
        ({"dynamic_kv_cache": True}, NotImplementedError),
        ({"mtp_seq_len": 2}, NotImplementedError),
        ({"max_sequence_length": 1568}, NotImplementedError),
        ({"image_height": 256}, NotImplementedError),
        ({"image_width": 160}, NotImplementedError),
        ({"video_num_frames": 32}, NotImplementedError),
        # Fractional or float sizes are refused, not truncated.
        ({"video_num_frames": 16.5}, ValueError),
        ({"image_height": 224.0}, ValueError),
        ({"max_batch_size": 1.0}, ValueError),
    ],
)
def test_unsupported_build_requests_are_refused_before_the_engine(tmp_path, change, error):
    from tensorrt_model_connect import BuildRequest

    request = BuildRequest(**{
        "model_dir": _write(tmp_path / "model", BASE_CONFIG, BASE_PROCESSOR),
        "output_path": tmp_path / "videomae.bundle",
        "family": "videomae",
        "task": "masked_video_text_to_video",
        "precision": "fp32",
        **change,
    })
    writer = _Writer()
    with pytest.raises(error):
        model.build(request, writer)
    assert writer.sections == {}


# Small random checkpoints for the graph test: 4 frames of 32x32 pixels in
# 2x8x8 tubes (32 tubes), with and without the final encoder norm.
TINY_CONFIG = {
    **BASE_CONFIG,
    "image_size": 32,
    "patch_size": 8,
    "num_frames": 4,
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "intermediate_size": 96,
    "decoder_hidden_size": 32,
    "decoder_num_hidden_layers": 2,
    "decoder_num_attention_heads": 2,
    "decoder_intermediate_size": 48,
}
TINY_PROCESSOR = {**BASE_PROCESSOR, "size": 32, "crop_size": 32}


def _tiny_checkpoint(directory: Path, use_mean_pooling: bool, seed: int):
    import torch
    from transformers import VideoMAEConfig, VideoMAEForPreTraining

    torch.manual_seed(seed)
    reference = VideoMAEForPreTraining(VideoMAEConfig(
        **{**TINY_CONFIG, "use_mean_pooling": use_mean_pooling}, attn_implementation="eager"))
    # The default initialization (std 0.02, unit norms, zero biases) would hide
    # mistakes in the norms and biases, so every tensor gets a random value.
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.3)
    reference.save_pretrained(directory)
    _write(directory, json.loads((directory / "config.json").read_text()), TINY_PROCESSOR)
    return reference.eval()


def test_checkpoint_tensors_are_checked_before_the_engine(tmp_path):
    _tiny_checkpoint(tmp_path, use_mean_pooling=False, seed=3)
    cfg = config_module.resolve(tmp_path)
    weights = model._open_checkpoint(tmp_path, cfg)
    model._require_weights(weights, cfg)

    missing = dict(weights)
    del missing["decoder.norm.bias"]
    with pytest.raises(KeyError, match="missing expected tensors: decoder.norm.bias"):
        model._require_weights(missing, cfg)
    extra = {**weights, "decoder.extra.weight": weights["decoder.norm.bias"]}
    with pytest.raises(KeyError, match="does not run: decoder.extra.weight"):
        model._require_weights(extra, cfg)
    half = {**weights, "mask_token": weights["mask_token"].astype(np.float16)}
    with pytest.raises(TypeError, match="float32 checkpoints only: mask_token"):
        model._require_weights(half, cfg)
    reshaped = {**weights, "mask_token": np.zeros((1, 2, cfg["decoder"]["width"]), np.float32)}
    with pytest.raises(ValueError, match="mask_token"):
        model._require_weights(reshaped, cfg)
    # A bias that would broadcast, and an MLP that disagrees with the configuration.
    bias = "videomae.encoder.layer.0.attention.attention.q_bias"
    with pytest.raises(ValueError, match=bias):
        model._require_weights({**weights, bias: np.zeros((1,), np.float32)}, cfg)
    mlp = "decoder.decoder_layers.1.intermediate.dense.weight"
    wider = np.zeros((cfg["decoder"]["intermediate"] + 8, cfg["decoder"]["width"]), np.float32)
    with pytest.raises(ValueError, match=mlp):
        model._require_weights({**weights, mlp: wider}, cfg)


def test_checkpoint_header_is_checked_before_any_tensor_is_read(tmp_path, monkeypatch):
    import safetensors
    from safetensors.numpy import save_file

    _tiny_checkpoint(tmp_path, use_mean_pooling=False, seed=3)
    cfg = config_module.resolve(tmp_path)
    weights = model._open_checkpoint(tmp_path, cfg)
    bias = "decoder.decoder_layers.0.attention.attention.v_bias"
    save_file({**weights, bias: np.zeros((1,), np.float32)}, str(tmp_path / "model.safetensors"))
    opened, read = [], []
    real_open = safetensors.safe_open

    class Recording:
        """safe_open that records the files it opens and the tensors whose data it reads."""

        def __init__(self, path, **options):
            opened.append(path)
            self._handle = real_open(path, **options)

        def __enter__(self):
            self._handle.__enter__()
            return self

        def __exit__(self, *details):
            return self._handle.__exit__(*details)

        def keys(self):
            return self._handle.keys()

        def get_slice(self, key):
            return self._handle.get_slice(key)

        def get_tensor(self, key):
            read.append(key)
            return self._handle.get_tensor(key)

    monkeypatch.setattr(safetensors, "safe_open", Recording)
    with pytest.raises(ValueError, match=bias):
        model._open_checkpoint(tmp_path, cfg)
    assert opened == [str(tmp_path / "model.safetensors")]
    assert read == []

def _selections(tubes: int, rng: np.random.Generator):
    per_tubelet = tubes // 2
    tube_mask = np.zeros(per_tubelet, dtype=bool)
    tube_mask[: int(0.9 * per_tubelet)] = True
    rng.shuffle(tube_mask)
    single_visible = np.ones(tubes, dtype=bool)
    single_visible[rng.integers(tubes)] = False
    single_masked = np.zeros(tubes, dtype=bool)
    single_masked[rng.integers(tubes)] = True
    return {
        "tube_90": np.tile(tube_mask, 2),
        "random_half": rng.permutation(np.arange(tubes) % 2 == 0),
        "one_visible": single_visible,
        "one_masked": single_masked,
    }


def _run_plan(plan: bytes, feeds: dict) -> np.ndarray:
    import tensorrt as trt
    import torch

    engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(plan)
    context = engine.create_execution_context()
    stream = torch.cuda.current_stream()
    buffers = {}
    for name, value in feeds.items():
        buffers[name] = torch.from_numpy(np.ascontiguousarray(value)).cuda()
        assert context.set_input_shape(name, tuple(buffers[name].shape))
        context.set_tensor_address(name, buffers[name].data_ptr())
    logits = torch.empty(tuple(context.get_tensor_shape("logits")), dtype=torch.float32,
                         device="cuda")
    context.set_tensor_address("logits", logits.data_ptr())
    assert context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    return logits.cpu().numpy()


@pytest.mark.gpu
@pytest.mark.trt
@pytest.mark.parametrize("use_mean_pooling", [False, True], ids=["final-norm", "no-final-norm"])
def test_engine_logits_match_transformers_float32(tmp_path, use_mean_pooling):
    import torch

    reference = _tiny_checkpoint(tmp_path, use_mean_pooling, seed=7)
    cfg = config_module.resolve(tmp_path)
    weights = model._open_checkpoint(tmp_path, cfg)
    plan = model.build_engine(weights, cfg)

    rng = np.random.default_rng(11)
    pixel_values = rng.standard_normal((1, 4, 3, 32, 32)).astype(np.float32)
    for name, masked in _selections(cfg["tubes"], rng).items():
        with torch.no_grad():
            expected = reference(torch.from_numpy(pixel_values),
                                 bool_masked_pos=torch.from_numpy(masked)[None]).logits.numpy()
        actual = _run_plan(plan, {
            "pixel_values": pixel_values,
            "visible_index": np.flatnonzero(~masked).astype(np.int32),
            "masked_index": np.flatnonzero(masked).astype(np.int32),
        })
        assert actual.shape == expected.shape == (1, int(masked.sum()), cfg["tube_values"]), name
        error = float(np.max(np.abs(actual - expected)))
        assert error <= 1e-4 * max(1.0, float(np.max(np.abs(expected)))), (name, error)

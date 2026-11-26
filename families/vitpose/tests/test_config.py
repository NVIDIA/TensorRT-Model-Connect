# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The checkpoint configuration and tensor checks, and the build options the family rejects, on the CPU."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from families.vitpose import config as config_module
from families.vitpose import model as model_module

# A tiny checkpoint of the published layout: two 16-pixel patches (32 x 16 crop), width 8, two heads, one layer.
_CONFIG = {
    "architectures": ["VitPoseForPoseEstimation"],
    "model_type": "vitpose",
    "backbone_config": {
        "model_type": "vitpose_backbone",
        "image_size": [32, 16],
        "patch_size": [16, 16],
        "hidden_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "mlp_ratio": 2,
        "out_indices": [1],
        "out_features": ["stage1"],
        "part_features": 0,
    },
    "id2label": {"0": "Nose", "1": "L_Eye", "2": "R_Eye"},
    "label2id": {"Nose": 0, "L_Eye": 1, "R_Eye": 2},
    "scale_factor": 4,
    "use_simple_decoder": True,
}
_PROCESSOR = {
    "do_affine_transform": True,
    "do_normalize": True,
    "do_rescale": True,
    "image_mean": [0.485, 0.456, 0.406],
    "image_processor_type": "VitPoseImageProcessor",
    "image_std": [0.229, 0.224, 0.225],
    "normalize_factor": 200.0,
    "rescale_factor": 0.00392156862745098,
    "size": {"height": 32, "width": 16},
}


def _header() -> dict[str, tuple[str, tuple[int, ...]]]:
    header = {
        "backbone.embeddings.patch_embeddings.projection.weight": ("F32", (8, 3, 16, 16)),
        "backbone.embeddings.patch_embeddings.projection.bias": ("F32", (8,)),
        "backbone.embeddings.position_embeddings": ("F32", (1, 3, 8)),
        "backbone.layernorm.weight": ("F32", (8,)),
        "backbone.layernorm.bias": ("F32", (8,)),
        "head.conv.weight": ("F32", (3, 8, 3, 3)),
        "head.conv.bias": ("F32", (3,)),
    }
    layer = "backbone.encoder.layer.0"
    for name in (
        "attention.attention.query",
        "attention.attention.key",
        "attention.attention.value",
        "attention.output.dense",
    ):
        header[f"{layer}.{name}.weight"] = ("F32", (8, 8))
        header[f"{layer}.{name}.bias"] = ("F32", (8,))
    for name in ("layernorm_before", "layernorm_after"):
        header[f"{layer}.{name}.weight"] = ("F32", (8,))
        header[f"{layer}.{name}.bias"] = ("F32", (8,))
    header[f"{layer}.mlp.fc1.weight"] = ("F32", (16, 8))
    header[f"{layer}.mlp.fc1.bias"] = ("F32", (16,))
    header[f"{layer}.mlp.fc2.weight"] = ("F32", (8, 16))
    header[f"{layer}.mlp.fc2.bias"] = ("F32", (8,))
    return header


def _write(directory: Path, config: dict | str, processor: dict | str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for name, value in (("config.json", config), ("preprocessor_config.json", processor)):
        (directory / name).write_text(
            value if isinstance(value, str) else json.dumps(value), encoding="utf-8"
        )
    return directory


def test_the_published_layout_resolves(tmp_path: Path) -> None:
    cfg = config_module.resolve(_write(tmp_path, _CONFIG, _PROCESSOR), _header())
    assert (cfg.image_height, cfg.image_width, cfg.grid, cfg.heatmap_size) == (
        32,
        16,
        (2, 1),
        (8, 4),
    )
    assert (cfg.hidden_size, cfg.num_heads, cfg.head_dim, cfg.intermediate_size) == (8, 2, 4, 16)
    assert cfg.labels == ("Nose", "L_Eye", "R_Eye") and cfg.layer_norm_eps == 1e-12
    metadata = config_module.runtime_metadata(cfg)
    assert metadata == {
        "input_height": 32,
        "input_width": 16,
        "rescale_factor": 0.00392156862745098,
        "image_mean": [0.485, 0.456, 0.406],
        "image_std": [0.229, 0.224, 0.225],
        "box_padding": 1.25,
        "heatmap_height": 8,
        "heatmap_width": 4,
        "dark_kernel": 11,
        "labels": ["Nose", "L_Eye", "R_Eye"],
    }


def _backbone(**changes) -> dict:
    config = copy.deepcopy(_CONFIG)
    config["backbone_config"].update(changes)
    return config


def _model(**changes) -> dict:
    config = copy.deepcopy(_CONFIG)
    config.update(changes)
    return config


@pytest.mark.parametrize(
    "config,error",
    [
        (_model(model_type="vit"), NotImplementedError),
        (_model(architectures=["VitPoseBackbone"]), NotImplementedError),
        (_model(use_simple_decoder=False), NotImplementedError),  # the deconvolution head
        (_model(use_simple_decoder=None), ValueError),  # an explicit null is not the default
        (_model(use_simple_decoder=1), ValueError),
        (_model(backbone="vit_base_patch16_224"), NotImplementedError),
        (_model(use_timm_backbone=True), NotImplementedError),
        (_model(backbone_config=None), NotImplementedError),  # the reference would read stage 4
        (_model(scale_factor=4.0), ValueError),
        (_model(scale_factor=0), ValueError),
        (_model(scale_factor=1), NotImplementedError),  # 2 x 1 heatmaps: the runtime needs 2 x 2
        (_backbone(model_type="vit"), NotImplementedError),
        (_backbone(out_indices=[0]), NotImplementedError),
        (_backbone(out_indices=None, out_features=["stem"]), NotImplementedError),
        (_backbone(out_indices=[True]), ValueError),
        (_backbone(num_experts=6), NotImplementedError),  # ViTPose+ mixture of experts
        (_backbone(hidden_act="gelu_new"), NotImplementedError),
        (_backbone(qkv_bias=False), NotImplementedError),
        (_backbone(num_channels=1), NotImplementedError),
        (_backbone(layer_norm_eps=0.0), ValueError),
        (_backbone(layer_norm_eps=1e-50), ValueError),  # underflows float32
        (_backbone(num_attention_heads=3), ValueError),
        (_backbone(hidden_size=8.0), ValueError),
        (
            _backbone(image_size=[36, 16]),
            NotImplementedError,
        ),  # the padded convolution makes 3 rows, not 2
        (_backbone(image_size=[32]), ValueError),
        (_backbone(patch_size=None), ValueError),
        (
            _model(id2label={"0": "Nose", "2": "R_Eye", "1": "L_Eye", "3": "L_Ear"}),
            ValueError,
        ),  # head has 3
        (_model(id2label={"0": "Nose", "1": "Nose", "2": "R_Eye"}), ValueError),
        (_model(id2label={"0": "Nose", "1": "L_Eye", "3": "R_Eye"}), ValueError),
        (_model(id2label={"0": "Nose", "1": " L_Eye", "2": "R_Eye"}), ValueError),
        (_model(id2label={}), ValueError),
    ],
)
def test_configurations_the_graph_does_not_build_are_rejected(
    config, error, tmp_path: Path
) -> None:
    with pytest.raises(error):
        config_module.resolve(_write(tmp_path, config, _PROCESSOR), _header())


def _processor(**changes) -> dict:
    processor = copy.deepcopy(_PROCESSOR)
    processor.update(changes)
    return processor


@pytest.mark.parametrize(
    "processor,error",
    [
        (_processor(size={"height": 64, "width": 16}), NotImplementedError),
        (_processor(size={"shortest_edge": 32}), ValueError),
        (_processor(normalize_factor=100.0), NotImplementedError),
        (_processor(do_affine_transform=False), NotImplementedError),
        (_processor(do_rescale=False), NotImplementedError),
        (_processor(do_normalize=None), ValueError),
        (_processor(image_std=[0.229, 0.0, 0.225]), ValueError),
        # Positive float32 values whose folded normalization overflows: 255 / (1e-40 * 255) is infinite.
        (_processor(image_std=[0.229, 0.224, 1e-40]), ValueError),
        (_processor(rescale_factor=1e-40), ValueError),
        # Values a little above FLT_MAX: numpy rounds them to FLT_MAX, the runtime refuses them.
        (_processor(rescale_factor=1.0, image_std=[0.229, 0.224, 3.4028235e38]), ValueError),
        (
            _processor(
                rescale_factor=1 / 3.4028235e38,
                image_mean=[0.0, 0.0, 0.0],
                image_std=[1.0, 1.0, 1.0],
            ),
            ValueError,
        ),
        (_processor(image_std=[0.229, 0.224]), ValueError),
        (_processor(image_mean=[0.485, 0.456, 1e300]), ValueError),
        (_processor(rescale_factor="0.0039"), ValueError),
        (_processor(rescale_factor=-0.0039), ValueError),
        (_processor(image_processor_type="ViTImageProcessor"), NotImplementedError),
        (_processor(do_center_crop=True), NotImplementedError),
    ],
)
def test_processors_the_runtime_does_not_reproduce_are_rejected(
    processor, error, tmp_path: Path
) -> None:
    with pytest.raises(error):
        config_module.resolve(_write(tmp_path, _CONFIG, processor), _header())


def test_repeated_json_keys_are_rejected(tmp_path: Path) -> None:
    repeated = json.dumps(_CONFIG)[:-1] + ', "scale_factor": 2}'
    with pytest.raises(ValueError, match="repeats the key"):
        config_module.resolve(_write(tmp_path / "config", repeated, _PROCESSOR), _header())
    repeated = json.dumps(_PROCESSOR)[:-1] + ', "rescale_factor": 1.0}'
    with pytest.raises(ValueError, match="repeats the key"):
        config_module.resolve(_write(tmp_path / "processor", _CONFIG, repeated), _header())


def test_the_layer_count_is_checked_against_the_header_first(tmp_path: Path, monkeypatch) -> None:
    # A billion declared layers must be refused from the header, before any per-layer table is sized from them.
    def sized(cfg):
        raise AssertionError(f"per-layer tensors sized for {cfg.num_layers} layers")

    monkeypatch.setattr(config_module, "_expected_tensors", sized)
    for layers in (2, 10**9):
        config = _backbone(
            num_hidden_layers=layers, out_indices=[layers], out_features=[f"stage{layers}"]
        )
        with pytest.raises(ValueError, match="checkpoint holds 1"):
            config_module.resolve(_write(tmp_path / str(layers), config, _PROCESSOR), _header())
    header = _header()
    header["backbone.encoder.layer.01.mlp.fc1.bias"] = ("F32", (16,))
    with pytest.raises(ValueError, match="malformed encoder layer"):
        config_module.resolve(_write(tmp_path / "malformed", _CONFIG, _PROCESSOR), header)
    header = {name.replace("layer.0.", "layer.1."): value for name, value in _header().items()}
    with pytest.raises(ValueError, match="numbered 0"):
        config_module.resolve(_write(tmp_path / "gap", _CONFIG, _PROCESSOR), header)


def test_tensors_must_match_the_configuration(tmp_path: Path) -> None:
    directory = _write(tmp_path, _CONFIG, _PROCESSOR)
    header = _header()
    del header["head.conv.bias"]
    with pytest.raises(ValueError, match="lacks 1 tensors"):
        config_module.resolve(directory, header)
    header = _header()
    header["backbone.encoder.layer.0.mlp.experts.0.weight"] = ("F32", (2, 16))
    with pytest.raises(NotImplementedError, match="does not use"):
        config_module.resolve(directory, header)
    header = _header()
    header["backbone.embeddings.position_embeddings"] = (
        "F32",
        (1, 4, 8),
    )  # a class row the graph lacks
    with pytest.raises(ValueError, match="position_embeddings"):
        config_module.resolve(directory, header)
    header = _header()
    header["head.conv.weight"] = ("F32", (17, 8, 3, 3))  # 17 heatmaps for 3 labels
    with pytest.raises(ValueError, match="head.conv.weight"):
        config_module.resolve(directory, header)
    header = _header()
    header["backbone.encoder.layer.0.mlp.fc1.weight"] = ("BF16", (16, 8))
    with pytest.raises(NotImplementedError, match="float32"):
        config_module.resolve(directory, header)


def _request(model_dir: Path, **changes):
    values = {
        "model_dir": model_dir,
        "output_path": model_dir / "unused.bundle",
        "family": "vitpose",
        "task": "image_text_to_points",
        "backend": "trt",
        "precision": "fp32",
        "quantization": None,
        "fp32_layers": (),
        "max_batch_size": 1,
        "tensor_parallel_size": 1,
        "context_parallel_size": 1,
        "dynamic_kv_cache": False,
        "video_num_frames": None,
        "max_sequence_length": None,
        "mtp_seq_len": None,
        "graph_transform": None,
        "image_height": None,
        "image_width": None,
        "verbose": False,
    }
    values.update(changes)
    return SimpleNamespace(**values)


class _Writer:
    def __init__(self) -> None:
        self.calls = []

    def set_header(self, **fields) -> None:
        self.calls.append(("header", fields))

    def add_bytes(self, name: str, data: bytes) -> None:
        self.calls.append(("bytes", name, data))

    def add_json(self, name: str, value) -> None:
        self.calls.append(("json", name, value))


@pytest.fixture
def stub_engine(monkeypatch):
    """The tiny checkpoint's header, no weights and a stand-in plan: build() runs everything but TensorRT."""

    built = []
    monkeypatch.setattr(model_module.weights_module, "read_header", lambda _: _header())
    monkeypatch.setattr(model_module.weights_module, "load", lambda _directory, cfg: {"cfg": cfg})

    def engine(weights, cfg, *, verbose=False):
        built.append((weights, cfg, verbose))
        return b"plan"

    monkeypatch.setattr(model_module, "build_engine", engine)
    return built


@pytest.mark.parametrize(
    "changes",
    [
        {},
        {"precision": "FP32", "quantization": "none", "verbose": True},
        {"image_height": 32, "image_width": 16},
    ],
)
def test_a_supported_build_request_writes_the_bundle(changes, tmp_path: Path, stub_engine) -> None:
    directory = _write(tmp_path, _CONFIG, _PROCESSOR)
    writer = _Writer()
    model_module.build(_request(directory, **changes), writer)
    cfg = config_module.resolve(directory, _header())
    assert stub_engine == [({"cfg": cfg}, cfg, bool(changes.get("verbose")))]
    assert writer.calls == [
        ("header", {"family": "vitpose", "task": "image_text_to_points", "backend": "trt"}),
        ("bytes", "engine.plan", b"plan"),
        ("json", "runtime.json", config_module.runtime_metadata(cfg)),
    ]


@pytest.mark.parametrize(
    "changes,error",
    [
        ({"task": "image_to_boxes"}, ValueError),
        ({"backend": "trt_rtx"}, ValueError),
        ({"precision": "fp16"}, NotImplementedError),
        ({"precision": "bf16"}, NotImplementedError),
        ({"quantization": "fp8"}, NotImplementedError),
        ({"fp32_layers": (0,)}, NotImplementedError),
        ({"max_batch_size": 2}, NotImplementedError),
        ({"tensor_parallel_size": 2}, NotImplementedError),
        ({"context_parallel_size": 2}, NotImplementedError),
        ({"dynamic_kv_cache": True}, NotImplementedError),
        ({"video_num_frames": 8}, NotImplementedError),
        ({"max_sequence_length": 193}, NotImplementedError),
        ({"mtp_seq_len": 2}, NotImplementedError),
        ({"image_height": 128}, NotImplementedError),
        ({"image_width": 96}, NotImplementedError),
    ],
)
def test_build_options_the_family_does_not_consume_are_rejected(
    changes, error, tmp_path: Path, stub_engine
) -> None:
    with pytest.raises(error):
        model_module.build(_request(_write(tmp_path, _CONFIG, _PROCESSOR), **changes), _Writer())
    assert stub_engine == [], "a rejected request must not build an engine"

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for SAM3 family plugin contract.

Trace: ARCH-FAM-001, UD-FAM-SAM3
Intent: Validate SAM3 model-card text-prompt family routing and text encoder weight mapping
Preconditions: Synthetic SAM3 config and text-tower safetensors are available
Postconditions: Plugin keeps SAM3 separate from legacy SAM and exposes the text-prompt PCS contract
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")


from tensorrt_model_connect.config import ModelConfig

safetensors = pytest.importorskip("safetensors.numpy")


RNG = np.random.RandomState(123)


def _rand(*shape: int) -> np.ndarray:
    return RNG.randn(*shape).astype(np.float32)


def _write_config(model_dir: Path, config: dict) -> None:
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (model_dir / "tokenizer.json").write_text(
        json.dumps(
            {
                "model": {
                    "type": "BPE",
                    "vocab": {
                        ("token_0token_1" if index == 2 else f"token_{index}"): index
                        for index in range(17)
                    },
                    "merges": ["token_0 token_1"],
                }
            }
        ),
        encoding="utf-8",
    )


def _write_safetensors(model_dir: Path, tensors: dict[str, np.ndarray]) -> None:
    safetensors.save_file(tensors, str(model_dir / "model.safetensors"))


def _write_processor_config(model_dir: Path) -> None:
    (model_dir / "processor_config.json").write_text(
        json.dumps(
            {
                "image_processor": {
                    "image_mean": [0.5, 0.5, 0.5],
                    "image_std": [0.5, 0.5, 0.5],
                    "size": {"height": 28, "width": 28},
                    "mask_size": {"height": 12, "width": 12},
                },
                "processor_class": "Sam3Processor",
                "target_size": 28,
            }
        ),
        encoding="utf-8",
    )


def _sam3_config() -> dict:
    return {
        "architectures": ["Sam3VideoModel"],
        "model_type": "sam3_video",
        "low_res_mask_size": 12,
        "detector_config": {
            "model_type": "sam3",
            "text_config": {
                "model_type": "clip_text_model",
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_attention_heads": 2,
                "num_hidden_layers": 2,
                "max_position_embeddings": 5,
                "bos_token_id": 101,
                "eos_token_id": 102,
                "pad_token_id": 0,
                "layer_norm_eps": 1e-5,
                "hidden_act": "gelu",
                "vocab_size": 17,
            },
            "vision_config": {
                "backbone_config": {
                    "model_type": "sam3_vit_model",
                    "image_size": 28,
                    "pretrain_image_size": 28,
                    "patch_size": 14,
                    "hidden_size": 8,
                    "intermediate_size": 16,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 2,
                    "window_size": 2,
                    "global_attn_indexes": [1],
                },
                "fpn_hidden_size": 4,
            },
            "detr_encoder_config": {
                "hidden_size": 4,
                "num_layers": 2,
                "num_attention_heads": 2,
                "intermediate_size": 8,
                "layer_norm_eps": 1e-6,
                "hidden_act": "relu",
            },
            "detr_decoder_config": {
                "hidden_size": 4,
                "num_layers": 3,
                "num_queries": 7,
                "num_attention_heads": 2,
                "intermediate_size": 8,
                "layer_norm_eps": 1e-6,
                "hidden_act": "relu",
            },
            "mask_decoder_config": {
                "hidden_size": 4,
                "num_attention_heads": 2,
                "num_upsampling_stages": 2,
                "layer_norm_eps": 1e-6,
            },
        },
    }


def _sam3_text_tensors(prefix: str = "detector_model.") -> dict[str, np.ndarray]:
    hidden = 8
    intermediate = 16
    vocab = 17
    seq = 5
    projected = 4
    tensors: dict[str, np.ndarray] = {
        f"{prefix}text_encoder.text_model.embeddings.token_embedding.weight": _rand(vocab, hidden),
        f"{prefix}text_encoder.text_model.embeddings.position_embedding.weight": _rand(seq, hidden),
        f"{prefix}text_encoder.text_model.final_layer_norm.weight": _rand(hidden),
        f"{prefix}text_encoder.text_model.final_layer_norm.bias": _rand(hidden),
        f"{prefix}text_projection.weight": _rand(projected, hidden),
        f"{prefix}text_projection.bias": _rand(projected),
    }
    for layer_idx in range(2):
        base = f"{prefix}text_encoder.text_model.encoder.layers.{layer_idx}"
        for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
            tensors[f"{base}.self_attn.{proj}.weight"] = _rand(hidden, hidden)
            tensors[f"{base}.self_attn.{proj}.bias"] = _rand(hidden)
        for norm in ("layer_norm1", "layer_norm2"):
            tensors[f"{base}.{norm}.weight"] = _rand(hidden)
            tensors[f"{base}.{norm}.bias"] = _rand(hidden)
        tensors[f"{base}.mlp.fc1.weight"] = _rand(intermediate, hidden)
        tensors[f"{base}.mlp.fc1.bias"] = _rand(intermediate)
        tensors[f"{base}.mlp.fc2.weight"] = _rand(hidden, intermediate)
        tensors[f"{base}.mlp.fc2.bias"] = _rand(hidden)
    return tensors


def _sam3_vision_tensors(prefix: str = "detector_model.") -> dict[str, np.ndarray]:
    hidden = 8
    intermediate = 16
    fpn = 4
    grid = 2
    patch = 14
    tensors: dict[str, np.ndarray] = {
        f"{prefix}vision_encoder.backbone.embeddings.patch_embeddings.projection.weight": _rand(
            hidden, 3, patch, patch
        ),
        f"{prefix}vision_encoder.backbone.embeddings.position_embeddings": _rand(
            1, grid * grid, hidden
        ),
        f"{prefix}vision_encoder.backbone.layer_norm.weight": _rand(hidden),
        f"{prefix}vision_encoder.backbone.layer_norm.bias": _rand(hidden),
    }
    for layer_idx in range(2):
        base = f"{prefix}vision_encoder.backbone.layers.{layer_idx}"
        for norm in ("layer_norm1", "layer_norm2"):
            tensors[f"{base}.{norm}.weight"] = _rand(hidden)
            tensors[f"{base}.{norm}.bias"] = _rand(hidden)
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            tensors[f"{base}.attention.{proj}.weight"] = _rand(hidden, hidden)
            tensors[f"{base}.attention.{proj}.bias"] = _rand(hidden)
        tensors[f"{base}.mlp.fc1.weight"] = _rand(intermediate, hidden)
        tensors[f"{base}.mlp.fc1.bias"] = _rand(intermediate)
        tensors[f"{base}.mlp.fc2.weight"] = _rand(hidden, intermediate)
        tensors[f"{base}.mlp.fc2.bias"] = _rand(hidden)

    base0 = f"{prefix}vision_encoder.neck.fpn_layers.0"
    tensors[f"{base0}.scale_layers.0.weight"] = _rand(hidden, hidden // 2, 2, 2)
    tensors[f"{base0}.scale_layers.0.bias"] = _rand(hidden // 2)
    tensors[f"{base0}.scale_layers.2.weight"] = _rand(hidden // 2, hidden // 4, 2, 2)
    tensors[f"{base0}.scale_layers.2.bias"] = _rand(hidden // 4)
    tensors[f"{base0}.proj1.weight"] = _rand(fpn, hidden // 4, 1, 1)
    tensors[f"{base0}.proj1.bias"] = _rand(fpn)
    tensors[f"{base0}.proj2.weight"] = _rand(fpn, fpn, 3, 3)
    tensors[f"{base0}.proj2.bias"] = _rand(fpn)

    base1 = f"{prefix}vision_encoder.neck.fpn_layers.1"
    tensors[f"{base1}.scale_layers.0.weight"] = _rand(hidden, hidden // 2, 2, 2)
    tensors[f"{base1}.scale_layers.0.bias"] = _rand(hidden // 2)
    tensors[f"{base1}.proj1.weight"] = _rand(fpn, hidden // 2, 1, 1)
    tensors[f"{base1}.proj1.bias"] = _rand(fpn)
    tensors[f"{base1}.proj2.weight"] = _rand(fpn, fpn, 3, 3)
    tensors[f"{base1}.proj2.bias"] = _rand(fpn)

    base2 = f"{prefix}vision_encoder.neck.fpn_layers.2"
    tensors[f"{base2}.proj1.weight"] = _rand(fpn, hidden, 1, 1)
    tensors[f"{base2}.proj1.bias"] = _rand(fpn)
    tensors[f"{base2}.proj2.weight"] = _rand(fpn, fpn, 3, 3)
    tensors[f"{base2}.proj2.bias"] = _rand(fpn)
    return tensors


def _linear_tensors(
    tensors: dict[str, np.ndarray], prefix: str, in_size: int, out_size: int
) -> None:
    tensors[f"{prefix}.weight"] = _rand(out_size, in_size)
    tensors[f"{prefix}.bias"] = _rand(out_size)


def _norm_tensors(tensors: dict[str, np.ndarray], prefix: str, hidden: int) -> None:
    tensors[f"{prefix}.weight"] = _rand(hidden)
    tensors[f"{prefix}.bias"] = _rand(hidden)


def _attention_tensors(tensors: dict[str, np.ndarray], prefix: str, hidden: int) -> None:
    for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
        _linear_tensors(tensors, f"{prefix}.{proj}", hidden, hidden)


def _sam3_core_tensors(prefix: str = "detector_model.") -> dict[str, np.ndarray]:
    hidden = 4
    intermediate = 8
    queries = 7
    heads = 2
    tensors: dict[str, np.ndarray] = {}

    tensors[f"{prefix}geometry_encoder.cls_embed.weight"] = _rand(1, hidden)
    _linear_tensors(tensors, f"{prefix}geometry_encoder.final_proj", hidden, hidden)
    _norm_tensors(tensors, f"{prefix}geometry_encoder.prompt_layer_norm", hidden)
    for layer_idx in range(3):
        base = f"{prefix}geometry_encoder.layers.{layer_idx}"
        for norm in ("layer_norm1", "layer_norm2", "layer_norm3"):
            _norm_tensors(tensors, f"{base}.{norm}", hidden)
        _attention_tensors(tensors, f"{base}.self_attn", hidden)
        _attention_tensors(tensors, f"{base}.cross_attn", hidden)
        _linear_tensors(tensors, f"{base}.mlp.fc1", hidden, intermediate)
        _linear_tensors(tensors, f"{base}.mlp.fc2", intermediate, hidden)
    _norm_tensors(tensors, f"{prefix}geometry_encoder.output_layer_norm", hidden)

    for layer_idx in range(2):
        base = f"{prefix}detr_encoder.layers.{layer_idx}"
        for norm in ("layer_norm1", "layer_norm2", "layer_norm3"):
            _norm_tensors(tensors, f"{base}.{norm}", hidden)
        _attention_tensors(tensors, f"{base}.self_attn", hidden)
        _attention_tensors(tensors, f"{base}.cross_attn", hidden)
        _linear_tensors(tensors, f"{base}.mlp.fc1", hidden, intermediate)
        _linear_tensors(tensors, f"{base}.mlp.fc2", intermediate, hidden)

    for layer_idx in range(3):
        base = f"{prefix}detr_decoder.layers.{layer_idx}"
        for norm in (
            "self_attn_layer_norm",
            "text_cross_attn_layer_norm",
            "vision_cross_attn_layer_norm",
            "mlp_layer_norm",
        ):
            _norm_tensors(tensors, f"{base}.{norm}", hidden)
        _attention_tensors(tensors, f"{base}.self_attn", hidden)
        _attention_tensors(tensors, f"{base}.text_cross_attn", hidden)
        _attention_tensors(tensors, f"{base}.vision_cross_attn", hidden)
        _linear_tensors(tensors, f"{base}.mlp.fc1", hidden, intermediate)
        _linear_tensors(tensors, f"{base}.mlp.fc2", intermediate, hidden)

    _norm_tensors(tensors, f"{prefix}detr_decoder.output_layer_norm", hidden)
    tensors[f"{prefix}detr_decoder.query_embed.weight"] = _rand(queries, hidden)
    tensors[f"{prefix}detr_decoder.reference_points.weight"] = _rand(queries, 4)
    tensors[f"{prefix}detr_decoder.presence_token.weight"] = _rand(1, hidden)
    for head in ("box_head", "presence_head"):
        output = 4 if head == "box_head" else 1
        _linear_tensors(tensors, f"{prefix}detr_decoder.{head}.layer1", hidden, hidden)
        _linear_tensors(tensors, f"{prefix}detr_decoder.{head}.layer2", hidden, hidden)
        _linear_tensors(tensors, f"{prefix}detr_decoder.{head}.layer3", hidden, output)
    _norm_tensors(tensors, f"{prefix}detr_decoder.presence_layer_norm", hidden)
    _linear_tensors(tensors, f"{prefix}detr_decoder.ref_point_head.layer1", hidden * 2, hidden)
    _linear_tensors(tensors, f"{prefix}detr_decoder.ref_point_head.layer2", hidden, hidden)
    for axis in ("x", "y"):
        _linear_tensors(tensors, f"{prefix}detr_decoder.box_rpb_embed_{axis}.layer1", 2, hidden)
        _linear_tensors(tensors, f"{prefix}detr_decoder.box_rpb_embed_{axis}.layer2", hidden, heads)

    _linear_tensors(tensors, f"{prefix}dot_product_scoring.text_mlp.layer1", hidden, intermediate)
    _linear_tensors(tensors, f"{prefix}dot_product_scoring.text_mlp.layer2", intermediate, hidden)
    _norm_tensors(tensors, f"{prefix}dot_product_scoring.text_mlp_out_norm", hidden)
    _linear_tensors(tensors, f"{prefix}dot_product_scoring.text_proj", hidden, hidden)
    _linear_tensors(tensors, f"{prefix}dot_product_scoring.query_proj", hidden, hidden)

    for layer_idx in range(2):
        base = f"{prefix}mask_decoder.pixel_decoder"
        tensors[f"{base}.conv_layers.{layer_idx}.weight"] = _rand(hidden, hidden, 3, 3)
        tensors[f"{base}.conv_layers.{layer_idx}.bias"] = _rand(hidden)
        _norm_tensors(tensors, f"{base}.norms.{layer_idx}", hidden)
    for layer_idx in range(3):
        _linear_tensors(
            tensors, f"{prefix}mask_decoder.mask_embedder.layers.{layer_idx}", hidden, hidden
        )
    tensors[f"{prefix}mask_decoder.instance_projection.weight"] = _rand(hidden, hidden, 1, 1)
    tensors[f"{prefix}mask_decoder.instance_projection.bias"] = _rand(hidden)
    _attention_tensors(tensors, f"{prefix}mask_decoder.prompt_cross_attn", hidden)
    _norm_tensors(tensors, f"{prefix}mask_decoder.prompt_cross_attn_norm", hidden)
    return tensors


def test_sam3_matches_sam3_video_not_legacy_sam() -> None:
    from tensorrt_model_connect.families import find_model
    from tensorrt_model_connect.families.sam3 import model

    assert model.matches("sam3")
    assert model.matches("sam3_video")
    assert not model.matches("sam")
    assert not model.matches("qwen3")
    assert model.runtime_strategy == "sam3_prompted_segmentation"
    assert model.requires_tokenizer is True
    assert find_model("sam3").name == "sam3"
    assert find_model("sam3_video").name == "sam3"


@pytest.mark.parametrize(
    ("tokenizer_payload", "message"),
    [
        (None, "requires tokenizer.json"),
        ("not-json", "not valid UTF-8 JSON"),
        (json.dumps({"model": {"type": "BPE"}}), "non-empty BPE vocab"),
    ],
)
def test_sam3_load_weights_fails_before_engine_build_for_incomplete_tokenizer(
    tmp_path: Path,
    tokenizer_payload: str | None,
    message: str,
) -> None:
    from tensorrt_model_connect.families.sam3 import model as plugin

    _write_config(tmp_path, _sam3_config())
    tokenizer_path = tmp_path / "tokenizer.json"
    if tokenizer_payload is None:
        tokenizer_path.unlink()
    else:
        tokenizer_path.write_text(tokenizer_payload, encoding="utf-8")
    config = ModelConfig.from_dir(tmp_path)

    with pytest.raises(RuntimeError, match=message):
        plugin.load_weights(str(tmp_path), config)


def test_sam3_load_weights_maps_text_encoder_prefix(tmp_path: Path) -> None:
    from tensorrt_model_connect.families.sam3 import model as plugin

    _write_config(tmp_path, _sam3_config())
    _write_safetensors(tmp_path, _sam3_text_tensors())
    config = ModelConfig.from_dir(tmp_path)

    weights = plugin.load_weights(str(tmp_path), config)

    assert "text_model.embeddings.token_embedding.weight" in weights
    assert "text_model.encoder.layers.0.self_attn.q_proj.weight" in weights
    assert "text_model.encoder.layers.1.mlp.fc2.weight" in weights
    assert weights["text_projection.weight"].shape == (8, 4)
    assert config.raw["_sam3_config"]["_text_projection_dim"] == 4


def test_sam3_segmentation_config_marks_text_prompt_variant(tmp_path: Path) -> None:
    from tensorrt_model_connect.families.sam3 import model as plugin

    _write_config(tmp_path, _sam3_config())
    _write_processor_config(tmp_path)
    _write_safetensors(tmp_path, _sam3_text_tensors(prefix=""))
    config = ModelConfig.from_dir(tmp_path)
    plugin.load_weights(str(tmp_path), config)

    seg_cfg = plugin.get_segmentation_config(config)

    assert seg_cfg["prompted_segmentation_variant"] == "sam3_text_prompt_pcs"
    assert seg_cfg["tokenizer_add_special_tokens"] == 1
    assert seg_cfg["tokenizer_special_prefix_ids"] == [101]
    assert seg_cfg["tokenizer_special_suffix_ids"] == [102]
    assert seg_cfg["sam3_text_max_position_embeddings"] == 5
    assert seg_cfg["sam3_text_projection_dim"] == 4
    assert seg_cfg["sam3_text_pad_token_id"] == 102
    assert seg_cfg["sam3_num_queries"] == 7
    assert seg_cfg["sam3_score_threshold"] == 0.5
    assert seg_cfg["sam3_mask_threshold"] == 0.5
    assert seg_cfg["sam3_video_tracking_supported"] is False
    assert seg_cfg["sam3_vision_pretrain_image_size"] == 28
    assert seg_cfg["sam3_vision_intermediate_size"] == 16
    assert seg_cfg["image_mean"] == [0.5, 0.5, 0.5]
    assert seg_cfg["image_std"] == [0.5, 0.5, 0.5]
    assert seg_cfg["input_image_h"] == 28


def test_sam3_segmentation_config_falls_back_to_meta_processor_defaults(
    tmp_path: Path,
) -> None:
    from tensorrt_model_connect.families.sam3 import model as plugin

    _write_config(tmp_path, _sam3_config())
    _write_safetensors(tmp_path, _sam3_text_tensors(prefix=""))
    config = ModelConfig.from_dir(tmp_path)
    plugin.load_weights(str(tmp_path), config)

    seg_cfg = plugin.get_segmentation_config(config)

    assert seg_cfg["image_mean"] == [0.5, 0.5, 0.5]
    assert seg_cfg["image_std"] == [0.5, 0.5, 0.5]


def test_sam3_build_engine_delegates_to_text_encoder_builder(tmp_path: Path, monkeypatch) -> None:
    from tensorrt_model_connect.families.sam3 import model as plugin

    _write_config(tmp_path, _sam3_config())
    _write_safetensors(tmp_path, _sam3_text_tensors())
    config = ModelConfig.from_dir(tmp_path)
    weights = plugin.load_weights(str(tmp_path), config)

    calls: list[dict[str, object]] = []

    def _fake_build(_weights, **kwargs):
        calls.append(kwargs)
        return b"sam3-text-plan"

    fake_module = types.SimpleNamespace(build_sam3_text_encoder_engine=_fake_build)
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_model_connect.families.sam3.text_encoder_builder",
        fake_module,
    )

    plan = plugin.build_engine(config, weights, max_cache_length=1, precision="fp16")

    assert plan == b"sam3-text-plan"
    assert calls[0]["hidden_size"] == 8
    assert calls[0]["projected_size"] == 4
    assert calls[0]["max_seq_len"] == 5
    assert calls[0]["precision"] == "fp16"


def test_sam3_build_vision_engine_delegates_to_vision_builder(tmp_path: Path, monkeypatch) -> None:
    from tensorrt_model_connect.families.sam3 import model as plugin

    _write_config(tmp_path, _sam3_config())
    tensors = _sam3_text_tensors()
    tensors.update(_sam3_vision_tensors())
    _write_safetensors(tmp_path, tensors)
    config = ModelConfig.from_dir(tmp_path)
    weights = plugin.load_weights(str(tmp_path), config)

    calls: list[dict[str, object]] = []

    def _fake_build(vision_weights, **kwargs):
        calls.append({"kwargs": kwargs, "keys": set(vision_weights.keys())})
        return b"sam3-vision-plan"

    fake_module = types.SimpleNamespace(build_sam3_vision_encoder_engine=_fake_build)
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_model_connect.families.sam3.vision_encoder_builder",
        fake_module,
    )

    plan = plugin.build_vision_engine(str(tmp_path), config, weights, precision="fp16")

    assert plan == b"sam3-vision-plan"
    assert calls[0]["kwargs"]["image_size"] == 28
    assert calls[0]["kwargs"]["pretrain_image_size"] == 28
    assert calls[0]["kwargs"]["hidden_size"] == 8
    assert calls[0]["kwargs"]["intermediate_size"] == 16
    assert calls[0]["kwargs"]["num_layers"] == 2
    assert calls[0]["kwargs"]["precision"] == "fp16"
    assert "vision.patch_embed.weight" in calls[0]["keys"]
    assert "vision.fpn.2.proj2.weight" in calls[0]["keys"]


def test_sam3_build_extra_engines_delegates_to_core_builder(tmp_path: Path, monkeypatch) -> None:
    from tensorrt_model_connect.families.sam3 import model as plugin

    _write_config(tmp_path, _sam3_config())
    tensors = _sam3_text_tensors()
    tensors.update(_sam3_core_tensors())
    _write_safetensors(tmp_path, tensors)
    config = ModelConfig.from_dir(tmp_path)
    config.raw["_model_dir"] = str(tmp_path)
    weights = plugin.load_weights(str(tmp_path), config)

    calls: list[dict[str, object]] = []

    def _fake_build(core_weights, **kwargs):
        calls.append({"kwargs": kwargs, "keys": set(core_weights.keys())})
        return b"sam3-core-plan"

    fake_module = types.SimpleNamespace(build_sam3_core_engine=_fake_build)
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_model_connect.families.sam3.core_builder",
        fake_module,
    )

    plans = plugin.build_extra_engines(config, weights, max_cache_length=1, precision="fp16")

    assert plans == {"sam3_core_engine_plan": b"sam3-core-plan"}
    assert calls[0]["kwargs"]["text_seq_len"] == 5
    assert calls[0]["kwargs"]["hidden_size"] == 4
    assert calls[0]["kwargs"]["fpn_shapes"] == ((8, 8), (4, 4), (2, 2))
    assert calls[0]["kwargs"]["num_queries"] == 7
    assert calls[0]["kwargs"]["detr_encoder_heads"] == 2
    assert calls[0]["kwargs"]["detr_decoder_intermediate_size"] == 8
    assert calls[0]["kwargs"]["geometry_encoder_layers"] == 3
    assert calls[0]["kwargs"]["geometry_encoder_heads"] == 2
    assert calls[0]["kwargs"]["geometry_encoder_intermediate_size"] == 8
    assert calls[0]["kwargs"]["geometry_encoder_layer_norm_eps"] == pytest.approx(1e-5)
    assert calls[0]["kwargs"]["precision"] == "fp16"
    assert "geometry_encoder.cls_embed.weight" in calls[0]["keys"]
    assert "geometry_encoder.layers.2.cross_attn.k_proj.weight" in calls[0]["keys"]
    assert "geometry_encoder.output_layer_norm.bias" in calls[0]["keys"]
    assert "detr_encoder.layers.0.self_attn.q_proj.weight" in calls[0]["keys"]
    assert "detr_decoder.layers.2.vision_cross_attn.o_proj.bias" in calls[0]["keys"]
    assert "mask_decoder.instance_projection.weight" in calls[0]["keys"]


def test_sam3_video_build_packages_all_tracker_plans(tmp_path: Path, monkeypatch) -> None:
    from tensorrt_model_connect.families.sam3 import model as plugin

    raw_config = _sam3_config()
    raw_config["tracker_config"] = {
        "model_type": "sam3_tracker_video",
        "image_size": 1008,
        "num_maskmem": 7,
        "max_object_pointers_in_encoder": 16,
    }
    _write_config(tmp_path, raw_config)
    tensors = _sam3_text_tensors()
    tensors.update(_sam3_core_tensors())
    _write_safetensors(tmp_path, tensors)
    config = ModelConfig.from_dir(tmp_path)
    config.raw["_model_dir"] = str(tmp_path)
    weights = plugin.load_weights(str(tmp_path), config)

    fake_core = types.SimpleNamespace(
        build_sam3_core_engine=lambda _weights, **_kwargs: b"sam3-core-plan"
    )
    tracker_calls: list[tuple[str, bool]] = []

    def _fake_tracker_build(
        model_dir: str,
        *,
        verbose: bool = False,
    ):
        tracker_calls.append((model_dir, verbose))
        return {
            "sam3_tracker_init_engine_plan": b"sam3-tracker-init-plan",
            "sam3_tracker_step_engine_plan": b"sam3-tracker-step-plan",
            "sam3_tracker_step_batch2_engine_plan": b"sam3-tracker-step-batch2-plan",
            "sam3_tracker_memory_engine_plan": b"sam3-tracker-memory-plan",
            "sam3_tracker_memory_batch2_engine_plan": b"sam3-tracker-memory-batch2-plan",
            "sam3_tracker_hard_memory_engine_plan": b"sam3-tracker-hard-memory-plan",
            "sam3_tracker_hard_memory_batch2_engine_plan": b"sam3-tracker-hard-memory-batch2-plan",
            "sam3_hard_mask_resize_engine_plan": b"sam3-hard-mask-resize-plan",
            "sam3_hard_mask_resize_batch2_engine_plan": b"sam3-hard-mask-resize-batch2-plan",
        }

    fake_tracker = types.SimpleNamespace(build_sam3_tracker_engines=_fake_tracker_build)
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_model_connect.families.sam3.core_builder",
        fake_core,
    )
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_model_connect.families.sam3.tracker_builder",
        fake_tracker,
    )

    plans = plugin.build_extra_engines(config, weights, max_cache_length=1, verbose=True)

    assert plans == {
        "sam3_core_engine_plan": b"sam3-core-plan",
        "sam3_tracker_init_engine_plan": b"sam3-tracker-init-plan",
        "sam3_tracker_step_engine_plan": b"sam3-tracker-step-plan",
        "sam3_tracker_step_batch2_engine_plan": b"sam3-tracker-step-batch2-plan",
        "sam3_tracker_memory_engine_plan": b"sam3-tracker-memory-plan",
        "sam3_tracker_memory_batch2_engine_plan": b"sam3-tracker-memory-batch2-plan",
        "sam3_tracker_hard_memory_engine_plan": b"sam3-tracker-hard-memory-plan",
        "sam3_tracker_hard_memory_batch2_engine_plan": b"sam3-tracker-hard-memory-batch2-plan",
        "sam3_hard_mask_resize_engine_plan": b"sam3-hard-mask-resize-plan",
        "sam3_hard_mask_resize_batch2_engine_plan": b"sam3-hard-mask-resize-batch2-plan",
    }
    assert tracker_calls == [(str(tmp_path), True)]
    assert all(not section.endswith(".pt2") for section in plans)
    assert all("aoti" not in section.lower() for section in plans)
    assert all("ffi" not in section.lower() for section in plans)
    assert plugin.get_segmentation_config(config)["sam3_video_tracking_supported"] is True
    tracking_config = plugin.get_segmentation_config(config)
    assert tracking_config["sam3_assoc_iou_threshold"] == 0.1
    assert tracking_config["sam3_tracker_assoc_iou_threshold"] == 0.5
    assert tracking_config["sam3_new_detection_threshold"] == 0.7
    assert tracking_config["sam3_detection_threshold"] == 0.5
    assert tracking_config["sam3_detection_nms_threshold"] == 0.1
    assert tracking_config["sam3_hotstart_delay"] == 15
    assert tracking_config["sam3_hotstart_unmatch_threshold"] == 8
    assert tracking_config["sam3_hotstart_duplicate_threshold"] == 8
    assert tracking_config["sam3_suppress_unmatched_only_within_hotstart"] is True
    assert tracking_config["sam3_initial_tracker_keep_alive"] == 30
    assert tracking_config["sam3_max_tracker_keep_alive"] == 30
    assert tracking_config["sam3_min_tracker_keep_alive"] == -1
    assert tracking_config["sam3_recondition_every_nth_frame"] == 16
    assert tracking_config["sam3_high_confidence_threshold"] == 0.8
    assert tracking_config["sam3_high_iou_threshold"] == 0.8
    assert tracking_config["sam3_overlap_suppression_threshold"] == 0.7
    assert tracking_config["sam3_num_mask_memory_frames"] == 7
    assert tracking_config["sam3_max_conditioning_frames"] == 4
    assert tracking_config["sam3_max_object_pointers"] == 16
    assert tracking_config["sam3_max_video_frames"] == 1024
    assert tracking_config["sam3_max_conditioning_pointers"] == 4
    assert tracking_config["sam3_max_pointer_inputs"] == 19

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Family identity, command, and checkpoint-integrity contracts."""

import json
import struct
from pathlib import Path
import numpy as np
import pytest
from tensorrt_model_connect.model_support import ModelMetadata, resolve_family
from families.flint.support import describe
from families.flint.weights import Weights


def test_exact_family_ownership():
    metadata = ModelMetadata(
        config={"model_type": "flint", "architectures": ["FlintForCausalLM"]}, model_index={}
    )
    family, support = resolve_family(metadata)
    assert family == "flint"
    assert support.tasks == ("text_generation",)
    assert support.default_task == "text_generation"
    assert support.default_precision == "bf16"
    assert describe(ModelMetadata(config={"model_type": "qwen3_moe"}, model_index={})) is None


def test_cli_routes_bf16_checkpoint(monkeypatch, tmp_path):
    from families.flint import cli

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "flint"}))
    calls = []
    monkeypatch.setattr(
        cli, "build_bundle", lambda request, output: calls.append((request, output))
    )
    assert (
        cli.build(model=str(tmp_path), output=tmp_path / "out.bundle", max_sequence_length=600) == 0
    )
    assert len(calls) == 1
    assert calls[0][0].precision == "bf16"
    assert calls[0][0].max_sequence_length == 600
    assert calls[0][1] == tmp_path / "out.bundle"


def make_checkpoint(root, offsets=(0, 4), payload=b"\x80\x3f\x00\x40"):
    header = json.dumps(
        {"x": {"dtype": "BF16", "shape": [2], "data_offsets": list(offsets)}}
    ).encode()
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"x": "part.safetensors"}})
    )
    (root / "part.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + payload)
    return Weights(root)


def test_weight_ranges_and_complete_mapping(tmp_path):
    weights = make_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="unmapped"):
        weights.finish()
    with weights.view("x") as view:
        assert bytes(view) == b"\x80\x3f\x00\x40"
    weights.finish()
    with (tmp_path / "copy").open("wb") as stream:
        weights.copy("x", stream)
    assert (tmp_path / "copy").read_bytes() == b"\x80\x3f\x00\x40"


def test_truncated_checkpoint_rejected(tmp_path):
    weights = make_checkpoint(tmp_path, payload=b"\x00\x00")
    with pytest.raises(ValueError, match="truncated"):
        weights.describe("x")


@pytest.mark.parametrize("offsets", [(-4, 0), (4, 0), (False, 4), (0.0, 4)])
def test_invalid_tensor_offsets_rejected_before_reading(tmp_path, offsets):
    weights = make_checkpoint(tmp_path, offsets=offsets)
    with pytest.raises(ValueError, match="safetensors offsets"):
        weights.describe("x")


def configuration():
    return {
        "model_type": "flint", "hidden_act": "silu", "attention_bias": False,
        "router_score_function": "sigmoid", "softmax_type": "learnable",
        "norm_topk_prob": True, "tie_word_embeddings": False,
        "hidden_size": 8, "intermediate_size": 16, "num_hidden_layers": 2,
        "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 4,
        "vocab_size": 32, "num_experts": 2, "num_experts_per_tok": 1,
        "moe_intermediate_size": 8, "shared_expert_intermediate_size": 8,
        "rms_norm_eps": 1e-6, "rope_theta": 10000, "sliding_window": 2,
        "max_position_embeddings": 32, "layer_types": ["sliding_attention", "full_attention"],
        "no_rope_layer_indices": [1], "mlp_only_layers": [], "eos_token_id": [2],
    }


def load_configuration(tmp_path, data):
    from families.flint.config import Config
    (tmp_path / "config.json").write_text(json.dumps(data))
    return Config.load(tmp_path)


@pytest.mark.parametrize("eos", [[], None, 2, [True], [-1], [32]])
def test_invalid_eos_is_rejected_before_building(tmp_path, eos):
    data = configuration()
    data["eos_token_id"] = eos
    with pytest.raises(ValueError, match="eos_token_id"):
        load_configuration(tmp_path, data)


def test_supported_eos_and_minimum_sliding_window(tmp_path):
    data = configuration()
    assert load_configuration(tmp_path, data).eos_token_id == (2,)


@pytest.mark.parametrize("window", [0, 1, True])
def test_sliding_window_must_retain_a_real_position(tmp_path, window):
    data = configuration()
    data["sliding_window"] = window
    with pytest.raises(ValueError, match="sliding_window"):
        load_configuration(tmp_path, data)


def test_full_attention_does_not_require_a_sliding_window(tmp_path):
    data = configuration()
    data.update(layer_types=["full_attention", "full_attention"],
                no_rope_layer_indices=[0, 1], sliding_window=1)
    assert load_configuration(tmp_path, data).sliding_window == 1


def test_one_changed_decode_token_fails_even_when_logits_are_close():
    from families.flint.tests.test_e2e import logit_metrics, assert_logit_metrics
    thresholds = json.loads((Path(__file__).parent / "thresholds/flint-bf16-logits.json").read_text())["threshold_overrides"]
    expected = np.tile([10.0, 9.999, 1.0, 0.0], (4, 1))
    actual = expected.copy()
    actual[-1, :2] = actual[-1, 1::-1]
    metrics = logit_metrics(actual, expected)
    assert max(metrics["relative_l2"]) < thresholds["max_relative_l2"]
    assert min(metrics["cosine_similarity"]) > thresholds["min_cosine_similarity"]
    assert metrics["top1_agreement"] == 0.75
    with pytest.raises(AssertionError):
        assert_logit_metrics(metrics, thresholds)
    assert_logit_metrics(logit_metrics(expected, expected), thresholds)


@pytest.mark.parametrize("name", ["present_k", "present_v"])
def test_new_cache_entry_error_cannot_be_diluted_by_old_positions(name):
    from families.flint.tests.test_graph import assert_layer_output
    reference = np.ones((2, 5, 4), np.float32)
    predicted = reference.copy()
    predicted[:, -1, :] *= 1.06
    assert np.linalg.norm(predicted - reference) / np.linalg.norm(reference) < 0.03
    with pytest.raises(AssertionError):
        assert_layer_output(name, predicted, reference)
    assert_layer_output(name, reference, reference)

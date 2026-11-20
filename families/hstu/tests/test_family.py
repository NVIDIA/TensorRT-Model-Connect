# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the HSTU configuration, synthetic checkpoint and parity judge
(no TensorRT, no torch)."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from families.hstu.config import load_config, parse_config
from families.hstu.tests.parity import LIMITS, compare, judge, parity_record, probe_logits
from families.hstu.weights import (
    REFERENCE_CONFIG,
    WEIGHTS_FILE,
    bf16_bits_to_fp32,
    fp32_to_bf16_bits,
    generate,
    load_checkpoint,
    tensor_layout,
    weights_record,
)


def test_reference_geometry_is_recsys_examples_kuairand_ranking():
    config = parse_config(copy.deepcopy(REFERENCE_CONFIG))
    assert (config.num_layers, config.num_attention_heads, config.head_dim) == (3, 4, 128)
    assert config.tokens == 6 + 2 * 4096 + 100 == 8298
    assert config.candidate_start == 8198
    assert config.scaling_seqlen == 8398
    assert config.uvqk_width == 2048


def test_position_ids_clamp_every_candidate_to_the_first_candidate():
    config = parse_config(copy.deepcopy(REFERENCE_CONFIG))
    ids = config.position_ids()
    assert ids[:8191] == list(range(8191))
    assert set(ids[8191:]) == {8191}  # N - C = 8198 > K - 1: the table clamps first
    small = parse_config(
        dict(REFERENCE_CONFIG, max_history_seqlen=16, max_num_candidates=8, scaling_seqlen=54)
    )
    assert small.position_ids() == list(range(38)) + [38] * 8


@pytest.mark.parametrize(
    "change, message",
    [
        ({"model_type": "hstu"}, "model_type"),
        ({"num_attention_heads": 3}, "hidden_size"),
        ({"prediction_head_arch": [512, 7]}, "num_tasks"),
        ({"prediction_head_act_type": "gelu"}, "biased ReLU"),
        ({"layernorm_epsilon": 0}, "layernorm_epsilon"),
        ({"max_history_seqlen": 0}, "max_history_seqlen"),
        ({"contextual_features": []}, "contextual_features"),
        ({"item_feature": {"name": "user_id", "vocab_size": 10}}, "unique"),
    ],
)
def test_config_refuses_what_the_graph_cannot_build(change, message):
    with pytest.raises(ValueError, match=message):
        parse_config(dict(copy.deepcopy(REFERENCE_CONFIG), **change))


def test_bf16_rounding_is_round_to_nearest_even():
    values = np.array([1.0, 1.0 + 2**-8, 1.0 + 3 * 2**-8, -2.5, 0.0], dtype=np.float32)
    assert bf16_bits_to_fp32(fp32_to_bf16_bits(values)).tolist() == [
        1.0,
        1.0,
        1.0 + 2**-6,
        -2.5,
        0.0,
    ]


def test_the_small_checkpoint_is_deterministic_and_holds_its_layout(tmp_path):
    first = generate(tmp_path / "a", small=True)
    second = generate(tmp_path / "b", small=True)
    assert (first / WEIGHTS_FILE).read_bytes() == (second / WEIGHTS_FILE).read_bytes()
    assert (first / "config.json").read_text() == (second / "config.json").read_text()
    config = load_config(first)
    assert config.raw["name"] == "hstu-kuairand-1k-synthetic-small"
    assert weights_record(config)["seed"] == 0
    tensors = load_checkpoint(first, config)
    assert [(name, tensors[name].shape) for name, _, _ in tensor_layout(config)] == [
        (name, shape) for name, shape, _ in tensor_layout(config)
    ]
    other = generate(tmp_path / "c", seed=1, small=True)
    assert (other / WEIGHTS_FILE).read_bytes() != (first / WEIGHTS_FILE).read_bytes()


def test_a_checkpoint_from_another_generator_is_refused(tmp_path):
    model_dir = generate(tmp_path / "ckpt", small=True)
    raw = json.loads((model_dir / "config.json").read_text())
    raw["weights"]["origin"] = "trained"
    (model_dir / "config.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="synthetic"):
        load_checkpoint(model_dir, load_config(model_dir))


def _result(logits: np.ndarray) -> dict:
    return {
        "scores": [
            {"question_id": f"task_{task}", "logits": logits[:, task].tolist()}
            for task in reversed(range(logits.shape[1]))
        ]
    }


def test_parity_judges_every_repeat_against_the_golden():
    rng = np.random.default_rng(0)
    golden_logits = rng.normal(size=(5, 3))
    golden = {"checkpoint": {"name": "x", "seed": 0}, "logits_fp32": golden_logits.tolist()}
    limits = {"max_abs": 1e-3, "min_task_cosine": 0.999}
    near, far = golden_logits + 5e-4, golden_logits + 5e-2
    assert np.array_equal(probe_logits(_result(golden_logits)), golden_logits)
    record = parity_record(
        {"results": [_result(near), _result(far)]},
        golden,
        limits,
        checkpoint={"name": "x", "seed": 0},
    )
    assert [repeat["passed"] for repeat in record["repeats"]] == [True, False]
    assert record["passed"] is False
    with pytest.raises(ValueError, match="golden"):
        parity_record(
            {"results": [_result(near)]}, golden, limits, checkpoint={"name": "x", "seed": 1}
        )
    assert set(LIMITS) == set(limits)


def test_compare_reports_the_worst_task():
    golden = np.array([[1.0, 0.0], [0.0, 1.0]])
    stats = compare(golden * [1.0, -1.0], golden)
    assert stats["max_abs"] == 2.0 and stats["min_task_cosine"] == pytest.approx(-1.0)
    assert not judge(stats, {"max_abs": 3.0, "min_task_cosine": 0.0})

"""Unit tests for tensorrt_model_connect.triattention_export."""

from __future__ import annotations

import json
import sys
import types

from tensorrt_model_connect.config import ModelConfig
from tensorrt_model_connect.triattention_export import export_triattention_stats_section


def test_export_rkv_stats_embeds_inv_freq_and_layer_stats(tmp_path, monkeypatch):
    stats_path = tmp_path / "triattention.pt"
    stats_path.write_bytes(b"stub")

    payload = {
        "metadata": {
            "head_dim": 4,
            "rope_style": "half",
            "sampled_heads": [[0, 0], [0, 1], [1, 2], [1, 3]],
        },
        "stats": {
            "layer00_head00": {
                "q_mean_real": [1.0, 2.0],
                "q_mean_imag": [0.5, 1.5],
                "q_abs_mean": [3.0, 4.0],
            },
            "layer00_head01": {
                "q_mean_real": [5.0, 6.0],
                "q_mean_imag": [2.5, 3.5],
                "q_abs_mean": [7.0, 8.0],
            },
            "layer01_head02": {
                "q_mean_real": [9.0, 10.0],
                "q_mean_imag": [4.5, 5.5],
                "q_abs_mean": [11.0, 12.0],
            },
            "layer01_head03": {
                "q_mean_real": [13.0, 14.0],
                "q_mean_imag": [6.5, 7.5],
                "q_abs_mean": [15.0, 16.0],
            },
        },
    }

    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(load=lambda *args, **kwargs: payload),
    )

    config = ModelConfig.from_json(
        json.dumps(
            {
                "model_type": "triattention_decoder",
                "hidden_size": 16,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "rope_scaling": {
                    "rope_type": "default",
                    "rope_theta": 1000000.0,
                },
            }
        )
    )

    exported = json.loads(
        export_triattention_stats_section(stats_path, config=config).decode("utf-8")
    )

    assert exported["num_attention_heads"] == 4
    assert exported["num_key_value_heads"] == 2
    assert exported["stats_head_count"] == 4
    assert exported["num_layers"] == 2
    assert exported["rope_theta"] == 1000000.0
    assert exported["inv_freq"] == [1.0, 0.001]
    assert exported["sampled_heads"] == [[0, 0], [0, 1], [1, 2], [1, 3]]

    layer0 = exported["layer_stats"]["0"]
    layer1 = exported["layer_stats"]["1"]
    assert layer0["q_mean_real"][0] == [1.0, 2.0]
    assert layer0["q_mean_real"][1] == [5.0, 6.0]
    assert layer0["q_mean_real"][2] == [0.0, 0.0]
    assert layer0["q_mean_real"][3] == [0.0, 0.0]
    assert layer0["q_mean_imag"][0] == [0.5, 1.5]
    assert layer0["q_mean_imag"][1] == [2.5, 3.5]
    assert layer0["q_abs_mean"][0] == [3.0, 4.0]
    assert layer0["q_abs_mean"][1] == [7.0, 8.0]
    assert layer1["q_mean_real"][0] == [0.0, 0.0]
    assert layer1["q_mean_real"][1] == [0.0, 0.0]
    assert layer1["q_mean_real"][2] == [9.0, 10.0]
    assert layer1["q_mean_real"][3] == [13.0, 14.0]
    assert layer1["q_mean_imag"][2] == [4.5, 5.5]
    assert layer1["q_mean_imag"][3] == [6.5, 7.5]
    assert layer1["q_abs_mean"][2] == [11.0, 12.0]
    assert layer1["q_abs_mean"][3] == [15.0, 16.0]
    assert exported["stats"]["layer00_head00"]["q_mean_real"] == [1.0, 2.0]
    assert exported["stats"]["layer00_head01"]["q_mean_real"] == [5.0, 6.0]
    assert exported["stats"]["layer01_head02"]["q_mean_real"] == [9.0, 10.0]
    assert exported["stats"]["layer01_head03"]["q_mean_real"] == [13.0, 14.0]
    assert layer0["freq_scale_sq"] == [[1.0, 1.0]] * 4

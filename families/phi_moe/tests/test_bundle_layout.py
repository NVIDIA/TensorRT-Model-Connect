# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phi-MoE bundle layout: single-GPU bundles carry a batched prefill engine (GPU-free)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for Phi-MoE builder tests")

from families.phi_moe import model  # noqa: E402


@pytest.mark.parametrize("tensor_parallel_size", [1, 2])
def test_build_writes_a_batched_prefill_engine_for_single_gpu_bundles(
    monkeypatch, tmp_path: Path, tensor_parallel_size: int
) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "phimoe",
                "vocab_size": 32,
                "hidden_size": 16,
                "intermediate_size": 4,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "max_position_embeddings": 4096,
                "bos_token_id": 1,
                "eos_token_id": 2,
                "pad_token_id": 0,
                "num_local_experts": 2,
                "num_experts_per_tok": 2,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{{ messages }}"}), encoding="utf-8"
    )
    calls = []
    monkeypatch.setattr(model._PhiMoEModel, "load_weights", lambda self, model_dir, config: {})

    def build_engine(self, config, weights, max_cache_length, **options):
        calls.append(("decode", max_cache_length, options["precision"]))
        return b"decode"

    def build_prefill_engine(self, config, weights, max_cache_length, **options):
        calls.append(("prefill", max_cache_length, options["precision"]))
        return b"prefill"

    monkeypatch.setattr(model._PhiMoEModel, "build_engine", build_engine)
    monkeypatch.setattr(model._PhiMoEModel, "build_prefill_engine", build_prefill_engine)
    sections: dict[str, object] = {}
    writer = SimpleNamespace(
        set_header=lambda **header: None,
        add_bytes=lambda name, data: sections.__setitem__(name, data),
        add_json=lambda name, value: sections.__setitem__(name, value),
    )
    request = SimpleNamespace(
        dynamic_kv_cache=False,
        image_height=None,
        image_width=None,
        video_num_frames=None,
        max_batch_size=1,
        context_parallel_size=1,
        task="text_generation",
        model_dir=tmp_path,
        precision="fp16",
        max_sequence_length=64,
        quantization=None,
        fp32_layers=(),
        mtp_seq_len=None,
        tensor_parallel_size=tensor_parallel_size,
        verbose=False,
        backend="tensorrt",
    )

    model.build(request, writer)

    runtime = sections["runtime.json"]
    if tensor_parallel_size == 1:
        assert calls == [("decode", 64, "fp16"), ("prefill", 64, "fp16")]
        assert sections["engine.plan"] == b"decode" and sections["prefill.plan"] == b"prefill"
        assert runtime["decoder_engine_layout"] == "split"
    else:
        assert calls == [("decode", 64, "fp16")] * tensor_parallel_size
        assert "prefill.plan" not in sections
        assert runtime["decoder_engine_layout"] == "dual_profile"

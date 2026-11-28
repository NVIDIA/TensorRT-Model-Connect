# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from io import BytesIO
from types import SimpleNamespace

import pytest

from families.ernie4_5_moe import model
from families.ernie4_5_moe.streaming import PlanWriter
from tensorrt_model_connect import BuildRequest


@pytest.mark.parametrize("eos", [None, [2], True, -1, 32])
def test_unsupported_eos_fails_before_engine_build(tmp_path, monkeypatch, eos):
    from transformers import Ernie4_5_MoeConfig

    cfg = dict(
        hidden_size=8,
        num_attention_heads=2,
        num_hidden_layers=1,
        moe_layer_interval=1,
        moe_layer_start_index=0,
        moe_layer_end_index=0,
        model_type="ernie4_5_moe",
        use_bias=False,
        hidden_act="silu",
        max_position_embeddings=128,
        vocab_size=32,
        eos_token_id=eos,
    )
    reference = SimpleNamespace(num_experts=2, num_experts_per_tok=1, to_dict=lambda: cfg.copy())
    monkeypatch.setattr(Ernie4_5_MoeConfig, "from_pretrained", lambda directory: reference)
    monkeypatch.setattr(
        model, "build_engine", lambda *args: pytest.fail("invalid EOS reached engine construction")
    )
    request = BuildRequest(
        model_dir=tmp_path,
        output_path=tmp_path / "model.bundle",
        family="ernie4_5_moe",
        task="text_continuation",
        precision="fp32",
        max_sequence_length=8,
    )
    with pytest.raises(ValueError, match="scalar EOS"):
        model.build(request, object())


def test_stream_writer_preserves_every_plan_byte():
    destination = BytesIO()
    writer = PlanWriter(destination)
    assert writer.write(b"first") == 5
    assert writer.write(b"\x00second") == 7
    assert destination.getvalue() == b"first\x00second"

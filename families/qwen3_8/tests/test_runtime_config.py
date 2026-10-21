# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""runtime.json keeps its old field set unless the prefill engine is dynamic."""

from __future__ import annotations

from types import SimpleNamespace

from families.qwen3_8 import model as qwen_model

_OVERRIDES = dict(
    hidden_size=12, num_hidden_layers=2, num_attention_heads=3, num_key_value_heads=1,
    head_dim=4, vocab_size=32, bos_token_id=1, num_attention_layers=1, num_mamba_layers=1,
    d_inner=12, mamba_d_state=4, mamba_d_conv=4, mamba_nheads=3, mamba_head_dim=4,
    conv_dim=20, layer_types=["deltanet", "attention"])


class _Model:
    def get_bundle_config_overrides(self, config):
        return dict(_OVERRIDES)


def _runtime(tmp_path, **updates) -> dict:
    base = dict(
        max_cache_length=256, precision="fp16", decoder_engine_layout="single",
        has_mtp=False, mtp_num_draft_tokens=0)
    base.update(updates)
    return qwen_model._runtime_config(
        tmp_path, SimpleNamespace(eos_token_id=2), _Model(), **base)


def test_static_prefill_runtime_json_has_no_dynamic_field(tmp_path) -> None:
    runtime = _runtime(tmp_path, prefill_chunk_len=128, prefill_dynamic=False)
    assert "prefill_dynamic" not in runtime
    assert runtime["prefill_chunk_len"] == 128
    assert len(runtime) == 23


def test_dynamic_prefill_runtime_json_publishes_the_flag(tmp_path) -> None:
    runtime = _runtime(tmp_path, prefill_chunk_len=1024, prefill_dynamic=True)
    assert runtime["prefill_dynamic"] is True
    assert runtime["prefill_chunk_len"] == 1024
    assert len(runtime) == 24

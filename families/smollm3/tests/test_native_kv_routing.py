# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU routing and NoPE contracts for the published SmolLM3 family."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from families.smollm3.build_routing import (
    native_kv_architecture_capability,
    native_kv_build_capability,
    native_kv_cache_geometry,
)
from families.smollm3.config import ModelConfig, resolve_rope_layer_schedule


def _published_checkpoint_config(**raw_updates: object) -> ModelConfig:
    """Return config.json fields from the revision pinned by the manifest."""
    raw = {
        "attention_bias": False,
        "attention_dropout": 0.0,
        "layer_types": ["full_attention"] * 36,
        "max_window_layers": 28,
        "mlp_bias": False,
        "no_rope_layer_interval": 4,
        "no_rope_layers": [int((index + 1) % 4 != 0) for index in range(36)],
        "pretraining_tp": 2,
        "rope_scaling": None,
        "sliding_window": None,
        "torch_dtype": "bfloat16",
        "use_cache": False,
        "use_sliding_window": False,
    }
    raw.update(raw_updates)
    return ModelConfig(
        model_type="smollm3",
        architectures=["SmolLM3ForCausalLM"],
        vocab_size=128256,
        hidden_size=2048,
        intermediate_size=11008,
        num_hidden_layers=36,
        num_attention_heads=16,
        num_key_value_heads=4,
        rms_norm_eps=1e-6,
        rope_theta=5_000_000.0,
        bos_token_id=128000,
        eos_token_id=128012,
        pad_token_id=128004,
        tie_word_embeddings=True,
        max_position_embeddings=65536,
        hidden_act="silu",
        raw=raw,
    )


def test_published_checkpoint_reaches_native_kv_despite_training_metadata() -> None:
    config = _published_checkpoint_config()

    architecture = native_kv_architecture_capability(config)
    build = native_kv_build_capability(
        config,
        precision="bf16",
        max_cache_length=config.max_position_embeddings,
    )

    assert config.raw["pretraining_tp"] == 2
    assert architecture.eligible, architecture.reason
    assert build.eligible, build.reason


def test_native_kv_requires_the_full_checkpoint_window() -> None:
    config = _published_checkpoint_config()

    decision = native_kv_build_capability(
        config,
        precision="bf16",
        max_cache_length=256,
    )

    assert not decision.eligible
    assert "max_position_embeddings (65536)" in decision.reason
    row_bytes, total_bytes = native_kv_cache_geometry(config, 65536)
    assert row_bytes == 2 * 36 * 4 * 128 * 2
    assert total_bytes == 65536 * row_bytes


def test_nope_schedule_matches_the_published_every_fourth_layer_pattern() -> None:
    schedule = resolve_rope_layer_schedule(_published_checkpoint_config())

    assert len(schedule) == 36
    assert [index for index, uses_rope in enumerate(schedule) if not uses_rope] == [
        3,
        7,
        11,
        15,
        19,
        23,
        27,
        31,
        35,
    ]


def test_published_schedule_overrides_the_interval() -> None:
    published = [1] * 36
    published[5] = 0
    schedule = resolve_rope_layer_schedule(
        _published_checkpoint_config(no_rope_layers=published)
    )

    assert [index for index, uses_rope in enumerate(schedule) if not uses_rope] == [5]


@pytest.mark.parametrize(
    ("raw_updates", "fragment"),
    [
        ({"no_rope_layers": None, "no_rope_layer_interval": 0}, "must be positive"),
        ({"no_rope_layers": [1, 1, 1]}, "at least num_hidden_layers"),
        ({"no_rope_layers": [1, None] + [1] * 34}, "int()"),
        ({"no_rope_layers": ["x"] * 36}, "invalid literal"),
    ],
)
def test_malformed_nope_schedule_fails_closed(
    raw_updates: dict[str, object], fragment: str
) -> None:
    decision = native_kv_architecture_capability(
        _published_checkpoint_config(**raw_updates)
    )

    assert not decision.eligible
    assert fragment in decision.reason


def test_both_builders_index_the_schedule_with_the_layer_loop_variable() -> None:
    family = Path(__file__).resolve().parents[1]
    for name in ("standard_decoder_builder.py", "dual_profile_decoder_builder.py"):
        tree = ast.parse((family / name).read_text(encoding="utf-8"))
        indices = {
            node.slice.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == "rope_schedule"
            and isinstance(node.slice, ast.Name)
        }
        assert indices == {"layer_idx"}, name


def test_standard_builder_forwards_the_resolved_nope_flag() -> None:
    tree = ast.parse(
        (Path(__file__).resolve().parents[1] / "standard_decoder_builder.py").read_text(
            encoding="utf-8"
        )
    )
    forwarded = [
        keyword.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg == "apply_rope"
    ]
    assert any(
        isinstance(value, ast.Subscript)
        and isinstance(value.value, ast.Name)
        and value.value.id == "rope_schedule"
        for value in forwarded
    )


class _FakeTensor:
    def __init__(self, name: str = "tensor", shape: tuple[int, ...] = (1, 1, 64)) -> None:
        self.name = name
        self.shape = shape
        self.dtype = None

    def __getattr__(self, _name: str):
        return None


class _FakeLayer:
    def get_output(self, _index: int) -> _FakeTensor:
        return _FakeTensor("out")

    def __getattr__(self, _name: str):
        return lambda *args, **kwargs: None

    def __setattr__(self, _name: str, _value: object) -> None:
        pass


class _FakeNetwork:
    def __getattr__(self, name: str):
        if name.startswith("add_"):
            return lambda *args, **kwargs: _FakeLayer()
        raise AttributeError(name)


def _rope_insertions(monkeypatch: pytest.MonkeyPatch, *, apply_rope: bool) -> int:
    pytest.importorskip("tensorrt")
    import numpy as np

    from families.smollm3 import graph_blocks, graph_ops

    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        graph_ops,
        "add_apply_rope_native",
        lambda *args, **kwargs: calls.append(args) or _FakeTensor("roped"),
    )
    hidden = attention = 64
    prefix = "layer.0"
    weights = {
        f"{prefix}.input_norm": np.ones(hidden, dtype=np.float32),
        f"{prefix}.w_q": np.zeros((hidden, attention), dtype=np.float32),
        f"{prefix}.w_k": np.zeros((hidden, attention), dtype=np.float32),
        f"{prefix}.w_v": np.zeros((hidden, attention), dtype=np.float32),
        f"{prefix}.w_o": np.zeros((attention, hidden), dtype=np.float32),
    }
    graph_blocks.add_attention_block(
        _FakeNetwork(),
        _FakeTensor("hidden"),
        _FakeTensor("cache_k"),
        _FakeTensor("cache_v"),
        _FakeTensor("mask"),
        _FakeTensor("position"),
        weights=weights,
        prefix=prefix,
        hidden_size=hidden,
        attention_size=attention,
        kv_attention_size=attention,
        num_heads=2,
        num_kv_heads=2,
        head_dim=32,
        max_cache_length=16,
        eps_tensor=_FakeTensor("eps"),
        apply_rope=apply_rope,
        cos_half_tensor=_FakeTensor("cos"),
        sin_half_tensor=_FakeTensor("sin"),
    )
    return len(calls)


def test_rope_gate_controls_real_graph_insertion(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _rope_insertions(monkeypatch, apply_rope=True) == 2
    assert _rope_insertions(monkeypatch, apply_rope=False) == 0

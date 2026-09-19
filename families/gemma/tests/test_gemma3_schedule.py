# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gemma 3's interleaved attention schedule and its two rope bases."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")

try:
    from families.gemma.config import ModelConfig
    from families.gemma.graph_blocks import (
        gemma3_attention_schedule,
        gemma_attention_scale,
    )
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires TensorRT", allow_module_level=True)


def _config(**overrides) -> ModelConfig:
    raw = {
        "model_type": "gemma3_text",
        "hidden_size": 1152,
        "num_hidden_layers": 26,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "head_dim": 256,
        "intermediate_size": 6912,
        "vocab_size": 262144,
        "max_position_embeddings": 32768,
        "rope_theta": 1000000,
    }
    raw.update(overrides)
    return ModelConfig.from_json(json.dumps(raw))


def _gemma3() -> ModelConfig:
    return _config(
        sliding_window=512,
        sliding_window_pattern=6,
        rope_local_base_freq=10000,
        query_pre_attn_scalar=256,
    )


def test_every_sixth_layer_attends_globally() -> None:
    """Read off the reference: layers 0-4 are local, layer 5 is global.

    transformers computes `is_sliding = bool((layer_idx + 1) % pattern)`, and
    `google/gemma-3-1b-it` reports exactly that per layer.
    """
    schedule = gemma3_attention_schedule(_gemma3(), 26)
    assert schedule["is_local"][:8] == [True, True, True, True, True, False, True, True]
    assert schedule["window"] == 512
    # 26 layers, global at indices 5, 11, 17, 23.
    assert sum(schedule["is_local"]) == 22


def test_the_local_layers_rotate_on_their_own_base() -> None:
    """10000 locally against 1000000 globally; using one base for both is wrong."""
    schedule = gemma3_attention_schedule(_gemma3(), 26)
    assert schedule["local_theta"] == 10000.0
    assert _gemma3().rope_theta == 1000000.0


def test_gemma_and_gemma2_keep_the_graph_they_had() -> None:
    """Neither declares a pattern, so every layer stays global on one base."""
    config = _config(model_type="gemma2")
    schedule = gemma3_attention_schedule(config, 26)
    assert schedule["is_local"] == [False] * 26
    assert schedule["window"] is None
    assert schedule["local_theta"] == config.rope_theta


def test_a_checkpoint_without_a_second_rope_base_stays_global() -> None:
    """Gemma 2 interleaves windows but rotates every layer on one base.

    It declares sliding_window but no rope_local_base_freq, so no second rope
    table may be built for it.

    This case previously passed a `gemma3_text` config, which made it assert
    that a Gemma 3 checkpoint missing its local rope base quietly becomes an
    all-global model. That is the shape of google/gemma-3-12b-it, and the
    resulting engine built cleanly and generated fluent, wrong text. The
    model_type below is the one the docstring always described.
    """
    schedule = gemma3_attention_schedule(_config(model_type="gemma2", sliding_window=4096), 26)
    assert schedule["is_local"] == [False] * 26
    assert schedule["window"] is None


def test_gemma3_without_a_second_rope_base_is_refused() -> None:
    """The same inputs under a Gemma 3 model_type must not be guessed at."""
    with pytest.raises(ValueError, match="rope_local_base_freq"):
        gemma3_attention_schedule(_config(sliding_window=4096), 26)


def test_an_absent_pattern_falls_back_to_the_gemma3_default() -> None:
    """google/gemma-3-270m-it omits sliding_window_pattern.

    transformers defaults it to 6, and that checkpoint's resolved layer_types
    are 3 global layers in 18 - exactly that default. Reading the absent key as
    "no schedule" would build every layer global and be wrong without failing,
    which is how this was nearly shipped.
    """
    config = _config(sliding_window=512, rope_local_base_freq=10000)
    schedule = gemma3_attention_schedule(config, 18)
    assert schedule["local_theta"] == 10000.0
    assert sum(1 for local in schedule["is_local"] if not local) == 3
    assert [index for index, local in enumerate(schedule["is_local"]) if not local] == [5, 11, 17]


@pytest.mark.parametrize("pattern", [0, -3])
def test_a_declared_nonsense_pattern_is_refused(pattern: int) -> None:
    """Only an absent key falls back to the default; a stated one must be usable."""
    config = _config(
        sliding_window=512, sliding_window_pattern=pattern, rope_local_base_freq=10000
    )
    with pytest.raises(ValueError, match="must be positive"):
        gemma3_attention_schedule(config, 26)


def test_the_query_scale_is_read_not_inferred() -> None:
    """`query_pre_attn_scalar` matches head_dim on some widths and not others."""
    assert gemma_attention_scale(_gemma3(), 256) == pytest.approx(1.0 / 16.0)
    # A width where the two differ: the declared scalar must win.
    config = _config(query_pre_attn_scalar=144)
    assert gemma_attention_scale(config, 128) == pytest.approx(1.0 / 12.0)


def test_the_query_scale_falls_back_to_head_dim() -> None:
    assert gemma_attention_scale(_config(), 256) == pytest.approx(1.0 / 16.0)


def test_a_non_positive_query_scale_is_refused() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        gemma_attention_scale(_config(query_pre_attn_scalar=0), 256)


def test_an_explicit_layer_types_list_wins() -> None:
    """A checkpoint stating its schedule outright is believed over a pattern."""
    config = _config(
        sliding_window=512,
        rope_local_base_freq=10000,
        sliding_window_pattern=6,
        layer_types=["full_attention", "sliding_attention", "sliding_attention"],
    )
    schedule = gemma3_attention_schedule(config, 3)
    assert schedule["is_local"] == [False, True, True]


def test_a_layer_types_list_of_the_wrong_length_is_refused() -> None:
    config = _config(
        sliding_window=512,
        rope_local_base_freq=10000,
        layer_types=["sliding_attention"] * 5,
    )
    with pytest.raises(ValueError, match="lists 5 entries"):
        gemma3_attention_schedule(config, 26)


def test_an_unknown_layer_type_is_refused() -> None:
    config = _config(
        sliding_window=512,
        rope_local_base_freq=10000,
        layer_types=["sliding_attention", "linear_attention"],
    )
    with pytest.raises(ValueError, match="unsupported entries"):
        gemma3_attention_schedule(config, 2)


def test_vocab_size_falls_back_to_the_embedding(tmp_path) -> None:
    """google/gemma-3-4b-it omits vocab_size from its text_config.

    The config parser then defaults it to 0 and the checkpoint mapper rejects
    the embedding it just loaded:

        AssertionError: Embedding shape (262208, 2560) != (0, 2560)

    The unsloth mirror of the same weights does state it, so this only appeared
    against the official checkpoint in CI.
    """
    from families.gemma.model import _embedding_vocab_size

    class _Slice:
        def get_shape(self):
            return [262208, 2560]

    class _Reader:
        def get_slice(self, _name):
            return _Slice()

    class _Readers:
        tensor_map = {"language_model.model.embed_tokens.weight": _Reader()}

    assert _embedding_vocab_size(_Readers(), "language_model.model") == 262208
    with pytest.raises(ValueError, match="has no model.embed_tokens.weight"):
        _embedding_vocab_size(_Readers(), "model")


class _FakeSlice:
    def __init__(self, rows):
        self._rows = rows

    def get_shape(self):
        return [self._rows, 2560]


class _FakeReader:
    def __init__(self, rows):
        self._rows = rows

    def get_slice(self, _name):
        return _FakeSlice(self._rows)


class _FakeReaders:
    """q_proj 2048 rows and k_proj 1024 rows: 8 and 4 heads at head_dim 256."""

    tensor_map = {
        "language_model.model.layers.0.self_attn.q_proj.weight": _FakeReader(2048),
        "language_model.model.layers.0.self_attn.k_proj.weight": _FakeReader(1024),
    }


def test_gemma3_defaults_fill_what_an_official_config_omits() -> None:
    """A published Gemma 3 config may leave out what transformers defaults.

    google/gemma-3-4b-it omits vocab_size, hidden_activation and
    num_key_value_heads; the unsloth mirror of the same weights states all
    three, which is why local runs passed and internal CI failed three times.
    The attention widths come from the q and k projections rather than a
    default, because the tensors are authoritative: falling back to
    num_attention_heads gave a K/V cache twice the checkpoint's real width and
    failed with "Compact K/V cache width must be ... (2048), got 1024".
    """
    from families.gemma.model import _apply_gemma3_config_defaults

    config = _config(model_type="gemma3_text")
    for key in ("rms_norm_eps", "head_dim", "num_key_value_heads", "num_attention_heads"):
        config.raw.pop(key, None)
    config.hidden_act = ""
    config.rms_norm_eps = 1e-5
    _apply_gemma3_config_defaults(config, _FakeReaders(), "language_model.model")
    assert config.hidden_act == "gelu_pytorch_tanh"
    assert config.rms_norm_eps == 1e-6
    assert config.head_dim == 256
    assert config.num_attention_heads == 8
    assert config.num_key_value_heads == 4


def test_gemma2_keeps_its_own_config_values() -> None:
    """The defaults are Gemma 3 only; Gemma 2 states its own and keeps them."""
    from families.gemma.model import _apply_gemma3_config_defaults

    config = _config(model_type="gemma2")
    config.hidden_act = ""
    config.rms_norm_eps = 1e-5
    _apply_gemma3_config_defaults(config, _FakeReaders(), "model")
    assert config.hidden_act == ""
    assert config.rms_norm_eps == 1e-5

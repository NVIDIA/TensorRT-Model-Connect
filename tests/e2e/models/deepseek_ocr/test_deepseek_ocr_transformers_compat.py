# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed contracts for the pinned DeepSeek-OCR reference shim."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorrt_model_connect.deepseek_ocr_reference_compat import (
    DEEPSEEK_OCR_REFERENCE_REVISION,
    DeepSeekOcrTransformersCompat,
    assert_deepseek_ocr_eager_attention,
    configure_deepseek_ocr_rotary_embeddings,
)


class _PositiveInvFreq:
    device = SimpleNamespace(type="cpu")

    def __gt__(self, value):
        assert value == 0
        return self

    def all(self):
        return self

    def item(self):
        return True

    def tolist(self):
        return [1.0]


class _Parameter:
    device = SimpleNamespace(type="cpu")


class _RotaryEmbedding:
    def __init__(self, *, config, device=None):
        del config, device
        self.inv_freq = _PositiveInvFreq()


class _EagerAttention:
    def __init__(self, *, implementation: str = "eager", use_mla: bool = False):
        self.config = SimpleNamespace(
            _attn_implementation=implementation,
            num_hidden_layers=1,
            use_mla=use_mla,
        )
        self.rotary_emb = _RotaryEmbedding(config=self.config)
        self._parameter = _Parameter()

    def parameters(self):
        return iter((self._parameter,))


class _FlashAttention(_EagerAttention):
    pass


class _Model:
    def __init__(self, module, *, implementation: str = "eager"):
        self.config = SimpleNamespace(_attn_implementation=implementation)
        self.module = module

    def named_modules(self):
        return (("", self), ("language.layers.0.self_attn", self.module))


class CustomQwen2ModelInner:
    __module__ = "transformers_modules.deepseek_ocr.deepencoderv2"

    def __init__(self):
        self.config = SimpleNamespace()
        self.rotary_emb = _RotaryEmbedding(config=self.config)
        self._parameter = _Parameter()

    def parameters(self):
        return iter((self._parameter,))


class _RotaryModel:
    def __init__(self):
        self.language = _EagerAttention()
        self.visual = CustomQwen2ModelInner()

    def named_modules(self):
        return (
            ("", self),
            ("language.layers.0.self_attn", self.language),
            ("vision.qwen2", self.visual),
        )


_COMPAT = DeepSeekOcrTransformersCompat(
    eager_attention_type=_EagerAttention,
    flash_attention_type=_FlashAttention,
    dynamic_cache_type=object,
)


def test_eager_attention_guard_accepts_only_the_pinned_mha_shape() -> None:
    assert assert_deepseek_ocr_eager_attention(_Model(_EagerAttention()), _COMPAT) == 1


@pytest.mark.parametrize(
    ("model", "message"),
    [
        (
            _Model(_EagerAttention(), implementation="flash_attention_2"),
            "must use eager attention",
        ),
        (
            _Model(_FlashAttention()),
            "unsupported flash-attention",
        ),
        (
            _Model(_EagerAttention(implementation="flash_attention_2")),
            "not configured for eager attention",
        ),
        (
            _Model(_EagerAttention(use_mla=True)),
            "did not select pinned MHA mode",
        ),
    ],
)
def test_eager_attention_guard_fails_closed(model, message: str) -> None:
    with pytest.raises(RuntimeError, match=message):
        assert_deepseek_ocr_eager_attention(model, _COMPAT)


def test_rotary_compat_rematerializes_exact_language_and_visual_modules() -> None:
    model = _RotaryModel()
    old_language = model.language.rotary_emb
    old_visual = model.visual.rotary_emb

    assert configure_deepseek_ocr_rotary_embeddings(model, _COMPAT) == (1, 1)
    assert model.language.rotary_emb is not old_language
    assert model.visual.rotary_emb is not old_visual
    assert model.language.rotary_emb.inv_freq.tolist() == [1.0]
    assert model.visual.rotary_emb.inv_freq.tolist() == [1.0]


def test_reference_installs_compat_before_loading_exact_remote_source() -> None:
    model_root = Path(__file__).resolve().parent
    script = (model_root / "hf_reference.py").read_text(encoding="utf-8")
    wrapper = (model_root / "e2e_plugins/references/hf_transformers.py").read_text(encoding="utf-8")

    assert script.index("install_deepseek_ocr_transformers_compat()") < script.index(
        "AutoTokenizer.from_pretrained"
    )
    guard_call = "eager_attention_layers = assert_deepseek_ocr_eager_attention"
    rotary_call = (
        "rotary_language_layers, rotary_visual_encoders = configure_deepseek_ocr_rotary_embeddings"
    )
    assert script.index("AutoModel.from_pretrained") < script.index(guard_call)
    assert script.index(guard_call) < script.index(rotary_call)
    assert script.index(rotary_call) < script.index("model.cuda()")
    assert 'attn_implementation="eager"' in script
    assert "revision=args.revision" in script
    assert '"--revision",\n            case.hf_revision,' in wrapper
    assert DEEPSEEK_OCR_REFERENCE_REVISION == ("aaa02f3811945a91062062994c5c4a3f4c0af2b0")

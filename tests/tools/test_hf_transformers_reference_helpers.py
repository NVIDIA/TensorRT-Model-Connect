"""Tests for Hugging Face reference helper logic."""

from __future__ import annotations

from tests.e2e_harness.references.hf_transformers import (
    _decode_vl_generated_text,
    _vl_fallback_prompt,
)


class _FakeProcessor:
    def __init__(self, mapping: dict[tuple[int, ...], str]) -> None:
        self.mapping = mapping

    def decode(self, token_ids, *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is True
        return self.mapping[tuple(int(token) for token in token_ids)]


def test_qwen_vl_fallback_prompt_includes_image_pad() -> None:
    assert _vl_fallback_prompt("Qwen/Qwen3-VL-2B-Instruct", "Describe it") == (
        "<|vision_start|><|image_pad|><|vision_end|>Describe it"
    )


def test_internvl_fallback_prompt_includes_image_placeholder() -> None:
    assert _vl_fallback_prompt("OpenGVLab/InternVL3-8B-hf", "Describe it") == (
        "<IMG_CONTEXT>\nDescribe it"
    )


def test_non_vl_fallback_prompt_is_unchanged() -> None:
    assert _vl_fallback_prompt("Qwen/Qwen3-0.6B", "Hello") == "Hello"


def test_vl_decode_uses_generated_suffix_for_full_sequences() -> None:
    processor = _FakeProcessor({
        (101, 102): "prompt",
        (201, 202): "blue",
    })

    assert _decode_vl_generated_text(processor, [101, 102, 201, 202], 2) == "blue"


def test_vl_decode_falls_back_when_model_returns_generated_only_ids() -> None:
    processor = _FakeProcessor({
        (): "",
        (201, 202): "blue",
    })

    assert _decode_vl_generated_text(processor, [201, 202], 4) == "blue"

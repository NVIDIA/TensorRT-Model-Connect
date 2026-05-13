"""Tests for vision-language Hugging Face reference guardrails."""

from __future__ import annotations

from tests.e2e_harness.references.hf_transformers import (
    _decode_vl_generated_text,
    _is_prompt_only_vl_text,
)


class _FakeProcessor:
    def __init__(self, mapping: dict[tuple[int, ...], str]) -> None:
        self.mapping = mapping

    def decode(self, token_ids, *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is True
        return self.mapping[tuple(int(token) for token in token_ids)]


def test_vl_decode_rejects_prompt_only_full_sequence() -> None:
    processor = _FakeProcessor({
        (): "",
        (101, 102): "What color is the vehicle in this image?",
    })

    assert _decode_vl_generated_text(
        processor,
        [101, 102],
        2,
        ("What color is the vehicle in this image?",),
    ) == ""


def test_vl_decode_keeps_generated_only_answer() -> None:
    processor = _FakeProcessor({
        (): "",
        (201, 202): "white",
    })

    assert _decode_vl_generated_text(
        processor,
        [201, 202],
        32,
        ("What color is the vehicle in this image?",),
    ) == "white"


def test_vl_decode_keeps_prompt_plus_answer() -> None:
    processor = _FakeProcessor({
        (201,): "",
        (101, 102, 201): "What color is the vehicle in this image? White",
    })

    assert _decode_vl_generated_text(
        processor,
        [101, 102, 201],
        2,
        ("What color is the vehicle in this image?",),
    ) == "What color is the vehicle in this image? White"


def test_vl_prompt_only_detects_chat_template_without_answer() -> None:
    assert _is_prompt_only_vl_text(
        "<image> What color is the vehicle? Assistant:",
        ("<image> What color is the vehicle?",),
    )

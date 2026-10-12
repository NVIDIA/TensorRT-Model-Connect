# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from families.nemotron_labs_diffusion.tests.benchmark.reference import _truncate


class _Tokenizer:
    def encode(self, prompt: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [int(value) for value in prompt.split()]

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        return " ".join(str(value) for value in token_ids)


def test_benchmark_reference_applies_declared_prompt_window() -> None:
    tokenizer = _Tokenizer()

    assert _truncate(tokenizer, "1 2 3 4", 2, "left") == "3 4"
    assert _truncate(tokenizer, "1 2 3 4", 2, "right") == "1 2"

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect

from tests.e2e.models.internlm.e2e_plugins.references import hf_transformers


def test_reference_retries_chat_template_without_thinking_kwarg() -> None:
    source = inspect.getsource(
        hf_transformers.HfTransformersReference._run_full_generation
    )
    assert "except TypeError:" in source
    assert 'chat_kwargs.pop("enable_thinking", None)' in source
    assert "hf_generated_tokens.json" in source
    assert 'json.dump({{"token_ids": generated_token_ids}}, f)' in source
    assert "_json_output_reader(token_path)" in source
    assert "PreTrainedTokenizerFast" in source
    assert 'tokenizer_file=str(tokenizer_dir / "tokenizer.json")' in source

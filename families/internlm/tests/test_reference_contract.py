# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from families.internlm.tests import test_e2e


def test_hf_reference_does_not_request_token_type_ids() -> None:
    class Tokenizer:
        def __init__(self) -> None:
            self.calls = []

        def __call__(self, text, **options):
            self.calls.append((text, options))
            return {"input_ids": [1]}

        def apply_chat_template(self, _messages, **_options):
            return "rendered"

    tokenizer = Tokenizer()
    test_e2e._render_prompt(tokenizer, "plain", {})
    test_e2e._render_prompt(tokenizer, "chat", {"use_chat_template": True})

    assert len(tokenizer.calls) == 2
    assert all(options["return_token_type_ids"] is False for _, options in tokenizer.calls)


def test_hf_reference_reuses_the_native_step_prover_tokenizer() -> None:
    source = (Path(__file__).resolve().parent / "test_e2e.py").read_text(encoding="utf-8")
    assert "ensure_tokenizer_json(model_dir)" in source
    assert "PreTrainedTokenizerFast(" in source
    assert "DynamicCache" not in source
    assert "use_fast=False" not in source
    assert "torch_dtype=dtypes[reference_precision]" in source
    assert "\n            dtype=dtypes[reference_precision]" not in source


def test_native_keeps_the_checkpoint_prompt_and_builder_policies() -> None:
    family = Path(__file__).resolve().parents[1]
    plugin = (family / "runtime/plugin.cpp").read_text(encoding="utf-8")
    model = (family / "model.py").read_text(encoding="utf-8")
    assert 'require_text_section(bundle, "tokenizer_config.json")' in plugin
    assert 'config.find("chat_template")' in plugin
    assert "chat_template.jinja" not in plugin
    assert '"chat_template.jinja"' not in model

    for path in (family / "utils.py", family / "dual_profile_decoder_tp_builder.py"):
        source = path.read_text(encoding="utf-8")
        assert "builder_optimization_level = 3" in source
        assert "builder_optimization_level = 1" not in source

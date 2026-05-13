"""Tests for compatibility shims around external Transformers model code."""

from __future__ import annotations

import json


def test_patch_legacy_dynamic_cache_api_restores_internlm_remote_code_hooks() -> None:
    from transformers.cache_utils import DynamicCache

    from tensorrt_model_connect.transformers_compat import (
        patch_legacy_dynamic_cache_api,
    )

    patch_legacy_dynamic_cache_api()

    assert hasattr(DynamicCache, "from_legacy_cache")
    assert hasattr(DynamicCache, "get_max_length")
    assert isinstance(DynamicCache.from_legacy_cache(None), DynamicCache)


def test_remap_token_ids_to_model_vocab_uses_tokenizer_config_ids(tmp_path) -> None:
    from tensorrt_model_connect.transformers_compat import (
        remap_token_ids_to_model_vocab,
    )

    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "added_tokens_decoder": {
                    "92542": {"content": "<|im_end|>", "special": True},
                    "92543": {"content": "<|im_start|>", "special": True},
                }
            }
        )
    )

    class FakeTokenizer:
        def get_added_vocab(self):
            return {"<|im_end|>": 92548, "<|im_start|>": 92549}

    assert remap_token_ids_to_model_vocab(
        FakeTokenizer(), tmp_path, [1, 92549, 1008, 92548], 92544
    ) == [1, 92543, 1008, 92542]


def test_patch_tokenizer_json_special_token_ids_rewrites_added_tokens(tmp_path) -> None:
    from tensorrt_model_connect.transformers_compat import (
        patch_tokenizer_json_special_token_ids,
    )

    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "added_tokens_decoder": {
                    "92542": {"content": "<|im_end|>", "special": True},
                    "92543": {"content": "<|im_start|>", "special": True},
                }
            }
        )
    )
    tokenizer_json = tmp_path / "tokenizer.json"
    tokenizer_json.write_text(
        json.dumps(
            {
                "model": {
                    "type": "BPE",
                    "vocab": {
                        "<s>": 1,
                        "[UNUSED_TOKEN_145]": 92542,
                        "[UNUSED_TOKEN_146]": 92543,
                    },
                },
                "added_tokens": [
                    {"id": 92542, "content": "[UNUSED_TOKEN_145]", "special": False},
                    {"id": 92543, "content": "[UNUSED_TOKEN_146]", "special": False},
                    {"id": 92548, "content": "<|im_end|>", "special": True},
                    {"id": 92549, "content": "<|im_start|>", "special": True},
                ],
            }
        )
    )

    assert patch_tokenizer_json_special_token_ids(
        tokenizer_json, tmp_path / "tokenizer_config.json", 92544
    )

    patched = json.loads(tokenizer_json.read_text())
    assert patched["model"]["vocab"]["<|im_end|>"] == 92542
    assert patched["model"]["vocab"]["<|im_start|>"] == 92543
    assert "[UNUSED_TOKEN_145]" not in patched["model"]["vocab"]
    assert "[UNUSED_TOKEN_146]" not in patched["model"]["vocab"]
    added_by_content = {
        token["content"]: token for token in patched["added_tokens"]
    }
    assert added_by_content["<|im_end|>"]["id"] == 92542
    assert added_by_content["<|im_start|>"]["id"] == 92543
    assert 92548 not in {token["id"] for token in patched["added_tokens"]}
    assert 92549 not in {token["id"] for token in patched["added_tokens"]}

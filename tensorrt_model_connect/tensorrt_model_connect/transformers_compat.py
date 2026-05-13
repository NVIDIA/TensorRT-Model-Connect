"""Compatibility shims for external Transformers model code."""

from __future__ import annotations

import json
from pathlib import Path


def patch_legacy_dynamic_cache_api() -> None:
    """Restore cache methods still used by some trusted remote-code models.

    Transformers 5.x removed a few legacy ``DynamicCache`` helpers that older
    remote-code repositories still call during generation.  The shim is small
    and idempotent so reference subprocesses can install it before loading a
    trusted remote-code model.
    """
    try:
        from transformers.cache_utils import DynamicCache
    except Exception:
        return

    if not hasattr(DynamicCache, "from_legacy_cache"):

        @classmethod
        def from_legacy_cache(cls, past_key_values=None, *args, **kwargs):
            if past_key_values is None:
                return cls()
            return cls(past_key_values)

        DynamicCache.from_legacy_cache = from_legacy_cache

    if (
        not hasattr(DynamicCache, "get_max_length")
        and hasattr(DynamicCache, "get_max_cache_shape")
    ):

        def get_max_length(self):
            return self.get_max_cache_shape()

        DynamicCache.get_max_length = get_max_length


def _special_token_model_ids(
    tokenizer_config_path: str | Path,
    vocab_size: int | None,
) -> dict[str, int]:
    """Return special-token content mapped to in-model token IDs."""
    if vocab_size is None:
        return {}

    try:
        tokenizer_config = json.loads(Path(tokenizer_config_path).read_text())
    except Exception:
        return {}

    remap: dict[str, int] = {}
    for token_id_str, spec in tokenizer_config.get("added_tokens_decoder", {}).items():
        if not isinstance(spec, dict):
            continue
        content = spec.get("content")
        try:
            token_id = int(token_id_str)
        except (TypeError, ValueError):
            continue
        if spec.get("special") and content and 0 <= token_id < vocab_size:
            remap[str(content)] = token_id
    return remap


def _tokenizer_config_path(model_ref: str | Path) -> Path:
    return Path(model_ref) / "tokenizer_config.json"


def remap_token_ids_to_model_vocab(
    tokenizer,
    model_ref: str | Path,
    token_ids: list[int],
    vocab_size: int | None,
) -> list[int]:
    """Remap tokenizer-added special IDs back into the model embedding vocab.

    Some trusted remote-code tokenizers register chat special tokens as added
    tokens beyond ``config.vocab_size`` even though tokenizer_config.json pins
    those tokens to reserved in-vocab IDs. Feeding the added IDs into the model
    crashes in the embedding lookup. This helper keeps the HF reference and TRT
    bundle tokenizer aligned on the in-vocab IDs.
    """
    content_to_model_id = _special_token_model_ids(
        _tokenizer_config_path(model_ref),
        vocab_size,
    )
    if not content_to_model_id:
        return token_ids

    try:
        added_vocab = tokenizer.get_added_vocab()
    except Exception:
        return token_ids

    out_of_vocab_to_model_id: dict[int, int] = {}
    for content, added_id in added_vocab.items():
        model_id = content_to_model_id.get(content)
        if model_id is None:
            continue
        try:
            added_id_int = int(added_id)
        except (TypeError, ValueError):
            continue
        if added_id_int >= (vocab_size or 0) and model_id != added_id_int:
            out_of_vocab_to_model_id[added_id_int] = model_id

    if not out_of_vocab_to_model_id:
        return token_ids
    return [
        out_of_vocab_to_model_id.get(int(token_id), int(token_id))
        for token_id in token_ids
    ]


def patch_tokenizer_json_special_token_ids(
    tokenizer_json_path: str | Path,
    tokenizer_config_path: str | Path,
    vocab_size: int | None,
) -> bool:
    """Patch tokenizer.json so special chat tokens use in-vocab model IDs."""
    content_to_model_id = _special_token_model_ids(tokenizer_config_path, vocab_size)
    if not content_to_model_id:
        return False

    tokenizer_json = Path(tokenizer_json_path)
    try:
        data = json.loads(tokenizer_json.read_text())
    except Exception:
        return False

    model = data.get("model", {})
    vocab = model.get("vocab")
    if not isinstance(vocab, dict):
        return False

    changed = False
    for content, model_id in content_to_model_id.items():
        old_token = next(
            (token for token, token_id in vocab.items() if token_id == model_id),
            None,
        )
        if old_token != content:
            if old_token is not None:
                del vocab[old_token]
            vocab[content] = model_id
            changed = True

    added_tokens = data.get("added_tokens")
    if isinstance(added_tokens, list):
        model_id_to_content = {v: k for k, v in content_to_model_id.items()}
        rewritten = []
        seen_model_ids: set[int] = set()
        for token in added_tokens:
            if not isinstance(token, dict):
                rewritten.append(token)
                continue
            token_id = token.get("id")
            content = token.get("content")
            if content in content_to_model_id and token_id != content_to_model_id[content]:
                changed = True
                continue
            if token_id in model_id_to_content:
                new_content = model_id_to_content[token_id]
                new_token = dict(token)
                if new_token.get("content") != new_content or not new_token.get("special"):
                    new_token["content"] = new_content
                    new_token["special"] = True
                    changed = True
                rewritten.append(new_token)
                seen_model_ids.add(int(token_id))
                continue
            rewritten.append(token)

        for model_id, content in model_id_to_content.items():
            if model_id not in seen_model_ids:
                rewritten.append(
                    {
                        "id": model_id,
                        "content": content,
                        "single_word": False,
                        "lstrip": False,
                        "rstrip": False,
                        "normalized": False,
                        "special": True,
                    }
                )
                changed = True
        if changed:
            data["added_tokens"] = rewritten

    if changed:
        tokenizer_json.write_text(json.dumps(data, ensure_ascii=False))
    return changed

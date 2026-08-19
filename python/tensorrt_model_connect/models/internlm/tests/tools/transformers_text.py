# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""InternLM-owned tokenizer policy for the native Transformers reference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


_TOKENIZER_REPO_ID = "internlm/internlm2-step-prover"
_TOKENIZER_REVISION = "6c727046190546168bf3aba9a1d78d5fb325ff14"
_TOKENIZER_FILENAME = "tokenizer.json"
_TOKENIZER_SHA256 = (
    "1193d3a1aa3d9f74866287ca3c1f7bf64fe54dd6ecf015e751f13ebce509e411"
)
_SOURCE_MODEL_SHA256 = (
    "f868398fc4e05ee1e8aeba95ddf18ddcc45b8bce55d5093bead5bbf80429b48b"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_cached_model_ref(
    model_id: str,
    *,
    revision: str,
    local_files_only: bool,
) -> Path:
    model_path = Path(model_id)
    if model_path.exists():
        return model_path

    try:
        from huggingface_hub import snapshot_download

        kwargs: dict[str, Any] = {
            "repo_id": model_id,
            "local_files_only": local_files_only,
            "allow_patterns": ["tokenizer.model", "tokenizer_config.json"],
        }
        if revision:
            kwargs["revision"] = revision
        return Path(snapshot_download(**kwargs))
    except Exception as exc:
        raise RuntimeError(
            f"could not resolve tokenizer files for {model_id!r} from the "
            "Hugging Face cache"
        ) from exc


def _resolve_reference_tokenizer_json(
    model_dir: Path,
    *,
    local_files_only: bool,
) -> Path | None:
    """Resolve the official fast tokenizer independently from the DUT."""
    source_model = model_dir / "tokenizer.model"
    if not source_model.is_file() or _sha256_file(source_model) != _SOURCE_MODEL_SHA256:
        return None

    try:
        from huggingface_hub import hf_hub_download

        tokenizer_path = Path(
            hf_hub_download(
                repo_id=_TOKENIZER_REPO_ID,
                filename=_TOKENIZER_FILENAME,
                revision=_TOKENIZER_REVISION,
                local_files_only=local_files_only,
            )
        )
    except Exception as exc:
        raise RuntimeError(
            "pinned official InternLM reference tokenizer is unavailable in "
            f"the local Hugging Face cache: {_TOKENIZER_REPO_ID}@{_TOKENIZER_REVISION}"
        ) from exc

    actual_sha256 = _sha256_file(tokenizer_path)
    if actual_sha256 != _TOKENIZER_SHA256:
        raise RuntimeError(
            "pinned official InternLM reference tokenizer SHA256 mismatch: "
            f"expected {_TOKENIZER_SHA256}, got {actual_sha256}: {tokenizer_path}"
        )
    return tokenizer_path


def load_tokenizer(arguments: Any, transformers_module: Any) -> Any | None:
    """Return InternLM's independent reference tokenizer when applicable."""
    model_ref = _resolve_cached_model_ref(
        arguments.model,
        revision=arguments.model_revision,
        local_files_only=arguments.local_files_only,
    )
    tokenizer_path = _resolve_reference_tokenizer_json(
        model_ref,
        local_files_only=arguments.local_files_only,
    )
    if tokenizer_path is None:
        return None
    tokenizer_config = json.loads(
        (model_ref / "tokenizer_config.json").read_text(encoding="utf-8")
    )
    return transformers_module.PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_path),
        model_input_names=["input_ids", "attention_mask"],
        bos_token=tokenizer_config.get("bos_token"),
        eos_token=tokenizer_config.get("eos_token"),
        unk_token=tokenizer_config.get("unk_token"),
        pad_token=tokenizer_config.get("pad_token"),
        chat_template=tokenizer_config.get("chat_template"),
        clean_up_tokenization_spaces=tokenizer_config.get(
            "clean_up_tokenization_spaces",
            False,
        ),
    )

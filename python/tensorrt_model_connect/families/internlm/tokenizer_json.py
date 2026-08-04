# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Install InternLM's pinned official native tokenizer artifact."""

from __future__ import annotations

import hashlib
import shutil
import sys
import tempfile
from pathlib import Path


PINNED_TOKENIZER_REPO_ID = "internlm/internlm2-step-prover"
PINNED_TOKENIZER_REVISION = "6c727046190546168bf3aba9a1d78d5fb325ff14"
PINNED_TOKENIZER_FILENAME = "tokenizer.json"
PINNED_TOKENIZER_SHA256 = (
    "1193d3a1aa3d9f74866287ca3c1f7bf64fe54dd6ecf015e751f13ebce509e411"
)
SOURCE_TOKENIZER_MODEL_SHA256 = (
    "f868398fc4e05ee1e8aeba95ddf18ddcc45b8bce55d5093bead5bbf80429b48b"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"{label} is missing: {path}")
    actual = _sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"{label} SHA256 mismatch: expected {expected}, got {actual}: {path}"
        )


def _matches_pinned_source(model_dir: Path) -> bool:
    model_path = model_dir / "tokenizer.model"
    return (
        model_path.is_file()
        and _sha256_file(model_path) == SOURCE_TOKENIZER_MODEL_SHA256
    )


def resolve_pinned_tokenizer_json(model_dir: str | Path) -> Path:
    """Resolve and verify the official JSON without permitting network access."""
    model_path = Path(model_dir) / "tokenizer.model"
    _require_sha256(
        model_path,
        SOURCE_TOKENIZER_MODEL_SHA256,
        "InternLM source tokenizer.model",
    )

    try:
        from huggingface_hub import hf_hub_download

        tokenizer_path = Path(
            hf_hub_download(
                repo_id=PINNED_TOKENIZER_REPO_ID,
                filename=PINNED_TOKENIZER_FILENAME,
                revision=PINNED_TOKENIZER_REVISION,
                local_files_only=True,
            )
        )
    except Exception as exc:
        raise RuntimeError(
            "pinned official InternLM tokenizer.json is unavailable in the "
            f"local Hugging Face cache: {PINNED_TOKENIZER_REPO_ID}@"
            f"{PINNED_TOKENIZER_REVISION}"
        ) from exc

    _require_sha256(
        tokenizer_path,
        PINNED_TOKENIZER_SHA256,
        "pinned official InternLM tokenizer.json",
    )
    return tokenizer_path


def ensure_tokenizer_json(
    model_dir: str | Path,
    *,
    previous_error: str | None = None,
) -> bool:
    """Atomically install the verified pinned JSON beside the source model."""
    path = Path(model_dir)
    tokenizer_path = path / PINNED_TOKENIZER_FILENAME
    temporary_path: Path | None = None

    # This family can match other InternLM checkpoints. Preserve their existing
    # tokenizer contract and apply the pinned artifact only to the exact source
    # model for which byte identity has been established.
    if not _matches_pinned_source(path):
        return tokenizer_path.is_file()

    try:
        if tokenizer_path.exists():
            _require_sha256(
                tokenizer_path,
                PINNED_TOKENIZER_SHA256,
                "installed InternLM tokenizer.json",
            )
            return True

        pinned_path = resolve_pinned_tokenizer_json(path)
        with tempfile.NamedTemporaryFile(
            dir=path,
            prefix=".trtmc-internlm-tokenizer-",
            suffix=".json",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            with pinned_path.open("rb") as source:
                shutil.copyfileobj(source, output)

        _require_sha256(
            temporary_path,
            PINNED_TOKENIZER_SHA256,
            "copied InternLM tokenizer.json",
        )
        temporary_path.replace(tokenizer_path)
        temporary_path = None
        _require_sha256(
            tokenizer_path,
            PINNED_TOKENIZER_SHA256,
            "installed InternLM tokenizer.json",
        )
        print(
            "[trtmc build] Installed pinned official InternLM tokenizer.json",
            file=sys.stderr,
        )
    except Exception as exc:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        detail = (
            f"{previous_error}; pinned family tokenizer install failed: {exc}"
            if previous_error
            else f"pinned family tokenizer install failed: {exc}"
        )
        raise RuntimeError(
            f"could not install tokenizer.json for {path}; {detail}"
        ) from exc

    return tokenizer_path.is_file()

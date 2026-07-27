# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""InternLM-owned fast-tokenizer serialization fallback."""

from __future__ import annotations

import os
import shutil
import stat
import tempfile
from pathlib import Path

from ...tokenizer_validation import native_tokenizer_json_error


def _path_is_present(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _remove_candidate(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _generated_tokenizer_file_is_safe(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError:
        return False
    return stat.S_ISREG(metadata.st_mode) and metadata.st_size > 0


def ensure_tokenizer_json(
    model_dir: str | Path,
    *,
    previous_error: str | None = None,
    trust_remote_code: bool = False,
) -> bool:
    if type(trust_remote_code) is not bool:
        raise TypeError(
            "trust_remote_code must be a bool, got "
            f"{type(trust_remote_code).__name__}"
        )
    del previous_error
    path = Path(model_dir)
    tokenizer_path = path / "tokenizer.json"
    if (
        _path_is_present(tokenizer_path)
        and native_tokenizer_json_error(tokenizer_path) is None
    ):
        return True

    had_original = _path_is_present(tokenizer_path)
    installed = False
    try:
        with tempfile.TemporaryDirectory(
            prefix=".internlm-tokenizer-repair-",
            dir=path,
        ) as temporary_dir:
            temporary_path = Path(temporary_dir)
            quarantined_path = temporary_path / "original-tokenizer.json"
            generated_dir = temporary_path / "generated"
            generated_dir.mkdir()
            if had_original:
                os.replace(tokenizer_path, quarantined_path)

            try:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(
                    str(path),
                    trust_remote_code=trust_remote_code,
                    use_fast=True,
                )
                if not getattr(tokenizer, "is_fast", False):
                    return False
                tokenizer.save_pretrained(str(generated_dir))
                candidate_path = generated_dir / "tokenizer.json"
                if (
                    not _generated_tokenizer_file_is_safe(candidate_path)
                    or native_tokenizer_json_error(candidate_path) is not None
                ):
                    return False
                os.replace(candidate_path, tokenizer_path)
                if (
                    not _generated_tokenizer_file_is_safe(tokenizer_path)
                    or native_tokenizer_json_error(tokenizer_path) is not None
                ):
                    return False
                installed = True
                return True
            finally:
                if not installed:
                    _remove_candidate(tokenizer_path)
                    if _path_is_present(quarantined_path):
                        os.replace(quarantined_path, tokenizer_path)
    except Exception:
        return False

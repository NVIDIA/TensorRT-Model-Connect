# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""InternLM-owned fast-tokenizer serialization fallback."""

from __future__ import annotations

import os
import shutil
import stat
import sys
import tempfile
from pathlib import Path

from ...tokenizer_validation import native_tokenizer_json_error


class _TokenizerRollbackError(RuntimeError):
    """A failed rollback whose message names the recoverable original."""


def _path_is_present(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _remove_candidate(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _reserve_recovery_path(model_dir: Path) -> tuple[Path, Path]:
    recovery_dir = Path(
        tempfile.mkdtemp(
            prefix=".internlm-tokenizer-recovery-",
            dir=model_dir,
        )
    )
    return recovery_dir, recovery_dir / "original-tokenizer.json"


def _discard_empty_recovery_dir(
    recovery_dir: Path | None,
    recovery_path: Path | None,
    *,
    discard_original: bool,
) -> None:
    if recovery_dir is None or recovery_path is None:
        return
    if discard_original:
        try:
            shutil.rmtree(recovery_dir)
        except Exception as exc:
            # The validated tokenizer is already installed. A recursive
            # cleanup can fail after partially deleting the previous artifact,
            # so report only possible residue and keep the successful result.
            try:
                print(
                    "[trtmc build] tokenizer.json repair succeeded, but "
                    "cleanup of the previous artifact was incomplete; "
                    "the recovery directory may contain residual files at "
                    f"'{recovery_dir}': {exc}",
                    file=sys.stderr,
                )
            except Exception:
                pass
        return
    if _path_is_present(recovery_path):
        return
    try:
        recovery_dir.rmdir()
    except OSError:
        pass


def _rollback_candidate(
    tokenizer_path: Path,
    quarantined_path: Path,
    *,
    recovery_path: Path | None,
) -> None:
    original_recoverable = _path_is_present(quarantined_path)
    if original_recoverable:
        if recovery_path is None:
            raise _TokenizerRollbackError(
                "cannot safely roll back tokenizer.json without a reserved "
                "recovery path"
            )
        if quarantined_path != recovery_path:
            try:
                os.replace(quarantined_path, recovery_path)
            except Exception as exc:
                raise _TokenizerRollbackError(
                    "failed to move the original tokenizer.json to its durable "
                    f"recovery path '{recovery_path}'; it remains preserved at "
                    f"'{quarantined_path}': {exc}"
                ) from exc

    try:
        _remove_candidate(tokenizer_path)
    except Exception as exc:
        recovery_detail = (
            f"; the original tokenizer.json is preserved at '{recovery_path}'"
            if original_recoverable
            else ""
        )
        raise _TokenizerRollbackError(
            "failed to remove the unsuccessful tokenizer.json candidate"
            f"{recovery_detail}: {exc}"
        ) from exc

    if not original_recoverable:
        return
    assert recovery_path is not None
    try:
        os.replace(recovery_path, tokenizer_path)
    except Exception as exc:
        raise _TokenizerRollbackError(
            "failed to restore the original tokenizer.json; it remains "
            f"preserved at '{recovery_path}': {exc}"
        ) from exc
    try:
        recovery_path.parent.rmdir()
    except OSError:
        pass


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
    recovery_dir: Path | None = None
    recovery_path: Path | None = None
    if had_original:
        try:
            recovery_dir, recovery_path = _reserve_recovery_path(path)
        except Exception:
            return False
    try:
        with tempfile.TemporaryDirectory(
            prefix=".internlm-tokenizer-repair-",
            dir=path,
        ) as temporary_dir:
            temporary_path = Path(temporary_dir)
            quarantined_path = (
                recovery_path
                if had_original
                else temporary_path / "original-tokenizer.json"
            )
            assert quarantined_path is not None
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
                    _rollback_candidate(
                        tokenizer_path,
                        quarantined_path,
                        recovery_path=recovery_path,
                    )
    except _TokenizerRollbackError:
        raise
    except Exception:
        return False
    finally:
        _discard_empty_recovery_dir(
            recovery_dir,
            recovery_path,
            discard_original=installed,
        )

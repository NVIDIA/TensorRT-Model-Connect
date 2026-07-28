# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tokenizer conversion primitives used by family-owned adapters."""

from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
from pathlib import Path
from typing import Iterable

from .tokenizer_validation import (
    native_tokenizer_json_error,
    tokenizer_repair_lock,
    tokenizer_repair_lock_present,
)


def _candidate_paths(model_dir: Path, candidates: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for candidate in candidates:
        if "*" in candidate or "?" in candidate:
            paths.extend(sorted(model_dir.glob(candidate)))
        else:
            path = model_dir / candidate
            if path.exists():
                paths.append(path)
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(path)
    return deduped


def ensure_unigram_tokenizer_json(
    model_dir: str | Path,
    *,
    sentencepiece_candidates: Iterable[str],
    vocab_json_name: str | None = None,
    previous_error: str | None = None,
) -> bool:
    """Generate tokenizer.json from an explicitly selected SentencePiece file."""
    path = Path(model_dir)
    tokenizer_path = path / "tokenizer.json"
    if (
        (tokenizer_path.exists() or tokenizer_path.is_symlink())
        and native_tokenizer_json_error(tokenizer_path) is None
        and not tokenizer_repair_lock_present(path)
    ):
        return True
    with tokenizer_repair_lock(path):
        if (
            (tokenizer_path.exists() or tokenizer_path.is_symlink())
            and native_tokenizer_json_error(tokenizer_path) is None
        ):
            return True
        return _ensure_unigram_tokenizer_json_under_lock(
            path,
            tokenizer_path,
            sentencepiece_candidates=sentencepiece_candidates,
            vocab_json_name=vocab_json_name,
            previous_error=previous_error,
        )


def _ensure_unigram_tokenizer_json_under_lock(
    path: Path,
    tokenizer_path: Path,
    *,
    sentencepiece_candidates: Iterable[str],
    vocab_json_name: str | None,
    previous_error: str | None,
) -> bool:
    candidates = _candidate_paths(path, sentencepiece_candidates)
    if not candidates:
        return False
    spm_path = candidates[0]

    try:
        import sentencepiece as spm_lib
        from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers
        from tokenizers.models import Unigram

        sp = spm_lib.SentencePieceProcessor()
        sp.Load(str(spm_path))
        scores = {sp.IdToPiece(i): sp.GetScore(i) for i in range(sp.GetPieceSize())}
        min_score = min(scores.values()) if scores else 0.0
        default_score = min_score - 10.0

        vocab_json_path = path / vocab_json_name if vocab_json_name else None
        combined_vocab = None
        if vocab_json_path is not None and vocab_json_path.exists():
            combined_vocab = json.loads(vocab_json_path.read_text(encoding="utf-8"))
            max_id = max(int(value) for value in combined_vocab.values())
            vocab = [("", default_score)] * (max_id + 1)
            for token, token_id in combined_vocab.items():
                vocab[int(token_id)] = (token, scores.get(token, default_score))
        else:
            vocab = [(sp.IdToPiece(i), sp.GetScore(i)) for i in range(sp.GetPieceSize())]

        unk_id = int(combined_vocab.get("<unk>", 0)) if combined_vocab else 0
        tokenizer = Tokenizer(Unigram(vocab, unk_id))
        tokenizer.normalizer = normalizers.Sequence([
            normalizers.Prepend(prepend="\u2581"),
            normalizers.Replace(" ", "\u2581"),
        ])
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([])
        tokenizer.decoder = decoders.Metaspace()
        with tempfile.TemporaryDirectory(
            prefix=".trtmc-unigram-tokenizer-repair-",
            dir=path,
        ) as temporary_dir:
            candidate_path = Path(temporary_dir) / "tokenizer.json"
            tokenizer.save(str(candidate_path))
            candidate_metadata = candidate_path.lstat()
            if (
                not stat.S_ISREG(candidate_metadata.st_mode)
                or candidate_metadata.st_size == 0
            ):
                raise RuntimeError(
                    "generated tokenizer.json must be a non-empty regular, "
                    "non-symlink file"
                )
            candidate_error = native_tokenizer_json_error(candidate_path)
            if candidate_error is not None:
                raise RuntimeError(
                    "generated tokenizer.json is not native-compatible: "
                    f"{candidate_error}"
                )
            os.replace(candidate_path, tokenizer_path)
        print(
            f"[trtmc build] Generated tokenizer.json from {spm_path.name} "
            f"({len(vocab)} tokens)",
            file=sys.stderr,
        )
        return True
    except Exception as exc:
        detail = (
            f"{previous_error}; family tokenizer conversion failed: {exc}"
            if previous_error
            else f"family tokenizer conversion failed: {exc}"
        )
        raise RuntimeError(
            f"could not generate tokenizer.json for {path}; {detail}. "
            "Install tokenizer conversion dependencies or provide tokenizer.json."
        ) from exc

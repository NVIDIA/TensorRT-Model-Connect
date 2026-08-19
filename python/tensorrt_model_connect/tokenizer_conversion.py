# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tokenizer conversion primitives used by family-owned adapters."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Callable, Iterable


def detect_tokenizer_add_special_tokens(model_dir: str | Path) -> bool:
    """Return whether the Hugging Face tokenizer adds framing tokens."""
    path = Path(model_dir)
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)
        return tokenizer.encode("hello") != tokenizer.encode(
            "hello", add_special_tokens=False
        )
    except Exception:
        pass

    tokenizer_config = path / "tokenizer_config.json"
    if tokenizer_config.is_file():
        try:
            with tokenizer_config.open(encoding="utf-8") as config_file:
                config = json.load(config_file)
            return bool(
                config.get("add_bos_token", False)
                or config.get("add_eos_token", False)
            )
        except Exception:
            pass
    return False


def detect_tokenizer_special_frame(
    model_dir: str | Path,
    *,
    revision: str | None = None,
    local_files_only: bool = False,
) -> tuple[list[int], list[int]] | None:
    """Return exact prefix/suffix IDs added by the source tokenizer."""
    try:
        from transformers import AutoTokenizer

        kwargs: dict[str, object] = {"trust_remote_code": True}
        if revision:
            kwargs["revision"] = revision
        if local_files_only:
            kwargs["local_files_only"] = True
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), **kwargs)
        framed = list(tokenizer.encode("hello"))
        plain = list(tokenizer.encode("hello", add_special_tokens=False))
    except Exception:
        return None

    if framed == plain:
        return [], []
    if not plain:
        return framed, []
    for start in range(len(framed) - len(plain) + 1):
        if framed[start : start + len(plain)] == plain:
            return framed[:start], framed[start + len(plain) :]
    return None


def _wordpiece_tokenizer_needs_rebuild(model_dir: Path) -> bool:
    tokenizer_path = model_dir / "tokenizer.json"
    vocab_path = model_dir / "vocab.txt"
    config_path = model_dir / "config.json"
    if not all(path.is_file() for path in (tokenizer_path, vocab_path, config_path)):
        return False
    try:
        tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        model = tokenizer.get("model", {})
        vocab = model.get("vocab", {})
        expected = int(config.get("vocab_size", 0))
        if model.get("type") != "WordPiece":
            return False
        if not isinstance(vocab, dict) or not vocab or expected <= 0:
            return False
        tokenizer_size = max(int(token_id) for token_id in vocab.values()) + 1
        source_size = len(vocab_path.read_text(encoding="utf-8").splitlines())
        return tokenizer_size < expected <= source_size
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def ensure_tokenizer_json(
    model_dir: str | Path,
    *,
    family_ensure: Callable[..., bool] | None = None,
    family_first: bool = False,
) -> None:
    """Ensure a native tokenizer JSON using generic or family conversion."""
    path = Path(model_dir)
    tokenizer_path = path / "tokenizer.json"
    rebuild_wordpiece = _wordpiece_tokenizer_needs_rebuild(path)
    if tokenizer_path.is_file() and not rebuild_wordpiece:
        if family_ensure is not None and not family_ensure(path, previous_error=None):
            raise RuntimeError(
                "family tokenizer validation rejected existing tokenizer.json"
            )
        return
    if family_first and family_ensure is not None:
        if family_ensure(path, previous_error=None):
            return
    if rebuild_wordpiece:
        print(
            "[trtmc build] Rebuilding undersized WordPiece tokenizer.json "
            "from vocab.txt",
            file=sys.stderr,
        )

    conversion_error: str | None = None
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(path), use_fast=True)
        with tempfile.TemporaryDirectory(prefix="trtmc-tokenizer-") as temporary:
            generated = Path(temporary) / "tokenizer.json"
            backend = getattr(tokenizer, "backend_tokenizer", None)
            if backend is None:
                backend = getattr(tokenizer, "_tokenizer", None)
            if backend is not None and hasattr(backend, "save"):
                backend.save(str(generated))
            if not generated.is_file():
                tokenizer.save_pretrained(temporary)
            if not generated.is_file():
                raise RuntimeError(
                    "tokenizer conversion did not create tokenizer.json"
                )
            with tempfile.NamedTemporaryFile(
                dir=path,
                prefix=".trtmc-tokenizer-",
                suffix=".json",
                delete=False,
            ) as output:
                temporary_path = Path(output.name)
                output.write(generated.read_bytes())
            temporary_path.replace(tokenizer_path)
        print(
            "[trtmc build] Generated tokenizer.json from source tokenizer",
            file=sys.stderr,
        )
        return
    except Exception as exc:
        conversion_error = f"fast tokenizer conversion failed: {exc}"

    if family_ensure is not None:
        if family_ensure(path, previous_error=conversion_error):
            return

    detail = conversion_error or "no tokenizer conversion was attempted"
    print(
        "[trtmc build] Warning: could not generate tokenizer.json "
        f"(C++ runtime may fail to create tokenizer): {detail}",
        file=sys.stderr,
    )


def prepare_tokenizer_special_frame(
    model_dir: str | Path,
    *,
    source_model_id_or_path: str | None = None,
    source_revision: str | None = None,
    family_ensure: Callable[..., bool] | None = None,
    family_first: bool = False,
) -> tuple[list[int], list[int]] | None:
    """Ensure tokenizer JSON while preserving the source framing contract."""
    path = Path(model_dir)
    source = source_model_id_or_path or str(path)
    source_frame = detect_tokenizer_special_frame(
        source,
        revision=source_revision,
        local_files_only=not Path(source).is_dir(),
    )
    ensure_tokenizer_json(
        path,
        family_ensure=family_ensure,
        family_first=family_first,
    )
    return source_frame if source_frame is not None else detect_tokenizer_special_frame(path)


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
    if (path / "tokenizer.json").exists():
        return True

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
        tokenizer.save(str(path / "tokenizer.json"))
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

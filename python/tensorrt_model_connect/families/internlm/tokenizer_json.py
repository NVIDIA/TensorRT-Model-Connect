# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""InternLM-owned fast-tokenizer serialization fallback."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def ensure_tokenizer_json(
    model_dir: str | Path,
    *,
    previous_error: str | None = None,
) -> bool:
    path = Path(model_dir)
    if (path / "tokenizer.json").exists():
        return True

    try:
        from sentencepiece import sentencepiece_model_pb2
        from tokenizers import AddedToken, Tokenizer, decoders, normalizers, processors
        from tokenizers.models import BPE
        from transformers.convert_slow_tokenizer import generate_merges

        model_path = path / "tokenizer.model"
        config_path = path / "tokenizer_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))

        proto = sentencepiece_model_pb2.ModelProto()
        proto.ParseFromString(model_path.read_bytes())
        if int(proto.trainer_spec.model_type) != 2:
            raise ValueError("InternLM2 tokenizer.model must use SentencePiece BPE")

        added_tokens = {
            int(token_id): AddedToken(**spec)
            for token_id, spec in config["added_tokens_decoder"].items()
        }
        vocab_scores = [
            ("<unk>", 0.0),
            ("<s>", 0.0),
            ("</s>", 0.0),
            *((piece.piece, piece.score) for piece in proto.pieces[3:]),
        ]
        for token_id, token in added_tokens.items():
            _, score = vocab_scores[token_id]
            vocab_scores[token_id] = (token.content, score)

        vocab = {
            piece: token_id
            for token_id, (piece, _score) in enumerate(vocab_scores)
        }
        tokenizer = Tokenizer(
            BPE(
                vocab,
                generate_merges(vocab, vocab_scores),
                unk_token=proto.trainer_spec.unk_piece,
                fuse_unk=True,
                byte_fallback=True,
            )
        )
        tokenizer.add_special_tokens(
            [added_tokens[token_id] for token_id in sorted(added_tokens)]
        )

        normalizer_steps = []
        if proto.normalizer_spec.add_dummy_prefix:
            normalizer_steps.append(normalizers.Prepend(prepend="\u2581"))
        normalizer_steps.append(
            normalizers.Replace(pattern=" ", content="\u2581")
        )
        tokenizer.normalizer = normalizers.Sequence(normalizer_steps)

        decoder_steps = [
            decoders.Replace("\u2581", " "),
            decoders.ByteFallback(),
            decoders.Fuse(),
        ]
        if proto.normalizer_spec.add_dummy_prefix:
            decoder_steps.append(decoders.Strip(content=" ", left=1))
        tokenizer.decoder = decoders.Sequence(decoder_steps)

        bos_token = str(config.get("bos_token", "<s>"))
        eos_token = str(config.get("eos_token", "</s>"))
        add_bos_token = bool(config.get("add_bos_token", True))
        add_eos_token = bool(config.get("add_eos_token", False))
        single = (
            f"{bos_token + ':0 ' if add_bos_token else ''}"
            f"$A:0{' ' + eos_token + ':0' if add_eos_token else ''}"
        )
        pair = (
            f"{single}"
            f"{' ' + bos_token + ':1' if add_bos_token else ''}"
            f" $B:1{' ' + eos_token + ':1' if add_eos_token else ''}"
        )
        special_tokens = []
        if add_bos_token:
            special_tokens.append((bos_token, vocab[bos_token]))
        if add_eos_token:
            special_tokens.append((eos_token, vocab[eos_token]))
        tokenizer.post_processor = processors.TemplateProcessing(
            single=single,
            pair=pair,
            special_tokens=special_tokens,
        )

        tokenizer.save(str(path / "tokenizer.json"))
        print(
            "[trtmc build] Generated InternLM BPE tokenizer.json "
            f"({len(vocab)} tokens)",
            file=sys.stderr,
        )
    except Exception as exc:
        detail = (
            f"{previous_error}; family tokenizer conversion failed: {exc}"
            if previous_error
            else f"family tokenizer conversion failed: {exc}"
        )
        raise RuntimeError(
            f"could not generate tokenizer.json for {path}; {detail}"
        ) from exc

    return (path / "tokenizer.json").exists()

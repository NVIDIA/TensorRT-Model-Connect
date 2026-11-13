#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch the pinned official SANA-WM reference with local checkpoint assets."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Sequence


_STAGE1_TEXT_ENCODER_NAME = "gemma-2-2b-it"


def _require_stage1_text_encoder(path: Path) -> Path:
    resolved = path.resolve()
    required = (resolved / "config.json", resolved / "tokenizer.json")
    missing = [str(value) for value in required if not value.is_file()]
    if not any(resolved.glob("model*.safetensors*")):
        missing.append(str(resolved / "model*.safetensors*"))
    if missing:
        raise FileNotFoundError(
            "SANA-WM local stage-1 text encoder is incomplete: " + ", ".join(missing)
        )
    return resolved


def load_local_stage1_text_encoder(path: Path, device: str) -> tuple[Any, Any]:
    """Load the exact prepared stage-1 tokenizer and decoder without Hub resolution."""

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved = _require_stage1_text_encoder(path)
    tokenizer = AutoTokenizer.from_pretrained(str(resolved), local_files_only=True)
    tokenizer.padding_side = "right"
    decoder = (
        AutoModelForCausalLM.from_pretrained(
            str(resolved),
            local_files_only=True,
            torch_dtype=torch.bfloat16,
        )
        .get_decoder()
        .to(device)
    )
    return tokenizer, decoder


def install_local_stage1_text_encoder(official: Any, path: Path) -> None:
    """Replace only the official pipeline's stage-1 asset locator."""

    resolved = _require_stage1_text_encoder(path)

    def load(name: str = "T5", device: str = "cuda") -> tuple[Any, Any]:
        if name != _STAGE1_TEXT_ENCODER_NAME:
            raise ValueError(
                "SANA-WM expected stage-1 text encoder "
                f"{_STAGE1_TEXT_ENCODER_NAME!r}, found {name!r}"
            )
        return load_local_stage1_text_encoder(resolved, device)

    official.get_tokenizer_and_text_encoder = load


def install_local_checkpoint(official: Any, repository: str, checkpoint: Path) -> None:
    """Resolve the checkpoint's own ``hf://`` URIs to its exact local snapshot.

    The checkpoint config names its VAE as ``hf://<repository>``. The official
    resolver would download the repository's default branch instead of the
    pinned revision, and fails offline when only the pinned snapshot exists.
    """

    root = checkpoint.resolve()
    prefix = f"hf://{repository}"
    resolve_other = official.resolve_hf_path

    def resolve(path: Any) -> Any:
        if not isinstance(path, str) or not path.startswith("hf://"):
            return resolve_other(path)
        if path != prefix and not path.startswith(prefix + "/"):
            raise FileNotFoundError(f"SANA-WM reference requested an undeclared Hub asset: {path}")
        local = root / path[len(prefix) :].lstrip("/")
        if not local.exists():
            raise FileNotFoundError(f"SANA-WM checkpoint asset is missing: {local}")
        return str(local)

    official.resolve_hf_path = resolve


def _official_module(reference_repository: Path) -> tuple[Path, Any]:
    repository = reference_repository.resolve()
    entrypoint = repository / "inference_video_scripts/wm/inference_sana_wm.py"
    if not entrypoint.is_file():
        raise FileNotFoundError(f"official SANA-WM entrypoint does not exist: {entrypoint}")
    if str(repository) not in sys.path:
        sys.path.insert(0, str(repository))
    from inference_video_scripts.wm import inference_sana_wm

    return entrypoint, inference_sana_wm


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(add_help=False)
    value.add_argument("--reference-repo", required=True, type=Path)
    value.add_argument("--stage1-text-encoder", required=True, type=Path)
    value.add_argument("--checkpoint-repo", required=True)
    value.add_argument("--checkpoint-dir", required=True, type=Path)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    arguments, official_arguments = parser().parse_known_args(argv)
    entrypoint, official = _official_module(arguments.reference_repo)
    install_local_stage1_text_encoder(official, arguments.stage1_text_encoder)
    install_local_checkpoint(official, arguments.checkpoint_repo, arguments.checkpoint_dir)
    previous = sys.argv
    try:
        sys.argv = [str(entrypoint), *official_arguments]
        result = official.main()
    finally:
        sys.argv = previous
    return int(result or 0)


if __name__ == "__main__":
    raise SystemExit(main())

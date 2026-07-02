#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare a local GitHub ELF checkpoint directory for TRTMC build.

Upstream ELF evaluates with separate ``--config`` and ``--checkpoint_path``
arguments. TRTMC builds from one model directory, so this tool assembles a
local directory containing the upstream YAML config, checkpoint files, and
local tokenizer files without downloading anything.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


CHECKPOINT_NAMES = ("model.npz", "elf_params.npz")
TOKENIZER_NAMES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "spiece.model",
    "tokenizer.model",
    "vocab.json",
)
ENCODER_CHECKPOINT_NAMES = (
    "t5_small_encoder_jax.pkl",
    "encoder_checkpoint.pkl",
    "text_encoder.pkl",
    "t5_encoder.pkl",
)


def _install_path(src: Path, dst: Path, *, copy: bool) -> None:
    if dst.exists() or dst.is_symlink():
        return
    if copy:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve(), target_is_directory=src.is_dir())


def _checkpoint_candidates(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"checkpoint path does not exist: {path}")
    candidates: list[Path] = []
    for name in CHECKPOINT_NAMES:
        candidate = path / name
        if candidate.exists():
            candidates.append(candidate)
    candidates.extend(sorted(path.glob("checkpoint_*")))
    if not candidates:
        raise FileNotFoundError(
            f"no ELF checkpoint files found in {path}; expected checkpoint_*, "
            "model.npz, or elf_params.npz"
        )
    return candidates


def _install_tokenizer(tokenizer: Path | None, output_dir: Path, *, copy: bool) -> list[str]:
    if tokenizer is None:
        return []
    if not tokenizer.exists():
        raise FileNotFoundError(f"tokenizer path does not exist: {tokenizer}")
    installed: list[str] = []
    if tokenizer.is_file():
        _install_path(tokenizer, output_dir / tokenizer.name, copy=copy)
        return [tokenizer.name]
    for name in TOKENIZER_NAMES:
        src = tokenizer / name
        if src.exists():
            _install_path(src, output_dir / name, copy=copy)
            installed.append(name)
    for src in sorted(tokenizer.glob("*.spm")):
        _install_path(src, output_dir / src.name, copy=copy)
        installed.append(src.name)
    return installed


def _install_encoder_checkpoint(
    checkpoint: Path | None, output_dir: Path, *, copy: bool
) -> str:
    if checkpoint is None:
        return ""
    if not checkpoint.exists() or not checkpoint.is_file():
        raise FileNotFoundError(f"encoder checkpoint does not exist: {checkpoint}")
    dst_name = (
        checkpoint.name
        if checkpoint.name in ENCODER_CHECKPOINT_NAMES
        else "t5_small_encoder_jax.pkl"
    )
    _install_path(checkpoint, output_dir / dst_name, copy=copy)
    return dst_name


def prepare_model_dir(args: argparse.Namespace) -> dict[str, object]:
    config = Path(args.config)
    checkpoint_path = Path(args.checkpoint_path)
    output_dir = Path(args.output)
    if not config.exists():
        raise FileNotFoundError(f"config file does not exist: {config}")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    config_name = config.name if config.suffix in {".yml", ".yaml"} else "config.yml"
    _install_path(config, output_dir / config_name, copy=args.copy)

    checkpoints = _checkpoint_candidates(checkpoint_path)
    installed_checkpoints: list[str] = []
    for src in checkpoints:
        _install_path(src, output_dir / src.name, copy=args.copy)
        installed_checkpoints.append(src.name)

    tokenizer = Path(args.tokenizer) if args.tokenizer else None
    installed_tokenizer = _install_tokenizer(tokenizer, output_dir, copy=args.copy)
    encoder_checkpoint = Path(args.encoder_checkpoint) if args.encoder_checkpoint else None
    installed_encoder = _install_encoder_checkpoint(
        encoder_checkpoint, output_dir, copy=args.copy)

    readme = output_dir / "README.trtmc.txt"
    readme.write_text(
        "\n".join(
            [
                "TRTMC GitHub ELF build directory",
                "",
                f"source_config: {config.resolve()}",
                f"source_checkpoint_path: {checkpoint_path.resolve()}",
                f"checkpoint_entries: {', '.join(installed_checkpoints)}",
                f"tokenizer_entries: {', '.join(installed_tokenizer) or '(none)'}",
                f"encoder_checkpoint: {installed_encoder or '(none)'}",
                "",
                "Build example:",
                f"python -m tensorrt_model_connect.__main__ build {output_dir} "
                "-o elf.trtfb --method trt",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "output_dir": str(output_dir),
        "config": config_name,
        "checkpoints": installed_checkpoints,
        "tokenizer_files": installed_tokenizer,
        "encoder_checkpoint": installed_encoder,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="upstream train_*.yml config")
    parser.add_argument("--checkpoint-path", required=True,
                        help="upstream ELF checkpoint file or directory")
    parser.add_argument("--tokenizer", default="",
                        help="local tokenizer file/dir, ideally containing tokenizer.json")
    parser.add_argument("--encoder-checkpoint", default="",
                        help="official ELF JAX T5 encoder checkpoint .pkl")
    parser.add_argument("--output", required=True, help="directory to create for TRTMC build")
    parser.add_argument("--copy", action="store_true",
                        help="copy files instead of creating symlinks")
    parser.add_argument("--force", action="store_true",
                        help="allow reusing a non-empty output directory")
    args = parser.parse_args()

    result = prepare_model_dir(args)
    print(f"Prepared ELF model directory: {result['output_dir']}")
    if not result["tokenizer_files"]:
        print("Warning: no tokenizer files were installed; runtime text decode needs tokenizer.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

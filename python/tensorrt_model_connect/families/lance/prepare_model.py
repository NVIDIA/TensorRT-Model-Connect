#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage a ``bytedance-research/Lance`` checkpoint into a buildable directory.

The Lance HF repo is not a flat HF checkpoint: the LLM lives under
``Lance_3B/`` (or ``Lance_3B_Video/``) as ``llm_config.json`` +
``model.safetensors``, and the Qwen2.5-VL ViT is a separate ``Qwen2.5-VL-ViT/``
dir. ``trtmc build`` expects a single directory with a ``config.json`` whose
``model_type`` selects a family plugin.

This script writes a staged directory (symlinks, no copies of large weights):

    <out>/
      config.json            # = <variant>/llm_config.json, model_type -> "lance"
      model.safetensors      # -> <variant>/model.safetensors  (Lance LLM)
      tokenizer.json, vocab.json, merges.txt, generation_config.json [, tokenizer_config.json]
      vision/model.safetensors  # -> Qwen2.5-VL-ViT/vit.safetensors

Then build it (understanding path):

    ./build/trtmc build <out> -o /tmp/lance.trtfb --max-cache-length 384 --precision bf16

Generation/editing tasks (t2i/t2v/edit) are not supported yet; this stages the
understanding sub-model that the ``lance`` family plugin builds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

_VARIANT_DIRS = {"image": "Lance_3B", "video": "Lance_3B_Video"}
# Files the native tokenizer / builder may look for at the top level.
_TOP_LEVEL_FILES = [
    "model.safetensors",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "generation_config.json",
    "tokenizer_config.json",
]


def _resolve_src(src: str | None) -> Path:
    if src:
        return Path(src).expanduser().resolve()
    # Fall back to the local HF cache snapshot.
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("error: --src not given and huggingface_hub is unavailable")
    return Path(snapshot_download(
        repo_id="bytedance-research/Lance",
        allow_patterns=["Lance_3B/*", "Lance_3B_Video/*", "Qwen2.5-VL-ViT/*"],
    )).resolve()


def _symlink(target: Path, link: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        link.unlink()
    os.symlink(target, link)


def stage_model_dir(src: Path, out: Path, *, variant: str = "image") -> Path:
    """Stage one Lance variant and return its buildable model directory."""
    src = src.expanduser().resolve()
    out = out.expanduser().resolve()
    llm_dir = src / _VARIANT_DIRS[variant]
    vit_path = src / "Qwen2.5-VL-ViT" / "vit.safetensors"

    if not (llm_dir / "llm_config.json").exists():
        raise FileNotFoundError(
            f"{llm_dir}/llm_config.json not found (bad source/variant?)")
    if not vit_path.exists():
        raise FileNotFoundError(f"{vit_path} not found")

    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((llm_dir / "llm_config.json").read_text())
    cfg["model_type"] = "lance"
    (out / "config.json").write_text(json.dumps(cfg, indent=2))

    for name in _TOP_LEVEL_FILES:
        srcf = llm_dir / name
        if srcf.exists():
            _symlink(srcf, out / name)
        elif name == "model.safetensors":
            raise FileNotFoundError(f"{srcf} not found")

    _symlink(vit_path, out / "vision" / "model.safetensors")
    return out


def resolve_model_dir(src: Path) -> Path | None:
    """Stage a downloaded non-flat Lance repository for normal builds."""
    if not (src / "Lance_3B" / "llm_config.json").exists():
        return None
    digest = hashlib.sha256(str(src.resolve()).encode()).hexdigest()[:12]
    staging_root = Path(os.environ.get(
        "TRTMC_FAMILY_MODEL_ROOT",
        str(Path(tempfile.gettempdir()) / "trtmc-family-models"),
    ))
    return stage_model_dir(src, staging_root / f"lance-image-{digest}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=None,
                    help="Lance HF snapshot dir (default: download via huggingface_hub)")
    ap.add_argument("--variant", choices=sorted(_VARIANT_DIRS), default="image",
                    help="image -> Lance_3B (default); video -> Lance_3B_Video")
    ap.add_argument("--out", required=True, help="Output staged directory")
    args = ap.parse_args()

    src = _resolve_src(args.src)
    out = Path(args.out).expanduser().resolve()
    stage_model_dir(src, out, variant=args.variant)
    llm_dir = src / _VARIANT_DIRS[args.variant]
    vit_path = src / "Qwen2.5-VL-ViT" / "vit.safetensors"

    print(f"Staged Lance ({args.variant}) at: {out}")
    print(f"  LLM : {llm_dir}")
    print(f"  ViT : {vit_path}")
    print("Build with:")
    print(f"  ./build/trtmc build {out} -o /tmp/lance.trtfb "
          f"--max-cache-length 384 --precision bf16")
    print("  ./build/trtmc run /tmp/lance.trtfb --prompt 'Describe this image.' "
          "--image <img> --max-new-tokens 40 --greedy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

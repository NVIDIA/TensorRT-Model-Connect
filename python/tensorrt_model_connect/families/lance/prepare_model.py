#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage a bytedance-research/Lance checkpoint for Model Connect."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tensorrt_model_connect.families.lance.config import (
    LANCE_VARIANT_DIRS,
    stage_model_dir,
)


def _resolve_src(src: str | None) -> Path:
    if src:
        return Path(src).expanduser().resolve()
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("error: --src not given and huggingface_hub is unavailable")
    return Path(
        snapshot_download(
            repo_id="bytedance-research/Lance",
            allow_patterns=[
                "Lance_3B/*",
                "Lance_3B_Video/*",
                "Qwen2.5-VL-ViT/*",
            ],
        )
    ).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        help="Lance HF snapshot directory (default: download from Hugging Face)",
    )
    parser.add_argument(
        "--variant",
        choices=sorted(LANCE_VARIANT_DIRS),
        default="image",
    )
    parser.add_argument("--out", required=True, help="Output staged directory")
    args = parser.parse_args()

    src = _resolve_src(args.src)
    out = stage_model_dir(src, Path(args.out), variant=args.variant)
    llm_dir = src / LANCE_VARIANT_DIRS[args.variant]
    vit_path = src / "Qwen2.5-VL-ViT" / "vit.safetensors"
    print(f"Staged Lance ({args.variant}) at: {out}")
    print(f"  LLM : {llm_dir}")
    print(f"  ViT : {vit_path}")
    print(
        f"Build with:\n  ./build/trtmc build {out} -o /tmp/lance.trtfb "
        "--max-cache-length 384 --precision bf16"
    )
    print(
        "  ./build/trtmc run /tmp/lance.trtfb --prompt 'Describe this image.' "
        "--image <img> --max-new-tokens 40 --greedy"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Stage a ``bytedance-research/Lance`` checkpoint into a buildable directory.

``trtmc build bytedance-research/Lance`` now stages the nested Lance repo
automatically (see ``tensorrt_model_connect.families.lance.staging``), so this
script is mainly a convenience for pre-staging or inspecting the layout. The
staging logic lives in the family module and is shared with the builder.

    python scripts/prepare_lance_model.py --variant image --out downloads/lance-3b
    ./build/trtmc build downloads/lance-3b -o /tmp/lance.trtfb \
        --max-cache-length 384 --precision bf16
"""
from __future__ import annotations

import argparse
from pathlib import Path

from tensorrt_model_connect.families.lance.staging import (
    VARIANT_DIRS,
    download_lance_repo,
    stage_lance_repo,
)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=None,
                    help="Lance HF snapshot dir (default: download via huggingface_hub)")
    ap.add_argument("--variant", choices=sorted(VARIANT_DIRS), default="image",
                    help="image -> Lance_3B (default); video -> Lance_3B_Video")
    ap.add_argument("--out", required=True, help="Output staged directory")
    args = ap.parse_args()

    src = Path(args.src).expanduser().resolve() if args.src else download_lance_repo()
    out = stage_lance_repo(src, Path(args.out).expanduser().resolve(), variant=args.variant)

    print(f"Staged Lance ({args.variant}) at: {out}")
    print("Build with:")
    print(f"  ./build/trtmc build {out} -o /tmp/lance.trtfb "
          f"--max-cache-length 384 --precision bf16")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

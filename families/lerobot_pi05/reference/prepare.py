# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare the native pi0.5 reference environment: the public PaliGemma tokenizer copy the catalog manifests
name (tokenizer.py's size check) under ``<environment>/trtmc-reference/paligemma-tokenizer``, the directory
LeRobot's tokenizer processor reads in place of the gated repository. The copy is the bare SentencePiece model, so a
``tokenizer_config.json`` names the Gemma tokenizer class and its special pieces (``SPECIAL_IDS``) as the gated
repository's config does; without it the processor's max-length padding has no pad token."""

import json
from pathlib import Path
import shutil
import sys
import tempfile
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from families.lerobot_pi05.tokenizer import PUBLIC_COPY, TOKENIZER_BYTES, TOKENIZER_FILE  # noqa: E402


TOKENIZER_CONFIG = {"tokenizer_class": "GemmaTokenizer", "pad_token": "<pad>", "eos_token": "<eos>",
                    "bos_token": "<bos>", "unk_token": "<unk>", "add_bos_token": True, "add_eos_token": False}


def main() -> int:
    destination = Path(sys.prefix) / "trtmc-reference" / "paligemma-tokenizer"
    if not (destination / TOKENIZER_FILE).is_file():
        _download(destination)
    (destination / "tokenizer_config.json").write_text(json.dumps(TOKENIZER_CONFIG, indent=1) + "\n")
    return 0


def _download(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="paligemma.", dir=destination.parent))
    try:
        with urllib.request.urlopen(PUBLIC_COPY, timeout=600) as response:
            data = response.read()
        if len(data) != TOKENIZER_BYTES:
            raise RuntimeError(f"the PaliGemma tokenizer copy has {len(data)} bytes, not {TOKENIZER_BYTES}")
        (temporary / TOKENIZER_FILE).write_bytes(data)
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())

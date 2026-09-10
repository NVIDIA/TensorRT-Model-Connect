# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Download pinned upstream model assets locally; no model files ship in the example."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    os.environ["HF_HOME"] = str(workspace / "models" / "huggingface")
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    from huggingface_hub import snapshot_download

    model_path = workspace / "models" / "Nemotron-VoiceChat-11B"
    snapshot_download(
        "nvidia/NVIDIA-NemotronLabs-VoiceChat-11B",
        revision="359ada7b1c60851e40ff08065f9b0340244f27e0",
        local_dir=model_path,
        allow_patterns=["config.json", "model.safetensors", "rnnt_tokenizer/*", "LICENSE", "README.md"],
        max_workers=4,
    )
    snapshot_download(
        "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
        revision="6533e8de2c68e4536bf7c411d7a3ce5734111476",
        allow_patterns=["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"],
    )
    print("Verifying the checkpoint SHA256...", flush=True)
    with (model_path / "model.safetensors").open("rb") as checkpoint:
        digest = hashlib.file_digest(checkpoint, "sha256").hexdigest()
    expected = "d553750c29434a6bb524377e17634c6cafdbf621892e643a77f406e51570354b"
    if digest != expected:
        raise RuntimeError("The downloaded checkpoint does not match the pinned SHA256.")
    print(f"Checkpoint and tokenizer assets are ready. SHA256: {digest}", flush=True)


if __name__ == "__main__":
    main()

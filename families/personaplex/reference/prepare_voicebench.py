# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare a small source overlay and official prompt for the VoiceBench profile.

Checkpoints stay in the shared HF cache. The environment contains symlinks,
one modified config, the selected voice and a derived prompt artifact only.
"""

from __future__ import annotations

import io
import json
import math
from pathlib import Path
import sys
import tarfile

from datasets import Audio, load_dataset
from huggingface_hub import snapshot_download
import soundfile

from families.personaplex.reference.prepare import main as prepare_official
from families.personaplex.system_prompt import prepare

MODEL_REVISION = "fdaf4090a61cb315c138a1faee287ffd6c716309"
DATASET_REVISION = "b02edcef1330480be3a11bd6f7434ac32f05ad08"
TEXT_PROMPT = ("You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way. "
               "For multiple-choice questions, say the option letter and its answer. Keep your answer brief.")


def main() -> None:
    prepare_official()
    source = Path(snapshot_download("nvidia/personaplex-7b-v1", revision=MODEL_REVISION))
    target = Path(sys.prefix) / "trtmc-reference/personaplex-model"
    target.mkdir(parents=True, exist_ok=True)
    for p in source.iterdir():
        if p.name != "config.json" and not (target / p.name).exists():
            (target / p.name).symlink_to(p)
    voice = target / "NATM1.pt"
    with tarfile.open(source / "voices.tgz") as archive:
        voice.write_bytes(archive.extractfile("voices/NATM1.pt").read())
    prompt = target / "system_prompt.npz"
    metadata = prepare(source, voice, TEXT_PROMPT, "fp16", prompt)
    data = load_dataset("hlt-lab/voicebench", "openbookqa", split="test", revision=DATASET_REVISION)
    data = data.cast_column("audio", Audio(decode=False))
    if len(data) != 455:
        raise ValueError("the pinned VoiceBench OpenBookQA split must contain 455 questions")
    durations = [soundfile.info(io.BytesIO(row["audio"]["bytes"])).duration for row in data]
    required_frames = metadata["frames"] + math.ceil(max(durations) * 12.5) + 125
    capacity = math.ceil(required_frames / 128) * 128
    config = json.loads((source / "config.json").read_text())
    config.update(personaplex_system_prompt="system_prompt.npz", personaplex_benchmark_capacity=capacity,
                  personaplex_listen_before_answer=True)
    (target / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (target / "benchmark-capacity.json").write_text(json.dumps({"dataset_revision": DATASET_REVISION,
        "samples": len(data), "max_question_seconds": max(durations), "prompt_frames": metadata["frames"],
        "reply_frames": 125, "required_frames": required_frames, "capacity": capacity}, indent=2) + "\n")
    print(json.dumps({"source": str(target), "prompt": metadata, "capacity": capacity}))


if __name__ == "__main__":
    main()

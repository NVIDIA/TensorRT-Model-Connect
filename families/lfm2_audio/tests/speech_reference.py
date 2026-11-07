# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Transcribe saved original/native PCM with the pinned original ASR model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from liquid_audio import ChatState, LFM2AudioModel, LFM2AudioProcessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    model = LFM2AudioModel.from_pretrained(Path(args.checkpoint)).eval()
    processor = LFM2AudioProcessor.from_pretrained(Path(args.checkpoint)).eval()
    records = []
    snapshots = Path(args.output).with_suffix(".inputs")
    snapshots.mkdir(parents=True, exist_ok=True)
    for index, job in enumerate(json.loads(Path(args.jobs).read_text())):
        blob = Path(job["waveform"]).read_bytes()
        snapshot = snapshots / f"{index:03d}.f32"
        snapshot.write_bytes(blob)
        wave = np.frombuffer(blob, np.float32).copy()
        if not wave.size or not np.isfinite(wave).all():
            raise ValueError("speech-content proof requires a finite waveform")
        chat = ChatState(processor)
        chat.new_turn("system")
        chat.add_text("Perform ASR.")
        chat.end_turn()
        chat.new_turn("user")
        chat.add_audio(torch.from_numpy(wave)[None], job.get("sample_rate", 24000))
        chat.end_turn()
        chat.new_turn("assistant")
        output = list(
            model.generate_sequential(
                **chat,
                max_new_tokens=512,
                text_temperature=0.0,
                text_top_k=1,
                audio_temperature=0.0,
                audio_top_k=1,
            )
        )
        ids = [int(token) for token in output if token.numel() == 1]
        record = job | {
            "asr_input": str(snapshot),
            "samples": int(wave.size),
            "transcript": processor.text.decode(ids, skip_special_tokens=True),
            "raw_transcript": processor.text.decode(ids),
            "tokens": ids,
            "asr_ended": bool(ids and ids[-1] == 7),
            "unexpected_audio_frames": sum(token.numel() == 8 for token in output),
        }
        records.append(record)
        Path(args.output).write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    with torch.inference_mode():
        main()

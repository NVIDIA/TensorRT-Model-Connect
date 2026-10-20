# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Transcribe saved generated audio after benchmark timing, without consulting labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--last-seconds", type=float, default=0.0,
                        help="evaluate only the final reply window; zero keeps the complete audio")
    args = parser.parse_args()
    if args.last_seconds < 0:
        parser.error("--last-seconds must not be negative")

    import librosa
    import soundfile
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model, revision=args.revision, dtype=torch.float16,
        attn_implementation="sdpa").to("cuda").eval()
    rows = json.loads(args.input.read_text())
    for index, observation in rows.items():
        if observation.get("audio_artifact"):
            path = Path(observation["audio_artifact"])
            if not path.is_absolute():
                path = Path(observation["artifact_directory"]) / path
            audio, rate = soundfile.read(path, dtype="float32")
        else:
            path = Path(observation["artifact"])
            audio, rate = np.load(path, allow_pickle=False), int(observation["sample_rate"])
        audio = np.asarray(audio).reshape(-1)
        window_start = max(0, audio.size - round(args.last_seconds * rate)) if args.last_seconds else 0
        finite = np.isfinite(audio).all()
        audio = audio[window_start:]
        if not finite:
            text = ""
        elif not audio.size or not np.any(audio):
            text = ""
        else:
            if rate != 16000:
                audio = librosa.resample(audio, orig_sr=rate, target_sr=16000)
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt", return_attention_mask=True,
                               truncation=False, padding="longest")
            with torch.inference_mode():
                generated = model.generate(
                    inputs.input_features.to("cuda", dtype=torch.float16),
                    attention_mask=inputs.attention_mask.to("cuda"), language="english", task="transcribe",
                    do_sample=False, temperature=0.0, num_beams=1, max_new_tokens=256,
                    return_timestamps=inputs.input_features.shape[-1] > 3000,
                    no_speech_threshold=0.6, logprob_threshold=-1.0, compression_ratio_threshold=2.4)
            text = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        observation.update(text=text, asr={"model": args.model, "revision": args.revision,
                                          "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                          "window_start_sample": window_start, "source_sample_rate": rate,
                                          "last_seconds": args.last_seconds})
        print(json.dumps({"index": index, "text": text}), flush=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()

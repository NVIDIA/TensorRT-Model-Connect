# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare native signal processing directly with the pinned liquid-audio code."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
import torchaudio
from liquid_audio.detokenizer import ISTFT
from liquid_audio.model.conformer.processor import AudioToMelSpectrogramPreprocessor

from families.lfm2_audio.model import mel_filters


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--probe", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(exist_ok=True, parents=True)
    raw = json.loads((Path(args.checkpoint) / "config.json").read_text())
    proc = AudioToMelSpectrogramPreprocessor(**raw["preprocessor"]).eval()
    filters = out / "filters.f32"
    mel_filters().tofile(filters)
    rng = np.random.default_rng(42)
    report = {}

    def compare(mode, name, source, ref, rate):
        ip = out / (name + ".input.f32")
        op = out / (name + ".output.f32")
        np.asarray(source, np.float32).tofile(ip)
        subprocess.run(
            [args.probe, mode, str(ip), str(op), str(rate), str(filters)],
            check=True,
            capture_output=True,
        )
        actual = np.fromfile(op, np.float32).reshape(ref.shape)
        diff = actual - ref
        report[name] = {
            "max_abs": float(np.max(np.abs(diff))),
            "rms": float(np.sqrt(np.mean(diff**2))),
            "bf16_equal_fraction": float(
                (torch.tensor(actual).bfloat16() == torch.tensor(ref).bfloat16()).float().mean()
            ),
        }
        return actual

    for rate in (8000, 16000, 22050, 24000, 44100, 48000):
        audio = (0.1 * rng.standard_normal(rate * 2 + 33)).astype(np.float32)
        wave = torch.tensor(audio)[None]
        ref = torchaudio.functional.resample(wave, rate, 16000)
        compare("resample", f"resample_{rate}", audio, ref.numpy()[0], rate)
        mel, _ = proc(ref, torch.tensor([ref.shape[1]]))
        compare("mel", f"mel_{rate}", audio, mel.numpy()[0].T, rate)
    istft = ISTFT(1280, 320, 1280)
    for frames in (6, 12, 36):
        logabs = torch.tensor(rng.normal(-2, 0.7, (1, 641, frames)), dtype=torch.float32)
        angle = torch.tensor(rng.uniform(-np.pi, np.pi, (1, 641, frames)), dtype=torch.float32)
        ref = istft(torch.polar(logabs.exp(), angle)).numpy()[0]
        source = torch.cat([logabs, angle], dim=1)[0].T.numpy()
        compare("istft", f"istft_{frames}", source, ref, 24000)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

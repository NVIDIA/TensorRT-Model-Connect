# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regenerate the golden data of ``tests/cpp/test_runtime.cpp`` from the reference.

    PYTHONPATH=core/builder:. python3 -m families.nemotron3_diarization.tests.generate_runtime_golden

Needs torch, librosa and transformers (the release in ``requirements.txt``).
Writes ``tests/data/runtime_golden.json`` with three reference cases and the
front-end constants the family's Python side ships in a bundle:

- ``front_end``: a deterministic signal and the reference feature extractor's
  log-mel features and valid frame count;
- ``speaker_cache``: a small speaker-cache policy driven through the reference
  ``Nemotron3DiarizationSpeakerCache`` with random logits, and its state after
  every step, compressions included;
- ``intervals``: random logits and the reference processor's speaker segments,
  at two thresholds;

and ``tests/data/mel_filters.f32`` / ``tests/data/window.f32`` from
``frontend.py``, which ``tests/test_model.py`` holds equal to the files.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformers.models.nemotron3_diarization.configuration_nemotron3_diarization import (
    Nemotron3DiarizationStreamingConfig,
)
from transformers.models.nemotron3_diarization.modeling_nemotron3_diarization import (
    Nemotron3DiarizationSpeakerCache,
)
from transformers.models.nemotron3_diarization.processing_nemotron3_diarization import (
    Nemotron3DiarizationProcessor,
)
from transformers.models.nemotron_asr_streaming.feature_extraction_nemotron_asr_streaming import (
    NemotronAsrStreamingFeatureExtractor,
)

from families.nemotron3_diarization import frontend

DATA = Path(__file__).resolve().parent / "data"
SAMPLE_RATE, N_FFT, WIN_LENGTH, HOP, MELS = 16000, 512, 400, 160, 128


def _extractor() -> NemotronAsrStreamingFeatureExtractor:
    return NemotronAsrStreamingFeatureExtractor(
        feature_size=MELS, sampling_rate=SAMPLE_RATE, hop_length=HOP, n_fft=N_FFT,
        win_length=WIN_LENGTH, preemphasis=0.97)


def _float64_features(samples: np.ndarray, valid_frames: int) -> np.ndarray:
    """The reference's front end with a float64 transform.

    Pre-emphasis and the windowing products are rounded to float32 where the
    reference rounds them; the transform, the power, the mel sums and the log
    are exact to float64. The reference itself transforms in float32, which
    puts its own features up to 2.4e-4 away from these at the mel bins whose
    energy sits near the 2**-24 floor.
    """
    previous = (np.float32(0.97) * samples[:-1]).astype(np.float32)
    emphasised = np.concatenate([samples[:1], (samples[1:] - previous).astype(np.float32)])
    padded = np.concatenate([np.zeros(N_FFT // 2, np.float32), emphasised,
                             np.zeros(N_FFT // 2, np.float32)])
    window = np.zeros(N_FFT, np.float32)
    offset = (N_FFT - WIN_LENGTH) // 2
    window[offset: offset + WIN_LENGTH] = frontend.symmetric_hann(WIN_LENGTH)
    filters = frontend.mel_filters(MELS, N_FFT, SAMPLE_RATE).astype(np.float64)
    features = np.zeros((1 + samples.size // HOP, MELS))
    for frame in range(valid_frames):
        windowed = (padded[frame * HOP: frame * HOP + N_FFT] * window).astype(np.float64)
        power = np.abs(np.fft.rfft(windowed)) ** 2
        features[frame] = np.log(filters @ power + 2.0**-24)
    return features


def front_end_case() -> dict:
    # 0.05 s plus one sample: six frames, the last of them past the audio.
    time = np.arange(801, dtype=np.float64) / SAMPLE_RATE
    signal = (0.5 * np.sin(2 * np.pi * 440.0 * time)
              + 0.25 * np.sin(2 * np.pi * 3100.0 * time + 0.3)
              + 0.05 * np.sin(2 * np.pi * 900.0 * time * time * 40.0))
    samples = signal.astype(np.float32)
    output = _extractor()(samples, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    valid = int(output["attention_mask"][0].sum())
    return {
        "samples": samples.tolist(),
        "features": output["input_features"][0].numpy().tolist(),
        "features_float64": _float64_features(samples, valid).tolist(),
        "valid_frames": valid,
    }


def speaker_cache_case() -> dict:
    # The FIFO overflows by less than the update period at first and by more
    # later, so frames stay queued between steps and both pop sizes occur.
    policy = {
        "num_speakers": 3, "subsampling_factor": 2, "hidden_size": 6,
        "cache_length": 12, "silence_frames_per_speaker": 1, "fifo_length": 4,
        "update_period": 3, "score_threshold": 0.25, "latest_frames_boost": 0.05,
    }
    config = Nemotron3DiarizationStreamingConfig(
        fifo_length=policy["fifo_length"],
        speaker_cache_update_period=policy["update_period"],
        speaker_cache_length=policy["cache_length"],
        speaker_cache_silence_frames_per_speaker=policy["silence_frames_per_speaker"],
        prediction_score_threshold=policy["score_threshold"],
        latest_frames_score_boost=policy["latest_frames_boost"],
        num_speakers=policy["num_speakers"], subsampling_factor=policy["subsampling_factor"])
    cache = Nemotron3DiarizationSpeakerCache(config)
    policy.update(min_positive_scores=cache.min_positive_scores,
                  strong_boosted_frames=cache.num_strong_boosted_frames,
                  weak_boosted_frames=cache.num_weak_boosted_frames)

    generator = torch.Generator().manual_seed(234)
    width, factor, speakers = policy["hidden_size"], policy["subsampling_factor"], 3
    silence = torch.randn(width, generator=generator)
    chunk, lookahead, steps = 5, 2, []
    for index in range(7):
        # One step carries look-back frames between the queued rows and the chunk.
        lookback = 2 if index == 4 else 0
        chunk_embeds = torch.randn(1, lookback + chunk + lookahead, width, generator=generator)
        cached = cache.get_embeds(chunk_embeds)
        step_input = torch.cat([cached, chunk_embeds], dim=1)
        rows = step_input.shape[1]
        # Speaker activity that changes over the steps, so every speaker is
        # sometimes speaking, sometimes overlapped and sometimes silent.
        logits = torch.randn(1, rows * factor, speakers, generator=generator) * 3.0
        logits[..., index % speakers] += 2.5
        mask = torch.ones(1, rows, dtype=torch.bool)
        if index == 6:
            mask[0, -1] = False  # a padding frame at the end of the recording
        cache.update(step_input, logits, silence, chunk, mask=mask, num_lookback_frames=lookback)
        steps.append({
            "step_input": step_input[0].tolist(),
            "logits": logits[0].tolist(),
            "valid": mask[0].int().tolist(),
            "chunk_frames": chunk,
            "lookback_frames": lookback,
            "cache_rows": cache.num_cache_frames,
            "fifo_rows": cache.num_fifo_frames,
            "compressed": cache.is_compressed,
            "cache": cache.embeds[0, : cache.num_cache_frames].tolist(),
            "cache_probs": cache.probs[0, : cache.num_cache_frames].tolist(),
            "fifo": cache.fifo[0, : cache.num_fifo_frames].tolist(),
        })
    return {"policy": policy, "silence": silence.tolist(), "steps": steps}


def intervals_case() -> dict:
    processor = Nemotron3DiarizationProcessor(feature_extractor=_extractor())
    generator = torch.Generator().manual_seed(53)
    logits = torch.randn(1, 60, 4, generator=generator).cumsum(dim=1) * 0.7
    cases = []
    for threshold in (0.5, 0.3):
        segments = processor.extract_speaker_dict(logits, threshold=threshold)[0]
        cases.append({"threshold": threshold, "segments": [
            [segment["Start"], segment["End"], segment["Speaker"]] for segment in segments]})
    return {"logits": logits[0].tolist(), "cases": cases}


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    golden = {"front_end": front_end_case(), "speaker_cache": speaker_cache_case(),
              "intervals": intervals_case()}
    (DATA / "runtime_golden.json").write_text(json.dumps(golden) + "\n", encoding="utf-8")
    frontend.mel_filters(MELS, N_FFT, SAMPLE_RATE).tofile(DATA / "mel_filters.f32")
    frontend.symmetric_hann(WIN_LENGTH).tofile(DATA / "window.f32")


if __name__ == "__main__":
    main()

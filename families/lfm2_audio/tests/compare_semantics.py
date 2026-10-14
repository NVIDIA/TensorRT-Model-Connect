# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Evaluate saved native/public results and auditable original-ASR transcripts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .semantic_contract import (
        dialogue_content,
        interleaved_schedule,
        red_light_content,
        spoken_content,
        words,
    )
except ImportError:
    from semantic_contract import (
        dialogue_content,
        interleaved_schedule,
        red_light_content,
        spoken_content,
        words,
    )


def waveform(path):
    blob = Path(path).read_bytes()
    samples = np.frombuffer(blob, np.float32)
    return samples, blob


def compare(cases, native, reference, public, transcripts):
    native = {item["name"]: item for item in native}
    reference = {item["name"]: item for item in reference}
    public = {item["name"]: item for item in public}
    transcripts = {
        (item["name"], item.get("turn", 0), item["producer"]): item for item in transcripts
    }
    rows = []
    for case in cases:
        name, mode = case["name"], case["mode"]
        actual, original = native[name], reference[name]["turns"]
        checks, details = {}, []
        if mode in {"asr", "mixed", "text"}:
            output = public[name] if mode == "text" else actual
            checks["text_only_output"] = (
                bool(output["tokens"]) and not actual.get("trace", {}).get("codes", [[]])[0]
            )
            if mode == "asr":
                checks["transcription_content"] = words(output["text"]) == words(
                    original[0]["text"]
                )
            elif mode == "mixed":
                checks["mixed_request_content"] = all(
                    dialogue_content(text, 0) for text in (output["text"], original[0]["text"])
                )
                checks["public_api_result"] = (
                    public[name]["tokens"] == output["tokens"]
                    and public[name]["text"] == output["text"]
                )
            elif "context_tokens" in case:
                checks["context_capacity"] = (
                    output["input_tokens"]
                    == actual["prompt_tokens"]
                    == original[0]["prompt_tokens"]
                    == case["context_tokens"] - case["max_new_tokens"]
                )
                checks["context_output_budget"] = len(output["tokens"]) == case["max_new_tokens"]
            else:
                checks["red_signal_content"] = all(
                    red_light_content(text) for text in (output["text"], original[0]["text"])
                )
            if "context_tokens" not in case:
                checks["text_stop"] = output["tokens"][-1] == original[0]["tokens"][-1] == 7
            prompt = actual.get(
                "prompt_text_tokens", actual.get("trace", {}).get("prompt_text_tokens")
            )
            if "prompt_text_tokens" in original[0]:
                checks["prompt_ids"] = prompt == original[0]["prompt_text_tokens"]
            details.append(
                {
                    "text_tokens_equal_diagnostic": output["tokens"] == original[0]["tokens"],
                    "native_text": output["text"],
                    "reference_text": original[0]["text"],
                }
            )
        else:
            turns = actual["turns"] if mode == "chat" else [actual]
            checks["turn_count"] = len(turns) == len(original) == (2 if mode == "chat" else 1)
            if mode == "chat":
                first, second = (turn["trace"] for turn in turns)
                checks["retained_audio_code_history"] = (
                    first["prior_audio_code_frames"] == 0
                    and second["prior_audio_code_frames"] == len(first["codes"][0]) > 0
                )
                checks["original_audio_input_count"] = (
                    first["input_audio_encodings"] == second["input_audio_encodings"] == 1
                )
            for turn_index, (turn, ref) in enumerate(zip(turns, original, strict=True)):
                trace = turn["trace"]
                prefix = f"turn_{turn_index}_"
                checks[prefix + "completion"] = (
                    trace["ended_audio"]
                    and ref["ended_audio"]
                    and all(book and book[-1] == 2048 for book in trace["codes"])
                    and all(book and book[-1] == 2048 for book in ref["codes"])
                )
                checks[prefix + "budget"] = (
                    len(trace["tokens"]) + len(trace["codes"][0]) <= case["max_new_tokens"]
                )
                if "prompt_text_tokens" in ref and turn_index == 0:
                    checks[prefix + "prompt_ids"] = (
                        trace["prompt_text_tokens"] == ref["prompt_text_tokens"]
                    )
                path = (
                    case["output_prefix"] + str(turn_index) + ".f32"
                    if mode == "chat"
                    else case["output"]
                )
                wave, pcm = waveform(path)
                ref_wave, ref_pcm = waveform(ref["waveform"])
                checks[prefix + "waveform_geometry"] = (
                    wave.size == (len(trace["codes"][0]) - int(trace["ended_audio"])) * 1920 > 0
                    and np.isfinite(wave).all()
                )
                checks[prefix + "reference_waveform_geometry"] = (
                    ref_wave.size == (len(ref["codes"][0]) - int(ref["ended_audio"])) * 1920 > 0
                    and np.isfinite(ref_wave).all()
                )
                spoken, ref_spoken = (
                    transcripts[(name, turn_index, producer)]
                    for producer in ("native", "reference")
                )
                checks[prefix + "transcript_identity"] = (
                    Path(spoken["asr_input"]).read_bytes() == pcm
                    and Path(ref_spoken["asr_input"]).read_bytes() == ref_pcm
                )
                checks[prefix + "asr_completion"] = all(
                    record["asr_ended"] and record["unexpected_audio_frames"] == 0
                    for record in (spoken, ref_spoken)
                )
                recovery = spoken_content(
                    spoken["intended"],
                    spoken["transcript"],
                    alternatives=mode in {"chat", "speech_response"},
                )
                checks[prefix + "spoken_content"] = recovery["equivalent"]
                if mode in {"chat", "speech_response"}:
                    checks[prefix + "interleaving"] = interleaved_schedule(trace)
                    checks[prefix + "dialogue_content"] = all(
                        dialogue_content(text, turn_index)
                        for text in (
                            spoken["intended"],
                            spoken["transcript"],
                            ref_spoken["intended"],
                            ref_spoken["transcript"],
                        )
                    )
                else:
                    checks[prefix + "sequential_text_stop"] = (
                        bool(trace["tokens"]) and trace["tokens"][-1] == ref["tokens"][-1] == 7
                    )
                    checks[prefix + "requested_words"] = (
                        words(spoken["intended"])
                        == words(case["text"])
                        == words(ref_spoken["intended"])
                    )
                    checks[prefix + "voice_condition"] = (
                        case["voice"] in {"us_male", "us_female", "uk_male", "uk_female"}
                        and trace["prompt_text_tokens"] == ref["prompt_text_tokens"]
                    )
                if mode == "chat":
                    api = public[name]["turns"][turn_index]
                    checks[prefix + "native_displayed_speech_text"] = words(
                        spoken["intended"]
                    ) == words(turn["text"])
                    checks[prefix + "reference_displayed_speech_text"] = words(
                        ref_spoken["intended"]
                    ) == words(ref["text"])
                    checks[prefix + "public_api_result"] = api["text"] == turn[
                        "text"
                    ] and np.array_equal(waveform(api["waveform"])[0], wave)
                details.append(
                    {
                        "turn": turn_index,
                        "native_spoken_content": recovery,
                        "reference_spoken_content_diagnostic": spoken_content(
                            ref_spoken["intended"],
                            ref_spoken["transcript"],
                            alternatives=mode in {"chat", "speech_response"},
                        ),
                        "text_tokens_equal_diagnostic": trace["tokens"] == ref["tokens"],
                        "audio_codes_equal_diagnostic": trace["codes"] == ref["codes"],
                        "waveform_shape_equal_diagnostic": wave.shape == ref_wave.shape,
                        "native_transcript": spoken["transcript"],
                        "reference_transcript": ref_spoken["transcript"],
                    }
                )
        rows.append(
            {
                "name": name,
                "passed": all(checks.values()),
                "checks": {key: bool(value) for key, value in checks.items()},
                "details": details,
            }
        )
    offline = next((case for case in cases if case["name"] == "lfm25-audio-tts-us-male"), None)
    streamed = next((case for case in cases if case["mode"] == "tts_stream"), None)
    if offline and streamed:
        a, b = waveform(offline["output"])[0], waveform(streamed["output"])[0]
        same = a.shape == b.shape and np.max(np.abs(a - b)) <= 1e-6
        row = next(row for row in rows if row["name"] == streamed["name"])
        row["checks"]["streamed_offline_waveform"] = bool(same)
        chunks = native[streamed["name"]]["chunks"]
        row["checks"]["successive_stream_chunks"] = (
            len(chunks) > 1 and min(chunks) > 0 and sum(chunks) == b.size
        )
        row["passed"] &= bool(same)
        row["passed"] &= bool(row["checks"]["successive_stream_chunks"])
    voices = {
        case["voice"]: case
        for case in cases
        if case["mode"] == "tts" and case.get("audio_temperature", 0) == 0
    }
    if set(voices) == {"us_male", "us_female", "uk_male", "uk_female"}:
        recordings = [waveform(case["output"])[1] for case in voices.values()]
        distinct = all(recordings[i] != recordings[j] for i in range(4) for j in range(i))
        for case in voices.values():
            row = next(row for row in rows if row["name"] == case["name"])
            row["checks"]["distinct_voice_conditioned_waveforms"] = distinct
            row["passed"] &= distinct
    return {
        "passed": all(row["passed"] for row in rows),
        "cases": rows,
        "voice_scope": "Four original voice prompts and intended spoken content; no independent perceptual timbre/accent score.",
        "strict_sequences": "Preserved as diagnostics; task semantics are the acceptance contract.",
    }


def read_records(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines()]
    result = json.loads(path.read_text())
    return result.get("records", result) if isinstance(result, dict) else result


def main():
    parser = argparse.ArgumentParser()
    for name in ("cases", "native", "reference", "public", "transcripts", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    result = compare(
        *(
            read_records(getattr(args, name))
            for name in ("cases", "native", "reference", "public", "transcripts")
        )
    )
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()

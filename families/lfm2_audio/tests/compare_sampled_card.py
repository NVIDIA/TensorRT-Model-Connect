# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict numeric diagnostic; semantic acceptance lives in compare_semantics."""


def main():
    import json
    from pathlib import Path
    import numpy as np

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-dir", required=True, type=Path)
    root = parser.parse_args().evidence_dir
    cases = json.loads((root / "sampled-card-cases.json").read_text())
    native = {
        x["name"]: x
        for x in map(json.loads, (root / "sampled-card-native.jsonl").read_text().splitlines())
    }
    reference = {
        x["name"]: x for x in json.loads((root / "sampled-card-reference.json").read_text())
    }
    coupling = json.loads((root / "sampled-card-reference.json.coupling.json").read_text())
    uniforms = json.loads((root / "native-uniforms-seed0.json").read_text())["uniforms"]
    results = []
    for case in cases:
        name = case["name"]
        actual = native[name]
        row = {"name": name, "turns": []}
        for turn, ref in enumerate(reference[name]["turns"]):
            a = actual["turns"][turn] if case["mode"] == "chat" else actual
            trace = a["trace"]
            result = {
                "text_tokens_match": trace["tokens"] == ref["tokens"],
                "audio_codes_match": trace["codes"] == ref["codes"],
                "prompt_text_tokens_match": trace["prompt_text_tokens"]
                == ref["prompt_text_tokens"],
                "native_audio_frames": len(trace["codes"][0]),
                "reference_audio_frames": len(ref["codes"][0]),
                "native_uniforms_used": 8 * len(trace["codes"][0]),
                "native_ended_audio": trace["ended_audio"],
                "reference_ended_audio": ref["ended_audio"],
            }
            if "text" in a:
                result["text_match"] = a["text"] == ref["text"]
            na, nr = np.array(trace["codes"]).T, np.array(ref["codes"]).T
            common = min(len(na), len(nr))
            mismatches = np.argwhere(na[:common] != nr[:common])
            if len(mismatches):
                frame, book = map(int, mismatches[0])
                draw = frame * 8 + book
                with (root / f"{name}-{turn}.original-draws.jsonl").open() as source:
                    original = next(json.loads(line) for i, line in enumerate(source) if i == draw)
                assert original["uniform"] == uniforms[draw]
                probabilities = dict(original["nonzero_probabilities"])
                result["first_audio_mismatch"] = {
                    "frame": frame,
                    "codebook": book,
                    "draw": draw,
                    "uniform": uniforms[draw],
                    "native_code": int(na[frame, book]),
                    "reference_code": int(nr[frame, book]),
                    "reference_probability_of_native_code": probabilities.get(
                        int(na[frame, book]), 0.0
                    ),
                    "reference_selected_code": original["selected"],
                }
            path = (
                case["output_prefix"] + str(turn) + ".f32"
                if case["mode"] == "chat"
                else case["output"]
            )
            wave, expected = (np.fromfile(p, np.float32) for p in (path, ref["waveform"]))
            result["native_samples"] = int(wave.size)
            result["reference_samples"] = int(expected.size)
            result["waveform_samples_match"] = wave.shape == expected.shape
            result["native_waveform_finite"] = bool(np.isfinite(wave).all())
            if wave.shape == expected.shape and wave.size and np.linalg.norm(expected) > 0:
                result["waveform_nrmse"] = float(
                    np.linalg.norm(wave - expected) / np.linalg.norm(expected)
                )
                result["waveform_correlation"] = float(np.corrcoef(wave, expected)[0, 1])
            result["passes_strict_gate"] = bool(
                result["text_tokens_match"]
                and result["audio_codes_match"]
                and result["native_ended_audio"]
                and result["reference_ended_audio"]
                and result["waveform_samples_match"]
                and result.get("waveform_nrmse", float("inf")) <= 0.02
                and result.get("waveform_correlation", -1) >= 0.999
            )
            row["turns"].append(result)
        results.append(row)
    report = {
        "checkpoint_revision": "c362a0625dfe45aa588dce5f0ada28a7e5707628",
        "reference_revision": "19e65845923a7f136442c95137884ec61eb386aa",
        "precision": "bf16",
        "max_new_tokens": 512,
        "coupling": coupling,
        "exit": json.loads((root / "sampled-card-exit.json").read_text()),
        "cases": results,
    }
    (root / "sampled-card-comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    passed = all(turn["passes_strict_gate"] for row in results for turn in row["turns"])
    completed = report["exit"] == {"reference_exit": 0, "native_exit": 0}
    raise SystemExit(0 if completed and passed else 1)


if __name__ == "__main__":
    main()

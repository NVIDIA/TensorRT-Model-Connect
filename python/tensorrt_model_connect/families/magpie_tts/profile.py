#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Profile MagpieTTS inference pipeline with fine-grained timing.

Instruments the C++ backend via a Python debug runner that mirrors the
C++ pipeline stages (encoder, cross-KV, decoder loop, codec) and reports
per-stage and per-step latency breakdowns.
"""

import argparse
import json
import os
import subprocess
import time


def profile_cpp_binary(bundle_path: str, trtmc_binary: str, prompt: str,
                       max_new_tokens: int, num_runs: int, greedy: bool):
    """Run the C++ binary and capture timing from stderr."""
    env = os.environ.copy()

    cmd = [
        trtmc_binary, "generate-audio", bundle_path,
        "--prompt", prompt,
        "--output", "/tmp/magpie_profile_output.wav",
        "--max-new-tokens", str(max_new_tokens),
        "--set", "platform.trt_log_stderr=true",
    ]
    if greedy:
        cmd.extend(["--set", "audio_magpie.greedy=true"])

    results = []
    for run_idx in range(num_runs):
        print(f"\n{'='*60}")
        print(f"Run {run_idx + 1}/{num_runs}")
        print(f"{'='*60}")

        t_start = time.perf_counter()
        proc = subprocess.run(
            cmd, capture_output=True, text=True, env=env, timeout=120
        )
        t_end = time.perf_counter()

        wall_time = t_end - t_start
        stderr = proc.stderr

        # Parse stderr for stage timings
        result = {
            "run": run_idx + 1,
            "wall_time_s": wall_time,
            "returncode": proc.returncode,
            "stderr_lines": stderr.strip().split("\n") if stderr else [],
        }

        # Extract key info from stderr
        for line in result["stderr_lines"]:
            if "Encoder: processed" in line:
                result["encoder_info"] = line.strip()
            elif "Prefilling" in line:
                result["prefill_info"] = line.strip()
            elif "Generated" in line and "frames" in line:
                result["generated_info"] = line.strip()
            elif "Codec:" in line:
                result["codec_info"] = line.strip()
            elif "samples" in line and "Hz" in line:
                result["audio_info"] = line.strip()

        results.append(result)
        print(f"Wall time: {wall_time:.3f}s")
        print(f"Return code: {proc.returncode}")
        if proc.stderr:
            print("--- stderr ---")
            print(proc.stderr)
        if proc.stdout:
            print("--- stdout ---")
            print(proc.stdout[:500])

    return results


def profile_with_cuda_events(bundle_path: str, prompt: str,
                             max_new_tokens: int, greedy: bool):
    """Reserved for a future Magpie-owned Python debug runner."""
    _ = (bundle_path, prompt, max_new_tokens, greedy)
    print(
        "CUDA event profiling requires a Magpie-owned Python debug runner; "
        "use C++ binary profiling for now."
    )
    return None


def main():
    parser = argparse.ArgumentParser(description="Profile MagpieTTS")
    parser.add_argument("--bundle", required=True, help="Path to .trtfb bundle")
    parser.add_argument("--trtmc-binary", default="./build/trtmc",
                        help="Path to C++ trtmc binary")
    parser.add_argument("--prompt", default="Hello, this is a test of text to speech synthesis.",
                        help="Text prompt")
    parser.add_argument("--max-new-tokens", type=int, default=200,
                        help="Max audio frames to generate")
    parser.add_argument("--num-runs", type=int, default=3,
                        help="Number of profiling runs")
    parser.add_argument("--greedy", action="store_true", default=True,
                        help="Use greedy decoding (deterministic)")
    parser.add_argument("--json", type=str, help="Output JSON results file")
    args = parser.parse_args()

    print(f"Bundle: {args.bundle}")
    print(f"Binary: {args.trtmc_binary}")
    print(f"Prompt: {args.prompt!r}")
    print(f"Max frames: {args.max_new_tokens}")
    print(f"Runs: {args.num_runs}")
    print(f"Greedy: {args.greedy}")

    # Phase 1: C++ binary profiling (wall-clock per run)
    cpp_results = profile_cpp_binary(
        args.bundle, args.trtmc_binary, args.prompt,
        args.max_new_tokens, args.num_runs, args.greedy
    )

    # Summary
    wall_times = [r["wall_time_s"] for r in cpp_results if r["returncode"] == 0]
    if wall_times:
        print(f"\n{'='*60}")
        print("SUMMARY (C++ binary)")
        print(f"{'='*60}")
        print(f"Runs: {len(wall_times)}")
        print(f"Wall time (min/avg/max): {min(wall_times):.3f} / "
              f"{sum(wall_times)/len(wall_times):.3f} / {max(wall_times):.3f}s")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"cpp_results": cpp_results}, f, indent=2)
        print(f"\nResults written to {args.json}")


if __name__ == "__main__":
    main()

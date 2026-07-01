#!/usr/bin/env python3
"""Diff tool for Bark text-to-audio: C++ TRT pipeline vs HuggingFace.

Staged comparison to isolate audio quality issues:

  Stage 1: C++ sampling smoke test
    Run the C++ binary with sampling and check that
    the output waveform has speech-level energy (RMS > threshold).

  Stage 2: Token distribution comparison
    Run both C++ and HF pipelines, dump intermediate tokens (semantic, coarse),
    and compare distributions (count, range, entropy). Since sampling is
    stochastic, exact match is not expected -- we check that both produce
    valid tokens in the expected ranges.

  Stage 3: Codec comparison
    Take coarse tokens from C++ (dumped via audio_bark.dump_path), run them through
    both TRT codec (via C++ binary) and HF EnCodec, and compare waveforms
    sample-by-sample.

  Stage 4: Greedy token parity (TRT engine vs HF, per-stage)
    Build TRT engines from HF weights, run all 4 Bark stages with greedy
    decoding through both TRT (via TrtRunner) and HF, compare outputs
    token-by-token. Requires --model (builds engines from HF directly).

Usage:
    # Stage 1: Quick smoke test -- does C++ produce speech?
    python3 tools/diff_audio.py \\
      --bundle bark.trtfb --binary ./build/trtmc \\
      --prompt "Hello, my dog is cute" \\
      --hf-python .venv/bin/python --stage 1

    # Stage 2: Token distribution comparison
    python3 tools/diff_audio.py \\
      --bundle bark.trtfb --binary ./build/trtmc \\
      --model suno/bark-small \\
      --prompt "Hello, my dog is cute" \\
      --hf-python .venv/bin/python --stage 2

    # Stage 3: Codec waveform comparison
    python3 tools/diff_audio.py \\
      --bundle bark.trtfb --binary ./build/trtmc \\
      --model suno/bark-small \\
      --prompt "Hello, my dog is cute" \\
      --hf-python .venv/bin/python --stage 3

    # Stage 4: Greedy parity (TRT engine vs HF)
    python3 tools/diff_audio.py \\
      --model suno/bark-small \\
      --stage 4 --max-semantic-tokens 100

    # All stages 1-3 (default)
    python3 tools/diff_audio.py \\
      --bundle bark.trtfb --binary ./build/trtmc \\
      --model suno/bark-small \\
      --prompt "Hello, my dog is cute" \\
      --hf-python .venv/bin/python
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np


def handles_audio_diff_args(argv: list[str]) -> bool:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", default="")
    parser.add_argument("--bundle", default="")
    ns, _ = parser.parse_known_args(argv)
    text = f"{ns.model} {ns.bundle}".lower()
    return "bark" in text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_energy(waveform: np.ndarray) -> float:
    """Compute RMS energy of a waveform."""
    if len(waveform) == 0:
        return 0.0
    return float(np.sqrt(np.mean(waveform ** 2)))


def read_wav_f32(path: str) -> tuple[np.ndarray, int]:
    """Read a float32 WAV file. Returns (samples, sample_rate)."""
    with open(path, "rb") as f:
        riff = f.read(4)
        if riff != b"RIFF":
            raise ValueError(f"Not a RIFF file: {path}")
        f.read(4)  # chunk size
        wave = f.read(4)
        if wave != b"WAVE":
            raise ValueError(f"Not a WAVE file: {path}")

        sample_rate = 24000
        data_bytes = b""

        while True:
            chunk_id = f.read(4)
            if len(chunk_id) < 4:
                break
            chunk_size = struct.unpack("<I", f.read(4))[0]
            if chunk_id == b"fmt ":
                fmt_data = f.read(chunk_size)
                audio_format = struct.unpack("<H", fmt_data[0:2])[0]
                sample_rate = struct.unpack("<I", fmt_data[4:8])[0]
                if audio_format != 3:  # IEEE float
                    raise ValueError(
                        f"Expected IEEE float (3), got format {audio_format}")
            elif chunk_id == b"data":
                data_bytes = f.read(chunk_size)
            else:
                f.read(chunk_size)

    samples = np.frombuffer(data_bytes, dtype=np.float32)
    return samples, sample_rate


def write_wav_f32(path: str, samples: np.ndarray, sample_rate: int = 24000):
    """Write a float32 WAV file."""
    num_samples = len(samples)
    data_size = num_samples * 4
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH", 16, 3, 1, sample_rate,
                            sample_rate * 4, 4, 32))
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(samples.astype(np.float32).tobytes())


def read_token_file(path: str) -> np.ndarray:
    """Read a newline-delimited token file."""
    tokens = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                tokens.append(int(line))
    return np.array(tokens, dtype=np.int32)


def token_stats(tokens: np.ndarray, label: str) -> dict:
    """Compute and print basic token statistics."""
    if len(tokens) == 0:
        print(f"  {label}: EMPTY")
        return {"count": 0}

    unique = np.unique(tokens)
    # Entropy of the distribution
    counts = np.bincount(tokens[tokens >= 0])
    probs = counts[counts > 0] / counts[counts > 0].sum()
    entropy = -np.sum(probs * np.log2(probs + 1e-12))

    stats = {
        "count": len(tokens),
        "min": int(tokens.min()),
        "max": int(tokens.max()),
        "unique": len(unique),
        "entropy": float(entropy),
        "mean": float(tokens.mean()),
    }
    print(f"  {label}: count={stats['count']}, range=[{stats['min']}, "
          f"{stats['max']}], unique={stats['unique']}, "
          f"entropy={stats['entropy']:.2f} bits")
    return stats


def find_trt_lib_dir() -> str:
    """Find the TRT library directory from the Python tensorrt_libs package."""
    try:
        import importlib.util
        spec = importlib.util.find_spec("tensorrt_libs")
        if spec and spec.submodule_search_locations:
            return spec.submodule_search_locations[0]
    except ImportError:
        pass
    return ""


def run_cpp_bark(binary: str, bundle: str, prompt: str, output_wav: str,
                 hf_python: str, dump_dir: str | None = None,
                 greedy: bool = False, max_tokens: int = 0) -> bool:
    """Run the C++ Bark pipeline and return True on success."""
    env = os.environ.copy()

    # Set LD_LIBRARY_PATH for TRT
    trt_lib = find_trt_lib_dir()
    if trt_lib:
        env["LD_LIBRARY_PATH"] = (
            f"{trt_lib}:/usr/local/cuda/lib64:"
            + env.get("LD_LIBRARY_PATH", ""))

    cmd = [
        binary, "generate-audio", bundle,
        "--prompt", prompt,
        "--output", output_wav,
    ]
    if hf_python:
        cmd += ["--hf-python", hf_python]
    if max_tokens > 0:
        cmd += ["--max-new-tokens", str(max_tokens)]
    if greedy:
        cmd += ["--set", "audio_bark.greedy=true"]
    if dump_dir:
        cmd += ["--set", f"audio_bark.dump_path={dump_dir}"]

    print(f"  Running: {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  C++ pipeline FAILED (rc={result.returncode})", file=sys.stderr)
        if result.stderr:
            # Print last 20 lines of stderr
            lines = result.stderr.strip().split("\n")
            for line in lines[-20:]:
                print(f"    {line}", file=sys.stderr)
        return False

    # Print key info from stderr
    if result.stderr:
        for line in result.stderr.strip().split("\n"):
            if any(k in line for k in ["semantic:", "coarse:", "codec:",
                                        "generated", "Audio saved"]):
                print(f"    {line}", file=sys.stderr)
    return True


# ---------------------------------------------------------------------------
# Stage 1: C++ sampling smoke test
# ---------------------------------------------------------------------------

def stage1_cpp_smoke_test(args) -> bool:
    """Run C++ pipeline with sampling and check output has speech energy."""
    print("\n=== Stage 1: C++ Sampling Smoke Test ===", file=sys.stderr)

    if not args.bundle or not args.binary:
        print("  SKIP: --bundle and --binary required", file=sys.stderr)
        return True

    with tempfile.TemporaryDirectory(prefix="diff_audio_") as tmpdir:
        wav_path = os.path.join(tmpdir, "bark_sampled.wav")

        ok = run_cpp_bark(
            args.binary, args.bundle, args.prompt, wav_path,
            args.hf_python, greedy=False)

        if not ok:
            print("  FAIL: C++ pipeline returned error", file=sys.stderr)
            return False

        if not os.path.exists(wav_path):
            print("  FAIL: No output WAV file", file=sys.stderr)
            return False

        waveform, sr = read_wav_f32(wav_path)
        energy = compute_energy(waveform)

        print(f"  Output: {len(waveform)} samples @ {sr} Hz, "
              f"energy={energy:.6f}", file=sys.stderr)

        if energy < args.min_energy:
            print(f"  FAIL: Output is near-silent "
                  f"(energy={energy:.6f} < {args.min_energy})",
                  file=sys.stderr)

            # Save for inspection
            save_path = "/tmp/bark_diff_stage1.wav"
            write_wav_f32(save_path, waveform, sr)
            print(f"  Saved output to {save_path} for inspection",
                  file=sys.stderr)
            return False

        print(f"  PASS: Output has speech-level energy ({energy:.6f})",
              file=sys.stderr)

        # Save for reference
        save_path = "/tmp/bark_diff_stage1.wav"
        write_wav_f32(save_path, waveform, sr)
        print(f"  Saved to {save_path}", file=sys.stderr)

    return True


# ---------------------------------------------------------------------------
# Stage 2: Token distribution comparison
# ---------------------------------------------------------------------------

def stage2_token_comparison(args) -> bool:
    """Compare token distributions between C++ and HF pipelines."""
    print("\n=== Stage 2: Token Distribution Comparison ===", file=sys.stderr)

    if not args.bundle or not args.binary:
        print("  SKIP: --bundle and --binary required for C++ side",
              file=sys.stderr)
        return True

    if not args.model:
        print("  SKIP: --model required for HF reference", file=sys.stderr)
        return True

    cpp_sem = None
    cpp_coarse = None

    # --- C++ side: run with audio_bark.dump_path ---
    with tempfile.TemporaryDirectory(prefix="diff_audio_") as tmpdir:
        wav_path = os.path.join(tmpdir, "bark_cpp.wav")
        dump_prefix = os.path.join(tmpdir, "bark_dump")

        ok = run_cpp_bark(
            args.binary, args.bundle, args.prompt, wav_path,
            args.hf_python, dump_dir=dump_prefix, greedy=False)

        if not ok:
            print("  FAIL: C++ pipeline failed", file=sys.stderr)
            return False

        sem_file = dump_prefix + ".sem_tokens"
        coarse_file = dump_prefix + ".coarse_tokens"

        if os.path.exists(sem_file):
            cpp_sem = read_token_file(sem_file)
        else:
            print("  WARNING: No semantic token dump", file=sys.stderr)

        if os.path.exists(coarse_file):
            cpp_coarse = read_token_file(coarse_file)
        else:
            print("  WARNING: No coarse token dump", file=sys.stderr)

    # --- HF side: run Bark with sampling ---
    hf_sem = None
    hf_coarse = None
    print("  Loading HF model...", file=sys.stderr)
    try:
        from transformers import AutoProcessor, BarkModel
        from transformers.models.bark.generation_configuration_bark import (
            BarkSemanticGenerationConfig, BarkCoarseGenerationConfig,
        )
        import torch

        processor = AutoProcessor.from_pretrained(args.model)
        model = BarkModel.from_pretrained(args.model)
        model.eval()

        inputs = processor(args.prompt, return_tensors="pt")

        # Build typed generation configs from the model's config dicts
        sem_gen_cfg = BarkSemanticGenerationConfig(
            **model.generation_config.semantic_config)
        coarse_gen_cfg = BarkCoarseGenerationConfig(
            **model.generation_config.coarse_acoustics_config)

        with torch.no_grad():
            # Step 1: semantic tokens
            semantic_output = model.semantic.generate(
                inputs["input_ids"],
                semantic_generation_config=sem_gen_cfg,
            )
            sem_flat = semantic_output.cpu().numpy().flatten()
            # Filter to valid semantic range [0, 10000)
            hf_sem = sem_flat[sem_flat < 10000]

            # Step 2: coarse tokens
            coarse_output = model.coarse_acoustics.generate(
                semantic_output,
                semantic_generation_config=sem_gen_cfg,
                coarse_generation_config=coarse_gen_cfg,
            )
            hf_coarse = coarse_output.cpu().numpy().flatten()

    except Exception as e:
        print(f"  HF generation failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        print("  Skipping HF comparison, showing C++ stats only",
              file=sys.stderr)

    # --- Compare ---
    passed = True

    print("\n  Semantic tokens:", file=sys.stderr)
    if cpp_sem is not None:
        cpp_sem_stats = token_stats(cpp_sem, "C++")
        # Semantic tokens should be in [0, 10000)
        if cpp_sem_stats["count"] > 0:
            if cpp_sem_stats["min"] < 0 or cpp_sem_stats["max"] >= 10000:
                print("    FAIL: C++ semantic tokens out of range [0, 10000)",
                      file=sys.stderr)
                passed = False
            if cpp_sem_stats["count"] < 10:
                print("    WARNING: Very few semantic tokens generated",
                      file=sys.stderr)
    if hf_sem is not None:
        token_stats(hf_sem, "HF ")

    print("\n  Coarse tokens:", file=sys.stderr)
    if cpp_coarse is not None:
        cpp_coarse_stats = token_stats(cpp_coarse, "C++")
        # Coarse tokens: codebook 0 in [10000, 11024), codebook 1 in [11024, 12048)
        if cpp_coarse_stats["count"] > 0:
            if cpp_coarse_stats["min"] < 10000 or cpp_coarse_stats["max"] >= 12048:
                print("    FAIL: C++ coarse tokens out of range [10000, 12048)",
                      file=sys.stderr)
                passed = False

            # Check interleaving: even indices should be CB0, odd should be CB1
            cb0 = cpp_coarse[0::2]
            cb1 = cpp_coarse[1::2]
            cb0_in_range = np.all((cb0 >= 10000) & (cb0 < 11024))
            cb1_in_range = np.all((cb1 >= 11024) & (cb1 < 12048))
            if not cb0_in_range:
                print("    FAIL: CB0 (even) tokens not in [10000, 11024)",
                      file=sys.stderr)
                passed = False
            if not cb1_in_range:
                print("    FAIL: CB1 (odd) tokens not in [11024, 12048)",
                      file=sys.stderr)
                passed = False
    if hf_coarse is not None:
        token_stats(hf_coarse, "HF ")

    if passed:
        print("\n  PASS: Token distributions look valid", file=sys.stderr)
    else:
        print("\n  FAIL: Token distribution issues detected", file=sys.stderr)

    return passed


# ---------------------------------------------------------------------------
# Stage 3: Codec waveform comparison
# ---------------------------------------------------------------------------

def stage3_codec_comparison(args) -> bool:
    """Compare TRT codec vs HF codec on same coarse tokens."""
    print("\n=== Stage 3: Codec Waveform Comparison ===", file=sys.stderr)

    if not args.bundle or not args.binary:
        print("  SKIP: --bundle and --binary required", file=sys.stderr)
        return True

    if not args.model:
        print("  SKIP: --model required for HF codec", file=sys.stderr)
        return True

    # --- Step 1: Generate coarse tokens via C++ and get TRT codec output ---
    with tempfile.TemporaryDirectory(prefix="diff_audio_") as tmpdir:
        wav_path = os.path.join(tmpdir, "bark_trt.wav")
        dump_prefix = os.path.join(tmpdir, "bark_dump")

        ok = run_cpp_bark(
            args.binary, args.bundle, args.prompt, wav_path,
            args.hf_python, dump_dir=dump_prefix, greedy=False)

        if not ok:
            print("  FAIL: C++ pipeline failed", file=sys.stderr)
            return False

        coarse_file = dump_prefix + ".coarse_tokens"
        if not os.path.exists(coarse_file):
            print("  FAIL: No coarse token dump from C++", file=sys.stderr)
            return False

        cpp_coarse = read_token_file(coarse_file)
        if len(cpp_coarse) == 0:
            print("  FAIL: Empty coarse tokens", file=sys.stderr)
            return False

        # Read TRT codec waveform
        if not os.path.exists(wav_path):
            print("  FAIL: No TRT output WAV", file=sys.stderr)
            return False

        trt_waveform, trt_sr = read_wav_f32(wav_path)

        # --- Step 2: Run same coarse tokens through HF EnCodec ---
        # Use only the first codec_frames tokens (matching TRT truncation)
        # to ensure a fair comparison.
        n_total_frames = len(cpp_coarse) // 2
        # Detect TRT codec frame limit from the wav length
        codec_frames = len(trt_waveform) // 320  # upsample_factor=320
        n_use = min(n_total_frames, codec_frames)
        coarse_subset = cpp_coarse[:n_use * 2]

        print(f"  Total coarse frames: {n_total_frames}, "
              f"TRT codec limit: {codec_frames}, using: {n_use}",
              file=sys.stderr)
        print("  Running HF codec on C++ coarse tokens...", file=sys.stderr)
        try:
            import torch
            from transformers import BarkModel

            bark = BarkModel.from_pretrained(args.model).eval()
            codec = bark.codec_model

            # De-interleave coarse tokens into codes [n_q, 1, T]
            codes = torch.zeros(8, 1, n_use, dtype=torch.long)
            for t in range(len(coarse_subset)):
                cb = t % 2
                frame = t // 2
                if frame < n_use:
                    raw = int(coarse_subset[t]) - 10000 - cb * 1024
                    codes[cb, 0, frame] = max(0, min(raw, 1023))

            with torch.no_grad():
                emb = codec.quantizer.decode(codes)
                hf_audio = codec.decoder(emb)
                hf_waveform = hf_audio.squeeze().cpu().numpy()

        except Exception as e:
            print(f"  HF codec failed: {e}", file=sys.stderr)
            return False

        # --- Step 3: Compare waveforms ---
        min_len = min(len(trt_waveform), len(hf_waveform))
        if min_len == 0:
            print("  FAIL: One or both waveforms are empty", file=sys.stderr)
            return False

        trt_trimmed = trt_waveform[:min_len]
        hf_trimmed = hf_waveform[:min_len]
        diff = np.abs(trt_trimmed - hf_trimmed)

        trt_energy = compute_energy(trt_trimmed)
        hf_energy = compute_energy(hf_trimmed)

        print(f"  TRT codec: {len(trt_waveform)} samples, "
              f"energy={trt_energy:.6f}", file=sys.stderr)
        print(f"  HF  codec: {len(hf_waveform)} samples, "
              f"energy={hf_energy:.6f}", file=sys.stderr)
        print(f"  Compared:  {min_len} samples", file=sys.stderr)
        print(f"  Diff: max={diff.max():.6f}, mean={diff.mean():.6f}, "
              f"median={np.median(diff):.6f}", file=sys.stderr)

        # Save files for manual inspection
        write_wav_f32("/tmp/bark_diff_trt_codec.wav", trt_trimmed, trt_sr)
        write_wav_f32("/tmp/bark_diff_hf_codec.wav", hf_trimmed, 24000)
        print("  Saved TRT codec output: /tmp/bark_diff_trt_codec.wav",
              file=sys.stderr)
        print("  Saved HF codec output:  /tmp/bark_diff_hf_codec.wav",
              file=sys.stderr)

        # Spectral similarity: both should have speech-like frequency content
        from numpy.fft import rfft
        def _band_energy(wav, lo, hi, sr=24000):
            spec = np.abs(rfft(wav)) ** 2
            freqs = np.arange(len(spec)) * sr / (2 * len(spec))
            mask = (freqs >= lo) & (freqs < hi)
            return np.sqrt(np.sum(spec[mask]) / (np.sum(spec) + 1e-12))

        trt_ratio = _band_energy(trt_trimmed, 0, 4000) / (
            _band_energy(trt_trimmed, 4000, 12000) + 1e-12)
        hf_ratio = _band_energy(hf_trimmed, 0, 4000) / (
            _band_energy(hf_trimmed, 4000, 12000) + 1e-12)
        print(f"  Speech band ratio: TRT={trt_ratio:.1f}, HF={hf_ratio:.1f} "
              f"(>2 = speech-like)", file=sys.stderr)

        # Check: codec outputs should be reasonably close (same tokens,
        # same weights, but TRT LSTM unrolling vs PyTorch LSTM may differ)
        atol = args.codec_atol
        if diff.mean() > atol:
            print(f"  FAIL: Mean diff {diff.mean():.6f} > atol {atol}",
                  file=sys.stderr)
            return False

        print(f"  PASS: Codec outputs match (mean diff {diff.mean():.6f} "
              f"<= {atol})", file=sys.stderr)

    return True


# ---------------------------------------------------------------------------
# Stage 4: Greedy token parity (TRT engine vs HF, per-stage)
# ---------------------------------------------------------------------------

def stage4_greedy_parity(args) -> bool:
    """Build TRT engines from HF, run all 4 Bark stages with greedy decoding,
    compare TRT vs HF outputs token-by-token."""
    print("\n=== Stage 4: Greedy Token Parity ===", file=sys.stderr)

    if not args.model:
        print("  SKIP: --model required for stage 4", file=sys.stderr)
        return True

    max_sem_tokens = getattr(args, "max_semantic_tokens", 100)

    # --- Lazy imports (heavy deps only needed for stage 4) ---
    try:
        import torch
        from transformers import AutoProcessor, BarkModel
        from transformers.models.bark.generation_configuration_bark import (
            BarkSemanticGenerationConfig,
            BarkCoarseGenerationConfig,
        )
        try:
            from transformers.models.bark.generation_configuration_bark import (
                BarkFineGenerationConfig,
            )
        except ImportError:
            BarkFineGenerationConfig = None
        from tensorrt_model_connect.engine_builder import _resolve_model
        from tensorrt_model_connect.families.bark.config import ModelConfig
        from tensorrt_model_connect.families import find_plugin
        from tensorrt_model_connect.families.bark.debug_runner import (
            TrtRunner,
            VisionTrtRunner,
        )
    except ImportError as e:
        print(f"  SKIP: missing dependency: {e}", file=sys.stderr)
        return True

    # --- Load HF model + processor ---
    print("  Loading HF model...", file=sys.stderr)
    processor = AutoProcessor.from_pretrained(args.model)
    model = BarkModel.from_pretrained(args.model).eval()

    # --- Build TRT engines from HF weights ---
    print("  Building TRT engines from HF weights...", file=sys.stderr)
    model_dir = _resolve_model(args.model)
    config = ModelConfig.from_dir(model_dir)
    plg = find_plugin(config.model_type)
    weights = plg.load_weights(model_dir, config)
    # Bark semantic generation starts after a fixed 256-token text/history
    # prefix plus the semantic infer token. Keep the full diagnostic sequence
    # in cache so Stage 4 compares against HF full-context generation rather
    # than introducing a TRT-only sliding window before the first token.
    max_cache = max(1024, 257 + max_sem_tokens)

    sem_plan = plg.build_engine(config, weights, max_cache)
    extra = plg.build_extra_engines(config, weights, max_cache)
    audio_cfg = plg.get_audio_config(config)

    # Sub-model configs
    sem_cfg = weights["_semantic_cfg"]
    coarse_cfg = weights["_coarse_cfg"]
    fine_cfg = weights["_fine_cfg"]

    # Embedding tables
    sem_embed = np.frombuffer(
        extra["semantic_embed"], dtype=np.float32
    ).reshape(sem_cfg["vocab_size"], sem_cfg["hidden_size"]).copy()
    coarse_embed = np.frombuffer(
        extra["coarse_embed"], dtype=np.float32
    ).reshape(coarse_cfg["vocab_size"], coarse_cfg["hidden_size"]).copy()

    # Audio config constants
    semantic_pad_token = audio_cfg["semantic_pad_token"]
    semantic_infer_token = audio_cfg["semantic_infer_token"]
    text_encoding_offset = audio_cfg["text_encoding_offset"]
    text_pad_token = audio_cfg["text_pad_token"]
    semantic_vocab_size = audio_cfg["semantic_vocab_size"]
    coarse_semantic_pad_token = audio_cfg["coarse_semantic_pad_token"]
    coarse_infer_token = audio_cfg["coarse_infer_token"]
    codebook_size = audio_cfg["codebook_size"]
    n_coarse_codebooks = audio_cfg.get("n_coarse_codebooks", 2)

    # Coarse generation constants (matching C++ BarkConfig defaults)
    coarse_rate_hz = 75
    semantic_rate_hz = 49.9
    max_coarse_history = 630
    max_coarse_input_length = 256
    sliding_window_len = 60

    # Tokenize prompt
    inputs = processor(args.prompt, return_tensors="pt")
    input_ids = inputs["input_ids"][0].tolist()

    # Build generation configs (greedy)
    sem_gen_cfg = BarkSemanticGenerationConfig(
        **model.generation_config.semantic_config)
    sem_gen_cfg.do_sample = False
    sem_gen_cfg.temperature = 1.0
    sem_gen_cfg.max_new_tokens = max_sem_tokens

    coarse_gen_cfg = BarkCoarseGenerationConfig(
        **model.generation_config.coarse_acoustics_config)
    coarse_gen_cfg.do_sample = False
    coarse_gen_cfg.temperature = 1.0

    all_pass = True

    # =================================================================
    # Stage 4a: Semantic greedy parity
    # =================================================================
    print("\n  Stage 4a: Semantic greedy parity...", file=sys.stderr)

    # --- HF semantic (greedy) ---
    with torch.no_grad():
        hf_semantic_output = model.semantic.generate(
            inputs["input_ids"],
            semantic_generation_config=sem_gen_cfg,
            attention_mask=inputs.get("attention_mask"),
        )
    hf_sem_all = hf_semantic_output[0].cpu().numpy()
    hf_sem_tokens = hf_sem_all[hf_sem_all < semantic_pad_token]

    # --- TRT semantic ---
    sem_runner = TrtRunner(
        engine_plan=sem_plan,
        max_cache_length=max_cache,
        num_layers=sem_cfg["num_layers"],
    )
    hidden = sem_cfg["hidden_size"]

    # Prefill: 256 positions (text context + semantic history)
    n_text = len(input_ids)
    for pos in range(256):
        if pos < n_text and input_ids[pos] != 0:
            text_tok = input_ids[pos] + text_encoding_offset
        else:
            text_tok = text_pad_token
        embed = (sem_embed[text_tok] + sem_embed[semantic_pad_token]
                 ).reshape(1, hidden)
        sem_runner.step(token_id=0, input_embed=embed, use_input_embed=1.0)

    # Feed infer token
    embed = sem_embed[semantic_infer_token].reshape(1, hidden)
    result = sem_runner.step(
        token_id=0, input_embed=embed, use_input_embed=1.0)

    # Autoregressive decode (greedy, matching C++ run_semantic)
    trt_sem_tokens = []
    for _ in range(max_sem_tokens):
        logits = result["logits"].flatten().copy()
        logits[semantic_pad_token + 1:] = -1e9
        token = int(np.argmax(logits[:semantic_pad_token + 1]))
        if token == semantic_pad_token:
            break
        trt_sem_tokens.append(token)
        # C++ uses token_id mode for decode (use_input_embed=0.0)
        result = sem_runner.step(token_id=token)

    trt_sem_tokens = np.array(trt_sem_tokens, dtype=np.int32)
    del sem_runner

    # Compare
    sem_n = min(len(hf_sem_tokens), len(trt_sem_tokens))
    sem_matched = (int(np.sum(hf_sem_tokens[:sem_n] == trt_sem_tokens[:sem_n]))
                   if sem_n > 0 else 0)
    sem_pass = (sem_matched == sem_n
                and len(hf_sem_tokens) == len(trt_sem_tokens))
    print(f"    semantic:  {sem_matched}/{max(len(hf_sem_tokens), len(trt_sem_tokens))} "
          f"tokens match   {'PASS' if sem_pass else 'FAIL'}",
          file=sys.stderr)
    if not sem_pass:
        all_pass = False
        if len(hf_sem_tokens) != len(trt_sem_tokens):
            print(f"    Length mismatch: HF={len(hf_sem_tokens)}, "
                  f"TRT={len(trt_sem_tokens)}", file=sys.stderr)
        for i in range(sem_n):
            if hf_sem_tokens[i] != trt_sem_tokens[i]:
                print(f"    First mismatch at index {i}: "
                      f"HF={hf_sem_tokens[i]}, TRT={trt_sem_tokens[i]}",
                      file=sys.stderr)
                break

    # =================================================================
    # Stage 4b: Coarse greedy parity
    # =================================================================
    print("\n  Stage 4b: Coarse greedy parity...", file=sys.stderr)

    # --- HF coarse (greedy) ---
    with torch.no_grad():
        hf_coarse_output = model.coarse_acoustics.generate(
            hf_semantic_output,
            semantic_generation_config=sem_gen_cfg,
            coarse_generation_config=coarse_gen_cfg,
        )
    hf_coarse_all = hf_coarse_output[0].cpu().numpy()
    hf_coarse_tokens = hf_coarse_all[
        (hf_coarse_all >= semantic_vocab_size)
        & (hf_coarse_all < semantic_vocab_size + n_coarse_codebooks * codebook_size)
    ]

    # --- TRT coarse (replicate C++ sliding window logic) ---
    coarse_plan = extra["coarse_engine_plan"]
    coarse_hidden = coarse_cfg["hidden_size"]

    # Use HF semantic tokens as input (ensures same input regardless of 4a)
    x_semantic = [
        coarse_semantic_pad_token if t == semantic_pad_token else int(t)
        for t in hf_sem_tokens
    ]
    sem_len = len(x_semantic)
    n_steps = max(
        int(math.floor(sem_len * coarse_rate_hz / semantic_rate_hz))
        * n_coarse_codebooks,
        0,
    )

    x_coarse = []
    n_window_steps = (int(math.ceil(n_steps / sliding_window_len))
                      if n_steps > 0 else 0)

    for win in range(n_window_steps):
        gen_this_window = min(sliding_window_len, n_steps - len(x_coarse))
        if gen_this_window <= 0:
            break

        # Build semantic context (matching C++ bark_backend.cpp:344-376)
        total_generated = len(x_coarse)
        max_sem_hist = int(math.floor(
            max_coarse_history * semantic_rate_hz / coarse_rate_hz))
        semantic_idx = int(round(
            total_generated * semantic_rate_hz / coarse_rate_hz))
        sem_start = max(0, semantic_idx - max_sem_hist)
        sem_context_len = min(sem_len - sem_start, max_coarse_input_length)

        # Build input token sequence
        input_tokens = []
        for i in range(sem_start, sem_start + sem_context_len):
            input_tokens.append(x_semantic[i])
        for _ in range(sem_context_len, max_coarse_input_length):
            input_tokens.append(coarse_semantic_pad_token)
        input_tokens.append(coarse_infer_token)
        hist_start = max(0, len(x_coarse) - max_coarse_history)
        for i in range(hist_start, len(x_coarse)):
            input_tokens.append(x_coarse[i])

        # New KV cache per window
        coarse_runner = TrtRunner(
            engine_plan=coarse_plan,
            max_cache_length=max_cache,
            num_layers=coarse_cfg["num_layers"],
        )

        # Prefill all but last token
        for i in range(len(input_tokens) - 1):
            embed = coarse_embed[input_tokens[i]].reshape(1, coarse_hidden)
            coarse_runner.step(
                token_id=0, input_embed=embed, use_input_embed=1.0)

        # Feed last prefill token and get first logits
        embed = coarse_embed[input_tokens[-1]].reshape(1, coarse_hidden)
        result = coarse_runner.step(
            token_id=0, input_embed=embed, use_input_embed=1.0)

        # Generate window tokens
        window_start = len(x_coarse)
        for step in range(gen_this_window):
            logits = result["logits"].flatten().copy()
            total_gen = window_start + step
            cb_idx = total_gen % n_coarse_codebooks

            # Mask to valid codebook range
            cb_start = semantic_vocab_size + cb_idx * codebook_size
            cb_end = cb_start + codebook_size
            masked = np.full_like(logits, -1e9)
            masked[cb_start:cb_end] = logits[cb_start:cb_end]

            token = int(np.argmax(masked))
            x_coarse.append(token)

            if step + 1 < gen_this_window:
                embed = coarse_embed[token].reshape(1, coarse_hidden)
                result = coarse_runner.step(
                    token_id=0, input_embed=embed, use_input_embed=1.0)

        del coarse_runner

    trt_coarse_tokens = np.array(x_coarse, dtype=np.int32)

    # Compare
    coarse_n = min(len(hf_coarse_tokens), len(trt_coarse_tokens))
    coarse_matched = (int(np.sum(
        hf_coarse_tokens[:coarse_n] == trt_coarse_tokens[:coarse_n]))
        if coarse_n > 0 else 0)
    coarse_pass = (coarse_matched == coarse_n
                   and len(hf_coarse_tokens) == len(trt_coarse_tokens))
    print(f"    coarse:    {coarse_matched}/"
          f"{max(len(hf_coarse_tokens), len(trt_coarse_tokens))} "
          f"codes match  {'PASS' if coarse_pass else 'FAIL'}",
          file=sys.stderr)
    if not coarse_pass:
        all_pass = False
        if len(hf_coarse_tokens) != len(trt_coarse_tokens):
            print(f"    Length mismatch: HF={len(hf_coarse_tokens)}, "
                  f"TRT={len(trt_coarse_tokens)}", file=sys.stderr)
        for i in range(coarse_n):
            if hf_coarse_tokens[i] != trt_coarse_tokens[i]:
                print(f"    First mismatch at index {i}: "
                      f"HF={hf_coarse_tokens[i]}, TRT={trt_coarse_tokens[i]}",
                      file=sys.stderr)
                break

    # =================================================================
    # Stage 4c: Fine greedy parity
    # =================================================================
    print("\n  Stage 4c: Fine greedy parity...", file=sys.stderr)

    fine_plan = extra.get("fine_engine_plan")
    if fine_plan is None:
        print("    SKIP: No fine engine built", file=sys.stderr)
        fine_pass = True
    else:
        # --- HF fine (always deterministic — argmax) ---
        fine_gen_cfg_dict = getattr(
            model.generation_config, "fine_acoustics_config", {})
        if BarkFineGenerationConfig is not None and fine_gen_cfg_dict:
            fine_gen_cfg = BarkFineGenerationConfig(**fine_gen_cfg_dict)
            # BarkFineModel uses argmax only when temperature is exactly 1.0;
            # its default 0.5 path samples with torch.multinomial.
            fine_gen_cfg.temperature = 1.0
        else:
            fine_gen_cfg = None

        with torch.no_grad():
            generate_kwargs = {
                "semantic_generation_config": sem_gen_cfg,
                "coarse_generation_config": coarse_gen_cfg,
                "codebook_size": codebook_size,
            }
            if fine_gen_cfg is not None:
                generate_kwargs["fine_generation_config"] = fine_gen_cfg
            hf_fine_output = model.fine_acoustics.generate(
                hf_coarse_output, **generate_kwargs)

        # hf_fine_output: [1, n_codebooks, seq_len]
        hf_fine_codes = hf_fine_output[0].cpu().numpy()  # [8, seq_len]

        # --- TRT fine (replicate C++ run_fine) ---
        fine_hidden = fine_cfg["hidden_size"]
        fine_cb_size = fine_cfg.get("codebook_size", 1056)
        fine_seq_length = audio_cfg.get("fine_seq_length", 256)

        # Load fine embedding tables
        n_embed_tables = fine_cfg.get("n_embed_tables", 8)
        fine_embed_flat = np.frombuffer(
            extra["fine_embed"], dtype=np.float32).copy()
        fine_embed = fine_embed_flat.reshape(
            n_embed_tables, fine_cb_size, fine_hidden)
        fine_pos_embed = np.frombuffer(
            extra["fine_position_embed"], dtype=np.float32
        ).reshape(-1, fine_hidden).copy()

        # De-interleave HF coarse tokens into codes [8, n_frames]
        n_frames_raw = len(hf_coarse_tokens) // n_coarse_codebooks
        n_frames = min(n_frames_raw, fine_seq_length)

        # CB0-1 from coarse, CB2-7 initialized to codebook_size (padding)
        trt_fine_codes = np.full((8, n_frames), codebook_size, dtype=np.int32)
        for t in range(n_frames * n_coarse_codebooks):
            cb = t % n_coarse_codebooks
            frame = t // n_coarse_codebooks
            if frame < n_frames:
                raw = (int(hf_coarse_tokens[t]) - semantic_vocab_size
                       - cb * codebook_size)
                trt_fine_codes[cb, frame] = max(0, min(raw, codebook_size - 1))

        # Run TRT fine engine
        fine_runner = VisionTrtRunner(fine_plan)

        for cb_idx in range(2, 8):
            # Sum embeddings for CB 0..cb_idx + position (matching C++)
            input_embeds = np.zeros(
                (fine_seq_length, fine_hidden), dtype=np.float32)
            actual_frames = min(n_frames, fine_seq_length)
            for frame in range(fine_seq_length):
                for cb in range(cb_idx + 1):
                    code = (int(trt_fine_codes[cb, frame])
                            if frame < actual_frames else codebook_size)
                    input_embeds[frame] += fine_embed[cb, code]
                input_embeds[frame] += fine_pos_embed[frame]

            outputs = fine_runner.encode(input_embeds=input_embeds)

            # Read logits for this codebook's head.
            # Engine naming: logits_cb1..logits_cb7, using w_lm_head_0..6.
            # C++ reads head_idx = cb_idx - 1, i.e. logits_cb{cb_idx}.
            head_name = f"logits_cb{cb_idx}"
            if head_name not in outputs:
                print(f"    WARNING: output {head_name} not found in engine",
                      file=sys.stderr)
                continue
            head_logits = outputs[head_name]  # [seq_length, codebook_size]
            valid_range = min(codebook_size, fine_cb_size)
            for frame in range(actual_frames):
                best = int(np.argmax(head_logits[frame, :valid_range]))
                trt_fine_codes[cb_idx, frame] = best

        del fine_runner

        # Compare codebooks 2-7
        n_compare = min(n_frames, hf_fine_codes.shape[1])
        fine_total = 0
        fine_matched = 0
        for cb in range(2, 8):
            for frame in range(n_compare):
                fine_total += 1
                if trt_fine_codes[cb, frame] == hf_fine_codes[cb, frame]:
                    fine_matched += 1

        fine_pass = fine_total > 0 and fine_matched == fine_total
        print(f"    fine:      {fine_matched}/{fine_total} codes match "
              f"(6x{n_compare})  {'PASS' if fine_pass else 'FAIL'}",
              file=sys.stderr)
        if not fine_pass:
            all_pass = False
            # Show first mismatch per codebook
            for cb in range(2, 8):
                for frame in range(n_compare):
                    if trt_fine_codes[cb, frame] != hf_fine_codes[cb, frame]:
                        print(f"    CB{cb} mismatch at frame {frame}: "
                              f"HF={hf_fine_codes[cb, frame]}, "
                              f"TRT={trt_fine_codes[cb, frame]}",
                              file=sys.stderr)
                        break

    # =================================================================
    # Stage 4d: Codec parity
    # =================================================================
    print("\n  Stage 4d: Codec parity...", file=sys.stderr)

    codec_plan = extra.get("codec_engine_plan")
    if codec_plan is None:
        print("    SKIP: No codec engine built", file=sys.stderr)
    elif fine_plan is None:
        print("    SKIP: No fine codes available (fine engine skipped)",
              file=sys.stderr)
    else:
        codec_seq_length = audio_cfg.get("codec_seq_length", 256)
        upsample = audio_cfg.get("codec_upsample_factor", 320)

        # Use HF fine codes as input to both sides
        hf_fine_codes_np = hf_fine_codes  # [8, seq_len]
        n_codec_frames = min(hf_fine_codes_np.shape[1], codec_seq_length)

        # --- TRT codec ---
        codec_runner = VisionTrtRunner(codec_plan)
        audio_codes_input = np.zeros(
            (1, 8, codec_seq_length), dtype=np.int32)
        audio_codes_input[0, :, :n_codec_frames] = (
            hf_fine_codes_np[:, :n_codec_frames])
        codec_output = codec_runner.encode(audio_codes=audio_codes_input)
        trt_waveform = codec_output["waveform"].flatten()
        trt_waveform = trt_waveform[:n_codec_frames * upsample]
        del codec_runner

        # --- HF codec ---
        codec_model = model.codec_model
        codes_tensor = torch.from_numpy(
            hf_fine_codes_np[:, :n_codec_frames].astype(np.int64)
        ).unsqueeze(1)  # [n_q, 1, T]
        with torch.no_grad():
            emb = codec_model.quantizer.decode(codes_tensor)
            hf_audio = codec_model.decoder(emb)
            hf_waveform = hf_audio.squeeze().cpu().numpy()

        # Compare waveforms
        min_len = min(len(trt_waveform), len(hf_waveform))
        if min_len > 0:
            diff = np.abs(trt_waveform[:min_len] - hf_waveform[:min_len])
            cos_num = np.dot(trt_waveform[:min_len], hf_waveform[:min_len])
            cos_den = (np.linalg.norm(trt_waveform[:min_len])
                       * np.linalg.norm(hf_waveform[:min_len]) + 1e-12)
            cosine = float(cos_num / cos_den)
            max_diff = float(diff.max())
            codec_pass = cosine > 0.99 and max_diff < 0.5
        else:
            cosine = 0.0
            max_diff = float("inf")
            codec_pass = False

        print(f"    codec:     cos={cosine:.3f} max={max_diff:.3f}  "
              f"{'PASS' if codec_pass else 'FAIL'}", file=sys.stderr)
        if not codec_pass:
            all_pass = False

    # =================================================================
    # Summary + JSON output
    # =================================================================
    stage_results = {
        "semantic": {
            "pass": bool(sem_pass),
            "hf_tokens": int(len(hf_sem_tokens)),
            "trt_tokens": int(len(trt_sem_tokens)),
            "matched": int(sem_matched),
        },
        "coarse": {
            "pass": bool(coarse_pass),
            "hf_tokens": int(len(hf_coarse_tokens)),
            "trt_tokens": int(len(trt_coarse_tokens)),
            "matched": int(coarse_matched),
        },
        "fine": {
            "pass": bool(fine_pass),
            "total": int(fine_total) if fine_plan else 0,
            "matched": int(fine_matched) if fine_plan else 0,
            "n_frames": int(n_compare) if fine_plan else 0,
        },
    }
    if codec_plan is not None and fine_plan is not None:
        stage_results["codec"] = {
            "pass": bool(codec_pass),
            "cosine": float(cosine),
            "max_diff": float(max_diff),
            "n_samples": int(min_len),
        }

    # Write JSON if --json specified
    json_path = getattr(args, "json", None)
    if json_path:
        with open(json_path, "w") as f:
            json.dump({
                "model": args.model,
                "prompt": args.prompt,
                "max_semantic_tokens": max_sem_tokens,
                "all_pass": all_pass,
                "stages": stage_results,
            }, f, indent=2)
        print(f"  Results written to {json_path}", file=sys.stderr)

    print(file=sys.stderr)
    if all_pass:
        print("  Stage 4: ALL PASSED", file=sys.stderr)
    else:
        print("  Stage 4: SOME STAGES FAILED", file=sys.stderr)

    return all_pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Bark TRT vs HF staged diff test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Stages:
  1  C++ sampling smoke test (needs --bundle, --binary)
  2  Token distribution comparison (needs --bundle, --binary, --model)
  3  Codec waveform comparison (needs --bundle, --binary, --model)
  4  Greedy token parity (needs --model; builds TRT engines from HF)

Examples:
  # Quick smoke test
  python3 tools/diff_audio.py --bundle bark.trtfb --binary ./build/trtmc \\
    --prompt "Hello, my dog is cute" --hf-python .venv/bin/python --stage 1

  # Greedy parity (TRT engine vs HF)
  python3 tools/diff_audio.py --model suno/bark-small --stage 4 \\
    --max-semantic-tokens 100

  # Stages 1-3 comparison
  python3 tools/diff_audio.py --bundle bark.trtfb --binary ./build/trtmc \\
    --model suno/bark-small --prompt "Hello, my dog is cute" \\
    --hf-python .venv/bin/python
""")
    parser.add_argument("--model", default=None,
                        help="HF model ID (e.g. suno/bark-small)")
    parser.add_argument("--bundle", default=None,
                        help="Path to .trtfb bundle")
    parser.add_argument("--binary", default=None,
                        help="Path to trtmc binary (e.g. ./build/trtmc)")
    parser.add_argument("--prompt", default="Hello, my dog is cute.",
                        help="Text prompt")
    parser.add_argument("--hf-python", default="",
                        help="Path to Python with HF tokenizers installed")
    parser.add_argument("--min-energy", type=float, default=0.005,
                        help="Min RMS energy for speech detection (stage 1)")
    parser.add_argument("--codec-atol", type=float, default=0.15,
                        help="Max mean diff for codec comparison (stage 3). "
                        "Note: when the bundle has a fine model, TRT feeds "
                        "8 codebooks while Stage 3 HF reference uses only 2, "
                        "so larger diffs are expected.")
    parser.add_argument("--stage", type=int, default=0,
                        help="Run specific stage (0=all 1-3, 1/2/3/4)")
    parser.add_argument("--max-semantic-tokens", type=int, default=100,
                        help="Max semantic tokens for stage 4 (default: 100)")
    parser.add_argument("--json", default=None,
                        help="Write stage 4 results to this JSON file")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    stages = [1, 2, 3] if args.stage == 0 else [args.stage]
    results = {}
    all_pass = True

    for stage in stages:
        if stage == 1:
            ok = stage1_cpp_smoke_test(args)
        elif stage == 2:
            ok = stage2_token_comparison(args)
        elif stage == 3:
            ok = stage3_codec_comparison(args)
        elif stage == 4:
            ok = stage4_greedy_parity(args)
        else:
            print(f"Unknown stage: {stage}", file=sys.stderr)
            ok = False

        results[stage] = ok
        if not ok:
            all_pass = False

    # Summary
    print("\n=== Summary ===", file=sys.stderr)
    for stage, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  Stage {stage}: {status}", file=sys.stderr)

    if all_pass:
        print("\nAll stages PASSED", file=sys.stderr)
    else:
        print("\nSome stages FAILED", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

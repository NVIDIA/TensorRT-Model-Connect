#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Diff test: Compare TRT PersonaPlex pipeline against the official NVIDIA/personaplex code.

Runs both the official PyTorch pipeline and our C++ TRT pipeline on the same input audio
and compares intermediate values at each stage:
  1. Mimi encode: codec tokens from user audio
  2. Temporal transformer: hidden states
  3. Depth transformer: generated codebook tokens
  4. Mimi decode: output waveform

Usage:
    # Stage-by-stage comparison (needs GPU + official repo installed)
    python -m tensorrt_model_connect.models.personaplex.diff_personaplex \
        --input-wav test_input.wav \
        --bundle /path/to/personaplex.bundle \
        --trtmc-binary ./build/trtmc \
        --hf-python .venv/bin/python \
        --official-repo /path/to/personaplex/moshi

    # Quick reference-only run (no TRT needed, dumps official intermediate values)
    python -m tensorrt_model_connect.models.personaplex.diff_personaplex \
        --input-wav test_input.wav \
        --official-repo /path/to/personaplex/moshi \
        --reference-only

    # TRT-only run (compares against saved reference)
    python -m tensorrt_model_connect.models.personaplex.diff_personaplex \
        --input-wav test_input.wav \
        --bundle /path/to/personaplex.bundle \
        --trtmc-binary ./build/trtmc \
        --hf-python .venv/bin/python \
        --reference-dir /path/to/saved_reference
"""

import argparse
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np

# ---------------------------------------------------------------------------
# Official PersonaPlex reference pipeline
# ---------------------------------------------------------------------------

def run_official_reference(
    input_wav: str,
    official_repo: str,
    output_dir: str,
    device: str = "cuda",
    hf_repo: str = "nvidia/personaplex-7b-v1",
    greedy: bool = True,
    num_frames: int = 50,
) -> dict:
    """Run the official PersonaPlex pipeline and capture intermediate values.

    Hooks into the official LMGen.step() to capture:
      - Mimi encoder output tokens
      - Temporal transformer hidden states (per-frame)
      - Depth transformer tokens (per-frame, per-codebook)
      - Final decoded audio

    Returns dict with keys: mimi_tokens, temporal_hidden, depth_tokens, audio_out
    """
    # Add official repo to path
    sys.path.insert(0, official_repo)

    from moshi.models import loaders, LMGen
    from moshi.models.lm import (
        load_audio as lm_load_audio,
        _iterate_audio as lm_iterate_audio,
        encode_from_sphn as lm_encode_from_sphn,
    )

    print(f"[ref] Loading official PersonaPlex from {hf_repo}")
    print(f"[ref] Device: {device}")

    # Load Mimi
    from huggingface_hub import hf_hub_download
    mimi_weight = hf_hub_download(hf_repo, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(mimi_weight, device)
    other_mimi = loaders.get_mimi(mimi_weight, device)

    # Load LM
    moshi_weight = hf_hub_download(hf_repo, loaders.MOSHI_NAME)
    lm = loaders.get_moshi_lm(moshi_weight, device=device)
    lm.eval()

    # Load tokenizer
    import sentencepiece
    tokenizer_path = hf_hub_download(hf_repo, loaders.TEXT_TOKENIZER_NAME)
    _text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)

    # Create LMGen (greedy decoding for reproducibility)
    frame_size = int(mimi.sample_rate / mimi.frame_rate)
    lm_gen = LMGen(
        lm,
        audio_silence_frame_cnt=0,  # no silence padding for diff test
        sample_rate=mimi.sample_rate,
        device=device,
        frame_rate=mimi.frame_rate,
        use_sampling=not greedy,
        temp=0.8,
        temp_text=0.7,
        top_k=250,
        top_k_text=25,
    )

    # Start streaming
    mimi.streaming_forever(1)
    other_mimi.streaming_forever(1)
    lm_gen.streaming_forever(1)

    # Warmup
    print("[ref] Warming up...")
    from moshi.offline import warmup
    warmup(mimi, other_mimi, lm_gen, device, frame_size)

    # Reset streaming
    mimi.reset_streaming()
    other_mimi.reset_streaming()
    lm_gen.reset_streaming()

    # Encode user audio
    print(f"[ref] Loading input audio: {input_wav}")
    user_audio = lm_load_audio(input_wav, mimi.sample_rate)
    print(f"[ref] Audio shape: {user_audio.shape}, sample_rate={mimi.sample_rate}")

    # Collect intermediate values
    all_mimi_tokens = []  # [num_frames, K, 1]
    all_temporal_hidden = []  # [num_frames, hidden_dim]
    all_depth_tokens = []  # [num_frames, dep_q]
    all_text_tokens = []
    generated_frames = []

    # Monkey-patch to capture temporal hidden states
    original_forward_codes = lm.forward_codes
    captured_transformer_out = []

    def patched_forward_codes(sequence):
        result = original_forward_codes(sequence)
        transformer_out, text_logits = result
        captured_transformer_out.append(transformer_out.detach().cpu().float().numpy())
        return result

    lm.forward_codes = patched_forward_codes

    # Also capture depth tokens from depformer_step
    original_depformer_step = lm_gen.depformer_step
    captured_depth_tokens = []

    def patched_depformer_step(text_token, transformer_out, audio_tokens, audio_provided):
        result = original_depformer_step(text_token, transformer_out, audio_tokens, audio_provided)
        if isinstance(result, tuple):
            tokens, logits = result
        else:
            tokens = result
        captured_depth_tokens.append(tokens.detach().cpu().numpy())
        return result

    lm_gen.depformer_step = patched_depformer_step

    # Run frame-by-frame
    print("[ref] Processing audio frames...")
    frame_count = 0

    for user_encoded in lm_encode_from_sphn(
        mimi,
        lm_iterate_audio(
            user_audio,
            sample_interval_size=frame_size,
            pad=True,
        ),
        max_batch=1,
    ):
        if frame_count >= num_frames:
            break

        all_mimi_tokens.append(user_encoded.detach().cpu().numpy())

        steps = user_encoded.shape[-1]
        for c in range(steps):
            step_in = user_encoded[:, :, c : c + 1]
            tokens = lm_gen.step(step_in)

            if tokens is None:
                continue

            # Save text token
            text_token = tokens[0, 0, 0].item()
            all_text_tokens.append(text_token)

            # Decode audio
            pcm = mimi.decode(tokens[:, 1:9])
            _ = other_mimi.decode(tokens[:, 1:9])
            pcm_np = pcm.detach().cpu().numpy()[0, 0]
            generated_frames.append(pcm_np)

            frame_count += 1

    # Collect captured values
    if captured_transformer_out:
        all_temporal_hidden = np.concatenate(captured_transformer_out, axis=1)  # [1, T, hidden]
        all_temporal_hidden = all_temporal_hidden[0]  # [T, hidden]
    else:
        all_temporal_hidden = np.array([])

    if captured_depth_tokens:
        all_depth_tokens = np.stack(captured_depth_tokens, axis=0)  # [T, dep_q]
    else:
        all_depth_tokens = np.array([])

    # Mimi tokens
    if all_mimi_tokens:
        mimi_tokens_np = np.concatenate(all_mimi_tokens, axis=-1)  # [1, K, T]
        mimi_tokens_np = mimi_tokens_np[0]  # [K, T]
    else:
        mimi_tokens_np = np.array([])

    # Audio output
    if generated_frames:
        audio_out = np.concatenate(generated_frames)
    else:
        audio_out = np.array([])

    # Save reference data
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "mimi_tokens.npy"), mimi_tokens_np)
    np.save(os.path.join(output_dir, "temporal_hidden.npy"), all_temporal_hidden)
    np.save(os.path.join(output_dir, "depth_tokens.npy"), all_depth_tokens)
    np.save(os.path.join(output_dir, "audio_out.npy"), audio_out)
    np.save(os.path.join(output_dir, "text_tokens.npy"), np.array(all_text_tokens))

    print(f"[ref] Saved reference data to {output_dir}")
    print(f"[ref] Mimi tokens shape: {mimi_tokens_np.shape}")
    print(f"[ref] Temporal hidden shape: {all_temporal_hidden.shape}")
    print(f"[ref] Depth tokens shape: {all_depth_tokens.shape}")
    print(f"[ref] Audio output: {len(audio_out)} samples")

    return {
        "mimi_tokens": mimi_tokens_np,
        "temporal_hidden": all_temporal_hidden,
        "depth_tokens": all_depth_tokens,
        "audio_out": audio_out,
        "text_tokens": np.array(all_text_tokens),
    }


# ---------------------------------------------------------------------------
# TRT pipeline runner
# ---------------------------------------------------------------------------

def run_trt_pipeline(
    input_wav: str,
    bundle: str,
    trtmc_binary: str,
    hf_python: str,
    output_dir: str,
) -> dict:
    """Run our TRT C++ pipeline and capture output.

    Uses `trtmc speak` command to process audio, then parses the output.
    Also captures intermediate debug output if available.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_wav = os.path.join(output_dir, "trt_output.wav")

    # Set up library paths
    try:
        import importlib.util
        spec = importlib.util.find_spec("tensorrt_libs")
        if spec and spec.submodule_search_locations:
            trt_lib_dir = spec.submodule_search_locations[0]
        else:
            trt_lib_dir = ""
    except (ImportError, AttributeError):
        trt_lib_dir = ""

    env = os.environ.copy()
    ld_path = env.get("LD_LIBRARY_PATH", "")
    if trt_lib_dir:
        ld_path = f"{trt_lib_dir}:/usr/local/cuda/lib64:{ld_path}"
    env["LD_LIBRARY_PATH"] = ld_path

    cmd = [
        trtmc_binary, "speak",
        bundle,
        "--audio-in", input_wav,
        "--audio-out", output_wav,
    ]
    if hf_python:
        cmd += ["--hf-python", hf_python]

    print(f"[trt] Running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)

    print(f"[trt] Return code: {proc.returncode}")
    if proc.stderr:
        # Parse debug output from stderr
        for line in proc.stderr.split("\n"):
            if line.strip():
                print(f"[trt] {line}")

    if proc.returncode != 0:
        print("[trt] ERROR: trtmc speak failed")
        print(proc.stderr[-2000:] if len(proc.stderr) > 2000 else proc.stderr)
        return {"audio_out": np.array([])}

    # Read output WAV
    audio_out = _read_wav(output_wav)
    np.save(os.path.join(output_dir, "trt_audio_out.npy"), audio_out)

    # Parse depth tokens from stderr (if debug output enabled)
    depth_tokens = _parse_depth_tokens_from_stderr(proc.stderr)
    if depth_tokens is not None:
        np.save(os.path.join(output_dir, "trt_depth_tokens.npy"), depth_tokens)

    # Parse Mimi encoder tokens from stderr
    mimi_tokens = _parse_mimi_tokens_from_stderr(proc.stderr)

    print(f"[trt] Audio output: {len(audio_out)} samples")
    if depth_tokens is not None:
        print(f"[trt] Depth tokens shape: {depth_tokens.shape}")

    return {
        "audio_out": audio_out,
        "depth_tokens": depth_tokens,
        "mimi_tokens": mimi_tokens,
    }


def _read_wav(path: str) -> np.ndarray:
    """Read a WAV file and return float32 samples."""
    if not os.path.exists(path):
        return np.array([], dtype=np.float32)

    with open(path, "rb") as f:
        data = f.read()

    # Simple WAV parser
    if len(data) < 44 or data[:4] != b"RIFF":
        # Try reading as raw float32
        return np.frombuffer(data, dtype=np.float32)

    # Find "data" chunk
    pos = 12
    while pos < len(data) - 8:
        chunk_id = data[pos:pos+4]
        chunk_size = struct.unpack("<I", data[pos+4:pos+8])[0]
        if chunk_id == b"data":
            audio_data = data[pos+8:pos+8+chunk_size]
            # Check format
            fmt_pos = data.find(b"fmt ")
            if fmt_pos >= 0:
                fmt_code = struct.unpack("<H", data[fmt_pos+8:fmt_pos+10])[0]
                bits = struct.unpack("<H", data[fmt_pos+22:fmt_pos+24])[0]
                if fmt_code == 3:  # IEEE float
                    return np.frombuffer(audio_data, dtype=np.float32)
                elif fmt_code == 1 and bits == 16:  # PCM 16-bit
                    samples = np.frombuffer(audio_data, dtype=np.int16)
                    return samples.astype(np.float32) / 32768.0
            return np.frombuffer(audio_data, dtype=np.float32)
        pos += 8 + chunk_size

    return np.array([], dtype=np.float32)


def _parse_depth_tokens_from_stderr(stderr: str) -> np.ndarray:
    """Parse depth token debug output from TRT stderr.

    Expects lines like: [speech] Frame N hidden L2=X depth: t0 t1 t2 ...
    """
    import re
    pattern = r"\[speech\] Frame \d+ hidden L2=[\d.]+ depth:([\d\s]+)"
    matches = re.findall(pattern, stderr)
    if not matches:
        return None
    tokens = []
    for m in matches:
        frame_tokens = [int(x) for x in m.strip().split()]
        tokens.append(frame_tokens)
    return np.array(tokens) if tokens else None


def _parse_mimi_tokens_from_stderr(stderr: str) -> np.ndarray:
    """Parse Mimi encoder token debug output from stderr."""
    import re
    pattern = r"\[speech\] Encoder tokens \[0:16\]: ([\d\s]+)"
    match = re.search(pattern, stderr)
    if match:
        tokens = [int(x) for x in match.group(1).strip().split()]
        return np.array(tokens)
    return None


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_results(ref: dict, trt: dict, tolerances: dict = None) -> dict:
    """Compare reference and TRT results, printing a detailed report.

    Returns dict with pass/fail status for each stage.
    """
    if tolerances is None:
        tolerances = {
            "mimi_token_match": 0.9,   # 90% token match
            "depth_token_match": 0.5,   # 50% token match (generous for now)
            "audio_rms_ratio": 0.1,     # TRT audio RMS should be at least 10% of ref
            "audio_cosine_sim": 0.1,    # Loose cosine similarity
        }

    results = {}
    print("\n" + "=" * 70)
    print("PersonaPlex Diff Test Report")
    print("=" * 70)

    # --- Mimi Encode Tokens ---
    if "mimi_tokens" in ref and ref["mimi_tokens"] is not None and ref["mimi_tokens"].size > 0:
        ref_tokens = ref["mimi_tokens"]
        print("\n--- Mimi Encode ---")
        print(f"  Reference shape: {ref_tokens.shape}")
        if "mimi_tokens" in trt and trt["mimi_tokens"] is not None:
            trt_tokens = trt["mimi_tokens"]
            print(f"  TRT first 16: {trt_tokens[:16]}")
            print(f"  Ref first 16 (cb0): {ref_tokens[0, :16]}")
        else:
            print(f"  Ref first 16 (cb0): {ref_tokens[0, :16] if ref_tokens.ndim > 1 else ref_tokens[:16]}")
            print("  TRT: not available (no debug tokens in stderr)")

    # --- Temporal Hidden States ---
    if "temporal_hidden" in ref and ref["temporal_hidden"] is not None and ref["temporal_hidden"].size > 0:
        th = ref["temporal_hidden"]
        print("\n--- Temporal Transformer Hidden States ---")
        print(f"  Reference shape: {th.shape}")
        if th.ndim == 2:
            for i in range(min(5, th.shape[0])):
                l2 = np.linalg.norm(th[i])
                print(f"  Frame {i}: L2={l2:.4f}, first 5: {th[i, :5]}")

    # --- Depth Tokens ---
    ref_depth = ref.get("depth_tokens")
    trt_depth = trt.get("depth_tokens")

    if ref_depth is not None and ref_depth.size > 0:
        print("\n--- Depth Transformer Tokens ---")
        print(f"  Reference shape: {ref_depth.shape}")
        for i in range(min(5, ref_depth.shape[0])):
            print(f"  Ref  Frame {i}: {ref_depth[i]}")

    if trt_depth is not None and trt_depth.size > 0:
        if ref_depth is None:
            print("\n--- Depth Transformer Tokens ---")
        print(f"  TRT shape: {trt_depth.shape}")
        for i in range(min(5, trt_depth.shape[0])):
            print(f"  TRT  Frame {i}: {trt_depth[i]}")

    if (ref_depth is not None and ref_depth.size > 0 and
        trt_depth is not None and trt_depth.size > 0):
        min_frames = min(ref_depth.shape[0], trt_depth.shape[0])
        min_cb = min(ref_depth.shape[1], trt_depth.shape[1])
        match_count = 0
        total_count = min_frames * min_cb
        for f in range(min_frames):
            for cb in range(min_cb):
                if ref_depth[f, cb] == trt_depth[f, cb]:
                    match_count += 1
        match_rate = match_count / max(total_count, 1)
        results["depth_token_match"] = match_rate
        passed = match_rate >= tolerances["depth_token_match"]
        print(f"  Token match rate: {match_rate:.1%} ({match_count}/{total_count})"
              f" {'PASS' if passed else 'FAIL'} (threshold: {tolerances['depth_token_match']:.0%})")

    # --- Text Tokens ---
    if "text_tokens" in ref and ref["text_tokens"].size > 0:
        print("\n--- Text Tokens ---")
        tt = ref["text_tokens"]
        print(f"  Reference: first 20 = {tt[:20].tolist()}")

    # --- Audio Output ---
    ref_audio = ref.get("audio_out", np.array([]))
    trt_audio = trt.get("audio_out", np.array([]))

    print("\n--- Audio Output ---")
    if ref_audio.size > 0:
        ref_rms = np.sqrt(np.mean(ref_audio ** 2))
        ref_peak = np.max(np.abs(ref_audio))
        print(f"  Reference: {ref_audio.shape[0]} samples, RMS={ref_rms:.4f}, Peak={ref_peak:.4f}")

    if trt_audio.size > 0:
        trt_rms = np.sqrt(np.mean(trt_audio ** 2))
        trt_peak = np.max(np.abs(trt_audio))
        print(f"  TRT:       {trt_audio.shape[0]} samples, RMS={trt_rms:.4f}, Peak={trt_peak:.4f}")

        if ref_audio.size > 0:
            # Compare audio
            min_len = min(len(ref_audio), len(trt_audio))
            r = ref_audio[:min_len]
            t = trt_audio[:min_len]

            rms_ratio = trt_rms / max(ref_rms, 1e-10)
            results["audio_rms_ratio"] = rms_ratio
            passed = rms_ratio >= tolerances["audio_rms_ratio"]
            print(f"  RMS ratio (TRT/Ref): {rms_ratio:.4f}"
                  f" {'PASS' if passed else 'FAIL'} (threshold: {tolerances['audio_rms_ratio']})")

            # Cosine similarity
            norm_r = np.linalg.norm(r)
            norm_t = np.linalg.norm(t)
            if norm_r > 0 and norm_t > 0:
                cos_sim = np.dot(r, t) / (norm_r * norm_t)
                results["audio_cosine_sim"] = cos_sim
                passed = cos_sim >= tolerances["audio_cosine_sim"]
                print(f"  Cosine similarity: {cos_sim:.4f}"
                      f" {'PASS' if passed else 'FAIL'} (threshold: {tolerances['audio_cosine_sim']})")

    # --- Summary ---
    print(f"\n{'=' * 70}")
    all_passed = True
    for key, val in results.items():
        threshold = tolerances.get(key, 0)
        passed = val >= threshold
        if not passed:
            all_passed = False
        print(f"  {key}: {val:.4f} {'PASS' if passed else 'FAIL'}")
    if not results:
        print("  No quantitative comparisons performed (missing TRT or reference data)")
        all_passed = None
    print(f"\n  Overall: {'PASS' if all_passed else ('FAIL' if all_passed is False else 'INCOMPLETE')}")
    print("=" * 70)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Diff test: TRT PersonaPlex vs official code")
    parser.add_argument("--input-wav", required=True, help="Path to input WAV file")
    parser.add_argument("--bundle", help="Path to PersonaPlex .bundle artifact")
    parser.add_argument("--trtmc-binary", default="./build/trtmc", help="Path to trtmc binary")
    parser.add_argument("--hf-python", default="", help="Path to Python with HF transformers")
    parser.add_argument("--official-repo", help="Path to cloned NVIDIA/personaplex/moshi directory")
    parser.add_argument("--hf-repo", default="nvidia/personaplex-7b-v1", help="HF repo ID")
    parser.add_argument("--device", default="cuda", help="Device for official model")
    parser.add_argument("--num-frames", type=int, default=50, help="Max frames to process")
    parser.add_argument("--output-dir", default=None, help="Directory to save intermediate data")
    parser.add_argument("--reference-only", action="store_true",
                        help="Only run official reference, save data, skip TRT")
    parser.add_argument("--reference-dir", help="Load pre-saved reference data instead of running official")
    parser.add_argument("--greedy", action="store_true", default=True,
                        help="Use greedy decoding (default: True)")

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = tempfile.mkdtemp(prefix="personaplex_diff_")
    os.makedirs(args.output_dir, exist_ok=True)

    ref_dir = os.path.join(args.output_dir, "reference")
    trt_dir = os.path.join(args.output_dir, "trt")

    # --- Run official reference ---
    ref_data = {}
    if args.reference_dir:
        print(f"[main] Loading pre-saved reference from {args.reference_dir}")
        for name in ["mimi_tokens", "temporal_hidden", "depth_tokens", "audio_out", "text_tokens"]:
            path = os.path.join(args.reference_dir, f"{name}.npy")
            if os.path.exists(path):
                ref_data[name] = np.load(path)
                print(f"  Loaded {name}: {ref_data[name].shape}")
    elif args.official_repo:
        ref_data = run_official_reference(
            input_wav=args.input_wav,
            official_repo=args.official_repo,
            output_dir=ref_dir,
            device=args.device,
            hf_repo=args.hf_repo,
            greedy=args.greedy,
            num_frames=args.num_frames,
        )
    else:
        print("[main] No --official-repo or --reference-dir specified; skipping reference")

    if args.reference_only:
        print(f"\n[main] Reference-only mode. Data saved to: {ref_dir}")
        return

    # --- Run TRT pipeline ---
    trt_data = {}
    if args.bundle:
        trt_data = run_trt_pipeline(
            input_wav=args.input_wav,
            bundle=args.bundle,
            trtmc_binary=args.trtmc_binary,
            hf_python=args.hf_python,
            output_dir=trt_dir,
        )
    else:
        print("[main] No --bundle specified; skipping TRT")

    # --- Compare ---
    if ref_data or trt_data:
        compare_results(ref_data, trt_data)
    else:
        print("[main] Nothing to compare (no reference or TRT data)")

    print(f"\n[main] Output saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

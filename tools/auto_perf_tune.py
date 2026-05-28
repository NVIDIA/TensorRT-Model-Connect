#!/usr/bin/env python3
"""Automated performance tuning: profile → classify → optimize → verify.

Runs the full /perf-tune loop non-interactively for any model and pipeline type.
No agent needed — pure Python orchestration.

Usage:
    # Single model
    python3 tools/auto_perf_tune.py --model Qwen/Qwen3-0.6B

    # With specific pipeline type and precision
    python3 tools/auto_perf_tune.py --model Qwen/Qwen2.5-7B-Instruct --precision fp16

    # Diffusion model
    python3 tools/auto_perf_tune.py --model black-forest-labs/FLUX.1-schnell \
        --pipeline-type diffusion_flux

    # Batch validation across models
    python3 tools/auto_perf_tune.py --batch models.json --output results/

    # Dry run (show what would be done)
    python3 tools/auto_perf_tune.py --model Qwen/Qwen3-0.6B --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NSYS_DEB_URL = (
    "https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/"
    "NsightSystems-linux-cli-public-2026.2.1.210-3763964.deb"
)

# Pipeline type → performance mode (same as sol_estimate.PIPELINE_MODES)
PIPELINE_MODES = {
    "decoder_kv_cache": "decode", "decoder_moe": "decode",
    "ssm_recurrent": "decode", "rwkv_recurrent": "decode",
    "hybrid_mamba_attention": "decode",
    "diffusion_flux": "diffusion", "diffusion_wan": "diffusion",
    "diffusion_zimage": "diffusion", "diffusion_pixart": "diffusion",
    "speech_to_text": "enc_dec", "text_to_text": "enc_dec",
    "vision_language": "enc_dec", "seq2seq": "enc_dec",
    "seq2seq_encoder_decoder": "enc_dec", "marian_translation": "enc_dec",
    "encoder_only": "single_pass", "embedding": "single_pass",
    "reranking": "single_pass", "segmentation": "single_pass",
    "prompted_segmentation": "single_pass", "object_detection": "single_pass",
    "neural_operator": "single_pass",
    "text_to_audio_bark": "multi_stage", "text_to_audio_magpie": "multi_stage",
    "speech_to_speech": "multi_stage", "omni_multimodal": "multi_stage",
}

# Default prompts by mode
DEFAULT_PROMPTS = {
    "decode": "The capital of France is",
    "diffusion": "a beautiful sunset over mountains",
    "enc_dec": "The capital of France is",
    "single_pass": "The quick brown fox jumps over the lazy dog",
    "multi_stage": "Hello, how are you today?",
}


@dataclass
class TuneResult:
    """Result of one auto-tune run."""
    model: str
    pipeline_type: str
    mode: str
    # Baseline
    baseline_tps: float = 0
    baseline_precision: str = "fp32"
    # Classification
    bottleneck: str = ""
    confidence: str = ""
    evidence: list[str] = field(default_factory=list)
    recommended_techniques: list[str] = field(default_factory=list)
    # After optimization
    optimized_tps: float = 0
    optimized_precision: str = ""
    optimized_runtime_flags: str = ""
    speedup: float = 0
    # SOL (both rooflines)
    sol_tps: float = 0
    bw_sol_tps: float = 0
    compute_sol_tps: float = 0
    sol_bottleneck: str = ""       # "bandwidth" or "compute"
    utilization_before: float = 0
    utilization_after: float = 0
    bw_sol_tps_fp16: float = 0
    compute_sol_tps_fp16: float = 0
    sol_bottleneck_fp16: str = ""
    # Nsys
    nsys_kernel_count: int = 0
    nsys_top_kernel: str = ""
    nsys_top_kernel_pct: float = 0
    # Status
    status: str = "pending"  # pending, success, failed, skipped
    error: str = ""


def run_cmd(cmd: str, timeout: int = 600, dry_run: bool = False) -> tuple[int, str]:
    """Run a shell command, return (exit_code, stdout+stderr)."""
    if dry_run:
        print(f"  [dry-run] {cmd[:120]}")
        return 0, ""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return result.returncode, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return -1, "TIMEOUT"


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def step_build(model: str, output: str, precision: str = "fp32",
               max_cache: int = 256, dry_run: bool = False) -> bool:
    """Build a .trtfb bundle."""
    cmd = (f"./build/trtmc build {model} -o {output} "
           f"--max-cache-length {max_cache} --precision {precision}")
    print(f"\n[build] {precision.upper()} bundle: {model}")
    rc, out = run_cmd(cmd, timeout=600, dry_run=dry_run)
    if rc != 0 and not dry_run:
        print(f"[build] FAILED: {out[-300:]}")
        return False
    print(f"[build] OK → {output}")
    return True


def _generate_test_image(path: str, width: int = 512, height: int = 512):
    """Generate a simple test PNG for segmentation benchmarking."""
    try:
        import numpy as np
        # Random RGB image
        img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        # Write as raw PPM then convert — or just use PIL if available
        from PIL import Image
        Image.fromarray(img).save(path)
    except ImportError:
        # Fallback: write a minimal 1x1 PNG
        import struct
        import zlib
        def _png_chunk(tag, data):
            c = tag + data
            return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xffffffff)
        with open(path, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n")
            f.write(_png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)))
            raw = b""
            for _ in range(height):
                raw += b"\x00" + b"\x80\x80\x80" * width  # gray
            f.write(_png_chunk(b"IDAT", zlib.compress(raw)))
            f.write(_png_chunk(b"IEND", b""))
    print(f"[bench] Generated test image: {path} ({width}x{height})")


def _generate_test_wav(path: str, duration_s: float = 2.0, sample_rate: int = 16000):
    """Generate a short sine-wave WAV for Whisper benchmarking."""
    import math
    import struct
    n_samples = int(duration_s * sample_rate)
    # 440Hz sine wave
    samples = [math.sin(2 * math.pi * 440 * i / sample_rate) * 0.5
               for i in range(n_samples)]
    # Write WAV
    with open(path, "wb") as f:
        data = struct.pack(f"<{n_samples}h",
                           *[int(s * 32767) for s in samples])
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + len(data)))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, 1, sample_rate,
                            sample_rate * 2, 2, 16))
        f.write(b"data")
        f.write(struct.pack("<I", len(data)))
        f.write(data)
    print(f"[bench] Generated test WAV: {path} ({duration_s}s)")


def _build_bench_cmd(
    bundle: str, prompt: str, mode: str, pipeline_type: str,
    max_tokens: int, gpu_argmax: bool,
) -> tuple[str, str]:
    """Build the trtmc CLI command and output dir for the given mode.

    Returns (shell_command, metric_name).
    metric_name is 'tok/s', 'pipeline_ms', or 'rtf'.
    """
    config_flags = "--set platform.trt_log_stderr=true"
    if gpu_argmax:
        config_flags += " --set runtime.prefer_gpu_greedy=true"
    binary = "/tmp/build/trtmc"
    hf = "--hf-python /opt/venv/bin/python"
    env = ""

    if mode == "decode" or pipeline_type in (
        "text_to_text", "vision_language", "seq2seq_encoder_decoder", "marian_translation"):
        # Autoregressive: trtmc run → tok/s
        cmd = (f'{binary} run {bundle} '
               f'--prompt "{prompt}" --max-new-tokens {max_tokens} {hf} {config_flags} 2>&1')
        return cmd, "tok/s"

    elif pipeline_type == "speech_to_text":
        # Whisper: trtmc transcribe --audio <wav>
        # Generate a short test WAV if none exists
        test_wav = "/tmp/bench_whisper_test.wav"
        if not os.path.exists(test_wav):
            _generate_test_wav(test_wav)
        cmd = (f'{env} {binary} transcribe {bundle} '
               f'--audio {test_wav} --max-new-tokens {max_tokens} {hf} 2>&1')
        return cmd, "pipeline_ms"

    elif pipeline_type in ("segmentation", "prompted_segmentation"):
        # Segmentation: trtmc segment --image <img> --output <out>
        test_img = "/tmp/bench_seg_test.png"
        if not os.path.exists(test_img):
            _generate_test_image(test_img)
        out_path = f"/tmp/bench_seg_out_{os.getpid()}.png"
        cmd = (f'{env} {binary} segment {bundle} '
               f'--image {test_img} --output {out_path} {hf} 2>&1')
        return cmd, "pipeline_ms"

    elif pipeline_type == "embedding":
        # Embedding: trtmc embed → measure latency
        cmd = (f'{env} {binary} embed {bundle} '
               f'--prompt "{prompt}" {hf} 2>&1')
        return cmd, "pipeline_ms"

    elif pipeline_type == "reranking":
        # Reranking: trtmc rerank → measure latency
        cmd = (f'{env} {binary} rerank {bundle} '
               f'--prompt "query" --document "{prompt}" {hf} 2>&1')
        return cmd, "pipeline_ms"

    elif mode == "single_pass":
        # Encoder-only (BERT, etc.): trtmc encode → measure latency
        cmd = (f'{env} {binary} encode {bundle} '
               f'--prompt "{prompt}" {hf} 2>&1')
        return cmd, "pipeline_ms"

    elif pipeline_type == "speech_to_speech":
        # Speech-to-speech: trtmc speak --audio-in <wav> --audio-out <wav>
        test_wav = "/tmp/bench_whisper_test.wav"
        if not os.path.exists(test_wav):
            _generate_test_wav(test_wav)
        out_wav = f"/tmp/bench_speak_out_{os.getpid()}.wav"
        cmd = (f'{env} {binary} speak {bundle} '
               f'--audio-in {test_wav} --audio-out {out_wav} {hf} 2>&1')
        return cmd, "pipeline_ms"

    elif mode == "multi_stage":
        # Audio generation: trtmc generate-audio → RTF and pipeline_ms
        out_wav = f"/tmp/bench_{os.getpid()}.wav"
        cmd = (f'{env} {binary} generate-audio {bundle} '
               f'--prompt "{prompt}" --output {out_wav} '
               f'--max-new-tokens {max_tokens} {hf} 2>&1')
        return cmd, "pipeline_ms"

    elif mode == "diffusion":
        # Diffusion: trtmc generate-video/generate-image → pipeline_ms
        out_dir = f"/tmp/bench_diff_{os.getpid()}"
        cmd = (f'{env} {binary} generate-video {bundle} '
               f'--prompt "{prompt}" --output {out_dir} {hf} 2>&1')
        return cmd, "pipeline_ms"

    else:
        # Fallback: trtmc run
        cmd = (f'{env} {binary} run {bundle} '
               f'--prompt "{prompt}" --max-new-tokens {max_tokens} {hf} 2>&1')
        return cmd, "tok/s"


def _parse_metric(out: str, metric_name: str) -> float:
    """Parse performance metric from trtmc output.

    Supported metrics:
        tok/s       — from "Decode: N tokens, X ms, Y tok/s"
        pipeline_ms — from "Total pipeline: X ms" or timed externally
        rtf         — from "RTF (real-time factor): X"
    """
    for line in out.split("\n"):
        # tok/s: "[trtmc] Decode: 20 tokens, 67.4 ms, 296.6 tok/s"
        if metric_name == "tok/s" and "Decode:" in line and "tok/s" in line:
            parts = line.split(",")
            for part in parts:
                if "tok/s" in part:
                    try:
                        return float(part.strip().split()[0])
                    except (ValueError, IndexError):
                        pass

        # pipeline_ms: "[magpie-tts]   Total pipeline: 867.361 ms"
        # or "[trtmc] Inference: X ms"
        if metric_name == "pipeline_ms" and "Total pipeline:" in line:
            try:
                ms = float(line.split("Total pipeline:")[1].strip().split()[0])
                return ms
            except (ValueError, IndexError):
                pass

        # rtf: "RTF (real-time factor): 0.889"
        if metric_name == "rtf" and "RTF" in line and "real-time" in line:
            try:
                return float(line.split(":")[1].strip().split()[0])
            except (ValueError, IndexError):
                pass

    return 0


def step_benchmark(bundle: str, prompt: str, max_tokens: int = 100,
                   gpu_argmax: bool = False, dry_run: bool = False,
                   mode: str = "decode",
                   pipeline_type: str = "decoder_kv_cache") -> float:
    """Benchmark with C++ binary, return performance metric.

    Returns tok/s for decode/enc_dec, or pipeline_ms for others.
    The metric type depends on the mode.
    """
    cmd, metric_name = _build_bench_cmd(
        bundle, prompt, mode, pipeline_type, max_tokens, gpu_argmax)

    label = "GPU argmax" if gpu_argmax else "CPU argmax"
    if mode == "single_pass":
        label = "encode"
    elif mode == "multi_stage":
        label = "generate-audio"
    elif mode == "diffusion":
        label = "generate"
    print(f"[bench] {label}")

    # For pipeline_ms: measure wall-clock time if output doesn't contain timing
    t0 = time.time()
    rc, out = run_cmd(cmd, timeout=300, dry_run=dry_run)
    elapsed_ms = (time.time() - t0) * 1000
    if dry_run:
        return 0

    value = _parse_metric(out, metric_name)

    # Fallback: use wall-clock time if no metric parsed and command succeeded
    if value == 0 and rc == 0 and elapsed_ms > 0:
        value = elapsed_ms
        if metric_name == "tok/s":
            # Convert wall-clock to approximate tok/s
            # (inaccurate — includes loading, but better than 0)
            value = max_tokens / (elapsed_ms / 1000) if elapsed_ms > 0 else 0
            metric_name = "tok/s"  # keep unit consistent
    if value == 0 and metric_name == "pipeline_ms":
        # Fallback: try to extract any "X ms" timing from output
        for line in out.split("\n"):
            if "ms" in line and ("pipeline" in line.lower() or "inference" in line.lower()):
                try:
                    for word in line.split():
                        if word.replace(".", "").isdigit():
                            value = float(word)
                            break
                except (ValueError, IndexError):
                    pass
                if value > 0:
                    break

    if value == 0:
        print(f"[bench] Could not parse {metric_name} from output")

    return value


def step_nsys_profile(bundle: str, prompt: str, output_prefix: str,
                      max_tokens: int = 50, dry_run: bool = False,
                      mode: str = "decode",
                      pipeline_type: str = "decoder_kv_cache") -> str | None:
    """Run nsys profile, return path to .sqlite or None."""
    nsys = "/tmp/nsys_install/opt/nvidia/nsight-systems-cli/2026.2.1/target-linux-x64/nsys"

    # Check nsys exists
    if not dry_run and not os.path.exists(nsys):
        print("[nsys] Not installed, skipping profile")
        return None

    # Build the trtmc command for profiling (same logic as benchmark)
    trtmc_cmd, _ = _build_bench_cmd(
        bundle, prompt, mode, pipeline_type, max_tokens, gpu_argmax=False)
    # Strip the "2>&1" from the end and env prefix — nsys wraps the command
    # Extract just the trtmc binary + args part
    trtmc_part = trtmc_cmd
    # Remove env vars prefix (everything before /tmp/build/trtmc)
    if "/tmp/build/trtmc" in trtmc_part:
        idx = trtmc_part.index("/tmp/build/trtmc")
        env_part = trtmc_part[:idx].strip()
        trtmc_part = trtmc_part[idx:]
    else:
        env_part = ""
    # Remove trailing 2>&1
    trtmc_part = trtmc_part.replace("2>&1", "").strip()

    rep = f"{output_prefix}.nsys-rep"
    cmd = (f"{env_part} {nsys} profile -t cuda,nvtx --cuda-graph-trace=node "
           f"-o {output_prefix} --force-overwrite true "
           f"{trtmc_part} 2>&1")
    print("[nsys] Profiling...")
    rc, out = run_cmd(cmd, timeout=300, dry_run=dry_run)
    if dry_run:
        return f"{output_prefix}.sqlite"

    if rc != 0 or not os.path.exists(rep):
        print("[nsys] Profile failed")
        return None

    # Export to SQLite
    cmd2 = f"{nsys} stats {rep} --force-export true 2>&1"
    run_cmd(cmd2, timeout=60)
    sqlite = f"{output_prefix}.sqlite"
    if os.path.exists(sqlite):
        print(f"[nsys] OK → {sqlite}")
        return sqlite
    return None


def step_classify(sqlite_path: str | None, pipeline_type: str,
                  results_jsonl: str | None = None,
                  engine_section: str = "all",
                  dry_run: bool = False) -> dict:
    """Run classify_bottleneck.py via direct import (avoids subprocess issues)."""
    if not sqlite_path and not dry_run:
        return {"classification": "unknown", "confidence": "low",
                "techniques": [], "evidence": ["no nsys data"]}

    print(f"[classify] pipeline_type={pipeline_type}, engine_section={engine_section}")
    if dry_run:
        return {"classification": "pending", "techniques": []}

    try:
        from classify_bottleneck import classify_from_nsys, to_json
        result = classify_from_nsys(sqlite_path, pipeline_type,
                                    engine_section=engine_section)
        return to_json(result)
    except Exception as e:
        print(f"[classify] Error: {e}")
        return {"classification": "error", "confidence": "low",
                "techniques": [], "evidence": [str(e)]}


def step_sol(model: str, dtype: str, cache_length: int,
             benchmark_json: str | None = None,
             actual_tps: float = 0,
             dry_run: bool = False) -> dict:
    """Run sol_estimate via direct import."""
    print(f"[sol] {dtype.upper()}")
    if dry_run:
        return {}

    try:
        from sol_estimate import (
            load_model_arch_from_hf, GPU_SPECS, estimate_sol,
            detect_gpu, to_json, parse_benchmark_json,
        )
        arch = load_model_arch_from_hf(model)
        gpu_key = detect_gpu() or "B200"
        gpu = GPU_SPECS[gpu_key]

        tps = actual_tps
        if benchmark_json and os.path.exists(benchmark_json):
            try:
                tps = parse_benchmark_json(benchmark_json)
            except Exception:
                pass

        est = estimate_sol(arch, gpu, dtype, cache_length, tps)
        return to_json(est)
    except Exception as e:
        print(f"[sol] Error: {e}")
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def auto_tune_model(
    model: str,
    pipeline_type: str = "decoder_kv_cache",
    max_cache: int = 256,
    max_tokens: int = 100,
    output_dir: str = "/tmp/auto_perf",
    engine_section: str = "all",
    dry_run: bool = False,
) -> TuneResult:
    """Run full auto-tune loop for one model."""
    mode = PIPELINE_MODES.get(pipeline_type, "decode")
    prompt = DEFAULT_PROMPTS.get(mode, "Hello")
    safe_name = model.split("/")[-1].lower().replace("-", "_")

    os.makedirs(output_dir, exist_ok=True)

    result = TuneResult(model=model, pipeline_type=pipeline_type, mode=mode)

    print(f"\n{'='*60}")
    print(f"  Auto Perf-Tune: {model}")
    print(f"  Pipeline: {pipeline_type} → mode: {mode}")
    print(f"{'='*60}")

    # --- Step 1: Build FP32 baseline ---
    fp32_bundle = f"{output_dir}/{safe_name}_fp32.trtfb"
    if not step_build(model, fp32_bundle, "fp32", max_cache, dry_run):
        result.status = "failed"
        result.error = "FP32 build failed"
        return result

    # Determine metric name for this mode
    _, metric_name = _build_bench_cmd(
        fp32_bundle, prompt, mode, pipeline_type, max_tokens, False)
    metric_unit = metric_name  # "tok/s" or "pipeline_ms"

    # --- Step 2: Baseline benchmark ---
    result.baseline_tps = step_benchmark(
        fp32_bundle, prompt, max_tokens, gpu_argmax=False, dry_run=dry_run,
        mode=mode, pipeline_type=pipeline_type)
    result.baseline_precision = "fp32"
    print(f"[baseline] FP32: {result.baseline_tps:.1f} {metric_unit}")

    # --- Step 3: Nsys profile ---
    nsys_prefix = f"{output_dir}/{safe_name}_nsys"
    sqlite = step_nsys_profile(
        fp32_bundle, prompt, nsys_prefix, max_tokens=50, dry_run=dry_run,
        mode=mode, pipeline_type=pipeline_type)

    # --- Step 4: Classify bottleneck ---
    classification = step_classify(sqlite, pipeline_type,
                                   engine_section=engine_section, dry_run=dry_run)
    result.bottleneck = classification.get("classification", "unknown")
    result.confidence = classification.get("confidence", "low")
    result.evidence = classification.get("evidence", [])
    result.recommended_techniques = [
        t.get("name", "") for t in classification.get("techniques", [])]

    print(f"[classify] {result.bottleneck}-bound "
          f"(confidence: {result.confidence})")
    print(f"[classify] Recommended: {result.recommended_techniques}")

    # --- Step 5: SOL estimation (both BW and compute rooflines) ---
    sol_data = step_sol(
        model, "fp32", max_cache, actual_tps=result.baseline_tps, dry_run=dry_run)
    result.sol_tps = sol_data.get("sol_tps", 0)
    result.utilization_before = sol_data.get("utilization_pct", 0)
    result.bw_sol_tps = sol_data.get("bw_sol_tps", 0)
    result.compute_sol_tps = sol_data.get("compute_sol_tps", 0)
    result.sol_bottleneck = sol_data.get("bottleneck", "")

    if result.sol_tps > 0:
        print(f"[sol] FP32 BW SOL: {result.bw_sol_tps:.0f} tok/s, "
              f"Compute SOL: {result.compute_sol_tps:.0f} tok/s")
        print(f"[sol] Bottleneck: {result.sol_bottleneck}, "
              f"SOL: {result.sol_tps:.0f} tok/s, "
              f"util: {result.utilization_before:.1f}%")

    # --- Step 6: Apply optimizations (based on mode) ---
    # Try GPU argmax (decode/enc_dec modes only — single_pass and diffusion don't use argmax)
    if mode in ("decode", "enc_dec", "multi_stage"):
        argmax_tps = step_benchmark(
            fp32_bundle, prompt, max_tokens, gpu_argmax=True, dry_run=dry_run,
            mode=mode, pipeline_type=pipeline_type)
        if argmax_tps > result.baseline_tps:
            pct = (argmax_tps / result.baseline_tps - 1) * 100 if result.baseline_tps > 0 else 0
            print(f"[optimize] GPU argmax: {argmax_tps:.1f} {metric_unit} (+{pct:.0f}%)")

    # Try all precision variants
    best_tps = result.baseline_tps
    best_precision = "fp32"
    best_runtime_flags = ""
    all_results = []

    for precision in ("fp16", "bf16"):
        bundle = f"{output_dir}/{safe_name}_{precision}.trtfb"
        if not step_build(model, bundle, precision, max_cache, dry_run):
            print(f"[optimize] {precision.upper()} build failed, skipping")
            continue

        tps = step_benchmark(
            bundle, prompt, max_tokens, gpu_argmax=False, dry_run=dry_run,
            mode=mode, pipeline_type=pipeline_type)
        all_results.append((precision, False, tps))
        print(f"[optimize] {precision.upper()}: {tps:.1f} {metric_unit}")

        if mode in ("decode", "enc_dec", "multi_stage"):
            argmax_tps = step_benchmark(
                bundle, prompt, max_tokens, gpu_argmax=True, dry_run=dry_run,
                mode=mode, pipeline_type=pipeline_type)
            all_results.append((precision, True, argmax_tps))
            print(f"[optimize] {precision.upper()} + GPU argmax: {argmax_tps:.1f} {metric_unit}")

    # Find best config.
    # For tok/s: higher is better. For pipeline_ms: lower is better.
    lower_is_better = metric_name == "pipeline_ms"

    for prec, argmax, tps in all_results:
        if tps <= 0:
            continue
        if lower_is_better:
            if best_tps <= 0 or tps < best_tps:
                best_tps = tps
                best_precision = prec
                best_runtime_flags = "--set runtime.prefer_gpu_greedy=true" if argmax else ""
        else:
            if tps > best_tps:
                best_tps = tps
                best_precision = prec
                best_runtime_flags = "--set runtime.prefer_gpu_greedy=true" if argmax else ""

    result.optimized_tps = best_tps
    result.optimized_precision = best_precision
    result.optimized_runtime_flags = best_runtime_flags
    if result.baseline_tps > 0 and best_tps > 0:
        if lower_is_better:
            result.speedup = result.baseline_tps / best_tps  # ms ratio (baseline/optimized)
        else:
            result.speedup = best_tps / result.baseline_tps  # tok/s ratio
    else:
        result.speedup = 0

    argmax_label = " + GPU argmax" if result.optimized_runtime_flags else ""
    print(f"[optimize] Best: {best_tps:.1f} {metric_unit} "
          f"({result.optimized_precision.upper()}{argmax_label}, "
          f"{result.speedup:.2f}x vs baseline)")

    # --- Step 7: SOL re-check (best precision, both rooflines) ---
    if result.optimized_tps > 0:
        sol16 = step_sol(
            model, result.optimized_precision, max_cache,
            actual_tps=result.optimized_tps, dry_run=dry_run)
        result.utilization_after = sol16.get("utilization_pct", 0)
        result.bw_sol_tps_fp16 = sol16.get("bw_sol_tps", 0)
        result.compute_sol_tps_fp16 = sol16.get("compute_sol_tps", 0)
        result.sol_bottleneck_fp16 = sol16.get("bottleneck", "")
        if result.utilization_after > 0:
            print(f"[sol] FP16 BW SOL: {result.bw_sol_tps_fp16:.0f}, "
                  f"Compute SOL: {result.compute_sol_tps_fp16:.0f}")
            print(f"[sol] Bottleneck: {result.sol_bottleneck_fp16}, "
                  f"util: {result.utilization_after:.1f}% "
                  f"(was {result.utilization_before:.1f}%)")

    # --- Step 8: Nsys kernel summary ---
    if sqlite and not dry_run:
        try:
            import sqlite3
            db = sqlite3.connect(sqlite)
            rows = db.execute("""
                SELECT s.value, SUM(k.end - k.start) as total_ns, COUNT(*)
                FROM CUPTI_ACTIVITY_KIND_KERNEL k
                JOIN StringIds s ON k.shortName = s.id
                GROUP BY s.value ORDER BY total_ns DESC
            """).fetchall()
            db.close()
            if rows:
                total = sum(r[1] for r in rows)
                result.nsys_kernel_count = sum(r[2] for r in rows)
                result.nsys_top_kernel = rows[0][0][:60]
                result.nsys_top_kernel_pct = rows[0][1] / total * 100 if total > 0 else 0
        except Exception:
            pass

    result.status = "success"

    # --- Report ---
    print(f"\n{'='*70}")
    print(f"  Result: {model}")
    print(f"{'='*70}")
    print(f"  Mode:           {mode}")
    print(f"  Bottleneck (nsys):  {result.bottleneck}-bound ({result.confidence})")
    print(f"  Bottleneck (SOL):   {result.sol_bottleneck} (FP32) → "
          f"{result.sol_bottleneck_fp16} (FP16)")
    print(f"  Baseline:       {result.baseline_tps:.1f} {metric_unit} (FP32)")
    print(f"  Optimized:      {result.optimized_tps:.1f} {metric_unit} "
          f"({result.optimized_precision.upper()}"
          f"{' + GPU argmax' if result.optimized_runtime_flags else ''})")
    print(f"  Speedup:        {result.speedup:.2f}x")
    print(f"  SOL (FP32):     BW={result.bw_sol_tps:.0f}  "
          f"Compute={result.compute_sol_tps:.0f}  "
          f"→ {result.sol_bottleneck} {result.sol_tps:.0f} tok/s  "
          f"util={result.utilization_before:.1f}%")
    print(f"  SOL (FP16):     BW={result.bw_sol_tps_fp16:.0f}  "
          f"Compute={result.compute_sol_tps_fp16:.0f}  "
          f"→ {result.sol_bottleneck_fp16}  "
          f"util={result.utilization_after:.1f}%")
    print(f"  Kernels:        {result.nsys_kernel_count} total, "
          f"top: {result.nsys_top_kernel} ({result.nsys_top_kernel_pct:.0f}%)")
    print(f"{'='*70}")

    return result


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------

VALIDATION_MODELS = [
    # === A. Autoregressive Decode ===
    # Small decoder (<1B) — sync-bound, GPU argmax high ROI
    {"model": "Qwen/Qwen3-0.6B", "pipeline_type": "decoder_kv_cache",
     "label": "A1-decode-small"},
    # Medium decoder (1-4B)
    {"model": "microsoft/Phi-3-mini-4k-instruct", "pipeline_type": "decoder_kv_cache",
     "label": "A2-decode-medium"},
    # Large decoder (7B+) — compute-bound, GEMM dominant
    {"model": "Qwen/Qwen2.5-7B-Instruct", "pipeline_type": "decoder_kv_cache",
     "label": "A3-decode-large"},
    # MoE decoder — sparse routing, different kernel mix
    {"model": "ggml-org/stories15M_MOE", "pipeline_type": "decoder_moe",
     "label": "A4-decode-moe"},
    # SSM — recurrent state, no KV cache
    {"model": "state-spaces/mamba-130m-hf", "pipeline_type": "ssm_recurrent",
     "label": "A5-ssm"},
    # RWKV — another recurrent variant
    {"model": "RWKV/rwkv-4-169m-pile", "pipeline_type": "rwkv_recurrent",
     "label": "A6-rwkv"},

    # === B. Iterative Denoising (Diffusion) ===
    # FLUX — flow matching, image generation
    {"model": "black-forest-labs/FLUX.1-schnell", "pipeline_type": "diffusion_flux",
     "label": "B1-diffusion-flux"},
    # Wan — video generation, 3D latent
    {"model": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "pipeline_type": "diffusion_wan",
     "label": "B2-diffusion-wan"},
    # Z-Image — image generation
    {"model": "Tongyi-MAI/Z-Image-Turbo", "pipeline_type": "diffusion_zimage",
     "label": "B3-diffusion-zimage"},

    # === C. Encoder + Decoder ===
    # Whisper — speech-to-text (mel → encoder → decoder)
    {"model": "openai/whisper-tiny", "pipeline_type": "speech_to_text",
     "label": "C1-whisper"},
    # T5 — text-to-text (encoder-decoder seq2seq)
    {"model": "google-t5/t5-small", "pipeline_type": "text_to_text",
     "label": "C2-t5"},
    # Vision-Language — image encoder + text decoder
    {"model": "Qwen/Qwen2.5-VL-3B-Instruct", "pipeline_type": "vision_language",
     "label": "C3-vision-language"},

    # === D. Single Forward Pass ===
    # BERT — classic encoder-only
    {"model": "google-bert/bert-base-uncased", "pipeline_type": "encoder_only",
     "label": "D1-bert"},
    # Sentence embedding
    {"model": "sentence-transformers/all-MiniLM-L6-v2", "pipeline_type": "encoder_only",
     "label": "D2-embedding"},
    # Segmentation — image → mask
    {"model": "nvidia/segformer-b0-finetuned-ade-512-512", "pipeline_type": "segmentation",
     "label": "D3-segmentation"},
    # SAM — prompted segmentation
    {"model": "facebook/sam-vit-base", "pipeline_type": "prompted_segmentation",
     "label": "D4-sam"},

    # === E. Multi-Stage Pipeline ===
    # Bark — 3-stage audio generation
    {"model": "suno/bark-small", "pipeline_type": "text_to_audio_bark",
     "label": "E1-bark"},
    # Magpie TTS — multi-codebook + CFG
    {"model": "nvidia/magpie_tts_multilingual_357m", "pipeline_type": "text_to_audio_magpie",
     "label": "E2-magpie"},
]


def run_batch(models: list[dict], output_dir: str, dry_run: bool) -> list[TuneResult]:
    """Run auto-tune on multiple models."""
    results = []
    for entry in models:
        try:
            r = auto_tune_model(
                model=entry["model"],
                pipeline_type=entry.get("pipeline_type", "decoder_kv_cache"),
                output_dir=f"{output_dir}/{entry.get('label', 'model')}",
                dry_run=dry_run,
            )
            results.append(r)
        except Exception as e:
            print(f"\n[ERROR] {entry['model']}: {e}")
            r = TuneResult(
                model=entry["model"],
                pipeline_type=entry.get("pipeline_type", ""),
                mode=PIPELINE_MODES.get(entry.get("pipeline_type", ""), ""),
                status="failed", error=str(e))
            results.append(r)

    # Summary table
    print(f"\n{'='*110}")
    print(f"  Batch Summary ({len(results)} models)")
    print(f"{'='*110}")
    print(f"  {'Model':<25} {'Mode':<10} {'Nsys':<10} {'SOL':<8} "
          f"{'Baseline':>8} {'Best':>8} {'Speedup':>7} "
          f"{'BW SOL':>8} {'CMP SOL':>8} {'Util':>5}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*8} "
          f"{'-'*8} {'-'*8} {'-'*7} "
          f"{'-'*8} {'-'*8} {'-'*5}")
    for r in results:
        print(f"  {r.model.split('/')[-1]:<25} {r.mode:<10} "
              f"{r.bottleneck:<10} {r.sol_bottleneck_fp16 or r.sol_bottleneck:<8} "
              f"{r.baseline_tps:>7.1f} {r.optimized_tps:>7.1f} "
              f"{r.speedup:>6.2f}x "
              f"{r.bw_sol_tps_fp16 or r.bw_sol_tps:>7.0f} "
              f"{r.compute_sol_tps_fp16 or r.compute_sol_tps:>7.0f} "
              f"{r.utilization_after or r.utilization_before:>4.1f}%")
    print(f"{'='*110}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Automated performance tuning for TRT models.")
    parser.add_argument("--model",
                        help="HuggingFace model ID")
    parser.add_argument("--pipeline-type", default="decoder_kv_cache",
                        help="Runtime strategy")
    parser.add_argument("--max-cache-length", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--output-dir", default="/tmp/auto_perf")
    parser.add_argument("--batch", action="store_true",
                        help="Run validation across all 5 representative models")
    parser.add_argument("--batch-json",
                        help="Custom batch config JSON file")
    parser.add_argument("--engine-section", default="all",
                        help="Engine section for multi-engine analysis: "
                             "'all' (default), 'primary', 'secondary', or graph ID")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without running")

    args = parser.parse_args()

    if args.batch or args.batch_json:
        if args.batch_json:
            with open(args.batch_json) as f:
                models = json.load(f)
        else:
            models = VALIDATION_MODELS
        results = run_batch(models, args.output_dir, args.dry_run)
        # Save results
        out_path = f"{args.output_dir}/batch_results.json"
        os.makedirs(args.output_dir, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump([vars(r) for r in results], f, indent=2)
        print(f"\nResults saved: {out_path}")
    elif args.model:
        result = auto_tune_model(
            model=args.model,
            pipeline_type=args.pipeline_type,
            max_cache=args.max_cache_length,
            max_tokens=args.max_tokens,
            output_dir=args.output_dir,
            engine_section=args.engine_section,
            dry_run=args.dry_run,
        )
        # Save result
        out_path = f"{args.output_dir}/result.json"
        with open(out_path, "w") as f:
            json.dump(vars(result), f, indent=2)
        print(f"\nResult saved: {out_path}")
    else:
        parser.error("Specify --model or --batch")


if __name__ == "__main__":
    main()

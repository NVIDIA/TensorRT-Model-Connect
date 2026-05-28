#!/usr/bin/env python3
"""Speed-of-Light (SOL) estimation for TRT decoder inference.

Calculates the theoretical maximum throughput for a given model on a given
GPU, based on the roofline model. For batch=1 autoregressive decode, the
bottleneck is almost always memory bandwidth (GEMV reads all weights per
token), so SOL ≈ GPU_bandwidth / model_bytes_per_token.

Usage:
    # From HuggingFace model config
    python3 tools/sol_estimate.py --model Qwen/Qwen3-0.6B

    # With specific GPU and precision
    python3 tools/sol_estimate.py --model Qwen/Qwen2.5-1.5B --gpu B200 --dtype fp16

    # With actual measured throughput for utilization %
    python3 tools/sol_estimate.py --model Qwen/Qwen2.5-1.5B --actual-tps 265.7

    # Closed-loop: auto-extract TPS from benchmark JSON
    python3 tools/sol_estimate.py --model Qwen/Qwen2.5-1.5B --benchmark-json results.json

    # From bundle metadata
    python3 tools/sol_estimate.py --bundle /tmp/qwen3.trtfb

    # JSON output
    python3 tools/sol_estimate.py --model Qwen/Qwen3-0.6B --json

    # With KV cache size for attention bandwidth
    python3 tools/sol_estimate.py --model Qwen/Qwen2.5-1.5B --cache-length 2048

    # Per-layer roofline analysis (requires profiler JSON)
    python3 tools/sol_estimate.py --model Qwen/Qwen2.5-1.5B --gpu B200 --dtype fp16 \\
        --cache-length 256 --layer-timing-json profile.json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# GPU Specifications Database
# ---------------------------------------------------------------------------

@dataclass
class GpuSpec:
    """Hardware specifications for a GPU."""
    name: str
    hbm_bandwidth_gb_s: float   # GB/s (peak theoretical)
    fp32_tflops: float          # TFLOPS (CUDA cores, not tensor cores)
    fp16_tflops: float          # TFLOPS (tensor cores)
    hbm_capacity_gb: float      # GB

    # Practical bandwidth is ~85-90% of peak due to ECC, refresh, etc.
    practical_bw_ratio: float = 0.85


# Peak specs from NVIDIA datasheets.
# For batch=1 decode GEMV, CUDA core FLOPS are more relevant than tensor core.
GPU_SPECS: dict[str, GpuSpec] = {
    "B200": GpuSpec(
        name="NVIDIA B200",
        hbm_bandwidth_gb_s=8000,
        fp32_tflops=180,
        fp16_tflops=4500,
        hbm_capacity_gb=192,
    ),
    "H200": GpuSpec(
        name="NVIDIA H200",
        hbm_bandwidth_gb_s=4800,
        fp32_tflops=134,
        fp16_tflops=1979,
        hbm_capacity_gb=141,
    ),
    "H100": GpuSpec(
        name="NVIDIA H100 SXM",
        hbm_bandwidth_gb_s=3350,
        fp32_tflops=134,
        fp16_tflops=1979,
        hbm_capacity_gb=80,
    ),
    "A100": GpuSpec(
        name="NVIDIA A100 SXM",
        hbm_bandwidth_gb_s=2039,
        fp32_tflops=19.5,
        fp16_tflops=312,
        hbm_capacity_gb=80,
    ),
    "L40S": GpuSpec(
        name="NVIDIA L40S",
        hbm_bandwidth_gb_s=864,
        fp32_tflops=91.6,
        fp16_tflops=733,
        hbm_capacity_gb=48,
    ),
}

DEFAULT_GPU = "B200"

BYTES_PER_PARAM = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1, "int4": 0.5}


# ---------------------------------------------------------------------------
# Model Architecture Parameters
# ---------------------------------------------------------------------------

@dataclass
class ModelArch:
    """Model architecture parameters needed for SOL calculation."""
    name: str
    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int

    @property
    def total_params(self) -> int:
        """Estimate total parameter count."""
        per_layer = (
            # Attention: Q + K + V + O projections
            self.hidden_size * self.hidden_size  # Q
            + self.hidden_size * self.num_kv_heads * self.head_dim  # K
            + self.hidden_size * self.num_kv_heads * self.head_dim  # V
            + self.hidden_size * self.hidden_size  # O
            # MLP: gate + up + down (SwiGLU)
            + self.hidden_size * self.intermediate_size  # gate
            + self.hidden_size * self.intermediate_size  # up
            + self.intermediate_size * self.hidden_size  # down
            # Norms: 2 per layer (pre-attn + pre-mlp), each = hidden_size
            + 2 * self.hidden_size
        )
        # Embedding + LM head (often tied)
        embedding = self.vocab_size * self.hidden_size
        lm_head = self.vocab_size * self.hidden_size  # assume untied
        final_norm = self.hidden_size

        return self.num_layers * per_layer + embedding + lm_head + final_norm


def load_model_arch_from_hf(model_id: str) -> ModelArch:
    """Load model architecture from HuggingFace config."""
    try:
        from transformers import AutoConfig
    except ImportError:
        print("ERROR: transformers not installed. Use --bundle instead.",
              file=sys.stderr)
        sys.exit(1)

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

    hidden = getattr(config, "hidden_size", 0)
    num_layers = getattr(config, "num_hidden_layers",
                         getattr(config, "n_layer", 0))
    num_heads = getattr(config, "num_attention_heads",
                        getattr(config, "n_head", 0))
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    head_dim = getattr(config, "head_dim", hidden // num_heads if num_heads else 0)
    intermediate = getattr(config, "intermediate_size",
                           getattr(config, "n_inner", 4 * hidden))
    vocab = getattr(config, "vocab_size", 0)

    return ModelArch(
        name=model_id,
        hidden_size=hidden,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        intermediate_size=intermediate,
        vocab_size=vocab,
    )


def load_model_arch_from_bundle(bundle_path: str) -> ModelArch:
    """Load model architecture from a .trtfb bundle."""
    try:
        sys.path.insert(0, "python")
        from tensorrt_model_connect.bundle_writer import BundleReader
        reader = BundleReader(bundle_path)
        config = json.loads(reader.read_section("config"))
    except Exception:
        # Fallback: read bundle as binary, find config JSON
        with open(bundle_path, "rb") as f:
            data = f.read()
        # Find JSON config section
        import re
        match = re.search(rb'\{"[^"]*model_type[^}]+\}', data)
        if not match:
            print("ERROR: Could not extract config from bundle.",
                  file=sys.stderr)
            sys.exit(1)
        config = json.loads(match.group())

    hidden = config.get("hidden_size", 0)
    num_heads = config.get("num_attention_heads", 0)
    num_kv_heads = config.get("num_key_value_heads", num_heads)
    head_dim = config.get("head_dim", hidden // num_heads if num_heads else 0)

    return ModelArch(
        name=config.get("model_id", bundle_path),
        hidden_size=hidden,
        num_layers=config.get("num_hidden_layers", 0),
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        intermediate_size=config.get("intermediate_size", 4 * hidden),
        vocab_size=config.get("vocab_size", 0),
    )


# ---------------------------------------------------------------------------
# SOL Calculation
# ---------------------------------------------------------------------------

@dataclass
class SolEstimate:
    """Speed-of-Light estimation result."""
    model_name: str
    gpu_name: str
    dtype: str

    # Model stats
    total_params: int
    model_bytes: float          # total weight bytes at given precision

    # Bandwidth-bound SOL (almost always the bottleneck for batch=1)
    bw_sol_tps: float           # tokens/sec limited by bandwidth
    bw_utilization_pct: float   # actual / bw_sol * 100

    # Compute-bound SOL
    compute_sol_tps: float      # tokens/sec limited by compute
    compute_utilization_pct: float

    # Overall
    sol_tps: float              # min(bw_sol, compute_sol)
    bottleneck: str             # "bandwidth" or "compute"
    actual_tps: float           # measured throughput (0 if not provided)
    utilization_pct: float      # actual / sol * 100

    # KV cache overhead (if cache_length > 0)
    kv_bytes_per_token: float = 0
    cache_length: int = 0

    # Breakdown
    weight_read_bytes: float = 0
    kv_read_bytes: float = 0
    total_read_bytes: float = 0


def estimate_sol(
    arch: ModelArch,
    gpu: GpuSpec,
    dtype: str = "fp32",
    cache_length: int = 0,
    actual_tps: float = 0,
) -> SolEstimate:
    """Calculate SOL for batch=1 autoregressive decode.

    For batch=1, each token generation requires:
    1. Reading ALL model weights (GEMV, not GEMM)
    2. Reading KV cache for attention (scales with sequence length)
    3. Minimal compute (arithmetic intensity ≈ 2 for GEMV)

    Therefore SOL ≈ bandwidth / bytes_per_token.
    """
    bpp = BYTES_PER_PARAM[dtype]
    practical_bw = gpu.hbm_bandwidth_gb_s * gpu.practical_bw_ratio  # GB/s

    # --- Weight reads per token ---
    # For GEMV (batch=1), all weights are read once per token
    weight_bytes = arch.total_params * bpp

    # --- KV cache reads per token ---
    # Attention: read K and V cache for all layers
    # Per layer: 2 (K+V) * num_kv_heads * head_dim * cache_length * bpp
    kv_bytes = 0
    if cache_length > 0:
        kv_bytes = (
            2 * arch.num_kv_heads * arch.head_dim
            * cache_length * bpp * arch.num_layers
        )

    total_bytes = weight_bytes + kv_bytes

    # --- Bandwidth-bound SOL ---
    # SOL = bandwidth / bytes_per_token
    bw_sol = (practical_bw * 1e9) / total_bytes  # tokens/sec

    # --- Compute-bound SOL ---
    # FLOPS per token (batch=1 decode)
    flops_per_token = _compute_flops_per_token(arch, cache_length)

    # For batch=1, CUDA core FLOPS (not tensor core) are relevant
    if dtype in ("fp16", "bf16"):
        peak_flops = gpu.fp16_tflops * 1e12
    else:
        peak_flops = gpu.fp32_tflops * 1e12

    compute_sol = peak_flops / flops_per_token if flops_per_token > 0 else float("inf")

    # --- Overall SOL ---
    sol = min(bw_sol, compute_sol)
    bottleneck = "bandwidth" if bw_sol <= compute_sol else "compute"

    # --- Utilization ---
    bw_util = (actual_tps / bw_sol * 100) if actual_tps > 0 and bw_sol > 0 else 0
    compute_util = (actual_tps / compute_sol * 100) if actual_tps > 0 and compute_sol > 0 else 0
    util = (actual_tps / sol * 100) if actual_tps > 0 and sol > 0 else 0

    return SolEstimate(
        model_name=arch.name,
        gpu_name=gpu.name,
        dtype=dtype,
        total_params=arch.total_params,
        model_bytes=weight_bytes,
        bw_sol_tps=bw_sol,
        bw_utilization_pct=bw_util,
        compute_sol_tps=compute_sol,
        compute_utilization_pct=compute_util,
        sol_tps=sol,
        bottleneck=bottleneck,
        actual_tps=actual_tps,
        utilization_pct=util,
        kv_bytes_per_token=kv_bytes,
        cache_length=cache_length,
        weight_read_bytes=weight_bytes,
        kv_read_bytes=kv_bytes,
        total_read_bytes=total_bytes,
    )


def _compute_flops_per_token(arch: ModelArch, cache_length: int) -> float:
    """Compute total FLOPS for generating one token (batch=1 decode).

    For GEMV (batch=1), each matmul of shape (1, M) x (M, N) = 2*M*N FLOPS.
    """
    h = arch.hidden_size
    kv_h = arch.num_kv_heads
    d = arch.head_dim
    ff = arch.intermediate_size
    L = arch.num_layers
    seq = max(cache_length, 1)

    per_layer = (
        # Q projection: (1, h) x (h, h) = 2 * h * h
        2 * h * h
        # K projection: (1, h) x (h, kv_h*d) = 2 * h * kv_h * d
        + 2 * h * kv_h * d
        # V projection: same as K
        + 2 * h * kv_h * d
        # Attention QK^T: (1, d) x (d, seq) per head = 2 * num_heads * d * seq
        + 2 * arch.num_heads * d * seq
        # Attention V: (1, seq) x (seq, d) per head = 2 * num_heads * seq * d
        + 2 * arch.num_heads * seq * d
        # O projection: (1, h) x (h, h) = 2 * h * h
        + 2 * h * h
        # MLP gate: (1, h) x (h, ff) = 2 * h * ff
        + 2 * h * ff
        # MLP up: same
        + 2 * h * ff
        # MLP down: (1, ff) x (ff, h) = 2 * ff * h
        + 2 * ff * h
    )

    # LM head: (1, h) x (h, vocab) = 2 * h * vocab
    lm_head = 2 * h * arch.vocab_size

    return L * per_layer + lm_head


# ---------------------------------------------------------------------------
# Workload-Aware SOL Estimation
# ---------------------------------------------------------------------------

# Pipeline type → performance mode mapping
PIPELINE_MODES: dict[str, str] = {
    # A. Autoregressive decode
    "decoder_kv_cache": "decode",
    "decoder_moe": "decode",
    "ssm_recurrent": "decode",
    "rwkv_recurrent": "decode",
    "hybrid_mamba_attention": "decode",
    # B. Iterative denoising
    "diffusion_flux": "diffusion",
    "diffusion_wan": "diffusion",
    "diffusion_zimage": "diffusion",
    "diffusion_pixart": "diffusion",
    # C. Encoder + Decoder
    "speech_to_text": "enc_dec",
    "text_to_text": "enc_dec",
    "vision_language": "enc_dec",
    "seq2seq": "enc_dec",
    # D. Single forward pass
    "encoder_only": "single_pass",
    "embedding": "single_pass",
    "reranking": "single_pass",
    "segmentation": "single_pass",
    "prompted_segmentation": "single_pass",
    "object_detection": "single_pass",
    "neural_operator": "single_pass",
    # E. Multi-stage pipeline
    "text_to_audio_bark": "multi_stage",
    "text_to_audio_magpie": "multi_stage",
    "speech_to_speech": "multi_stage",
    "omni_multimodal": "multi_stage",
}


def estimate_sol_for_workload(
    arch: ModelArch,
    gpu: GpuSpec,
    dtype: str = "fp32",
    pipeline_type: str = "decoder_kv_cache",
    cache_length: int = 0,
    actual_throughput: float = 0,
    sequence_length: int = 1,
    num_denoising_steps: int = 20,
) -> dict:
    """Workload-aware SOL estimation.

    Returns a dict with SOL analysis tailored to the pipeline's performance mode.
    For autoregressive decode, this wraps estimate_sol(). For other modes, it
    computes the appropriate roofline.

    Args:
        arch: Model architecture.
        gpu: GPU specifications.
        dtype: Precision string.
        pipeline_type: One of the runtime_strategy strings (e.g., "decoder_kv_cache").
        cache_length: KV cache length (decode mode).
        actual_throughput: Measured throughput (tok/s, img/s, or samples/s).
        sequence_length: Input sequence length (for encoder/single-pass modes).
        num_denoising_steps: Number of denoising steps (diffusion mode).
    """
    mode = PIPELINE_MODES.get(pipeline_type, "decode")
    bpp = BYTES_PER_PARAM[dtype]

    if mode == "decode":
        est = estimate_sol(arch, gpu, dtype, cache_length, actual_throughput)
        return {
            "mode": "decode",
            "pipeline_type": pipeline_type,
            "sol_tps": round(est.sol_tps, 1),
            "bottleneck": est.bottleneck,
            "utilization_pct": round(est.utilization_pct, 1),
            "bw_sol_tps": round(est.bw_sol_tps, 1),
            "compute_sol_tps": round(est.compute_sol_tps, 1),
            "actual": actual_throughput,
            "unit": "tok/s",
            "detail": to_json(est),
        }

    if mode == "diffusion":
        return _estimate_sol_diffusion(
            arch, gpu, dtype, bpp, num_denoising_steps, actual_throughput)

    if mode == "enc_dec":
        return _estimate_sol_enc_dec(
            arch, gpu, dtype, bpp, cache_length, sequence_length,
            actual_throughput)

    if mode == "single_pass":
        return _estimate_sol_single_pass(
            arch, gpu, dtype, bpp, sequence_length, actual_throughput)

    if mode == "multi_stage":
        return {
            "mode": "multi_stage",
            "pipeline_type": pipeline_type,
            "note": "Multi-stage pipeline: profile each stage separately. "
                    "SOL = bottleneck of slowest stage.",
            "recommendation": "Run nsys profile and use classify_bottleneck.py "
                              "per stage.",
            "actual": actual_throughput,
            "unit": "samples/s",
        }

    # Fallback
    return {"mode": mode, "pipeline_type": pipeline_type, "error": "unknown mode"}


def _estimate_sol_diffusion(
    arch: ModelArch,
    gpu: GpuSpec,
    dtype: str,
    bpp: float,
    num_steps: int,
    actual_throughput: float,
) -> dict:
    """SOL for diffusion models: compute-bound (large GEMMs on latent tensors).

    For diffusion, each denoising step runs a full DiT/UNet forward pass.
    The bottleneck is usually compute (tensor core GEMM), not bandwidth.
    SOL ≈ compute_peak / (FLOPS_per_step × num_steps).
    """
    # FLOPS per forward pass (all layers, full sequence — not batch=1 GEMV)
    # For DiT: similar to transformer but with full sequence (latent patches)
    # Approximate: 2 × total_params × sequence_elements per forward pass
    # For diffusion, sequence_elements ≈ latent_size (e.g., 1024-4096 patches)
    flops_per_forward = 2 * arch.total_params  # rough: 2 FLOPS per param per element

    if dtype in ("fp16", "bf16"):
        peak_flops = gpu.fp16_tflops * 1e12
    else:
        peak_flops = gpu.fp32_tflops * 1e12

    # Time for one forward pass (compute-bound)
    time_per_step_s = flops_per_forward / peak_flops if peak_flops > 0 else float("inf")
    total_time_s = time_per_step_s * num_steps
    sol_samples_per_s = 1.0 / total_time_s if total_time_s > 0 else 0

    # Also compute bandwidth-bound estimate
    weight_bytes = arch.total_params * bpp
    practical_bw = gpu.hbm_bandwidth_gb_s * gpu.practical_bw_ratio * 1e9
    bw_time_per_step = weight_bytes / practical_bw
    bw_total_time = bw_time_per_step * num_steps
    bw_sol = 1.0 / bw_total_time if bw_total_time > 0 else 0

    sol = min(sol_samples_per_s, bw_sol)
    bottleneck = "compute" if sol_samples_per_s <= bw_sol else "bandwidth"

    util = (actual_throughput / sol * 100) if actual_throughput > 0 and sol > 0 else 0

    return {
        "mode": "diffusion",
        "pipeline_type": "diffusion",
        "sol": round(sol, 3),
        "bottleneck": bottleneck,
        "utilization_pct": round(util, 1),
        "compute_sol": round(sol_samples_per_s, 3),
        "bw_sol": round(bw_sol, 3),
        "num_denoising_steps": num_steps,
        "time_per_step_ms": round(time_per_step_s * 1000, 2),
        "actual": actual_throughput,
        "unit": "samples/s",
    }


def _estimate_sol_enc_dec(
    arch: ModelArch,
    gpu: GpuSpec,
    dtype: str,
    bpp: float,
    cache_length: int,
    sequence_length: int,
    actual_throughput: float,
) -> dict:
    """SOL for encoder-decoder: encoder is single-pass, decoder is autoregressive.

    Encoder: compute-bound (full sequence GEMM).
    Decoder: bandwidth-bound (batch=1 GEMV per token).
    Total latency = encoder_time + decoder_time × num_output_tokens.
    """
    if dtype in ("fp16", "bf16"):
        peak_flops = gpu.fp16_tflops * 1e12
    else:
        peak_flops = gpu.fp32_tflops * 1e12

    practical_bw = gpu.hbm_bandwidth_gb_s * gpu.practical_bw_ratio * 1e9
    weight_bytes = arch.total_params * bpp

    # Encoder: compute-bound, full sequence
    # FLOPS ≈ 2 × params × sequence_length (GEMM, not GEMV)
    seq = max(sequence_length, 1)
    encoder_flops = 2 * arch.total_params * seq
    encoder_time_s = encoder_flops / peak_flops if peak_flops > 0 else 0
    encoder_time_ms = encoder_time_s * 1000

    # Decoder: bandwidth-bound, batch=1 per token
    # SOL = bandwidth / weight_bytes (same as decode mode)
    decoder_sol_tps = practical_bw / weight_bytes if weight_bytes > 0 else 0

    return {
        "mode": "enc_dec",
        "pipeline_type": "encoder_decoder",
        "encoder_time_ms": round(encoder_time_ms, 2),
        "encoder_bottleneck": "compute",
        "decoder_sol_tps": round(decoder_sol_tps, 1),
        "decoder_bottleneck": "bandwidth",
        "note": "Total latency = encoder_time + (output_tokens / decoder_sol_tps). "
                "Optimize encoder with FP16 (compute-bound). "
                "Optimize decoder with GPU argmax + FP16 (bandwidth-bound).",
        "actual": actual_throughput,
        "unit": "tok/s (decoder phase)",
    }


def _estimate_sol_single_pass(
    arch: ModelArch,
    gpu: GpuSpec,
    dtype: str,
    bpp: float,
    sequence_length: int,
    actual_throughput: float,
) -> dict:
    """SOL for single-pass models (BERT, SegFormer, etc.).

    Full sequence processed in one forward pass. Typically compute-bound
    for moderate sequence lengths, bandwidth-bound for very short sequences.
    """
    if dtype in ("fp16", "bf16"):
        peak_flops = gpu.fp16_tflops * 1e12
    else:
        peak_flops = gpu.fp32_tflops * 1e12

    practical_bw = gpu.hbm_bandwidth_gb_s * gpu.practical_bw_ratio * 1e9
    weight_bytes = arch.total_params * bpp

    seq = max(sequence_length, 1)

    # Compute-bound: 2 × params × sequence_length FLOPS
    flops = 2 * arch.total_params * seq
    compute_time_s = flops / peak_flops if peak_flops > 0 else float("inf")
    compute_sol = 1.0 / compute_time_s if compute_time_s > 0 else 0

    # Bandwidth-bound: read all weights once
    bw_time_s = weight_bytes / practical_bw
    bw_sol = 1.0 / bw_time_s if bw_time_s > 0 else 0

    sol = min(compute_sol, bw_sol)
    bottleneck = "compute" if compute_sol <= bw_sol else "bandwidth"

    util = (actual_throughput / sol * 100) if actual_throughput > 0 and sol > 0 else 0

    return {
        "mode": "single_pass",
        "pipeline_type": "single_pass",
        "sol": round(sol, 1),
        "bottleneck": bottleneck,
        "utilization_pct": round(util, 1),
        "compute_sol": round(compute_sol, 1),
        "bw_sol": round(bw_sol, 1),
        "sequence_length": seq,
        "latency_ms": round(min(compute_time_s, bw_time_s) * 1000, 3),
        "actual": actual_throughput,
        "unit": "samples/s",
    }


def parse_benchmark_json(path: str) -> float:
    """Parse benchmark JSON and extract throughput TPS.

    Supports multiple formats:
    - perf_compare.py: json["trt"]["throughput_tps"]["mean"]
    - trtmc run stderr (future): json["throughput_tps"]
    - Flat format: json["throughput_tps"] or json["actual_tps"]

    Raises:
        FileNotFoundError: if path does not exist
        ValueError: if no TPS field found in JSON
    """
    with open(path) as f:
        data = json.load(f)

    # perf_compare.py format
    if "trt" in data and isinstance(data["trt"], dict):
        trt = data["trt"]
        if "throughput_tps" in trt and isinstance(trt["throughput_tps"], dict):
            val = trt["throughput_tps"].get("mean")
            if val is not None:
                return float(val)

    # Flat format: throughput_tps or actual_tps
    if "throughput_tps" in data:
        return float(data["throughput_tps"])
    if "actual_tps" in data:
        return float(data["actual_tps"])

    raise ValueError(
        f"No throughput field found in {path}. Expected "
        "trt.throughput_tps.mean, throughput_tps, or actual_tps."
    )


# ---------------------------------------------------------------------------
# Per-Layer Roofline
# ---------------------------------------------------------------------------

@dataclass
class LayerRoofline:
    """Per-layer roofline analysis result."""
    layer_name: str
    measured_ms: float
    theoretical_ms: float
    utilization_pct: float
    weight_bytes: float
    kv_bytes: float


def per_layer_roofline(
    arch: ModelArch,
    gpu: GpuSpec,
    dtype: str,
    layer_timing: dict,
    cache_length: int = 0,
) -> list[LayerRoofline]:
    """Compute per-layer roofline utilization from profiler timing.

    Args:
        arch: Model architecture parameters.
        gpu: GPU specifications.
        dtype: Data type string (e.g., "fp16").
        layer_timing: Dict with "layers" list and optional "lm_head_ms".
            Each layer: {"name": "layer_0", "time_ms": 1.5}
        cache_length: KV cache length.

    Returns:
        List of LayerRoofline sorted by utilization (worst first).
    """
    bpp = BYTES_PER_PARAM[dtype]
    practical_bw = gpu.hbm_bandwidth_gb_s * gpu.practical_bw_ratio * 1e9  # bytes/s

    h = arch.hidden_size
    kv_h = arch.num_kv_heads
    d = arch.head_dim
    ff = arch.intermediate_size

    # Per-layer weight bytes
    layer_weight_bytes = bpp * (
        h * h + h * kv_h * d + h * kv_h * d + h * h  # Q + K + V + O
        + h * ff + h * ff + ff * h                      # gate + up + down
        + 2 * h                                          # norms
    )

    # Per-layer KV cache bytes
    layer_kv_bytes = 0.0
    if cache_length > 0:
        layer_kv_bytes = 2 * kv_h * d * cache_length * bpp

    # LM head
    lm_head_weight_bytes = bpp * h * arch.vocab_size

    results = []
    for layer_info in layer_timing.get("layers", []):
        name = layer_info["name"]
        measured_ms = layer_info["time_ms"]
        total_bytes = layer_weight_bytes + layer_kv_bytes
        theoretical_ms = (total_bytes / practical_bw) * 1000  # ms
        util = (theoretical_ms / measured_ms * 100) if measured_ms > 0 else 0

        results.append(LayerRoofline(
            layer_name=name,
            measured_ms=measured_ms,
            theoretical_ms=theoretical_ms,
            utilization_pct=util,
            weight_bytes=layer_weight_bytes,
            kv_bytes=layer_kv_bytes,
        ))

    # LM head entry
    lm_head_ms = layer_timing.get("lm_head_ms", 0)
    if lm_head_ms > 0:
        theoretical_ms = (lm_head_weight_bytes / practical_bw) * 1000
        util = (theoretical_ms / lm_head_ms * 100) if lm_head_ms > 0 else 0
        results.append(LayerRoofline(
            layer_name="lm_head",
            measured_ms=lm_head_ms,
            theoretical_ms=theoretical_ms,
            utilization_pct=util,
            weight_bytes=lm_head_weight_bytes,
            kv_bytes=0,
        ))

    # Sort by utilization (worst first = lowest utilization = most headroom)
    results.sort(key=lambda r: r.utilization_pct)
    return results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _fmt_bytes(n: float) -> str:
    """Format byte count as human-readable."""
    if n >= 1e9:
        return f"{n / 1e9:.2f} GB"
    if n >= 1e6:
        return f"{n / 1e6:.1f} MB"
    return f"{n:.0f} B"


def _fmt_params(n: int) -> str:
    """Format parameter count."""
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    return f"{n:,}"


def print_report(est: SolEstimate) -> None:
    """Print human-readable SOL report."""
    print()
    print("=" * 60)
    print("  Speed-of-Light (SOL) Estimation")
    print("=" * 60)
    print(f"  Model:      {est.model_name}")
    print(f"  GPU:        {est.gpu_name}")
    print(f"  Precision:  {est.dtype}")
    print(f"  Parameters: {_fmt_params(est.total_params)}")
    print(f"  Weight size: {_fmt_bytes(est.model_bytes)}")
    if est.cache_length > 0:
        print(f"  KV cache:   {est.cache_length} tokens "
              f"({_fmt_bytes(est.kv_bytes_per_token)} per token)")
        print(f"  Total read: {_fmt_bytes(est.total_read_bytes)} per token")
    print()

    print("  Theoretical Maximum (batch=1 decode):")
    print(f"    Bandwidth-bound:  {est.bw_sol_tps:,.0f} tok/s")
    print(f"    Compute-bound:    {est.compute_sol_tps:,.0f} tok/s")
    print(f"    SOL (bottleneck): {est.sol_tps:,.0f} tok/s  [{est.bottleneck}]")
    print()

    if est.actual_tps > 0:
        print("  Actual vs SOL:")
        print(f"    Measured:      {est.actual_tps:,.1f} tok/s")
        print(f"    Utilization:   {est.utilization_pct:.1f}%")
        gap = est.sol_tps - est.actual_tps
        headroom = (gap / est.sol_tps * 100) if est.sol_tps > 0 else 0
        print(f"    Headroom:      {gap:,.0f} tok/s ({headroom:.0f}% room to improve)")
        print()

        if est.utilization_pct > 80:
            print("  --> Near SOL. Diminishing returns from optimization.")
        elif est.utilization_pct > 50:
            print("  --> Moderate utilization. Runtime optimizations (CUDA Graphs,")
            print("      async ops) can close the gap.")
        else:
            print("  --> Low utilization. Significant optimization opportunity.")
            print("      Check: kernel launch overhead, CPU sync, memory allocation.")
    else:
        print("  (No actual measurement provided. Use --actual-tps to compare.)")

    print("=" * 60)


def print_layer_roofline(layers: list[LayerRoofline]) -> None:
    """Print per-layer roofline table (sorted by utilization, worst first)."""
    if not layers:
        return
    print()
    print("  Per-Layer Roofline (sorted by utilization, worst first):")
    print(f"  {'Layer':<20} {'Measured':>10} {'Theoretical':>12} {'Util%':>8} {'Headroom':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*12} {'-'*8} {'-'*10}")
    for lr in layers:
        headroom = lr.measured_ms - lr.theoretical_ms
        print(f"  {lr.layer_name:<20} {lr.measured_ms:>9.3f}ms {lr.theoretical_ms:>11.3f}ms "
              f"{lr.utilization_pct:>7.1f}% {headroom:>9.3f}ms")
    print()


def to_json(
    est: SolEstimate,
    layer_roofline: list[LayerRoofline] | None = None,
) -> dict:
    """Convert SOL estimate to JSON-serializable dict."""
    result = {
        "model": est.model_name,
        "gpu": est.gpu_name,
        "dtype": est.dtype,
        "total_params": est.total_params,
        "model_bytes": est.model_bytes,
        "cache_length": est.cache_length,
        "kv_bytes_per_token": est.kv_bytes_per_token,
        "total_read_bytes_per_token": est.total_read_bytes,
        "bw_sol_tps": round(est.bw_sol_tps, 1),
        "compute_sol_tps": round(est.compute_sol_tps, 1),
        "sol_tps": round(est.sol_tps, 1),
        "bottleneck": est.bottleneck,
        "actual_tps": est.actual_tps,
        "utilization_pct": round(est.utilization_pct, 1),
    }
    if layer_roofline is not None:
        result["layer_roofline"] = [
            {
                "layer_name": lr.layer_name,
                "measured_ms": round(lr.measured_ms, 3),
                "theoretical_ms": round(lr.theoretical_ms, 3),
                "utilization_pct": round(lr.utilization_pct, 1),
                "weight_bytes": lr.weight_bytes,
                "kv_bytes": lr.kv_bytes,
            }
            for lr in layer_roofline
        ]
    return result


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------

def detect_gpu() -> str | None:
    """Detect GPU model from nvidia-smi."""
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            name = result.stdout.strip().split("\n")[0]
            # Match to known specs
            for key in GPU_SPECS:
                if key.lower() in name.lower():
                    return key
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Estimate Speed-of-Light (SOL) throughput for TRT decode.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", help="HuggingFace model ID")
    group.add_argument("--bundle", help="Path to .trtfb bundle")

    parser.add_argument("--gpu", default=None, choices=list(GPU_SPECS.keys()),
                        help=f"GPU model (default: auto-detect or {DEFAULT_GPU})")
    parser.add_argument("--dtype", default="fp32",
                        choices=list(BYTES_PER_PARAM.keys()),
                        help="Compute precision (default: fp32)")
    parser.add_argument("--cache-length", type=int, default=0,
                        help="KV cache length for attention bandwidth (default: 0)")
    parser.add_argument("--actual-tps", type=float, default=0,
                        help="Actual measured throughput for utilization comparison")
    parser.add_argument("--benchmark-json", default=None,
                        help="Path to benchmark JSON (auto-extracts TPS; "
                        "overrides --actual-tps if both given)")
    parser.add_argument("--layer-timing-json", default=None,
                        help="Path to per-layer timing JSON from profiler "
                        "(enables per-layer roofline analysis)")
    parser.add_argument("--engine-section", default="primary",
                        help="Engine section for multi-engine bundles (metadata only, "
                             "default: primary)")
    parser.add_argument("--json", action="store_true",
                        help="Output JSON instead of text report")

    args = parser.parse_args()

    # Load model architecture
    if args.model:
        arch = load_model_arch_from_hf(args.model)
    else:
        arch = load_model_arch_from_bundle(args.bundle)

    # Resolve GPU
    gpu_key = args.gpu
    if gpu_key is None:
        gpu_key = detect_gpu() or DEFAULT_GPU
    gpu = GPU_SPECS[gpu_key]

    # Resolve actual TPS: benchmark-json overrides --actual-tps
    actual_tps = args.actual_tps
    if args.benchmark_json:
        actual_tps = parse_benchmark_json(args.benchmark_json)

    # Calculate SOL
    est = estimate_sol(
        arch=arch,
        gpu=gpu,
        dtype=args.dtype,
        cache_length=args.cache_length,
        actual_tps=actual_tps,
    )

    # Per-layer roofline (optional)
    layer_results = None
    if args.layer_timing_json:
        with open(args.layer_timing_json) as f:
            layer_timing = json.load(f)
        layer_results = per_layer_roofline(
            arch, gpu, args.dtype, layer_timing, args.cache_length)

    # Output
    if args.json:
        print(json.dumps(to_json(est, layer_results), indent=2))
    else:
        print_report(est)
        if layer_results:
            print_layer_roofline(layer_results)


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FlashInfer vs PyTorch SDPA — fused attention kernel benchmark.

Intent:
    Compares FlashInfer's fused attention kernel against PyTorch's
    scaled_dot_product_attention to quantify the speedup from using
    fused external kernels via the TVM-FFI bridge plugin.

Preconditions:
    - FlashInfer Python package available
    - CUDA GPU available
    - PyTorch available

Postconditions:
    - Prints latency comparison for decode and prefill attention
    - Reports speedup ratio

Trace IDs: ARCH-TVM-FFI-001, UD-TVM-FFI-PERF-001, IT-FLASHINFER-BENCH-001
"""

from __future__ import annotations

import time

import pytest

try:
    import torch
    import torch.nn.functional as F
    _has_torch = True
except ImportError:
    _has_torch = False

try:
    import flashinfer
    _has_flashinfer = True
except ImportError:
    _has_flashinfer = False


requires_flashinfer = pytest.mark.skipif(
    not (_has_torch and _has_flashinfer),
    reason="FlashInfer + PyTorch not available",
)


def _benchmark_fn(fn, warmup=10, iters=100):
    """Benchmark a CUDA function, return mean latency in ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return (elapsed / iters) * 1000.0  # ms


@requires_flashinfer
def test_flashinfer_decode_attention_speedup():
    """Compare FlashInfer single_decode vs PyTorch SDPA for decode step."""
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    configs = [
        # (num_heads, num_kv_heads, head_dim, cache_length)
        (16, 16, 128, 256),
        (16, 16, 128, 1024),
        (16, 16, 128, 2048),
        (32, 8, 128, 512),   # GQA
        (32, 8, 128, 2048),  # GQA
    ]

    print("\n" + "=" * 85)
    print("FlashInfer vs PyTorch SDPA — Decode (seq_len=1)")
    print("=" * 85)
    print(f"{'Config':<40} {'SDPA (ms)':<14} {'FlashInfer (ms)':<18} {'Speedup':<10}")
    print("-" * 85)

    speedups = []

    for num_heads, num_kv_heads, head_dim, cache_len in configs:
        # SDPA format: [batch, heads, seq, head_dim]
        q = torch.randn(1, num_heads, 1, head_dim, device=device, dtype=torch.float16)
        k = torch.randn(1, num_kv_heads, cache_len, head_dim, device=device, dtype=torch.float16)
        v = torch.randn(1, num_kv_heads, cache_len, head_dim, device=device, dtype=torch.float16)
        scale = 1.0 / (head_dim ** 0.5)

        # Expand KV for GQA to match SDPA requirements
        repeats = num_heads // num_kv_heads
        k_expanded = k.repeat(1, repeats, 1, 1) if repeats > 1 else k
        v_expanded = v.repeat(1, repeats, 1, 1) if repeats > 1 else v

        # FlashInfer format: q=[num_heads, head_dim], kv=[cache_len, num_kv_heads, head_dim]
        q_fi = q.squeeze(0).squeeze(1)  # [num_heads, head_dim]
        k_fi = k.squeeze(0).transpose(0, 1).contiguous()  # [cache_len, num_kv_heads, head_dim]
        v_fi = v.squeeze(0).transpose(0, 1).contiguous()

        # Benchmark SDPA
        def sdpa():
            return F.scaled_dot_product_attention(q, k_expanded, v_expanded, scale=scale)

        t_sdpa = _benchmark_fn(sdpa)

        # Benchmark FlashInfer
        def fi_decode():
            return flashinfer.single_decode_with_kv_cache(
                q_fi, k_fi, v_fi, sm_scale=scale,
            )

        t_flashinfer = _benchmark_fn(fi_decode)

        speedup = t_sdpa / t_flashinfer if t_flashinfer > 0 else 0
        speedups.append(speedup)

        gqa_str = " GQA" if repeats > 1 else ""
        config_str = f"h={num_heads} kv_h={num_kv_heads} d={head_dim} cache={cache_len}{gqa_str}"
        print(f"{config_str:<40} {t_sdpa:<14.4f} {t_flashinfer:<18.4f} {speedup:<10.2f}x")

    print("-" * 85)
    avg_speedup = sum(speedups) / len(speedups) if speedups else 0
    print(f"{'Average speedup:':<58} {'':<14} {avg_speedup:.2f}x")
    print("=" * 85)

    # FlashInfer should be at least competitive
    assert avg_speedup > 0.5, f"FlashInfer unexpectedly slow: {avg_speedup:.2f}x"


@requires_flashinfer
def test_flashinfer_prefill_attention_speedup():
    """Compare FlashInfer single_prefill vs PyTorch SDPA for prefill."""
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    configs = [
        # (num_heads, num_kv_heads, head_dim, seq_length)
        (16, 16, 128, 64),
        (16, 16, 128, 256),
        (16, 16, 128, 512),
        (32, 8, 128, 256),   # GQA
    ]

    print("\n" + "=" * 85)
    print("FlashInfer vs PyTorch SDPA — Prefill (causal)")
    print("=" * 85)
    print(f"{'Config':<40} {'SDPA (ms)':<14} {'FlashInfer (ms)':<18} {'Speedup':<10}")
    print("-" * 85)

    speedups = []

    for num_heads, num_kv_heads, head_dim, seq_len in configs:
        q = torch.randn(1, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
        k = torch.randn(1, num_kv_heads, seq_len, head_dim, device=device, dtype=torch.float16)
        v = torch.randn(1, num_kv_heads, seq_len, head_dim, device=device, dtype=torch.float16)
        scale = 1.0 / (head_dim ** 0.5)

        repeats = num_heads // num_kv_heads
        k_expanded = k.repeat(1, repeats, 1, 1) if repeats > 1 else k
        v_expanded = v.repeat(1, repeats, 1, 1) if repeats > 1 else v

        # FlashInfer format: [seq_len, num_heads, head_dim]
        q_fi = q.squeeze(0).transpose(0, 1).contiguous()  # [seq_len, num_heads, head_dim]
        k_fi = k.squeeze(0).transpose(0, 1).contiguous()
        v_fi = v.squeeze(0).transpose(0, 1).contiguous()

        def sdpa():
            return F.scaled_dot_product_attention(
                q, k_expanded, v_expanded, scale=scale, is_causal=True,
            )

        t_sdpa = _benchmark_fn(sdpa)

        def fi_prefill():
            return flashinfer.single_prefill_with_kv_cache(
                q_fi, k_fi, v_fi, sm_scale=scale, causal=True,
            )

        t_flashinfer = _benchmark_fn(fi_prefill)

        speedup = t_sdpa / t_flashinfer if t_flashinfer > 0 else 0
        speedups.append(speedup)

        gqa_str = " GQA" if repeats > 1 else ""
        config_str = f"h={num_heads} kv_h={num_kv_heads} d={head_dim} seq={seq_len}{gqa_str}"
        print(f"{config_str:<40} {t_sdpa:<14.4f} {t_flashinfer:<18.4f} {speedup:<10.2f}x")

    print("-" * 85)
    avg_speedup = sum(speedups) / len(speedups) if speedups else 0
    print(f"{'Average speedup:':<58} {'':<14} {avg_speedup:.2f}x")
    print("=" * 85)

    assert avg_speedup > 0.5, f"FlashInfer unexpectedly slow: {avg_speedup:.2f}x"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

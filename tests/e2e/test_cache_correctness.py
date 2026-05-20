"""E2E tests for KV cache correctness -- overflow and consistency.

These tests verify that the KV cache behaves correctly when prompts approach
or exceed the configured max_cache_length. They require GPU + the unified
trtmc binary.

Usage:
    pytest tests/e2e/test_cache_correctness.py -v \
      --trtmc-binary ./build/trtmc --hf-python .venv/bin/python \
      --engine-dir /mnt/storage/tensorrt-model-connect/engines

The tests build small bundles with specific cache sizes, so they are slower
than pure-binary tests but faster than the full E2E suite.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[2]

# Small, fast model for cache tests (Qwen3-0.6B is the standard canary)
CACHE_TEST_MODEL = "Qwen/Qwen3-0.6B"
CACHE_TEST_FAMILY = "qwen"

# Long prompt -- enough tokens to overflow small cache sizes.
# ~80 tokens when tokenised by most models.
LONG_PROMPT = (
    "The history of artificial intelligence began in antiquity, "
    "with myths, stories and rumors of artificial beings endowed "
    "with intelligence or consciousness by master craftsmen. The "
    "seeds of modern AI were planted by philosophers who attempted "
    "to describe the process of human thinking as the mechanical "
    "manipulation of symbols. This work culminated in the invention "
    "of the programmable digital computer in the 1940s, a machine "
    "based on the abstract essence of mathematical reasoning."
)

SHORT_PROMPT = "The capital of France is"


def _build_bundle(trtmc_binary, hf_id, output_path, max_cache_length, timeout=600):
    """Build a .trtfb bundle with a specific cache size."""
    cmd = [
        str(trtmc_binary), "build",
        hf_id, "-o", str(output_path),
        "--max-cache-length", str(max_cache_length),
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        pytest.fail(
            f"Bundle build failed (cache={max_cache_length}):\n"
            f"{result.stderr[-2000:]}")
    return output_path


def _run_inference(trtmc_binary, bundle_path, prompt, max_new_tokens,
                   hf_python, ld_library_path, timeout=120):
    """Run C++ inference and return (returncode, stdout, stderr)."""
    cmd = [
        str(trtmc_binary), "run", str(bundle_path),
        "--prompt", prompt,
        "--max-new-tokens", str(max_new_tokens),
        "--hf-python", str(hf_python),
    ]
    env = {"LD_LIBRARY_PATH": ld_library_path}
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, env=env)
    return result.returncode, result.stdout.strip(), result.stderr


def _has_tensorrt_model_connect():
    """Check if the Python builder package is available."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "tensorrt_model_connect", "version"],
            capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# Skip the entire module if the Python builder package is not installed.
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        not _has_tensorrt_model_connect(),
        reason="Python builder package not available (pip install -e tensorrt_model_connect/)"),
]


class TestCacheOverflow:
    """Verify inference doesn't crash when the prompt exceeds max_cache_length."""

    def test_cache_overflow_produces_output(self, trtmc_binary, hf_python,
                                            ld_library_path, engine_dir):
        """Generate with prompt exceeding max_cache_length=32 -> non-empty output."""
        bundle_path = engine_dir / "cache_test_32.trtfb"

        # Build with tiny cache (32 tokens, LONG_PROMPT is ~80 tokens)
        _build_bundle(trtmc_binary, CACHE_TEST_MODEL, bundle_path, max_cache_length=32)

        rc, stdout, stderr = _run_inference(
            trtmc_binary, bundle_path, LONG_PROMPT, max_new_tokens=10,
            hf_python=hf_python, ld_library_path=ld_library_path)

        assert rc == 0, (
            f"Inference crashed with cache overflow (rc={rc}):\n{stderr}")
        assert len(stdout) > 0, (
            "Inference produced no output with cache overflow")

    def test_cache_overflow_no_segfault(self, trtmc_binary, hf_python,
                                        ld_library_path, engine_dir):
        """Specifically verify no segfault (signal -11) on cache overflow."""
        bundle_path = engine_dir / "cache_test_32.trtfb"

        # Reuse if already built by previous test, otherwise build fresh
        if not bundle_path.is_file():
            _build_bundle(trtmc_binary, CACHE_TEST_MODEL, bundle_path, max_cache_length=32)

        rc, stdout, stderr = _run_inference(
            trtmc_binary, bundle_path, LONG_PROMPT, max_new_tokens=10,
            hf_python=hf_python, ld_library_path=ld_library_path)

        assert rc != -11, f"Segfault on cache overflow: {stderr}"


class TestCacheConsistency:
    """Verify that different cache sizes produce consistent first tokens
    for short prompts that fit in all cache sizes."""

    def test_first_tokens_match_across_cache_sizes(
            self, trtmc_binary, hf_python, ld_library_path, engine_dir):
        """Same short prompt with cache=64 vs cache=256 -> first N tokens match.

        When the prompt fits in both caches, the first generated tokens
        should be identical because greedy decoding is deterministic.
        """
        bundle_64 = engine_dir / "cache_test_64.trtfb"
        bundle_256 = engine_dir / "cache_test_256.trtfb"

        # Build both bundles (skip if build fails -- probably no GPU)
        _build_bundle(trtmc_binary, CACHE_TEST_MODEL, bundle_64, max_cache_length=64)
        _build_bundle(trtmc_binary, CACHE_TEST_MODEL, bundle_256, max_cache_length=256)

        max_new = 10

        rc_64, out_64, stderr_64 = _run_inference(
            trtmc_binary, bundle_64, SHORT_PROMPT, max_new,
            hf_python=hf_python, ld_library_path=ld_library_path)
        rc_256, out_256, stderr_256 = _run_inference(
            trtmc_binary, bundle_256, SHORT_PROMPT, max_new,
            hf_python=hf_python, ld_library_path=ld_library_path)

        assert rc_64 == 0, f"cache=64 inference failed:\n{stderr_64}"
        assert rc_256 == 0, f"cache=256 inference failed:\n{stderr_256}"
        assert len(out_64) > 0, "cache=64 produced empty output"
        assert len(out_256) > 0, "cache=256 produced empty output"

        # First generated tokens should match (greedy decoding is deterministic)
        assert out_64 == out_256, (
            f"Output mismatch across cache sizes:\n"
            f"  cache=64:  {out_64!r}\n"
            f"  cache=256: {out_256!r}")

        print("\n[cache_consistency] Outputs match (cache=64 vs cache=256):")
        print(f"  {out_64!r}")


class TestCacheBoundary:
    """Edge case: prompt length exactly equals max_cache_length."""

    def test_prompt_at_cache_boundary(self, trtmc_binary, hf_python,
                                      ld_library_path, engine_dir):
        """Prompt tokenising to ~64 tokens with cache=64 -> should succeed.

        This exercises the exact boundary condition where every cache slot
        is used by the prompt, leaving zero slots for generation.  The
        runtime should either succeed (generating from the last token)
        or fail cleanly.
        """
        bundle_path = engine_dir / "cache_test_64.trtfb"
        if not bundle_path.is_file():
            _build_bundle(trtmc_binary, CACHE_TEST_MODEL, bundle_path, max_cache_length=64)

        # Use a moderate-length prompt that's around 60-70 tokens
        boundary_prompt = (
            "In the beginning was the Word, and the Word was with God, "
            "and the Word was God. The same was in the beginning with God. "
            "All things were made by him; and without him was not any thing "
            "made that was made."
        )

        rc, stdout, stderr = _run_inference(
            trtmc_binary, bundle_path, boundary_prompt, max_new_tokens=5,
            hf_python=hf_python, ld_library_path=ld_library_path)

        # Must not crash
        assert rc != -11, f"Segfault at cache boundary: {stderr}"

        # Either succeeds or fails with a clean error
        if rc == 0:
            print(f"\n[cache_boundary] Succeeded at boundary: {stdout!r}")
        else:
            combined = (stderr + stdout).lower()
            assert ("error" in combined
                    or "cache" in combined
                    or "overflow" in combined
                    or "exceed" in combined
                    or "failed" in combined), (
                f"Non-zero exit without clear error: {stderr}")
            print("\n[cache_boundary] Clean failure at boundary (expected)")

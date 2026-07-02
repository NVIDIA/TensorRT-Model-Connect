# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for error handling -- malformed/missing bundles.

These tests verify that the C++ trtmc binary fails gracefully with informative
error messages when given bad inputs.  Most tests do NOT require a GPU or a
real engine bundle; they exercise the early-exit error paths.

Usage:
    pytest tests/e2e/test_error_handling.py -v --trtmc-binary ./build/trtmc
"""

from __future__ import annotations

import os
import subprocess

import pytest


# ---------------------------------------------------------------------------
# Tests that only need the binary (no bundle / no GPU)
# ---------------------------------------------------------------------------

class TestMissingBundle:
    """Error paths for bundles that don't exist or can't be read."""

    def test_missing_bundle_file(self, trtmc_binary, ld_library_path):
        """Non-existent bundle path -> non-zero exit with error message."""
        env = {"LD_LIBRARY_PATH": ld_library_path}
        result = subprocess.run(
            [str(trtmc_binary), "run", "/nonexistent/path.trtfb",
             "--prompt", "hello"],
            capture_output=True, text=True, timeout=30, env=env)

        assert result.returncode != 0, (
            "Expected non-zero exit for missing bundle")
        combined = (result.stderr + result.stdout).lower()
        assert ("error" in combined
                or "not found" in combined
                or "no such" in combined
                or "failed" in combined), (
            f"Expected error message, got:\nstdout={result.stdout}\n"
            f"stderr={result.stderr}")

    def test_directory_as_bundle(self, tmp_path, trtmc_binary, ld_library_path):
        """Passing a directory instead of a file -> non-zero exit."""
        env = {"LD_LIBRARY_PATH": ld_library_path}
        result = subprocess.run(
            [str(trtmc_binary), "run", str(tmp_path),
             "--prompt", "hello"],
            capture_output=True, text=True, timeout=30, env=env)

        assert result.returncode != 0, (
            "Expected non-zero exit when bundle path is a directory")


class TestMalformedBundle:
    """Error paths for bundles that exist but have invalid content."""

    def test_truncated_bundle(self, tmp_path, trtmc_binary, ld_library_path):
        """Truncated/corrupt bundle -> non-zero exit with error."""
        bad_bundle = tmp_path / "truncated.trtfb"
        bad_bundle.write_bytes(b"NOT_A_VALID_BUNDLE_FILE")

        env = {"LD_LIBRARY_PATH": ld_library_path}
        result = subprocess.run(
            [str(trtmc_binary), "run", str(bad_bundle),
             "--prompt", "hello"],
            capture_output=True, text=True, timeout=30, env=env)

        assert result.returncode != 0, (
            "Expected non-zero exit for corrupt bundle")

    def test_empty_file_as_bundle(self, tmp_path, trtmc_binary, ld_library_path):
        """Zero-byte file -> non-zero exit."""
        empty = tmp_path / "empty.trtfb"
        empty.write_bytes(b"")

        env = {"LD_LIBRARY_PATH": ld_library_path}
        result = subprocess.run(
            [str(trtmc_binary), "run", str(empty),
             "--prompt", "hello"],
            capture_output=True, text=True, timeout=30, env=env)

        assert result.returncode != 0, (
            "Expected non-zero exit for empty bundle")

    def test_random_bytes_bundle(self, tmp_path, trtmc_binary, ld_library_path):
        """Random bytes -> non-zero exit (no crash/segfault)."""
        bad_bundle = tmp_path / "random.trtfb"
        bad_bundle.write_bytes(os.urandom(4096))

        env = {"LD_LIBRARY_PATH": ld_library_path}
        result = subprocess.run(
            [str(trtmc_binary), "run", str(bad_bundle),
             "--prompt", "hello"],
            capture_output=True, text=True, timeout=30, env=env)

        assert result.returncode != 0, (
            "Expected non-zero exit for random-bytes bundle")
        # Specifically check for no segfault (signal 11)
        assert result.returncode != -11, (
            "Binary segfaulted on random-bytes bundle")


class TestBadArguments:
    """Error paths for invalid CLI arguments."""

    def test_no_subcommand(self, trtmc_binary, ld_library_path):
        """No subcommand at all -> non-zero exit or help text."""
        env = {"LD_LIBRARY_PATH": ld_library_path}
        result = subprocess.run(
            [str(trtmc_binary)],
            capture_output=True, text=True, timeout=30, env=env)

        # Either non-zero exit or prints usage info
        assert (result.returncode != 0
                or "usage" in (result.stdout + result.stderr).lower()
                or "help" in (result.stdout + result.stderr).lower()), (
            "Expected error or usage info with no subcommand")

    def test_unknown_subcommand(self, trtmc_binary, ld_library_path):
        """Unknown subcommand -> non-zero exit."""
        env = {"LD_LIBRARY_PATH": ld_library_path}
        result = subprocess.run(
            [str(trtmc_binary), "nonexistent_command"],
            capture_output=True, text=True, timeout=30, env=env)

        assert result.returncode != 0, (
            "Expected non-zero exit for unknown subcommand")

    def test_run_missing_prompt(self, tmp_path, trtmc_binary, ld_library_path):
        """run subcommand without --prompt -> non-zero exit."""
        # Create a dummy file so we get past the "file not found" check
        dummy = tmp_path / "dummy.trtfb"
        dummy.write_bytes(b"dummy")

        env = {"LD_LIBRARY_PATH": ld_library_path}
        result = subprocess.run(
            [str(trtmc_binary), "run", str(dummy)],
            capture_output=True, text=True, timeout=30, env=env)

        assert result.returncode != 0, (
            "Expected non-zero exit when --prompt is missing")


# ---------------------------------------------------------------------------
# Tests that need a real bundle (GPU required)
# ---------------------------------------------------------------------------

class TestEmptyPrompt:
    """Tests for edge-case prompts that require a real bundle."""

    @pytest.mark.e2e
    def test_empty_prompt_string(self, model_entry, trtmc_binary,
                                 hf_python, ld_library_path):
        """Empty prompt string -> should either produce output or fail cleanly."""
        # Skip non-text models
        if model_entry.get("test_type") in ("diffusion", "segmentation", "audio"):
            pytest.skip("Non-text model")
        runtime_strategy = str(model_entry.get("runtime_strategy") or "")
        task_strategy = str(model_entry.get("task_strategy") or "")
        if (
            runtime_strategy.endswith("_vision_language")
            or runtime_strategy == "text_to_audio"
            or task_strategy in ("segmentation", "prompted_segmentation")
        ):
            pytest.skip("Non-text runtime strategy")

        env = {"LD_LIBRARY_PATH": ld_library_path}
        result = subprocess.run(
            [str(trtmc_binary), "run", model_entry["bundle_path"],
             "--prompt", "",
             "--max-new-tokens", "5",
             "--hf-python", str(hf_python)],
            capture_output=True, text=True, timeout=120, env=env)

        # Must not crash (signal -11 = segfault)
        assert result.returncode != -11, (
            f"Segfault on empty prompt: {result.stderr}")

        # Either succeeds (some models can generate from empty) or
        # returns a clean error
        if result.returncode != 0:
            combined = (result.stderr + result.stdout).lower()
            assert ("error" in combined
                    or "empty" in combined
                    or "invalid" in combined
                    or "failed" in combined), (
                f"Non-zero exit without error message: {result.stderr}")

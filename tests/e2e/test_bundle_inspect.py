"""E2E: Inspect bundles and verify header fields."""

from __future__ import annotations

import subprocess
import pytest


@pytest.mark.e2e
def test_inspect_produces_output(model_entry, trtmc_binary, ld_library_path):
    """trtmc inspect <bundle> should produce valid output."""
    env = {"LD_LIBRARY_PATH": ld_library_path}
    result = subprocess.run(
        [str(trtmc_binary), "inspect", model_entry["bundle_path"]],
        capture_output=True, text=True, timeout=30, env=env)
    assert result.returncode == 0, f"inspect failed: {result.stderr}"
    assert len(result.stdout.strip()) > 0, "inspect produced no output"


@pytest.mark.e2e
def test_inspect_shows_runtime_strategy(model_entry, trtmc_binary, ld_library_path):
    """Inspect output should mention the runtime strategy."""
    env = {"LD_LIBRARY_PATH": ld_library_path}
    result = subprocess.run(
        [str(trtmc_binary), "inspect", model_entry["bundle_path"]],
        capture_output=True, text=True, timeout=30, env=env)
    assert result.returncode == 0
    # Verify runtime_strategy is printed and matches the expected value.
    expected = str(model_entry.get("runtime_strategy") or "")
    assert expected, "model entry must declare runtime_strategy"
    assert "Runtime strategy:" in result.stdout, (
        "Expected 'Runtime strategy:' field in inspect output")
    actual_line = [l for l in result.stdout.splitlines()
                   if "Runtime strategy:" in l]
    assert actual_line, "Expected Runtime strategy line in inspect output"
    actual = actual_line[0].split(":")[-1].strip()
    assert expected == actual, (
        f"Expected runtime_strategy '{expected}', got '{actual}'")

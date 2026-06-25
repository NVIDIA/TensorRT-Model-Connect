"""Performance parity test: C++ binary vs Python TrtRunner.

Verifies that the C++ binary and Python TrtRunner produce matching
text output, and reports timing as informational (not a hard assertion).

Requires GPU + engine bundle + built C++ binary.
Skips if any prerequisite is missing.

Trace: ARCH-PERF-001, UD-PERF-PARITY
Intent: Validate C++ binary and Python TrtRunner produce matching text output with informational timing
Preconditions: GPU, engine bundle, and built C++ binary are available (skips otherwise)
Postconditions: C++ and Python outputs match textually; timing difference is reported but not asserted
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[4]
DEFAULT_ENGINE_DIR = Path("/mnt/storage/tensorrt-model-connect/engines")
DEFAULT_BINARY = PROJECT_DIR / "build" / "trtmc"
DEFAULT_HF_PYTHON = PROJECT_DIR / ".venv" / "bin" / "python"

BUNDLE_NAME = "qwen3-0.6b.trtfb"
PROMPT = "The capital of France is"
MAX_NEW_TOKENS = 10


def pytest_addoption(parser):
    try:
        parser.addoption(
            "--engine-dir", default=None,
            help="Directory containing .trtfb bundles")
    except ValueError:
        pass  # already registered by another conftest
    try:
        parser.addoption(
            "--trtmc-binary", default=None,
            help="Path to the C++ trtmc binary")
    except ValueError:
        pass
    try:
        parser.addoption(
            "--hf-python", default=None,
            help="Python interpreter with HuggingFace tokenizers")
    except ValueError:
        pass


def _get_paths(request):
    """Resolve paths from CLI options or defaults, skip if missing."""
    engine_dir = request.config.getoption("--engine-dir", default=None)
    engine_dir = Path(engine_dir) if engine_dir else DEFAULT_ENGINE_DIR

    binary = request.config.getoption("--trtmc-binary", default=None)
    binary = Path(binary) if binary else DEFAULT_BINARY

    hf_python = request.config.getoption("--hf-python", default=None)
    hf_python = Path(hf_python) if hf_python else DEFAULT_HF_PYTHON

    bundle = engine_dir / BUNDLE_NAME
    if not bundle.exists():
        pytest.skip(f"Bundle not found: {bundle}")
    if not binary.exists():
        pytest.skip(f"C++ binary not found: {binary}")
    if not hf_python.exists():
        pytest.skip(f"HF python not found: {hf_python}")

    return bundle, binary, hf_python


def _run_cpp(binary: Path, bundle: Path, hf_python: Path) -> tuple[str, float]:
    """Run C++ binary and return (output_text, wall_clock_seconds)."""
    env = os.environ.copy()
    # Resolve TRT libs using the same interpreter passed to --hf-python.
    trt_probe = (
        "import importlib.util; "
        "s=importlib.util.find_spec('tensorrt_libs'); "
        "print(s.submodule_search_locations[0] if s and s.submodule_search_locations else '')"
    )
    probe = subprocess.run(
        [str(hf_python), "-c", trt_probe],
        capture_output=True,
        text=True,
        timeout=10,
    )
    trt_lib_dir = probe.stdout.strip()

    base = env.get("LD_LIBRARY_PATH", "")
    parts = [p for p in [trt_lib_dir, "/usr/local/cuda/lib64", base] if p]
    if parts:
        env["LD_LIBRARY_PATH"] = ":".join(parts)

    cmd = [
        str(binary),
        "run",
        str(bundle),
        "--prompt",
        PROMPT,
        "--max-new-tokens",
        str(MAX_NEW_TOKENS),
        "--hf-python",
        str(hf_python),
    ]

    t0 = time.perf_counter()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(PROJECT_DIR),
        env=env,
    )
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        pytest.fail(
            f"C++ binary failed with rc={result.returncode}\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stderr: {result.stderr}"
        )

    return result.stdout.strip(), elapsed


def _run_python(hf_python: Path, bundle: Path) -> tuple[str, float]:
    """Run Python TrtRunner via subprocess and return (output_text, wall_clock_seconds)."""
    script = f"""
import sys
sys.path.insert(0, "{PROJECT_DIR / 'python'}")
from tensorrt_model_connect.families.qwen.debug_runner import load_engine_from_bundle
from tensorrt_model_connect.families.qwen.debug_runner import TrtRunner
import numpy as np

bundle_path = "{bundle}"
engine_plan, header = load_engine_from_bundle(bundle_path)
runner = TrtRunner(
    engine_plan=engine_plan,
    max_cache_length=header["max_cache_length"],
    num_layers=header["num_layers"],
)

# Tokenize using transformers
from transformers import AutoTokenizer
model_id = header.get("model_id", "Qwen/Qwen3-0.6B")
# model_id in bundle may be a commit hash; fall back to known model
try:
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
except Exception:
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
input_ids = tokenizer.encode("{PROMPT}")

# Prefill + decode
for tid in input_ids[:-1]:
    runner.step(tid)
result = runner.step(input_ids[-1])
output_ids = list(input_ids)

for _ in range({MAX_NEW_TOKENS}):
    logits = result["logits"].flatten()
    next_token = int(np.argmax(logits))
    output_ids.append(next_token)
    eos = header.get("eos_token_id")
    if isinstance(eos, list):
        if next_token in eos:
            break
    elif next_token == eos:
        break
    result = runner.step(next_token)

text = tokenizer.decode(output_ids, skip_special_tokens=True)
print(text)
"""
    t0 = time.perf_counter()
    result = subprocess.run(
        [str(hf_python), "-c", script],
        capture_output=True, text=True, timeout=120,
        cwd=str(PROJECT_DIR))
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        pytest.fail(f"Python runner failed:\nstderr: {result.stderr}")

    return result.stdout.strip(), elapsed


def test_perf_parity(request):
    """Verify C++ binary and Python TrtRunner produce matching output.

    Also reports timing comparison as informational.
    """
    bundle, binary, hf_python = _get_paths(request)

    cpp_text, cpp_time = _run_cpp(binary, bundle, hf_python)
    py_text, py_time = _run_python(hf_python, bundle)

    # Report timing (informational, not a hard assertion)
    print(f"\n  C++ time:    {cpp_time:.2f}s")
    print(f"  Python time: {py_time:.2f}s")
    if py_time > 0:
        print(f"  Ratio (C++/Python): {cpp_time / py_time:.2f}x")

    # Assert text output matches
    assert cpp_text == py_text, (
        f"Text mismatch!\n  C++:    {cpp_text!r}\n  Python: {py_text!r}")

"""Fixtures for E2E tests — engine directory, binary, model parametrization."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from tests.e2e_harness.manifest_loader import iter_manifest_paths

E2E_DIR = Path(__file__).resolve().parent
MODELS_DIR = E2E_DIR / "models"
PROJECT_DIR = E2E_DIR.parents[1]


def _load_manifest():
    """Load model manifests from flat and model-owned E2E layouts."""
    models = []
    engine_dir = "/mnt/storage/tensorrt-model-connect/engines"

    for model_file in iter_manifest_paths(MODELS_DIR):
        with open(model_file) as f:
            entry = json.load(f)
        models.append(entry)

    return {"engine_dir": engine_dir, "models": models}


def _default_engine_dir():
    manifest = _load_manifest()
    return Path(manifest.get("engine_dir", "/mnt/storage/tensorrt-model-connect/engines"))


def _models():
    manifest = _load_manifest()
    return manifest.get("models", [])


# ---------------------------------------------------------------------------
# CLI options
# ---------------------------------------------------------------------------

def pytest_addoption(parser):
    def addoption(*args, **kwargs):
        try:
            parser.addoption(*args, **kwargs)
        except ValueError:
            pass

    addoption(
        "--engine-dir", default=None,
        help="Directory containing .trtfb bundles (default: from engines.json)")
    addoption(
        "--trtmc-binary", default=None,
        help="Path to the C++ trtmc binary (default: build/trtmc)")
    addoption(
        "--hf-python", default=None,
        help="Python interpreter with HuggingFace tokenizers (default: .venv/bin/python)")
    addoption(
        "--model-plugin-dir", default=None,
        help="Directory containing libtrtmc_model_*.so")
    addoption(
        "--e2e-model", action="append", default=[],
        help="Filter by E2E case name or family; repeat or comma-separate values")
    addoption(
        "--rebuild-engines", action="store_true", default=False,
        help="Force rebuild of all engine bundles (default: use cached)")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def engine_dir(request):
    """Resolved engine directory. Creates it if it doesn't exist."""
    cli_val = request.config.getoption("--engine-dir")
    if cli_val:
        d = Path(cli_val)
    else:
        d = _default_engine_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture(scope="session")
def trtmc_binary(request):
    """Path to the C++ trtmc binary."""
    cli_val = request.config.getoption("--trtmc-binary")
    if cli_val:
        binary = Path(cli_val)
    else:
        binary = PROJECT_DIR / "build" / "trtmc"
    if not binary.is_file():
        pytest.skip(f"trtmc binary not found: {binary}")
    return binary


@pytest.fixture(scope="session")
def hf_python(request):
    """Python interpreter with HuggingFace tokenizers."""
    cli_val = request.config.getoption("--hf-python")
    if cli_val:
        return Path(cli_val)
    venv_python = PROJECT_DIR / ".venv" / "bin" / "python"
    if venv_python.is_file():
        return venv_python
    return Path(sys.executable)


@pytest.fixture(scope="session")
def ld_library_path():
    """LD_LIBRARY_PATH with TRT libs."""
    try:
        result = subprocess.run(
            [sys.executable, "-c",
             "import importlib.util; s=importlib.util.find_spec('tensorrt_libs'); "
             "print(s.submodule_search_locations[0])"],
            capture_output=True, text=True, timeout=10)
        trt_lib_dir = result.stdout.strip()
    except Exception:
        trt_lib_dir = ""
    base = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [p for p in [trt_lib_dir, "/usr/local/cuda/lib64", base] if p]
    return ":".join(parts)


# ---------------------------------------------------------------------------
# Model parametrization
# ---------------------------------------------------------------------------

def _model_ids():
    return [m["name"] for m in _models()]


def _model_by_name(name):
    for m in _models():
        if m["name"] == name:
            return m
    return None


@pytest.fixture(params=_model_ids() or ["__no_models__"])
def model_entry(request, engine_dir):
    """Parametrized fixture yielding one model entry at a time."""
    name = request.param
    if name == "__no_models__":
        pytest.skip("No models in engines.json")
    entry = _model_by_name(name)
    if entry.get("skip"):
        pytest.skip(entry["skip"])
    bundle_path = engine_dir / entry["bundle"]
    if not bundle_path.is_file():
        pytest.skip(f"Bundle not found: {bundle_path}")
    entry["bundle_path"] = str(bundle_path)
    return entry


# ---------------------------------------------------------------------------
# Built-bundle fixture (for full-pipeline tests)
# ---------------------------------------------------------------------------

def _build_bundle(trtmc_binary, hf_id, bundle_path, max_cache_length, precision="fp32"):
    """Build a .trtfb bundle as a subprocess to isolate GPU memory.

    Returns build time in seconds.
    """
    cmd = [
        str(trtmc_binary), "build",
        hf_id, "-o", str(bundle_path),
        "--max-cache-length", str(max_cache_length),
    ]
    if precision != "fp32":
        cmd.extend(["--precision", str(precision)])
    t0 = time.monotonic()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    elapsed = time.monotonic() - t0

    if result.returncode != 0:
        pytest.fail(
            f"Bundle build failed for {hf_id} (rc={result.returncode}):\n"
            f"{result.stderr[-2000:]}")

    return elapsed


# ---------------------------------------------------------------------------
# Frame quality helpers (for diffusion tests)
# ---------------------------------------------------------------------------

def compute_frame_stats(frame_dir: Path) -> dict:
    """Load PNG frames from a directory, return aggregate pixel statistics.

    Returns dict with keys: count, mean, std, min, max.
    Pixel values are normalized to [0, 1].
    """
    from PIL import Image

    frames = sorted(frame_dir.glob("frame_*.png"))
    if not frames:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

    all_pixels = []
    for fp in frames:
        img = Image.open(fp).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        all_pixels.append(arr.flatten())

    combined = np.concatenate(all_pixels)
    return {
        "count": len(frames),
        "mean": float(np.mean(combined)),
        "std": float(np.std(combined)),
        "min": float(np.min(combined)),
        "max": float(np.max(combined)),
    }


@pytest.fixture(params=_model_ids() or ["__no_models__"])
def built_bundle(request, engine_dir, trtmc_binary):
    """Parametrized fixture that ensures a bundle exists, building if needed.

    Returns a dict:
        path: str          — absolute path to the .trtfb file
        entry: dict        — the model manifest entry
        build_time_s: float | None  — build time if freshly built
        was_cached: bool   — True if the existing bundle was reused
    """
    name = request.param
    if name == "__no_models__":
        pytest.skip("No models in engines.json")

    entry = _model_by_name(name)
    if entry.get("skip"):
        pytest.skip(entry["skip"])
    bundle_path = engine_dir / entry["bundle"]
    rebuild = request.config.getoption("--rebuild-engines")

    if bundle_path.is_file() and not rebuild:
        entry["bundle_path"] = str(bundle_path)
        return {
            "path": str(bundle_path),
            "entry": entry,
            "build_time_s": None,
            "was_cached": True,
        }

    # Build the bundle
    hf_id = entry["hf_id"]
    max_cache = entry.get("max_cache_length", 256)
    precision = entry.get("precision", "fp32")
    build_time = _build_bundle(trtmc_binary, hf_id, bundle_path, max_cache, precision)

    entry["bundle_path"] = str(bundle_path)
    return {
        "path": str(bundle_path),
        "entry": entry,
        "build_time_s": build_time,
        "was_cached": False,
    }

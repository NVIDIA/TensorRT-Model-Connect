# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal reference fixtures used by the family correctness tests."""

from contextlib import contextmanager
import importlib
from pathlib import Path
import subprocess
import sys

import numpy as np

from families.cosyvoice3.config import SOURCE_REVISION

# Elementwise gates for one estimator call, calibrated on 2026-09-08 against
# a float64 evaluation of the pinned official model (the development FP64 audit). The
# official FP32 model itself deviates from float64 truth by up to 6.6x the
# former 1e-3 tolerance (synthetic CFG trajectories) and the native engine by
# up to 6.2x, so a native versus official-FP32 comparison must admit their
# sum; the worst observed pair differs by 9.7x. These gates detect wrong
# mathematics, not sub-reference rounding; the development FP64 audit addresses that.
ATOL = 2e-2
RTOL = 2e-2
# Ten-step Euler integration with classifier-free guidance amplifies rounding:
# on the 256-frame prompt-free acoustic case official FP32 differs from the
# float64 integration by 29x the former tolerance, the native engine by 16x,
# and the two FP32 results from each other by 43.5x.
INTEGRATED_ATOL = 1e-1
INTEGRATED_RTOL = 1e-1


@contextmanager
def _ieee_fp32_reference(torch):
    """Match the native engine's FP32 policy in the official reference.

    PyTorch enables TF32 for cuDNN convolutions by default on Ampere GPUs,
    while ``build_flow_engine`` explicitly clears TensorRT's TF32 flag.  A
    parity test must compare equal precision policies; otherwise the first
    causal convolution differs before any model-specific TensorRT math runs.
    """
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    matmul_precision = torch.get_float32_matmul_precision()
    try:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        yield
    finally:
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.set_float32_matmul_precision(matmul_precision)


def _official_dit(source: Path):
    revision = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if revision != SOURCE_REVISION:
        raise ValueError(f"CosyVoice source must be exactly {SOURCE_REVISION}; got {revision}")
    status = subprocess.run(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all", "--", "cosyvoice"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if status:
        raise ValueError("Official CosyVoice source has local changes; use a clean pinned checkout")
    sys.path.insert(0, str(source))
    module = importlib.import_module("cosyvoice.flow.DiT.dit")
    resolved = Path(module.__file__).resolve()
    if source.resolve() not in resolved.parents:
        raise ValueError(f"Imported CosyVoice from the wrong checkout: {resolved}")
    return module.DiT, revision


def _cases():
    # Synthetic independent batch rows stress the estimator. These are NOT
    # paired CFG inputs from a speech request, nor a measure of TTS success.
    rng = np.random.default_rng(2512)
    result = []
    for frames in (4, 17, 64, 128):
        for masked in (False, True):
            values = {
                "x": rng.normal(size=(2, 80, frames)).astype(np.float32),
                "mask": np.ones((2, 1, frames), np.float32),
                "mu": rng.normal(size=(2, 80, frames)).astype(np.float32),
                "t": np.array([0.0, 0.7], np.float32),
                "spks": rng.normal(size=(2, 80)).astype(np.float32),
                "cond": rng.normal(size=(2, 80, frames)).astype(np.float32),
            }
            if masked and frames > 1:
                values["mask"][1, :, -min(3, frames - 1):] = 0
            result.append((frames, masked, values))
    return result


def _cfg_cases():
    rng = np.random.default_rng(2512)
    for frames in (4, 17, 64, 128):
        for masked in (False, True):
            values = {name: rng.normal(size=(1, 80, frames)).astype(np.float32)
                      for name in ("mu", "cond", "noise")}
            values["spks"] = rng.normal(size=(1, 80)).astype(np.float32)
            values["mask"] = np.ones((1, 1, frames), np.float32)
            if masked:
                values["mask"][:, :, -min(3, frames - 1):] = 0
            # Synthetic prompt conditioning only in the prefix, not all frames.
            values["cond"][:, :, frames // 2:] = 0
            yield frames, masked, values


def _official_solver(source):
    """Verify the required pinned submodule, then import the real upstream class."""
    matcha = source / "third_party/Matcha-TTS"

    def git(root, *args):
        return subprocess.run(["git", "-C", str(root), *args], check=True,
                              capture_output=True, text=True).stdout.strip()

    expected = git(source, "rev-parse", "HEAD:third_party/Matcha-TTS")
    if not (matcha / ".git").exists() or git(matcha, "rev-parse", "HEAD") != expected:
        raise ValueError("Initialize the pinned official Matcha-TTS submodule")
    if git(matcha, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("Official Matcha-TTS submodule must be clean")
    sys.path.insert(0, str(matcha))
    module = importlib.import_module("cosyvoice.flow.flow_matching")
    dependency = importlib.import_module("matcha.models.components.flow_matching")
    for imported, root in ((module, source), (dependency, matcha)):
        if root.resolve() not in Path(imported.__file__).resolve().parents:
            raise ValueError("Official solver imported from the wrong source directory")
    return module.ConditionalCFM, expected


def acoustic_cases(tokens, features, speaker):
    """Fixed coverage declared before inference: 16/64/128/256 mel frames x prompt/no prompt."""
    if (tokens.ndim != 2 or tokens.shape[0] != 1 or tokens.shape[1] < 128
            or features.ndim != 3 or features.shape[0] != 1 or features.shape[1] < 256
            or features.shape[2] != 80 or speaker.shape != (1, 192)):
        raise ValueError("Audio must provide at least 128 speech tokens and 256 mel frames")
    for count in (8, 32, 64, 128):
        for with_prompt in (False, True):
            prompt = min(25, count // 2) if with_prompt else 0
            yield {"id": f"tokens{count}_prompt{prompt}", "frames": count * 2, "prompt_tokens_count": prompt,
                   "tokens": tokens[:, prompt:count].copy(), "prompt_tokens": tokens[:, :prompt].copy(),
                   "prompt_features": features[:, :prompt * 2].copy(), "speaker": speaker.copy()}


def compare_outputs(actual, expected, *, atol, rtol):
    """Reject invalid comparisons and explain an elementwise allclose result."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or actual.size == 0:
        raise ValueError("Parity outputs must have identical, nonempty shapes")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("Parity outputs must both be finite")
    difference = np.abs(actual - expected)
    tolerance = atol + rtol * np.abs(expected)
    return {
        "passed": bool(np.allclose(actual, expected, atol=atol, rtol=rtol)),
        "max_abs_error": float(difference.max()),
        "mean_abs_error": float(difference.mean()),
        "max_tolerance_ratio": float((difference / tolerance).max()),
        "elements_over_tolerance": int(np.count_nonzero(difference > tolerance)),
        "total_elements": int(difference.size),
        "max_abs_error_by_batch": [float(value.max()) for value in difference],
    }

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DINOv3 feature conversion criteria shared by E2E and dataset qualification."""

from functools import lru_cache
from pathlib import Path
import json

import numpy as np


def _cosine(left, right) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    assert a.shape == b.shape and a.size > 0
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    assert denominator > 0.0
    return float(np.dot(a, b) / denominator)


def _assert_parity(actual, expected, manifest: dict, case: dict, thresholds: dict) -> None:
    manifest["task"]
    actual_hidden = np.asarray(actual["last_hidden_state"], dtype=np.float64).reshape(
        actual["last_hidden_state_shape"]
    )
    expected_hidden = np.asarray(expected["last_hidden_state"], dtype=np.float64)
    actual_pooler = np.asarray(actual["pooler_output"], dtype=np.float64).reshape(
        actual["pooler_output_shape"]
    )
    expected_pooler = np.asarray(expected["pooler_output"], dtype=np.float64)
    assert actual_hidden.shape == expected_hidden.shape
    assert actual_pooler.shape == expected_pooler.shape
    assert np.isfinite(actual_hidden).all() and np.isfinite(actual_pooler).all()
    assert np.isfinite(expected_hidden).all() and np.isfinite(expected_pooler).all()
    assert np.array_equal(actual_pooler, actual_hidden[:, 0, :])
    assert np.array_equal(expected_pooler, expected_hidden[:, 0, :])
    assert _cosine(actual_hidden, expected_hidden) >= float(thresholds["full_cosine"])
    assert _cosine(actual_hidden[:, 0, :], expected_hidden[:, 0, :]) >= float(
        thresholds["cls_cosine"]
    )
    assert _cosine(actual_pooler, expected_pooler) >= float(thresholds["pooler_cosine"])
    relative_frobenius = np.linalg.norm(actual_hidden - expected_hidden) / max(
        float(np.linalg.norm(expected_hidden)), 1e-12
    )
    assert float(relative_frobenius) <= float(thresholds["relative_frobenius"])
    register_count = int(expected["num_register_tokens"])
    assert register_count == int(case["num_register_tokens"])
    patch_start = 1 + register_count
    assert 0 <= register_count and patch_start < actual_hidden.shape[1]
    if register_count:
        assert _cosine(
            actual_hidden[:, 1:patch_start, :], expected_hidden[:, 1:patch_start, :]
        ) >= float(thresholds["register_cosine"])
    actual_patches = actual_hidden[:, patch_start:, :].reshape(-1, actual_hidden.shape[-1])
    expected_patches = expected_hidden[:, patch_start:, :].reshape(-1, expected_hidden.shape[-1])
    assert actual_patches.shape[0] > 0 and expected_patches.shape[0] > 0
    denominators = np.linalg.norm(actual_patches, axis=1) * np.linalg.norm(expected_patches, axis=1)
    patch_cosines = np.divide(
        np.sum(actual_patches * expected_patches, axis=1),
        denominators,
        out=np.zeros_like(denominators),
        where=denominators != 0.0,
    )
    assert float(np.mean(patch_cosines)) >= float(thresholds["mean_patch_cosine"])
    assert float(np.percentile(patch_cosines, 1.0)) >= float(thresholds["p01_patch_cosine"])


@lru_cache(maxsize=None)
def _contract(profile: str) -> tuple[dict, dict, dict]:
    root = Path(__file__).resolve().parent / "tests"
    if Path(profile).name != profile:
        raise ValueError("DINOv3 validation profile must be a manifest name")
    manifest = json.loads((root / "manifests" / f"{profile}.json").read_text())
    case = manifest["testcases"][0]
    thresholds = json.loads((root / "thresholds" / f"{case['name']}.json").read_text())[
        "threshold_overrides"
    ]
    return manifest, case, thresholds


def compare(candidate: dict | None, native: dict | None, gate: dict) -> tuple[bool | None, str]:
    """Every image satisfies the existing complete-feature E2E criteria; no class-accuracy floor."""
    required = {
        "last_hidden_state",
        "last_hidden_state_shape",
        "pooler_output",
        "pooler_output_shape",
    }
    if any(not side or not required <= side.keys() for side in (candidate, native)):
        return None, "complete hidden and pooled feature outputs are required"
    manifest, case, thresholds = _contract(str(gate["profile"]))
    try:
        expected = {
            "last_hidden_state": np.asarray(native["last_hidden_state"], dtype=np.float64).reshape(
                native["last_hidden_state_shape"]
            ),
            "pooler_output": np.asarray(native["pooler_output"], dtype=np.float64).reshape(
                native["pooler_output_shape"]
            ),
            "num_register_tokens": int(native.get("num_register_tokens", case["num_register_tokens"])),
        }
        _assert_parity(candidate, expected, manifest, case, thresholds)
        return True, (
            f"full cosine {_cosine(candidate['last_hidden_state'], native['last_hidden_state']):.6f}; "
            "CLS, pooler, registers, patches and relative error satisfy the family criteria"
        )
    except (AssertionError, TypeError, ValueError, IndexError) as error:
        return False, f"DINOv3 full-feature criteria failed ({type(error).__name__}: {error})"

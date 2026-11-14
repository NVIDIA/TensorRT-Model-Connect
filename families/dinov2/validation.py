# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DINOv2 feature conversion criteria for dataset qualification.

The E2E gate of tests/test_e2e.py, applied per image against the FP32 native features: cosine at least
0.999 over every token feature, and the largest absolute difference within the build precision's
tolerance (5e-3 for FP32, 3.0 for FP16, set there from measured builds). The FP16 tanh-GELU share needs
a second native pass and stays an E2E check.
"""

import numpy as np

COSINE_FLOOR = 0.999
FEATURE_TOLERANCE = {"fp32": 5e-3, "fp16": 3.0}


def compare(candidate: dict | None, native: dict | None, gate: dict) -> tuple[bool | None, str]:
    required = {"last_hidden_state", "last_hidden_state_shape", "pooler_output", "pooler_output_shape"}
    if any(not side or not required <= side.keys() for side in (candidate, native)):
        return None, "complete hidden and pooled feature outputs are required"
    try:
        sides = []
        for side in (candidate, native):
            hidden = np.asarray(side["last_hidden_state"], dtype=np.float64).reshape(side["last_hidden_state_shape"])
            pooled = np.asarray(side["pooler_output"], dtype=np.float64).reshape(side["pooler_output_shape"])
            sides.append((hidden, pooled))
        (hidden, pooled), (expected, expected_pooled) = sides
        if hidden.shape != expected.shape or pooled.shape != expected_pooled.shape:
            return False, f"shape differs: {hidden.shape} vs {expected.shape}"
        if not (np.isfinite(hidden).all() and np.isfinite(pooled).all()):
            return False, "non-finite features"
        values, wanted = hidden.reshape(-1), expected.reshape(-1)
        cosine = float(values @ wanted / (np.linalg.norm(values) * np.linalg.norm(wanted)))
        worst = float(np.max(np.abs(values - wanted)))
        pooled_cosine = float(pooled.reshape(-1) @ expected_pooled.reshape(-1)
                              / (np.linalg.norm(pooled) * np.linalg.norm(expected_pooled)))
    except (KeyError, TypeError, ValueError) as error:
        return False, f"unreadable features ({type(error).__name__}: {error})"
    tolerance = FEATURE_TOLERANCE[str(gate.get("precision", "fp32"))] * float(gate.get("threshold_scale", 1.0))
    ok = cosine >= COSINE_FLOOR and pooled_cosine >= COSINE_FLOOR and worst <= tolerance
    return ok, (f"feature cosine {cosine:.6f}, pooled cosine {pooled_cosine:.6f}, "
                f"max abs {worst:.3e} (tolerance {tolerance:g})")

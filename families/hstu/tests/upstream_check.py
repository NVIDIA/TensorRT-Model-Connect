# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent original-model qualification with the public checkpoint."""

import os
import json
from pathlib import Path
import numpy as np
import pytest
from ..config import Config, ServingProfile
from ..checkpoint import load_weights
from ..reference import Reference, example_inputs


@pytest.mark.gpu
def test_original_dense_model_matches_full_sequence_reference():
    source = os.environ.get("DLRMV3_UPSTREAM")
    checkpoint = os.environ.get("DLRMV3_CHECKPOINT")
    converted = os.environ.get("DLRMV3_CONVERTED")
    if not all((source, checkpoint, converted)):
        pytest.skip("set DLRMv3 upstream source, raw checkpoint and converted weights")
    from .upstream import load_original, score_original

    model = load_original(Path(source), checkpoint)
    weights = load_weights(converted)
    for t, c in ((32, 8), (512, 128)):
        config = Config(serving=ServingProfile(history_length=t, candidates=c))
        profile = "short" if t == 32 else "default"
        gate = json.loads(
            (Path(__file__).parent / "thresholds" / f"dlrmv3-fp32-{profile}-b4.json").read_text()
        )["threshold_overrides"]
        for seed in (42, 43, 250, 1000):
            inputs = example_inputs(config, 4, seed=seed)
            if seed in (42, 43):
                inputs["history_lengths"][:2, 0] = [0, 18]
                inputs["num_candidates"][:2, 0] = [1, 2]
            original = score_original(model, config, inputs)
            actual = Reference(config, weights).score(inputs)["scores"]
            np.testing.assert_allclose(
                actual, original, atol=gate["score_atol"], rtol=gate["score_rtol"]
            )
            np.testing.assert_array_equal(actual.argmax(1), original.argmax(1))

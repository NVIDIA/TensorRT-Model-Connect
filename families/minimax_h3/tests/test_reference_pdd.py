# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PDD math contracts; these do not qualify a trained transformer or VSA kernel."""

from copy import deepcopy

import numpy as np
import pytest

from families.minimax_h3.reference_pdd import (
    CONTRACT,
    FINE_STEPS,
    HEAD_FEATURES,
    TRANSFORMER_CONFIG,
    fuse_output_projection,
    initial_target_noise,
    sampling_plan,
    scheduler_step,
    validate_contract,
    validate_transformer_config,
)


def test_exact_released_contract_is_explicit():
    validate_contract(deepcopy(CONTRACT))
    for key in CONTRACT:
        changed = deepcopy(CONTRACT)
        changed.pop(key)
        with pytest.raises(ValueError):
            validate_contract(changed)


def test_only_the_released_transformer_topology_is_admitted():
    validate_transformer_config(deepcopy(TRANSFORMER_CONFIG))
    for key in TRANSFORMER_CONFIG:
        changed = deepcopy(TRANSFORMER_CONFIG)
        changed.pop(key)
        with pytest.raises(ValueError):
            validate_transformer_config(changed)
    for key, value in [
        ("pdd_steps", 4),
        ("hidden_size", 5120),
        ("patch_size", [1, 2, True]),
        ("num_layers", 50.0),
        ("attention_head_dim", 96),
        ("unknown", 1),
    ]:
        changed = deepcopy(TRANSFORMER_CONFIG)
        changed[key] = value
        with pytest.raises(ValueError):
            validate_transformer_config(changed)


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_inference_steps", True),
        ("transformer_forwards", 4),
        ("grid_max_t", 1.0),
        ("video_scheduler_shift", 10.0),
        ("audio_scheduler_shift", 12.0),
        ("pdd_step_indices", [0, 4, 8, 12, 16, 20, 24, 28, True]),
        ("pdd_step_indices", [0, 4, 8, 12, 16, 20, 24, 28, 31]),
        ("attention_backend", "dense"),
        ("vsa_tile_size", 64),
        ("vsa_ref_keep_rate", 1.0),
        ("vsa_ref_policy", "single_region"),
    ],
)
def test_other_sampling_contracts_are_not_silently_adopted(field, value):
    changed = deepcopy(CONTRACT)
    changed[field] = value
    with pytest.raises(ValueError):
        validate_contract(changed)


@pytest.mark.parametrize("modality", ["video", "audio"])
def test_full_fine_grid_is_partitioned_without_losing_the_final_interval(modality):
    plan = sampling_plan(modality)
    assert plan.fine_grid.dtype == plan.integration_weights.dtype == np.float64
    assert plan.sigmas.dtype == plan.timesteps.dtype == np.float32
    assert plan.fine_grid.shape == (33,) and plan.sigmas.shape == (9,)
    assert plan.sigmas[0] == np.float32(0.999) and plan.sigmas[-1] == 0
    assert np.all(np.diff(plan.sigmas) < 0) and np.all(plan.integration_weights < 0)
    np.testing.assert_allclose(plan.integration_weights.sum(), -0.999, rtol=1e-14, atol=1e-16)
    with pytest.raises(ValueError):
        plan.sigmas[0] = 1


@pytest.mark.parametrize("modality", ["video", "audio"])
def test_constant_heads_stay_constant_and_do_not_change_checkpoint_inputs(modality):
    width = HEAD_FEATURES[modality]
    weight = np.ones((FINE_STEPS * width, 7), dtype=np.float32)
    bias = np.full(FINE_STEPS * width, 2.0, dtype=np.float32)
    for step in range(8):
        w, b = fuse_output_projection(weight, bias, modality, step)
        np.testing.assert_array_equal(w, np.ones((width, 7), dtype=np.float32))
        np.testing.assert_array_equal(b, np.full(width, 2.0, dtype=np.float32))
    assert np.all(weight == 1) and np.all(bias == 2)


@pytest.mark.parametrize("step", [True, -1, 8, 1.5])
def test_step_selection_cannot_change_the_trained_partition(step):
    with pytest.raises(ValueError):
        scheduler_step(np.ones(3, dtype=np.float32), np.ones(3, dtype=np.float32), "video", step)


def test_target_initialization_and_step_keep_the_H3_velocity_sign():
    noise = np.ones((3, 4), dtype=np.float32)
    initial = initial_target_noise(noise, "audio")
    np.testing.assert_array_equal(initial, np.full_like(noise, np.float32(0.999)))
    assert np.all(noise == 1)
    previous = initial.copy()
    for step in range(8):
        previous = scheduler_step(previous, np.ones_like(previous), "audio", step)
    assert np.all(previous > initial)
    with pytest.raises(ValueError):
        sampling_plan("conditioning")


@pytest.mark.parametrize("problem", ["shape", "dtype", "nonfinite", "bias"])
def test_malformed_parameters_fail_before_composition(problem):
    weight = np.ones((32 * 96, 7), dtype=np.float32)
    bias = None
    if problem == "shape":
        weight = weight[:-1]
    elif problem == "dtype":
        weight = weight.astype(np.float16)
    elif problem == "nonfinite":
        weight[0, 0] = np.nan
    else:
        bias = np.zeros(17, dtype=np.float32)
    with pytest.raises(ValueError):
        fuse_output_projection(weight, bias, "video", 0)

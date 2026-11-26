# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

PROFILES = ("dinov3-vits16-pretrain-lvd1689m", "dinov3-convnext-tiny-pretrain-lvd1689m")


def observation(registers=4):
    hidden = np.arange(1, (1 + registers + 4) * 8 + 1, dtype=np.float64).reshape(1, -1, 8)
    hidden[:, 1:, :] *= 0.0001
    return {
        "last_hidden_state": hidden.reshape(-1).tolist(),
        "last_hidden_state_shape": list(hidden.shape),
        "pooler_output": hidden[:, 0, :].reshape(-1).tolist(),
        "pooler_output_shape": [1, 8],
        "num_register_tokens": registers,
    }


@pytest.mark.parametrize("profile,registers", zip(PROFILES, (4, 0)))
def test_complete_feature_values_match_and_pooled_only_output_is_not_evidence(profile, registers):
    from families.dinov3.validation import compare

    native = observation(registers)
    gate = {"profile": profile}
    assert compare(native, native, gate)[0] is True
    candidate = {key: value for key, value in native.items() if key != "num_register_tokens"}
    # The public candidate output has no register-count metadata; its owning manifest supplies it.
    assert compare(candidate, candidate, gate)[0] is True
    assert compare(candidate, native, gate)[0] is True
    assert compare(candidate, {**native, "num_register_tokens": registers + 1}, gate)[0] is False
    incomplete = {key: value for key, value in native.items() if key != "last_hidden_state"}
    assert compare(incomplete, native, gate)[0] is None


@pytest.mark.parametrize("token", [1, 5])
def test_register_and_patch_errors_cannot_hide_in_a_good_full_cosine(token):
    from families.dinov3.validation import compare

    native = observation()
    mine = {**native, "last_hidden_state": list(native["last_hidden_state"])}
    mine["last_hidden_state"][token * 8 : (token + 1) * 8] = [
        -x for x in mine["last_hidden_state"][token * 8 : (token + 1) * 8]
    ]
    assert compare(mine, native, {"profile": PROFILES[0]})[0] is False


def test_wrong_shape_and_nonfinite_outputs_fail():
    from families.dinov3.validation import compare

    native = observation()
    assert (
        compare(
            {**native, "last_hidden_state_shape": [1, 3, 24]}, native, {"profile": PROFILES[0]}
        )[0]
        is False
    )
    mine = {**native, "last_hidden_state": list(native["last_hidden_state"])}
    mine["last_hidden_state"][0] = float("nan")
    assert compare(mine, native, {"profile": PROFILES[0]})[0] is False

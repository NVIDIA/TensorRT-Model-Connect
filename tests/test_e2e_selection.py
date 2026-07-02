# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from tests.e2e_harness.contracts import E2ECase
from tests.test_e2e import _case_matches_e2e_model, _parse_e2e_model_filters


def test_parse_e2e_model_filters_supports_repeat_and_csv():
    assert _parse_e2e_model_filters(["decoder_family, bark", "flux"]) == {
        "decoder_family",
        "bark",
        "flux",
    }


def test_case_matches_e2e_model_by_name_family_and_strategy():
    case = E2ECase(
        name="example-decoder-fp16",
        hf_id="example-org/example-decoder",
        family="example_decoder",
        runtime_strategy="example_decoder_decoder_kv_cache",
    )

    assert _case_matches_e2e_model(case, {"example-decoder-fp16"})
    assert _case_matches_e2e_model(case, {"example_decoder"})
    assert _case_matches_e2e_model(case, {"example_decoder_decoder_kv_cache"})
    assert not _case_matches_e2e_model(case, {"bark"})

from tests.e2e_harness.contracts import E2ECase
from tests.test_e2e import _case_matches_e2e_model, _parse_e2e_model_filters


def test_parse_e2e_model_filters_supports_repeat_and_csv():
    assert _parse_e2e_model_filters(["qwen, bark", "flux"]) == {
        "qwen",
        "bark",
        "flux",
    }


def test_case_matches_e2e_model_by_name_family_and_strategy():
    case = E2ECase(
        name="qwen3-0.6b-fp16",
        hf_id="Qwen/Qwen3-0.6B",
        family="qwen",
        runtime_strategy="decoder_kv_cache",
    )

    assert _case_matches_e2e_model(case, {"qwen3-0.6b-fp16"})
    assert _case_matches_e2e_model(case, {"qwen"})
    assert _case_matches_e2e_model(case, {"decoder_kv_cache"})
    assert not _case_matches_e2e_model(case, {"bark"})

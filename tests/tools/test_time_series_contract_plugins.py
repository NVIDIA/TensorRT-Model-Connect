from __future__ import annotations

from pathlib import Path

from tests.e2e_harness.contracts import E2ECase, StageOutput, ThresholdProfile
from tests.e2e_harness.plugins import find_plugin
from tests.e2e_harness.registry import activate_model_plugins, reset


REPO_ROOT = Path(__file__).resolve().parents[2]
E2E_MODELS = REPO_ROOT / "tests" / "e2e" / "models"


def _case(name: str, reference_family: str, user_contract: str) -> E2ECase:
    return E2ECase(
        name=name,
        hf_id="dummy/model",
        family="patchtst",
        runtime_strategy="patchtst_trt",
        reference_family=reference_family,
        user_contract=user_contract,
    )


def _activate_family(family: str, reference_family: str):
    reset()
    activate_model_plugins(E2E_MODELS / family)
    return find_plugin(reference_family)


def test_time_series_point_forecast_plugin_discovers_and_passes():
    plugin = _activate_family("patchtst", "time_series_point_forecast")
    assert plugin is not None
    assert plugin.user_contract == "time_series_point_forecast"

    case = _case("point-forecast-case", "time_series_point_forecast", plugin.user_contract)
    threshold = ThresholdProfile(
        task_strategy="neural_operator",
        metrics={"relative_l2": 1e-6, "max_pointwise_error": 1e-6},
    )
    trt = StageOutput(
        stage_name="full_inference",
        data={"output_field": [0.1, 0.2, 0.3, 0.4], "output_dim": 4},
    )
    ref = StageOutput(
        stage_name="full_inference",
        data={
            "output_field": [0.1, 0.2, 0.3, 0.4],
            "output_shape": [1, 2, 2],
            "reference_output_name": "prediction_outputs",
        },
    )
    result = plugin.verify(trt, ref, case, threshold)
    assert result.passed


def test_time_series_quantile_forecast_plugin_requires_quantile_output():
    plugin = _activate_family("chronos_bolt", "time_series_quantile_forecast")
    assert plugin is not None
    assert plugin.user_contract == "time_series_quantile_forecast"

    case = _case(
        "quantile-forecast-case",
        "time_series_quantile_forecast",
        plugin.user_contract,
    )
    threshold = ThresholdProfile(
        task_strategy="neural_operator",
        metrics={"relative_l2": 1e-6, "max_pointwise_error": 1e-6},
    )
    trt = StageOutput(
        stage_name="full_inference",
        data={"output_field": [0.1] * 6, "output_dim": 6},
    )
    ref = StageOutput(
        stage_name="full_inference",
        data={
            "output_field": [0.1] * 6,
            "output_shape": [1, 3, 2],
            "reference_output_name": "quantile_preds",
        },
    )
    result = plugin.verify(trt, ref, case, threshold)
    assert result.passed


def test_time_series_regression_plugin_discovers():
    plugin = _activate_family("patchtst", "time_series_regression")
    assert plugin is not None
    assert plugin.user_contract == "time_series_regression"


def test_time_series_classification_plugin_is_not_globally_shared():
    reset()
    plugin = find_plugin("time_series_classification")
    assert plugin is None

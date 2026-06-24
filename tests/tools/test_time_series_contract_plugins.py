from __future__ import annotations

from tests.e2e_harness.contracts import E2ECase, StageOutput, ThresholdProfile
from tests.e2e_harness.plugins import find_plugin


def _case(name: str, reference_family: str, user_contract: str) -> E2ECase:
    return E2ECase(
        name=name,
        hf_id="dummy/model",
        family="neural_operator",
        runtime_strategy="neural_operator",
        reference_family=reference_family,
        user_contract=user_contract,
    )


def test_time_series_point_forecast_plugin_discovers_and_passes():
    plugin = find_plugin("time_series_point_forecast")
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
    plugin = find_plugin("time_series_quantile_forecast")
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
    plugin = find_plugin("time_series_regression")
    assert plugin is not None
    assert plugin.user_contract == "time_series_regression"


def test_time_series_classification_plugin_discovers():
    plugin = find_plugin("time_series_classification")
    assert plugin is not None
    assert plugin.user_contract == "time_series_classification"

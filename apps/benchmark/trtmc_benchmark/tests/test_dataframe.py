# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
import struct
import subprocess
import sys

import pytest

from trtmc_benchmark.dataframe import prepare_forecast_frame, format_forecast_frame
from trtmc_benchmark.types import BenchmarkError, ModelDescriptor, MeasurementSpec, ResolvedCase
from trtmc_benchmark.worker import run_worker
from trtmc_benchmark.task_adapters import resolve_task_case

pd = pytest.importorskip("pandas")


def _case(tmp_path, task, request):
    path = tmp_path / "forecast.bundle"
    header = json.dumps(
        {"format": 1, "family": "numeric_fixture", "task": task, "backend": "fake", "sections": {}}
    ).encode()
    path.write_bytes(b"BUNDLE\x01\x00" + struct.pack("<Q", len(header)) + header)
    model = ModelDescriptor(
        "fixture",
        "fixture",
        "",
        path.name,
        "numeric_fixture",
        task,
        "fp32",
        tmp_path / "model.toml",
        (),
        {},
    )
    return ResolvedCase(
        "dataframe",
        model,
        "table",
        path,
        "solve",
        request,
        Path(os.environ["TRTMC_BENCH_RUNTIME_ROOT"]),
        MeasurementSpec(0, 1, telemetry="off"),
        {},
    )


@pytest.fixture
def worker():
    if not os.environ.get("TRTMC_BENCH_WORKER") or not os.environ.get("TRTMC_BENCH_RUNTIME_ROOT"):
        pytest.skip("set TRTMC_BENCH_WORKER and TRTMC_BENCH_RUNTIME_ROOT for loaded-DSO E2E")
    return Path(os.environ["TRTMC_BENCH_WORKER"])


def test_dataframe_joint_loaded_dso(tmp_path, worker):
    frame = pd.DataFrame(
        {
            "unique_id": ["z", "a", "z", "a", "z"],
            "ds": pd.to_datetime(
                ["2026-01-03", "2026-02-01", "2026-01-01", "2026-02-02", "2026-01-02"]
            ),
            "values": [3.0, 8.0, 1.0, 9.0, 2.0],
        }
    )
    prepared = prepare_forecast_frame(frame, freq="D", config_by_series={"a": {"frequency": 0}})
    assert prepared.series_ids == ("z", "a")
    assert [item["shape"] for item in prepared.request["items"]] == [[3, 1], [2, 1]]
    receipt = run_worker(
        _case(tmp_path, "batch_series_to_point_and_quantile_forecast", prepared.request),
        tmp_path,
        worker,
    )
    result = format_forecast_frame(prepared, receipt["output_summary"])
    assert result["unique_id"].tolist() == ["z", "z", "a", "a"]
    assert result["horizon_step"].tolist() == [1, 3, 1, 3]
    assert result["ds"].tolist() == list(
        pd.to_datetime(["2026-01-04", "2026-01-06", "2026-02-03", "2026-02-05"])
    )
    # Both items carry evaluation marker 1000: one native batch method, no scalar loop.
    assert result["forecast"].tolist() == [1001.0, 1014.0, 1008.0, 1020.0]
    assert result["forecast-q-0.5"].tolist() == [1006.0, 1019.0, 1013.0, 1025.0]
    assert len(receipt["observations"]) == 1
    assert result["channel_name"].isna().all()


def _frame():
    return pd.DataFrame(
        {
            "unique_id": ["b", "b", "a", "a"],
            "ds": pd.to_datetime(["2026-01-01", "2026-01-02"] * 2),
            "values": [1.0, 2.0, 8.0, 9.0],
        }
    )


@pytest.mark.parametrize(
    "task,point,quantile",
    [
        ("batch_series_to_point_forecast", True, False),
        ("batch_series_to_quantile_forecast", False, True),
        ("batch_series_to_point_and_quantile_forecast", True, True),
    ],
)
def test_all_batch_roles_multichannel(tmp_path, worker, task, point, quantile):
    frame = _frame().assign(second=[11.0, 12.0, 18.0, 19.0])
    prepared = prepare_forecast_frame(
        frame,
        freq="D",
        value_columns=("values", "second"),
        config_by_series={"a": {"frequency": 2}},
    )
    assert [item["shape"] for item in prepared.request["items"]] == [[2, 2], [2, 2]]
    resolved = resolve_task_case(task, {"inputs": prepared.request}, tmp_path)
    assert resolved.request == prepared.request and resolved.operation == "solve"
    receipt = run_worker(_case(tmp_path, task, resolved.request), tmp_path, worker)
    result = format_forecast_frame(prepared, receipt["output_summary"], model_name="model")
    assert len(result) == 8
    assert result["channel_index"].tolist() == [0, 1] * 4
    assert ("model" in result) == point
    assert ("model-q-0.5" in result) == quantile
    values = result["model"] if point else result["model-q-0.5"] - 5
    assert values.tolist() == [1001, 1002, 1013, 1014, 1208, 1209, 1220, 1221]


def test_masks_context_and_owned_request(tmp_path, worker):
    frame = _frame()
    frame.loc[0, "values"] = float("nan")
    frame["observed"] = [0, 1, 0, 1]
    configs = {"a": {"frequency": 0}}
    prepared = prepare_forecast_frame(
        frame, freq="D", observed_columns=("observed",), config_by_series=configs
    )
    assert prepared.request["items"][0]["past_values"] == [None, 2.0]
    assert prepared.request["items"][1]["past_values"] == [8.0, 9.0]
    assert prepared.request["items"][0]["config"] == {}
    frame.loc[:, "values"] = 777.0
    configs["a"]["frequency"] = 2
    assert prepared.request["items"][1]["config"] == {"frequency": 0}
    receipt = run_worker(
        _case(tmp_path, "batch_series_to_point_forecast", prepared.request), tmp_path, worker
    )
    result = format_forecast_frame(prepared, receipt["output_summary"])
    assert result["forecast"].tolist() == [990, 1002, 990, 1002]
    tail = prepare_forecast_frame(_frame(), freq="D", context_length=1)
    assert [item["shape"] for item in tail.request["items"]] == [[1, 1], [1, 1]]
    assert [item["past_values"] for item in tail.request["items"]] == [[2.0], [9.0]]


def _summary():
    part = {
        "shape": [2, 1],
        "axes": ["horizon", "channel"],
        "values": [1.0, 2.0],
        "horizon_steps": [1, 3],
        "channel_names": ["actual"],
        "channel_units": ["m"],
    }
    return {"items": [deepcopy(part), deepcopy(part)]}


@pytest.mark.parametrize(
    "start,freq,expected",
    [
        ("2026-01-31", "ME", ["2026-03-31", "2026-05-31"]),
        ("2026-01-01", "B", ["2026-01-05", "2026-01-07"]),
    ],
)
def test_calendar_offsets(start, freq, expected):
    frame = _frame()
    frame["ds"] = list(pd.date_range(start, periods=2, freq=freq)) * 2
    result = format_forecast_frame(prepare_forecast_frame(frame, freq=freq), _summary())
    assert result["ds"].tolist()[:2] == list(pd.to_datetime(expected))
    assert result["channel_name"].tolist() == ["actual"] * 4
    assert result["channel_unit"].tolist() == ["m"] * 4


def test_timezone_dst_and_per_series_frequency():
    times = pd.date_range("2026-03-06 12:00", periods=2, freq="D", tz="America/Los_Angeles")
    frame = _frame()
    frame["ds"] = list(times) * 2
    result = format_forecast_frame(
        prepare_forecast_frame(frame, freq={"a": "D", "b": "D"}), _summary()
    )
    assert [value.hour for value in result["ds"]] == [12] * 4
    assert str(result["ds"].dt.tz) == "America/Los_Angeles"
    assert result["ds"].iloc[0].utcoffset().total_seconds() == -7 * 3600


@pytest.mark.parametrize(
    "change,options,match",
    [
        (lambda f: f.iloc[:0], {}, "nonempty"),
        (lambda f: f.assign(unique_id=[None, "b", "a", "a"]), {}, "IDs"),
        (lambda f: f.assign(ds=[1, 2, 3, 4]), {}, "datetimes"),
        (lambda f: f.assign(ds=["2026-01-01"] * 4), {}, "datetimes"),
        (lambda f: f.assign(ds=pd.to_datetime(["2026-01-01"] * 4)), {}, "unique"),
        (lambda f: f.assign(ds=pd.to_datetime(["2026-01-01", "2026-01-03"] * 2)), {}, "grid"),
        (lambda f: f.assign(values=[float("inf"), 2, 3, 4]), {}, "finite"),
        (
            lambda f: f.assign(values=[float("nan"), 2, 3, 4], mask=[1, 1, 1, 1]),
            {"observed_columns": ("mask",)},
            "marked observed",
        ),
        (lambda f: f.assign(mask=[0, 0.5, 1, 1]), {"observed_columns": ("mask",)}, "zero/one"),
        (lambda f: f, {"value_columns": "values"}, "sequence"),
        (lambda f: f, {"context_length": 0}, "positive"),
        (lambda f: f, {"context_length": True}, "positive"),
        (lambda f: f, {"freq": None}, "explicit"),
        (lambda f: f, {"freq": "-1D"}, "positive"),
        (lambda f: f, {"freq": {"b": "D"}}, "series 'a'"),
        (lambda f: f, {"freq": {"b": "D", "a": "D", "missing": "D"}}, "unknown series"),
        (lambda f: f, {"config_by_series": {"missing": {}}}, "unknown series"),
        (
            lambda f: f.rename(columns={"unique_id": "channel_index"}),
            {"id_column": "channel_index"},
            "collide",
        ),
    ],
)
def test_input_rejection(change, options, match):
    with pytest.raises(ValueError, match=match):
        prepare_forecast_frame(change(_frame()), **({"freq": "D"} | options))


@pytest.mark.parametrize(
    "change,match",
    [
        (lambda s: s["items"].pop(), "count"),
        (lambda s: s["items"][1].pop("horizon_steps"), "item\\[1\\]"),
        (lambda s: s["items"][1].update(horizon_steps=[1, 1]), "horizon_steps"),
        (lambda s: s["items"][1].update(channel_names=["a", "b"]), "channel"),
        (lambda s: s["items"][1].update(shape=[3, 1]), "item\\[1\\]"),
    ],
)
def test_output_rejection(change, match):
    summary = _summary()
    change(summary)
    with pytest.raises(ValueError, match=match):
        format_forecast_frame(prepare_forecast_frame(_frame(), freq="D"), summary)


def test_heterogeneous_quantile_levels_and_no_point_invention():
    summary = _summary()
    for index, part in enumerate(summary["items"]):
        part.update(
            shape=[1, 2, 1],
            axes=["quantile", "horizon", "channel"],
            quantile_levels=[0.1 if index == 0 else 0.9],
        )
    result = format_forecast_frame(prepare_forecast_frame(_frame(), freq="D"), summary)
    assert "forecast" not in result
    assert result["forecast-q-0.1"].isna().tolist() == [False, False, True, True]
    assert result["forecast-q-0.9"].isna().tolist() == [True, True, False, False]
    assert result.attrs["quantile_levels_by_series"] == (("b", (0.1,)), ("a", (0.9,)))


def test_endpoint_quantiles_follow_the_native_contract():
    summary = _summary()
    for part in summary["items"]:
        part.update(
            shape=[2, 2, 1],
            axes=["quantile", "horizon", "channel"],
            quantile_levels=[0.0, 1.0],
            values=[0.0, 1.0, 9.0, 10.0],
        )
    result = format_forecast_frame(prepare_forecast_frame(_frame(), freq="D"), summary)
    assert result["forecast-q-0.0"].tolist() == [0.0, 1.0, 0.0, 1.0]
    assert result["forecast-q-1.0"].tolist() == [9.0, 10.0, 9.0, 10.0]


@pytest.mark.parametrize("level", [-0.1, 1.1, float("nan")])
def test_invalid_quantile_levels_are_not_rendered(level):
    summary = _summary()
    for part in summary["items"]:
        part.update(
            shape=[1, 2, 1], axes=["quantile", "horizon", "channel"], quantile_levels=[level]
        )
    with pytest.raises(ValueError, match="quantile levels"):
        format_forecast_frame(prepare_forecast_frame(_frame(), freq="D"), summary)


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -float("inf")])
def test_nonfinite_predictions_are_not_missing_quantile_cells(value):
    summary = _summary()
    summary["items"][1]["values"][0] = value
    with pytest.raises(ValueError, match="predictions must be finite"):
        format_forecast_frame(prepare_forecast_frame(_frame(), freq="D"), summary)


def test_mixed_numeric_ids_retain_original_identity():
    frame = _frame()
    frame["unique_id"] = pd.Series([9007199254740993, 9007199254740993, 2.5, 2.5], dtype=object)
    prepared = prepare_forecast_frame(
        frame, freq="D", config_by_series={9007199254740993: {"frequency": 2}}
    )
    assert prepared.series_ids == (9007199254740993, 2.5)
    assert isinstance(prepared.series_ids[0], int)
    assert prepared.request["items"][0]["config"] == {"frequency": 2}
    result = format_forecast_frame(prepared, _summary())
    assert result["unique_id"].tolist() == [9007199254740993, 9007199254740993, 2.5, 2.5]
    assert isinstance(result["unique_id"].iloc[0], int)


@pytest.mark.parametrize("object_dtype", [False, True])
def test_complex_values_are_not_silently_projected_to_real(object_dtype):
    frame = _frame()
    frame["values"] = pd.Series([1 + 2j] * 4, dtype=object if object_dtype else complex)
    with pytest.raises(ValueError, match="complex observations"):
        prepare_forecast_frame(frame, freq="D")


@pytest.mark.parametrize(
    "change,match",
    [
        (lambda i: i.update(shape=[3, 1]), "shape"),
        (lambda i: i.update(shape=[2.0, 1]), "shape"),
        (lambda i: i.update(observed_mask=[1]), "mask"),
        (lambda i: i.update(past_values=[None, 2]), "null"),
        (lambda i: i.update(past_values=[True, 2]), "numbers"),
        (lambda i: i.update(past_values=[1e100, 2]), "finite"),
        (lambda i: i.update(config={"not_declared": 0}), "not_declared"),
        (lambda i: i.update(config={"frequency": 0.25}), "int64"),
        (lambda i: i.update(config={"frequency": 3}), "frequency"),
    ],
)
def test_worker_later_item_errors(tmp_path, worker, change, match):
    prepared = prepare_forecast_frame(_frame(), freq="D")
    change(prepared.request["items"][1])
    with pytest.raises(BenchmarkError, match=match):
        run_worker(
            _case(tmp_path, "batch_series_to_point_forecast", prepared.request), tmp_path, worker
        )
    failure = json.loads((tmp_path / "worker-result.json").read_text())
    assert "item[1]" in failure["error"]
    assert "observations" not in failure and "output_summary" not in failure


def test_scalar_primary_rejects_batch_without_fallback(tmp_path, worker):
    prepared = prepare_forecast_frame(_frame().iloc[:2], freq="D")
    with pytest.raises(BenchmarkError, match="requires a native batch Task"):
        run_worker(_case(tmp_path, "series_to_point_forecast", prepared.request), tmp_path, worker)


def test_manifest_batch_global_config_is_not_ignored(tmp_path):
    prepared = prepare_forecast_frame(_frame(), freq="D")
    with pytest.raises(BenchmarkError, match="belongs to each"):
        resolve_task_case(
            "batch_series_to_point_forecast",
            {"inputs": prepared.request, "config": {"frequency": 0}},
            tmp_path,
        )


def test_package_and_helper_import_without_pandas():
    code = """
import sys
class NoPandas:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'pandas' or fullname.startswith('pandas.'):
            raise ModuleNotFoundError('pandas deliberately unavailable')
sys.meta_path.insert(0, NoPandas())
import trtmc_benchmark
from trtmc_benchmark.dataframe import prepare_forecast_frame
assert 'pandas' not in sys.modules
try:
    prepare_forecast_frame(None, freq='D')
except ImportError as error:
    assert 'pip install pandas' in str(error)
else:
    raise AssertionError('missing optional dependency was ignored')
"""
    subprocess.run([sys.executable, "-c", code], check=True)

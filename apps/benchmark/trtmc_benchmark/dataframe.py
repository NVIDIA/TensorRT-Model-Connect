# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure table conversions for the native batch forecast worker.

Calendar frequency only locates timestamps. Model frequency categories, missing
value treatment, and native batching remain the selected family's responsibility.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


def _pandas():
    try:
        import pandas as pd
    except ImportError as error:
        raise ImportError("DataFrame helpers require pandas: pip install pandas") from error
    return pd


@dataclass(frozen=True)
class PreparedForecastFrame:
    """Owned request data and its table index; contains no model/runtime state."""

    series_ids: tuple[Hashable, ...]
    last_times: tuple[Any, ...]
    offsets: tuple[Any, ...]
    value_columns: tuple[str, ...]
    id_column: str
    time_column: str
    request: dict[str, Any]


def prepare_forecast_frame(
    frame: pd.DataFrame,
    *,
    freq: str | Mapping[Hashable, str],
    value_columns: Sequence[str] = ("values",),
    id_column: str = "unique_id",
    time_column: str = "ds",
    observed_columns: Sequence[str] | None = None,
    context_length: int | None = None,
    config_by_series: Mapping[Hashable, Mapping[str, object]] | None = None,
) -> PreparedForecastFrame:
    """Group in first-seen ID order, sorting time within each contiguous series.

    The returned ``request`` is the existing worker's batch ``solve`` payload.
    Null values encode unobserved NaNs, not zero-filled observations. No model
    call, imputation, resampling, context-limit lookup, or Config default occurs.
    """
    pd = _pandas()
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("frame must be a nonempty DataFrame")
    if not frame.columns.is_unique:
        raise ValueError("frame column names must be unique")
    columns = tuple(value_columns)
    if isinstance(value_columns, str) or not columns or len(set(columns)) != len(columns):
        raise ValueError("value_columns must be a nonempty sequence of distinct names")
    mask_columns = None if observed_columns is None else tuple(observed_columns)
    if mask_columns is not None and (
        isinstance(observed_columns, str)
        or len(mask_columns) != len(columns)
        or len(set(mask_columns)) != len(mask_columns)
    ):
        raise ValueError("observed_columns must have one distinct column per value column")
    required = (id_column, time_column, *columns, *(mask_columns or ()))
    if id_column == time_column or id_column in columns or time_column in columns:
        raise ValueError("ID, time and value columns must have distinct roles")
    if {id_column, time_column} & {"horizon_step", "channel_index", "channel_name", "channel_unit"}:
        raise ValueError("ID/time column names collide with forecast index columns")
    for name in required:
        if name not in frame.columns:
            raise ValueError(f"missing frame column {name!r}")
    if frame[id_column].isna().any():
        raise ValueError("series IDs must not be missing")
    if context_length is not None and (
        isinstance(context_length, bool)
        or not isinstance(context_length, int)
        or context_length <= 0
    ):
        raise ValueError("context_length must be a positive integer")
    if not isinstance(freq, (str, Mapping)):
        raise ValueError("freq must be an explicit calendar frequency or per-series mapping")
    if config_by_series is not None and not isinstance(config_by_series, Mapping):
        raise ValueError("config_by_series must be a mapping")

    ids, last_times, offsets, items = [], [], [], []
    for _, group in frame.groupby(id_column, sort=False, observed=True):
        # GroupBy's result index may coerce mixed numeric IDs to float64.
        # Keep the original ID object, including integers beyond float precision.
        key = group[id_column].iloc[0]
        try:
            if not all(
                isinstance(value, (datetime, np.datetime64)) for value in group[time_column]
            ):
                raise ValueError("timestamps must be datetimes, not numbers or date strings")
            times = pd.DatetimeIndex(group[time_column])
            if times.hasnans or times.has_duplicates:
                raise ValueError("timestamps must be nonmissing and unique within a series")
            order = np.argsort(times.asi8, kind="stable")
            group = group.iloc[order]
            times = times.take(order)
            offset = pd.tseries.frequencies.to_offset(
                freq[key] if isinstance(freq, Mapping) else freq
            )
            if offset is None or offset.n <= 0:
                raise ValueError("freq must be a positive calendar offset")
            expected = pd.date_range(times[0], periods=len(times), freq=offset)
            if not times.equals(expected):
                raise ValueError("timestamps are not on the supplied contiguous frequency grid")
            if context_length is not None:
                group = group.tail(context_length)
            raw_values = group.loc[:, list(columns)].to_numpy()
            if np.iscomplexobj(raw_values) or (
                raw_values.dtype.kind == "O"
                and any(np.iscomplexobj(value) for value in raw_values.flat)
            ):
                raise ValueError("complex observations cannot be represented as float32 history")
            with np.errstate(over="ignore", invalid="ignore"):
                values = group.loc[:, list(columns)].to_numpy(
                    dtype=np.float32, na_value=np.nan, copy=True
                )
            if np.isinf(values).any():
                raise ValueError("values must be float32-representable finite numbers or missing")
            if mask_columns is None:
                observed = ~np.isnan(values)
            else:
                raw = group.loc[:, list(mask_columns)].to_numpy()
                if not pd.DataFrame(raw).isin([0, 1, False, True]).all().all():
                    raise ValueError("observed columns must contain only boolean/zero/one values")
                observed = raw.astype(bool)
                if (observed & np.isnan(values)).any():
                    raise ValueError("a missing value cannot be marked observed")
            config = {} if config_by_series is None else config_by_series.get(key, {})
            if not isinstance(config, Mapping):
                raise ValueError("per-series config must be an object")
            items.append(
                {
                    "past_values": [
                        None if np.isnan(value) else float(value) for value in values.flat
                    ],
                    "shape": list(values.shape),
                    "observed_mask": observed.astype(np.uint8).ravel().tolist(),
                    "config": deepcopy(dict(config)),
                }
            )
            ids.append(key)
            last_times.append(times[-1])
            offsets.append(offset)
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            raise ValueError(f"series {key!r}: {error}") from error
    for name, mapping in (("freq", freq), ("config_by_series", config_by_series)):
        if isinstance(mapping, Mapping) and any(key not in ids for key in mapping):
            raise ValueError(f"{name} contains an unknown series ID")
    return PreparedForecastFrame(
        tuple(ids),
        tuple(last_times),
        tuple(offsets),
        columns,
        id_column,
        time_column,
        {"items": items},
    )


def _forecast_part(part: Mapping[str, Any], quantile: bool):
    shape = part["shape"]
    if len(shape) != (3 if quantile else 2) or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in shape
    ):
        raise ValueError("invalid forecast shape")
    if part["axes"] != (["quantile", "horizon", "channel"] if quantile else ["horizon", "channel"]):
        raise ValueError("invalid forecast axis roles")
    values = np.asarray(part["values"], dtype=np.float64).reshape(shape)
    if not np.isfinite(values).all():
        raise ValueError("forecast predictions must be finite, not missing")
    steps = part["horizon_steps"]
    if (
        len(steps) != shape[-2]
        or any(isinstance(step, bool) or not isinstance(step, int) or step <= 0 for step in steps)
        or any(a >= b for a, b in zip(steps, steps[1:]))
    ):
        raise ValueError("forecast requires actual increasing positive horizon_steps")
    names, units = part["channel_names"], part["channel_units"]
    if any(
        len(labels) not in (0, shape[-1]) or any(not isinstance(v, str) for v in labels)
        for labels in (names, units)
    ):
        raise ValueError("forecast channel metadata does not match its channel axis")
    levels = part["quantile_levels"] if quantile else []
    if quantile and (
        len(levels) != shape[0]
        or any(
            isinstance(q, bool) or not isinstance(q, (int, float)) or not 0 <= q <= 1
            for q in levels
        )
        or any(a >= b for a, b in zip(levels, levels[1:]))
    ):
        raise ValueError("forecast quantile levels must match the quantile axis")
    return values, steps, names, units, levels


def format_forecast_frame(
    prepared: PreparedForecastFrame,
    output_summary: Mapping[str, object],
    *,
    model_name: str = "forecast",
) -> pd.DataFrame:
    """Copy native batch results into a table without changing predictions.

    Output rows follow series, actual horizon-step, then channel order. Missing
    labels remain missing. Quantile-only results have no fabricated point column.
    Per-series actual quantile levels are also retained in DataFrame attrs.
    """
    pd = _pandas()
    items = output_summary.get("items")
    if not isinstance(items, list) or len(items) != len(prepared.series_ids):
        raise ValueError("forecast result count does not match the prepared series")
    reserved = {
        prepared.id_column,
        prepared.time_column,
        "horizon_step",
        "channel_index",
        "channel_name",
        "channel_unit",
    }
    if not isinstance(model_name, str) or not model_name or model_name in reserved:
        raise ValueError("model_name must not collide with index columns")
    rows, actual_levels = [], []
    for index, item in enumerate(items):
        try:
            if "point" in item or "quantiles" in item:
                point = _forecast_part(item["point"], False)
                quantiles = _forecast_part(item["quantiles"], True)
                if point[0].shape != quantiles[0].shape[1:] or point[1:4] != quantiles[1:4]:
                    raise ValueError("joint point and quantile axes differ")
            elif item["axes"] == ["horizon", "channel"]:
                point, quantiles = _forecast_part(item, False), None
            else:
                point, quantiles = None, _forecast_part(item, True)
            reference = point if point is not None else quantiles
            values, steps, names, units, _ = reference
            levels = [] if quantiles is None else quantiles[4]
            actual_levels.append(tuple(levels))
            columns = [f"{model_name}-q-{q}" for q in levels]
            if any(column in reserved for column in columns):
                raise ValueError("quantile column name collides with index columns")
            for h, step in enumerate(steps):
                time = pd.date_range(
                    prepared.last_times[index], periods=2, freq=prepared.offsets[index] * step
                )[-1]
                for channel in range(values.shape[-1]):
                    row = {
                        prepared.id_column: prepared.series_ids[index],
                        prepared.time_column: time,
                        "horizon_step": step,
                        "channel_index": channel,
                        "channel_name": names[channel] if names else None,
                        "channel_unit": units[channel] if units else None,
                    }
                    if point is not None:
                        row[model_name] = float(point[0][h, channel])
                    if quantiles is not None:
                        for q, column in enumerate(columns):
                            row[column] = float(quantiles[0][q, h, channel])
                    rows.append(row)
        except (ValueError, TypeError, KeyError, IndexError, OverflowError) as error:
            raise ValueError(f"result item[{index}]: {error}") from error
    result = pd.DataFrame(rows)
    result[prepared.id_column] = pd.Series([row[prepared.id_column] for row in rows], dtype=object)
    result.attrs["quantile_levels_by_series"] = tuple(zip(prepared.series_ids, actual_levels))
    return result

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Server-reported model-call time as an AIPerf record metric."""

from __future__ import annotations

from aiperf.common.enums import MetricFlags, MetricTimeUnit
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import ParsedResponseRecord
from aiperf.metrics import BaseRecordMetric
from aiperf.metrics.metric_dicts import MetricRecordDict


class TrtmcModelCallTimeMetric(BaseRecordMetric[float]):
    """Backend model-call wall time reported in ``trtmc_timing.model_call_ms``."""

    tag = "trtmc_model_call_time"
    header = "TRTMC Model Call Time"
    short_header = "Model Call"
    unit = MetricTimeUnit.MILLISECONDS
    display_order = 305
    flags = MetricFlags.NONE

    def _parse_record(self, record: ParsedResponseRecord, record_metrics: MetricRecordDict) -> float:
        for response in record.responses:
            value = getattr(response.data, "model_call_ms", None)
            if value is not None:
                return float(value)
        raise NoMetricValue("trtmc_timing.model_call_ms not present in the response")

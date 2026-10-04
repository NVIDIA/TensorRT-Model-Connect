# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generic TRTMC task endpoint: JSON operation request in, JSON observation out."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from aiperf.common.models import InferenceServerResponse, ParsedResponse, RequestInfo
from aiperf.common.models.record_models import TextResponseData
from aiperf.endpoints.base_endpoint import BaseEndpoint


@dataclass(slots=True)
class TrtmcTaskResponseData(TextResponseData):
    """Observation as canonical JSON text plus the server-reported model-call time."""

    model_call_ms: float | None = None


# Importing the metric module registers it (aiperf registers metrics on subclass creation).
from trtmc_aiperf_plugins import metrics as _metrics  # noqa: E402,F401


class TrtmcTaskEndpoint(BaseEndpoint):
    def format_payload(self, request_info: RequestInfo) -> dict[str, Any]:
        turn = request_info.turns[-1]
        if not turn.texts or not turn.texts[0].contents:
            raise ValueError("trtmc_task requires the turn text to be a JSON operation request")
        payload = json.loads(turn.texts[0].contents[0])
        if not isinstance(payload, dict) or "request" not in payload:
            raise ValueError('trtmc_task turn text must be {"request": {...}}')
        return {"request": payload["request"]}

    def parse_response(self, response: InferenceServerResponse) -> ParsedResponse | None:
        body = response.get_json()
        if not body or "trtmc_observation" not in body:
            return None
        observation = body["trtmc_observation"]
        timing = body.get("trtmc_timing") or {}
        data = TrtmcTaskResponseData(text=json.dumps(observation, sort_keys=True),
                                     model_call_ms=timing.get("model_call_ms"))
        return ParsedResponse(perf_ns=response.perf_ns, data=data)

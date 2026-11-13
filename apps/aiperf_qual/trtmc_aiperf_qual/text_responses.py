# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify the supported stateless Responses wire subset against native records."""
from __future__ import annotations

import hashlib
import json

from .config import ConfigError


def prompt_text(payload):
    if not isinstance(payload, dict):
        return None
    if isinstance(payload.get("prompt"), str):
        return payload["prompt"]
    value = payload.get("input")
    if isinstance(value, str):
        return value
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], dict) or value[0].get("role") != "user":
        return None
    content = value[0].get("content")
    if isinstance(content, str):
        return content
    if (not isinstance(content, list) or not content or any(not isinstance(part, dict) or
            part.get("type") != "input_text" or not isinstance(part.get("text"), str) for part in content)):
        return None
    return "".join(part["text"] for part in content)


def verify(run, joined):
    path = run.directory / "responses-verification.json"
    try:
        result = verify_records(run, joined)
    except ConfigError as error:
        path.write_text(json.dumps({"verified": False, "error": str(error), "request_records": "joined.jsonl"}, indent=2) + "\n")
        raise
    path.write_text(json.dumps(result, indent=2) + "\n")


def verify_records(run, joined):
    from .text_profile import response_text
    native = {row["request_id"]: row["server"] for row in joined}
    requests, ids, response_ids = [], set(), set()
    for row in run.raw_records(phase=None):
        meta, payload = row.get("metadata", {}), row.get("payload")
        if not isinstance(meta, dict):
            raise ConfigError("Responses request metadata is not an object")
        request_id = meta.get("x_request_id")
        server = native.get(request_id)
        prompt, text = prompt_text(payload), response_text(row)
        try:
            body = json.loads(row["responses"][-1]["text"])
            message, = body["output"]
            part, = message["content"]
            valid = (row.get("status") == 200 and not row.get("error") and isinstance(prompt, str) and
                isinstance(text, str) and body["object"] == "response" and body["status"] == "completed" and
                body.get("error") is None and body.get("usage") is None and body["model"] == payload["model"] and
                isinstance(body["id"], str) and body["id"].startswith("resp_") and body["id"] not in response_ids and
                message["type"] == "message" and message["role"] == "assistant" and message["status"] == "completed" and
                isinstance(message["id"], str) and message["id"].startswith("msg_") and
                part["type"] == "output_text" and part["text"] == text and part["annotations"] == [] and
                server is not None and server.get("status") == 200 and server.get("timing_scope") == "public_task_call_wall" and
                server.get("prompt_sha256") == hashlib.sha256(prompt.encode()).hexdigest() and
                server.get("completion_sha256") == hashlib.sha256(text.encode()).hexdigest() and
                server.get("generation_config", {}).get("max_new_tokens") == payload["max_output_tokens"] and
                body["max_output_tokens"] == payload["max_output_tokens"] and
                type(server.get("completion_tokens")) is int and 0 <= server["completion_tokens"] <= payload["max_output_tokens"])
        except (KeyError, IndexError, TypeError, ValueError):
            valid = False
        if not valid or not isinstance(request_id, str) or request_id in ids:
            raise ConfigError("Responses wire object, input/output identity, native cap or Task timing is invalid")
        ids.add(request_id)
        response_ids.add(body["id"])
        requests.append({"request_id": request_id, "response_id": body["id"], "message_id": message["id"],
                         "phase": meta.get("benchmark_phase"), "completion_sha256": server["completion_sha256"]})
    if not requests:
        raise ConfigError("Responses export has no native completions")
    return {"verified": True, "requests": requests, "streaming": "unsupported; no events advertised",
            "usage": "unavailable; native input count is not provided; output count retained in server records"}

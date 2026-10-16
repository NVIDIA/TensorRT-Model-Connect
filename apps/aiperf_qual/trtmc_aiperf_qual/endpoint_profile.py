# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Profile an existing native bundle through the standard non-generation endpoints.

Reuse the persistent server, benchmark worker, published AIPerf runner, and
request/metric joins. Family correctness remains a separate prerequisite;
transport parity does not certify the model's numerical or task quality.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from .aiperf_runner import run_aiperf
from .config import ConfigError, Environment
from .endpoint_catalog import GROUP_TASKS, inventory
from .services import gpu_exclusive, text_serving
from .text_profile import check_aiperf, digest, git, join_records, statistics


ENDPOINTS = {
    "embeddings": ("embeddings", "/v1/embeddings"),
    "reranking": ("nim_rankings", "/v1/ranking"),
    "transcription": ("audio_transcription", "/v1/audio/transcriptions"),
    "detection": ("image_retrieval", "/v1/infer"),
}


def input_key(group: str, payload: dict[str, Any]) -> str:
    """Identify the effective wire input, including original media bytes."""
    if group == "transcription":
        value = base64.b64decode(payload["file"]["b64_data"], validate=True)
    elif group == "detection":
        value = base64.b64decode(payload["input"][0]["url"].split(",", 1)[1], validate=True)
    elif group == "embeddings":
        text = payload["input"]
        if isinstance(text, list):
            text, = text
        value = json.dumps(text, ensure_ascii=False).encode()
    else:
        value = json.dumps([payload["query"], payload["passages"]], sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(value).hexdigest()


def samples(group: str, inputs: Path, model: str, token_limit: int) -> list[dict[str, Any]]:
    result = []
    for line in inputs.read_text().split("\n"):
        if not line.strip():
            continue
        row = json.loads(line)
        if group == "embeddings":
            text, = row["texts"]
            payload = {"model": model, "input": text}
            native = {"prompt": text}
        elif group == "reranking":
            parts = {part["name"]: part["contents"] for part in row["texts"]}
            query, = parts["query"]
            payload = {"model": model, "query": {"text": query},
                       "passages": [{"text": text} for text in parts["passages"]]}
            native = {"query": query, "documents": parts["passages"]}
        elif group == "transcription":
            path = Path(row["audio"])
            payload = {"model": model, "max_output_tokens": token_limit,
                       "file": {"b64_data": base64.b64encode(path.read_bytes()).decode()}}
            native = {"audio_path": str(path), "max_new_tokens": token_limit, "streaming": False}
        else:
            payload = {"input": [{"type": "image_url", "url": row["image"]}], "model": model}
            native = {}  # The caller writes exactly these embedded bytes for the native image decoder.
        result.append({"key": input_key(group, payload), "payload": payload, "request": native})
    if not result or len({row["key"] for row in result}) != len(result):
        raise ValueError("the endpoint corpus must contain distinct, nonempty inputs")
    return result


def post(url: str, payload: dict[str, Any], request_id: str, group: str) -> Any:
    if group == "transcription":
        boundary = uuid.uuid4().hex
        body = bytearray()
        for key, value in payload.items():
            if key != "file":
                body.extend(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode())
        body.extend(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="clip.wav"\r\nContent-Type: audio/wav\r\n\r\n'.encode())
        body.extend(base64.b64decode(payload["file"]["b64_data"], validate=True))
        body.extend(f'\r\n--{boundary}--\r\n'.encode())
        content_type = f"multipart/form-data; boundary={boundary}"
    else:
        body = json.dumps(payload).encode()
        content_type = "application/json"
    request = urllib.request.Request(url, data=bytes(body), headers={
        "Content-Type": content_type, "X-Request-ID": request_id})
    with urllib.request.urlopen(request, timeout=900) as response:
        return json.load(response)


def check_output(group: str, actual: Any, native: dict[str, Any], dimensions=None) -> int:
    """Preserve the complete public Task output, not a summary or a success code."""
    if group == "transcription":
        if actual["text"] != native["text"]:
            raise ValueError("HTTP transcript differs from the complete native transcript")
        return 1
    if group == "embeddings":
        actual = actual["data"][0]["embedding"]
        expected = native["values"]
        if len(actual) != len(expected) or not expected:
            raise ValueError("HTTP vector dimensions differ from native output")
        pairs = zip(actual, expected, strict=True)
    elif group == "reranking":
        rankings = actual["rankings"]
        scores = native["scores"]
        indices = [row["index"] for row in rankings]
        if indices != sorted(range(len(scores)), key=lambda i: -scores[i]):
            raise ValueError("HTTP ranking lost or reordered native passage indices")
        pairs = ((row["logit"], scores[row["index"]]) for row in rankings)
    else:
        width, height = dimensions
        boxes = native["boxes"]
        boxes = boxes if not boxes or isinstance(boxes[0], list) else [boxes[i:i + 4] for i in range(0, len(boxes), 4)]
        expected: dict[str, list[dict[str, float]]] = {}
        for box, score, label in zip(boxes, native["scores"], native["class_ids"], strict=True):
            expected.setdefault(str(label), []).append(dict(zip(
                ("x_min", "y_min", "x_max", "y_max", "confidence"),
                (box[0] / width, box[1] / height, box[2] / width, box[3] / height, score), strict=True)))
        data, = actual["data"]
        actual = data["bounding_boxes"]
        if data["index"] != 0 or actual.keys() != expected.keys():
            raise ValueError("HTTP detections lost native class IDs")
        values = []
        for label, want in expected.items():
            if len(actual[label]) != len(want):
                raise ValueError("HTTP detections lost native boxes")
            for row, reference in zip(actual[label], want, strict=True):
                if row.keys() != reference.keys():
                    raise ValueError("HTTP detection has incomplete coordinates or confidence")
                values.extend((row[key], reference[key]) for key in reference)
        pairs = iter(values)
    count = 0
    for value, expected in pairs:
        if not math.isfinite(value) or not math.isfinite(expected) or abs(value - expected) > 1e-6:
            raise ValueError("HTTP values differ from the complete native output")
        count += 1
    return count


def profile(environment: Environment, name: str, bundle: Path, inputs: Path, out: Path,
            *, token_limit: int = 120) -> dict[str, Any]:
    """One serial model run: native verification, then 5 warmups and 50 measured requests."""
    from trtmc_benchmark.catalog import ManifestCatalog
    from trtmc_perf_serving.backends.trtmc import TrtmcWorkerBackend

    repo = environment.path("repo")
    entry = next((e for e in ManifestCatalog(repo / "families").entries() if e.name == name), None)
    group = next((g for g, tasks in GROUP_TASKS.items()
                  if entry and entry.model and entry.model.task in tasks), None)
    if group is None:
        raise ConfigError(f"{name!r} is not a native endpoint catalog profile")
    if not any(row["profile"] == name for row in inventory(repo, group)):
        raise ConfigError("select the group's normal full-model checkpoint profile")
    out.mkdir(parents=True, exist_ok=False)
    client = str(environment["aiperf_python"])
    client_info = check_aiperf(environment, {})
    runtime = environment.path("runtime_root")
    provenance = {"source_head": git(repo, "rev-parse", "HEAD"),
                  "source_status": git(repo, "status", "--porcelain"), "profile": name,
                  "manifest": json.loads(entry.model.manifest_path.read_text()),
                  "bundle_sha256": digest(bundle), "inputs_sha256": digest(inputs), "aiperf": client_info}
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    corpus = samples(group, inputs, entry.hf_id, token_limit)
    expected = {}
    with gpu_exclusive(environment):
        native = TrtmcWorkerBackend(worker=runtime / "trtmc_benchmark_worker", session={
            "schema_version": 2, "operation": entry.operation, "bundle": str(bundle),
            "runtime_root": str(runtime), "request": {}, "expected_family": entry.family,
            "expected_task": entry.model.task}, scratch=out / "native", full_observations=True)
        try:
            for row in corpus:
                dimensions = None
                if group == "detection":
                    from PIL import Image

                    raw = base64.b64decode(row["payload"]["input"][0]["url"].split(",", 1)[1], validate=True)
                    path = out / (row["key"] + ".image")
                    path.write_bytes(raw)
                    with Image.open(path) as image:
                        dimensions = image.size
                    row["request"] = {"image_path": str(path)}
                observed = native.invoke(row["request"], out / row["key"]).observation
                expected[row["key"]] = {"observation": observed, "dimensions": dimensions}
        finally:
            native.close()
        (out / "native-outputs.json").write_text(json.dumps(expected, indent=2) + "\n")
        settings = {"bundle": bundle, "model_name": entry.hf_id, "binary": runtime / "trtmc-server",
                    "runtime_root": runtime, "port": 18078, "startup_timeout": 900, "request_timeout": 900}
        with text_serving(environment, settings, out / "server") as server:
            endpoint_type, endpoint = ENDPOINTS[group]
            for index, row in enumerate(corpus):
                observed = expected[row["key"]]
                response = post(server["url"] + endpoint, row["payload"], f"verify-{index}", group)
                check_output(group, response, observed["observation"], observed["dimensions"])
            arguments = ["--url", server["url"], "--endpoint-type", endpoint_type, "--endpoint", endpoint,
                         "--input-file", str(inputs), "--custom-dataset-type", "single_turn", "--concurrency", "1",
                         "--warmup-request-count", "5", "--warmup-concurrency", "1", "--request-count", "50"]
            if group == "transcription":
                arguments += ["--extra-inputs", f"max_output_tokens:{token_limit}"]
            run = run_aiperf(environment, out / "client", arguments, model_name=entry.hf_id,
                             command_prefix=[client, "-m", "aiperf"], timeout_s=1800)
            if run.exit_code:
                raise RuntimeError(f"AIPerf failed with exit code {run.exit_code}")
            records = [json.loads(line) for line in server["records"].read_text().split("\n")
                       if line.strip()]
            joined = join_records(run, records)
        (out / "joined-records.json").write_text(json.dumps(joined, indent=2) + "\n")
        if len(joined) != 55 or not all(r["joined"] and r["metrics_joined"] and not r["client_error"] and
                                      r["client_status"] == 200 and r["server"]["status"] == 200 for r in joined):
            raise RuntimeError("client/server/metric records are incomplete or contain errors")
        checked = 0
        for record in run.raw_records(phase=None):
            key = input_key(group, record["payload"])
            if key not in expected:
                raise ConfigError("client input not in verified corpus")
            observed = expected[key]
            response = json.loads(record["responses"][-1]["text"])
            checked += check_output(group, response, observed["observation"], observed["dimensions"])
        measured = [r for r in joined if r["phase"] == "profiling"]
        if len(measured) != 50 or sum(r["phase"] == "warmup" for r in joined) != 5:
            raise RuntimeError("warmup and measurement counts differ from the workload")
        elapsed = (max(r["client_metadata"]["request_end_ns"] for r in measured) -
                   min(r["client_metadata"]["request_start_ns"] for r in measured)) / 1e9
        summary = {"profile": name, "group": group, "transport_verified": True,
                   "family_correctness": "separate prerequisite; see owning-family evidence",
                   "successful_requests": 50, "warmup_requests": 5, "complete_responses": 55,
                   "verification_inputs": len(corpus), "output_values_checked": checked,
                   "client_errors": 0, "unmatched_requests": 0, "missing_metric_exports": 0,
                   "http_latency_ms": statistics([r["client_metrics"]["request_latency"]["value"] for r in measured]),
                   "task_call_ms": statistics([r["server"]["model_call_ms"] for r in measured]),
                   "requests_per_second": 50 / elapsed, "bundle_sha256": provenance["bundle_sha256"]}
        if group == "transcription":
            seconds = sum(r["server"]["audio_duration_seconds"] for r in measured)
            summary.update(measured_audio_seconds=seconds, audio_seconds_per_second=seconds / elapsed)
        if group == "reranking":
            summary["passages_per_second"] = sum(r["server"]["input_items"] for r in measured) / elapsed
        if group == "detection":
            summary["images_per_second"] = summary["requests_per_second"]
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Published AIPerf image endpoint, with complete native PNG identity per request.

Owning-family quality on the exact bundle remains a prerequisite. This command
checks media transport and measures fixed work; it does not replace that gate.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import subprocess
from pathlib import Path
from typing import Any

from .aiperf_runner import run_aiperf
from .config import ConfigError, Environment
from .endpoint_catalog import inventory
from .endpoint_profile import source_identity
from .services import gpu_exclusive, text_serving
from .text_profile import check_aiperf, digest, join_records, statistics


def bundle_identity(runtime: Path, bundle: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Use the public inspector to bind a claimed profile to the actual artifact."""
    info = json.loads(subprocess.check_output([str(runtime / "trtmc"), "inspect", str(bundle)], text=True))
    if info.get("family") != manifest["family"] or info.get("task") != manifest["task"]:
        raise ConfigError("bundle family/Task differs from the selected image profile")
    provenance = info.get("provenance") or {}
    checkpoint, build = provenance.get("checkpoint", {}), provenance.get("build", {})
    if (checkpoint.get("hf_id") != manifest["hf_id"] or
            checkpoint.get("revision") != manifest["hf_revision"]):
        raise ConfigError("bundle checkpoint/revision differs from the selected image profile")
    if build.get("quantization") != manifest.get("quantization"):
        raise ConfigError("bundle quantization differs from the selected image profile")
    for key in ("precision", "quantization", "image_height", "image_width", "tensor_parallel_size",
                "context_parallel_size", "fp32_layers"):
        if key in manifest and build.get(key) != manifest[key]:
            raise ConfigError(f"bundle {key} differs from the selected image profile")
    return info


def check_image(response: dict[str, Any], record: dict[str, Any], output: Path) -> int:
    """Join the decoded client image to the PNG written by that native invocation."""
    from PIL import Image

    item, = response["data"]
    data = base64.b64decode(item["b64_json"], validate=True)
    if hashlib.sha256(data).hexdigest() != record["output_sha256"]:
        raise ValueError("HTTP PNG differs from the native artifact for this request")
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        if image.format != "PNG" or image.size != (record["output_width"], record["output_height"]):
            raise ValueError("HTTP PNG dimensions differ from the native result")
        if record["output_count"] != 1:
            raise ValueError("image workload requires one generated image")
        elements = image.width * image.height * len(image.getbands())
    output.write_bytes(data)
    return elements


def prepare_inputs(inputs: Path, destination: Path, base: dict[str, Any]) -> list[dict[str, Any]]:
    """Use catalog controls by default; retain each explicit AIPerf turn override."""
    from trtmc_perf_serving.openai import image_generation

    names = {"height", "width", "num_steps", "seed", "guidance_scale", "cfg_scale", "negative_prompt"}
    unknown = set(base) - names - {"prompt", "batch_size", "media_type"}
    if unknown:
        raise ConfigError(f"catalog image controls need an explicit adapter: {sorted(unknown)}")
    controls = {key: value for key, value in base.items() if key in names - {"height", "width", "num_steps"}
                and not (key in {"guidance_scale", "cfg_scale"} and value == -1)}
    if base.get("height") and base.get("width"):
        controls["size"] = f"{base['width']}x{base['height']}"
    if base.get("num_steps", -1) > 0:
        controls["num_inference_steps"] = base["num_steps"]
    rows = []
    effective = []
    for line in inputs.read_text().split("\n"):
        if not line.strip():
            continue
        row = json.loads(line)
        prompt = row.get("text")
        if prompt is None:
            prompt, = row["texts"]
        extra = {**controls, **row.get("extra", {})}
        image_generation({"prompt": prompt, **extra}, {})
        if ("size" not in extra or "num_inference_steps" not in extra or "seed" not in extra or
                "guidance_scale" not in extra):
            raise ConfigError("image workload must state dimensions, steps, seed and guidance")
        rows.append({"text": prompt, "extra": extra})
        effective.append({"prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "controls": extra})
    if not rows:
        raise ConfigError("image workload needs at least one prompt")
    destination.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return effective


def profile(environment: Environment, name: str, bundle: Path, inputs: Path, out: Path, *,
            requests: int = 5, warmup: int = 1) -> dict[str, Any]:
    from trtmc_benchmark.catalog import ManifestCatalog, resolve_case
    from .models import resolve_model

    if requests < 1 or warmup < 0:
        raise ConfigError("request count must be positive and warmup nonnegative")
    repo, runtime = environment.path("repo"), environment.path("runtime_root")
    rows = inventory(repo, "images")
    variants = [v for row in rows for v in row["variants"]
                if v["profile"] == name and v["status"] == "ready" and not v["partial"]]
    if len(variants) != 1:
        raise ConfigError("select an existing full single-process text-to-image profile")
    entry, = [e for e in ManifestCatalog(repo / "families").entries() if e.name == name]
    out.mkdir(parents=True, exist_ok=False)
    prepared = out / "inputs.jsonl"
    base = {**dict(resolve_case(entry.model, bundle).request),
            **resolve_model(name, environment).get("catalog_request", {})}
    controls = prepare_inputs(inputs, prepared, base)
    client = str(environment["aiperf_python"])
    worker = Path(environment.values.get("image_worker", runtime / "trtmc-server"))
    with gpu_exclusive(environment):
        source = source_identity(repo, out)
        manifest = json.loads(entry.model.manifest_path.read_text())
        actual_bundle = bundle_identity(runtime, bundle, manifest)
        provenance = {**source, "profile": name, "manifest": manifest, "actual_bundle": actual_bundle,
                      "bundle_sha256": digest(bundle), "inputs_sha256": digest(inputs),
                      "prepared_inputs_sha256": digest(prepared), "workloads": controls,
                      "aiperf": check_aiperf(environment, {}), "worker_sha256": digest(worker),
                      "family_dso_sha256": digest(runtime / f"libtrtmc_model_{entry.family}.so"),
                      "family_correctness": "separate prerequisite on this exact bundle",
                      "timing_scope": "public_task_call_wall", "transport": "lossless base64 PNG"}
        (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        settings = {"bundle": bundle, "model_name": entry.hf_id, "binary": worker,
                    "runtime_root": runtime, "port": int(environment.values.get("image_port", 18078)),
                    "startup_timeout": 1800, "request_timeout": 1800, "max_body_bytes": 16 * 1024 * 1024}
        with text_serving(environment, settings, out / "server") as server:
            run = run_aiperf(environment, out / "client", ["--url", server["url"],
                "--endpoint-type", "image_generation", "--input-file", str(prepared),
                "--custom-dataset-type", "single_turn", "--tokenizer", "builtin", "--concurrency", "1",
                "--warmup-request-count", str(warmup), "--warmup-concurrency", "1",
                "--request-count", str(requests)], model_name=entry.hf_id,
                command_prefix=[client, "-m", "aiperf"], timeout_s=1800 * (requests + warmup + 1))
            if run.exit_code:
                raise RuntimeError(f"AIPerf failed with exit code {run.exit_code}; raw evidence preserved")
            records = [json.loads(line) for line in server["records"].read_text().split("\n") if line.strip()]
            joined = join_records(run, records)
        (out / "joined-records.json").write_text(json.dumps(joined, indent=2) + "\n")
        if len(joined) != requests + warmup or not all(r["joined"] and r["metrics_joined"] and
            not r["client_error"] and r["client_status"] == 200 and r["server"]["status"] == 200 for r in joined):
            raise RuntimeError("image request/metric joins are incomplete or contain errors")
        by_id = {r["server"]["request_id"]: r for r in joined}
        images = out / "images"
        images.mkdir()
        checked, identities, seen = 0, [], set()
        for index, row in enumerate(run.raw_records(phase=None)):
            identifier = row["metadata"].get("x_request_id")
            if identifier not in by_id or identifier in seen:
                raise ValueError("raw image response has a duplicate or unknown request identity")
            seen.add(identifier)
            record = by_id[identifier]["server"]
            payload = row["payload"]
            if hashlib.sha256(payload["prompt"].encode()).hexdigest() != record["prompt_sha256"]:
                raise ValueError("image prompt differs between client and Task")
            response = json.loads(row["responses"][-1]["text"])
            checked += check_image(response, record, images / f"{index:04d}.png")
            identities.append({"request_id": identifier, "phase": by_id[identifier]["phase"],
                               "png_sha256": record["output_sha256"], "path": f"images/{index:04d}.png"})
        measured = [r for r in joined if r["phase"] == "profiling"]
        if len(seen) != requests + warmup or len(measured) != requests:
            raise RuntimeError("image response or phase counts differ from the workload")
        elapsed = (max(r["client_metadata"]["request_end_ns"] for r in measured) -
                   min(r["client_metadata"]["request_start_ns"] for r in measured)) / 1e9
        if source_identity(repo, out) != source or digest(bundle) != provenance["bundle_sha256"] or digest(inputs) != provenance["inputs_sha256"]:
            raise ConfigError("source, bundle or inputs changed during profiling; evidence preserved")
        summary = {"profile": name, "group": "images", "endpoint": "image_generation",
                   "family_correctness": provenance["family_correctness"], "transport_verified": True,
                   "successful_requests": requests, "warmup_requests": warmup, "client_errors": 0,
                   "complete_responses": len(seen), "output_values_checked": checked,
                   "http_latency_ms": statistics([r["client_metrics"]["request_latency"]["value"] for r in measured]),
                   "task_call_ms": statistics([r["server"]["model_call_ms"] for r in measured]),
                   "images_per_second": requests / elapsed, "bundle_sha256": provenance["bundle_sha256"]}
        (out / "images.json").write_text(json.dumps(identities, indent=2) + "\n")
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Published video endpoint qualification with actual complete clip downloads."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

from .aiperf_runner import run_aiperf
from .config import ConfigError, Environment
from .endpoint_catalog import inventory
from .endpoint_profile import source_identity
from .image_profile import bundle_identity
from .services import gpu_exclusive, text_serving
from .text_profile import check_aiperf, digest, join_records, statistics
from trtmc_perf_serving.openai import image_generation
from trtmc_perf_serving.video_media import decoded_identity


def prepare_inputs(inputs: Path, destination: Path, base: dict) -> list[dict]:
    controls = {key: base[key] for key in ("seed", "negative_prompt", "guidance_scale", "cfg_scale")
                if key in base and not (key in {"guidance_scale", "cfg_scale"} and base[key] == -1)}
    controls.update(size=f"{base['width']}x{base['height']}", num_frames=base["num_frames"],
                    num_inference_steps=base["num_steps"], fps=24)
    rows = []
    for line in inputs.read_text().split("\n"):
        if not line.strip():
            continue
        row = json.loads(line)
        if set(row) - {"text", "extra"} or not isinstance(row.get("text"), str) or not row["text"].strip():
            raise ConfigError("video inputs require prompt text and optional explicit controls")
        extra = {**controls, **row.get("extra", {})}
        if extra["num_frames"] != base["num_frames"] or extra["size"] != controls["size"]:
            raise ConfigError("video workload must preserve the built full clip geometry")
        image_generation({"prompt": row["text"], **extra}, {}, video=True)
        if set(extra) - {"size", "seed", "num_inference_steps", "guidance_scale", "cfg_scale",
                         "negative_prompt", "initial_latents_b64", "num_frames", "fps"}:
            raise ConfigError("unsupported video controls")
        rows.append({"text": row["text"], "extra": extra})
    if not rows:
        raise ConfigError("video workload requires prompts")
    destination.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return rows


def check_controls(payload: dict, record: dict, workloads: list[dict]):
    if not any(payload == {"model": payload["model"], "prompt": row["text"], **row["extra"]}
               for row in workloads):
        raise ValueError("client video request differs from the fixed workload")
    mapped = image_generation(payload, {}, video=True)
    prompt = mapped.pop("prompt")
    count = mapped.pop("num_frames")
    noise = mapped.pop("initial_latents_path", None)
    negative = mapped.pop("negative_prompt", "")
    if (mapped != record["generation_config"] or count != record["output_frames"] or
            payload["fps"] != record["fps"] or
            hashlib.sha256(prompt.encode()).hexdigest() != record["prompt_sha256"] or
            hashlib.sha256(negative.encode()).hexdigest() != record["negative_prompt_sha256"] or
            (hashlib.sha256(base64.b64decode(payload["initial_latents_b64"], validate=True)).hexdigest()
             if noise is not None else None) != record["initial_latents_sha256"]):
        raise ValueError("actual client video controls differ from the native request")


def profile(environment: Environment, name: str, bundle: Path, inputs: Path, out: Path, *,
            requests: int = 2, warmup: int = 1) -> dict:
    from trtmc_benchmark.catalog import ManifestCatalog, resolve_case

    if requests < 1 or warmup < 0:
        raise ConfigError("request count must be positive and warmup nonnegative")
    repo, runtime = environment.path("repo"), environment.path("runtime_root")
    variants = [v for row in inventory(repo, "videos") for v in row["variants"]
                if v["profile"] == name and v["status"] == "ready" and not v["partial"]]
    if len(variants) != 1:
        raise ConfigError("select an existing full single-process text-to-video profile")
    entry, = [e for e in ManifestCatalog(repo / "families").entries() if e.name == name]
    manifest = json.loads(entry.model.manifest_path.read_text())
    out.mkdir(parents=True, exist_ok=False)
    prepared = out / "inputs.jsonl"
    workloads = prepare_inputs(inputs, prepared, dict(resolve_case(entry.model, bundle).request))
    client = str(environment["aiperf_python"])
    worker = Path(environment.values.get("video_worker", runtime / "trtmc-server"))
    capture = out / "clips"
    capture.mkdir()
    with gpu_exclusive(environment):
        source = source_identity(repo, out)
        info = bundle_identity(runtime, bundle, manifest)
        if info["provenance"]["build"].get("video_num_frames") != manifest["video_num_frames"]:
            raise ConfigError("bundle frame count differs from the selected full profile")
        provenance = {**source, "profile": name, "manifest": manifest, "actual_bundle": info,
                      "bundle_sha256": digest(bundle), "inputs_sha256": digest(inputs),
                      "prepared_inputs_sha256": digest(prepared), "aiperf": check_aiperf(environment, {}),
                      "worker_sha256": digest(worker), "workloads": workloads,
                      "family_dso_sha256": digest(runtime / f"libtrtmc_model_{entry.family}.so"),
                      "family_correctness": "separate prerequisite on this exact bundle",
                      "timing_scope": "public_task_call_wall",
                      "client_scope": "submit, poll, download, and capture actual bytes; no timing subtraction"}
        (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        timeout = int(environment.values.get("video_request_timeout", 14400))
        settings = {"bundle": bundle, "model_name": entry.hf_id, "binary": worker,
                    "runtime_root": runtime, "port": int(environment.values.get("video_port", 18079)),
                    "startup_timeout": 1800, "request_timeout": timeout, "max_body_bytes": 16 * 1024 * 1024}
        with text_serving(environment, settings, out / "server") as server:
            run = run_aiperf(environment, out / "client", ["--url", server["url"],
                "--endpoint-type", "video_generation", "--transport", "trtmc_video_capture",
                "--download-video-content",
                "--input-file", str(prepared), "--custom-dataset-type", "single_turn",
                "--tokenizer", "builtin", "--concurrency", "1",
                "--request-count", str(requests),
                *(["--warmup-request-count", str(warmup), "--warmup-concurrency", "1"] if warmup else [])],
                env={"TRTMC_VIDEO_CAPTURE_DIR": str(capture.resolve())}, model_name=entry.hf_id,
                command_prefix=[client, "-m", "aiperf"], request_timeout_s=timeout,
                timeout_s=timeout * (requests + warmup + 1))
            if run.exit_code:
                raise RuntimeError("AIPerf failed; original client and server evidence preserved")
            records = [json.loads(line) for line in server["records"].read_text().split("\n") if line.strip()]
            joined = join_records(run, [r for r in records if r.get("record_kind") == "video_completion"])
        if len(joined) != requests + warmup or not all(r["joined"] and r["metrics_joined"] and
                not r["client_error"] and r["client_status"] == 200 and r["server"]["status"] == 200 for r in joined):
            raise ValueError("video request/metric/completion joins are incomplete or contain errors")
        by_id = {row["request_id"]: row for row in joined}
        seen, identities = set(), []
        for raw in run.raw_records(phase=None):
            identifier = raw["metadata"].get("x_request_id")
            if identifier not in by_id or identifier in seen:
                raise ValueError("duplicate or unknown video request identity")
            seen.add(identifier)
            record = by_id[identifier]["server"]
            response = json.loads(raw["responses"][-1]["text"])
            if response["id"] != record["job_id"] or response["status"] != "completed":
                raise ValueError("client video job differs from native completion")
            check_controls(raw["payload"], record, workloads)
            path = capture / f"{record['job_id']}.mp4"
            frames = out / f"frames-{record['job_id']}"
            if digest(path) != record["output_sha256"] or decoded_identity(path, record["output_width"],
                    record["output_height"], record["output_frames"], output=frames, fps=record["fps"]) != record["output_pixels_sha256"]:
                raise ValueError("downloaded clip differs from the complete native frame sequence")
            identities.append({"request_id": identifier, "job_id": record["job_id"],
                               "clip": str(path.relative_to(out)), "frames": str(frames.relative_to(out)),
                               "encoded_sha256": record["output_sha256"],
                               "pixels_sha256": record["output_pixels_sha256"], "num_frames": record["output_frames"]})
        measured = [r for r in joined if r["phase"] == "profiling"]
        if len(seen) != requests + warmup or len(measured) != requests or len(list(capture.glob("*.mp4"))) != len(seen):
            raise ValueError("video phase or complete download counts differ from the workload")
        if source_identity(repo, out) != source or digest(bundle) != provenance["bundle_sha256"] or digest(inputs) != provenance["inputs_sha256"]:
            raise ConfigError("source, bundle or inputs changed during video profiling")
        summary = {"profile": name, "group": "videos", "endpoint": "video_generation",
                   "family_correctness": provenance["family_correctness"], "transport_verified": True,
                   "successful_requests": requests, "warmup_requests": warmup, "client_errors": 0,
                   "complete_responses": len(seen), "frames_checked": sum(r["num_frames"] for r in identities),
                   "http_latency_ms": statistics([r["client_metrics"]["request_latency"]["value"] for r in measured]),
                   "task_call_ms": statistics([r["server"]["model_call_ms"] for r in measured]),
                   "bundle_sha256": provenance["bundle_sha256"], "client_scope": provenance["client_scope"]}
        for file, value in (("joined-records.json", joined), ("clips.json", identities), ("summary.json", summary)):
            (out / file).write_text(json.dumps(value, indent=2) + "\n")
        return summary

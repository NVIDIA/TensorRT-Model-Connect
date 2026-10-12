# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Text load profiles through the existing server; independent of qualification gates."""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import platform
import re
import zipfile
import subprocess
from pathlib import Path
from typing import Any

from .aiperf_runner import AiperfRun, _run, run_aiperf
from .config import ConfigError, Environment, _load
from .services import build_env, gpu_exclusive, serving, text_serving

SHA = re.compile(r"^[0-9a-f]{40}$")
AIPERF_VERSION = "0.13.0"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def check_aiperf(environment: Environment, pin: dict[str, Any]) -> dict[str, Any]:
    if pin.get("version", AIPERF_VERSION) != AIPERF_VERSION:
        raise ConfigError(f"text profiles require the tested AIPerf {AIPERF_VERSION} release")
    source = None
    if "source" in pin or "commit" in pin:
        if not pin.get("source") or not isinstance(pin.get("commit"), str):
            raise ConfigError("an optional AIPerf source pin requires both source and commit")
        source = Path(pin["source"]).resolve()
        if not SHA.fullmatch(pin["commit"]) or git(source, "rev-parse", "HEAD") != pin["commit"]:
            raise ConfigError("AIPerf source must be checked out at the configured exact commit")
        if git(source, "status", "--porcelain", "--untracked-files=no"):
            raise ConfigError("AIPerf source has tracked modifications")
    # Query the client interpreter, which can be separate from the GPU runtime.
    # RECORD and metadata fingerprints identify the installed distribution;
    # direct_url also records editable/VCS installs when one was used.
    code = """import hashlib, importlib.metadata, json, aiperf
dist = importlib.metadata.distribution('aiperf')
def fingerprint(name):
    value = dist.read_text(name)
    return hashlib.sha256(value.encode()).hexdigest() if value is not None else None
print(json.dumps({
    'version': dist.version, 'module': aiperf.__file__,
    'distribution_root': str(dist.locate_file('')),
    'metadata_sha256': fingerprint('METADATA'), 'record_sha256': fingerprint('RECORD'),
    'installer': (dist.read_text('INSTALLER') or '').strip(),
    'direct_url': json.loads(dist.read_text('direct_url.json') or 'null'),
}))
"""
    installed = json.loads(subprocess.check_output([str(environment["aiperf_python"]), "-c", code], text=True))
    if installed["version"] != AIPERF_VERSION:
        raise ConfigError(f"aiperf_python must import AIPerf {AIPERF_VERSION}; "
                          f"install it with pip install aiperf=={AIPERF_VERSION}")
    if source is not None:
        if not Path(installed["module"]).resolve().is_relative_to(source):
            raise ConfigError("aiperf_python must import AIPerf from the configured source checkout")
        installed.update(source=str(source), commit=pin["commit"])
    return installed


def cases(config: dict[str, Any]) -> list[dict[str, Any]]:
    workload = config["workload"]
    axes = {"endpoint": workload.get("endpoints", ["completions", "chat"]),
            "input_tokens": workload.get("input_tokens", [32]),
            "output_tokens": workload.get("output_tokens", [16]),
            "concurrency": workload.get("concurrency", [1]),
            "request_rate": workload.get("request_rate", [None]),
            "streaming": workload.get("streaming", [False])}
    result = [dict(zip(axes, values)) for values in itertools.product(*axes.values())]
    if not result:
        raise ConfigError("text workload axes must be nonempty")
    for item in result:
        if item["endpoint"] not in {"completions", "chat"} or type(item["streaming"]) is not bool:
            raise ConfigError("text profiles support completions/chat and boolean streaming only")
        for name in ("input_tokens", "output_tokens", "concurrency"):
            if type(item[name]) is not int or item[name] <= 0:
                raise ConfigError(f"{name} must be a positive integer")
        rate = item["request_rate"]
        if rate is not None and (isinstance(rate, bool) or not isinstance(rate, (int, float))
                                 or not math.isfinite(rate) or rate <= 0):
            raise ConfigError("request_rate must be positive or null")
    for name, default in (("requests", 20), ("warmup_requests", 3)):
        value = workload.get(name, default)
        if type(value) is not int or value < (1 if name == "requests" else 0):
            raise ConfigError(f"{name} has an invalid request count")
    return result


def arguments(url: str, config: dict[str, Any], case: dict[str, Any]) -> list[str]:
    workload = config["workload"]
    tokenizer = config["tokenizer"]
    result = ["--url", url, "--endpoint-type", case["endpoint"],
              "--tokenizer", str(tokenizer["name"]), "--tokenizer-revision", tokenizer["revision"],
              "--synthetic-input-tokens-mean", str(case["input_tokens"]),
              "--synthetic-input-tokens-stddev", "0", "--output-tokens-mean", str(case["output_tokens"]),
              "--output-tokens-stddev", "0", "--concurrency", str(case["concurrency"]),
              "--request-count", str(workload.get("requests", 20)),
              "--warmup-request-count", str(workload.get("warmup_requests", 3)),
              "--warmup-concurrency", "1", "--extra-inputs", "temperature:0", "seed:0", "top_k:1",
              ]
    if case["endpoint"] == "chat":
        result.append("enable_thinking:false")
    if case["request_rate"] is not None:
        result += ["--request-rate", str(case["request_rate"])]
    if case["streaming"]:
        result.append("--streaming")
    # Never use --use-server-token-count: the server's prompt count is unknown.
    return result


def join_records(run: AiperfRun, server_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in server_records:
        key = record["request_id"]
        if key in indexed:
            raise ConfigError(f"duplicate server request ID: {key}")
        indexed[key] = record
    def metric_key(metadata: dict[str, Any]) -> tuple:
        return (metadata.get("benchmark_phase"), metadata.get("session_num"), metadata.get("x_request_id"))
    metric_records = {metric_key(record["metadata"]): record for record in run.metric_records(phase=None)}
    result = []
    for record in run.raw_records(phase=None):
        metadata = record["metadata"]
        key = metadata.get("x_request_id")
        server = indexed.get(key)
        metric_record = metric_records.get(metric_key(metadata))
        result.append({"request_id": key, "phase": metadata.get("benchmark_phase"),
            "client_error": record.get("error") or (metric_record or {}).get("error"),
            "client_status": record.get("status"),
            "client_metrics": (metric_record or record).get("metrics", {}),
            "metrics_joined": metric_record is not None or "metrics" in record,
            "client_metadata": metadata, "server": server,
            "joined": server is not None})
    return result


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lo = math.floor(index)
    hi = math.ceil(index)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def statistics(values: list[float]) -> dict[str, Any]:
    return {"samples": len(values), **{f"p{int(p * 100)}": percentile(values, p)
                                      for p in (0.5, 0.9, 0.95, 0.99)}}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    profiling = [row for row in rows if row["phase"] == "profiling"]
    successes = [row for row in profiling if not row["client_error"] and row["client_status"] == 200 and row["server"]
                 and row["server"]["status"] == 200]
    def server_values(name: str) -> list[float]:
        return [row["server"][name] for row in successes if name in row["server"]]
    def client_values(name: str) -> list[float]:
        return [row["client_metrics"][name]["value"] for row in successes
                if name in row["client_metrics"]]
    starts = [row["client_metadata"].get("request_start_ns") for row in profiling]
    ends = [row["client_metadata"].get("request_end_ns") for row in profiling]
    elapsed = (max(ends) - min(starts)) / 1e9 if starts and all(starts) and all(ends) else None
    return {"profiling_requests": len(profiling), "successful_requests": len(successes),
        "unmatched_client_requests": sum(not row["joined"] for row in profiling),
        "status_counts": {str(status): sum(row["server"] is not None and row["server"]["status"] == status
            for row in profiling) for status in sorted({row["server"]["status"] for row in profiling if row["server"]})},
        "client_errors": sum(bool(row["client_error"]) for row in profiling),
        "missing_metric_exports": sum(not row["metrics_joined"] for row in successes),
        "successful_requests_per_second": len(successes) / elapsed if elapsed and elapsed > 0 else None,
        "actual_completion_tokens": statistics(server_values("completion_tokens")),
        "native_model_call_ms": statistics(server_values("model_call_ms")),
        "worker_roundtrip_ms": statistics(server_values("worker_roundtrip_ms")),
        "handler_ms": statistics(server_values("handler_ms")),
        "client_request_latency_ms": statistics(client_values("request_latency")),
        # AIPerf request_latency ends at the last content chunk. The worker
        # lifecycle also includes terminal SSE/usage delivery and cleanup.
        "client_request_lifecycle_ms": statistics([
            (row["client_metadata"]["request_end_ns"] - row["client_metadata"]["request_start_ns"]) / 1e6
            for row in successes if row["client_metadata"].get("request_start_ns")
            and row["client_metadata"].get("request_end_ns")]),
        "client_ttft_ms": statistics(client_values("time_to_first_token")),
        "client_itl_ms": statistics(client_values("inter_token_latency")),
        "client_input_tokens": statistics(client_values("input_sequence_length")),
        "client_output_tokens": statistics(client_values("output_sequence_length")),
        "client_output_tokens_per_second_estimate": sum(client_values("output_sequence_length")) / elapsed if elapsed and elapsed > 0 else None,
        "client_token_source": "client_tokenizer_estimate", "server_input_token_source": "unavailable"}


def profile(environment: Environment, config_path: Path, out: Path) -> dict[str, Any]:
    config = _load(config_path)
    matrix = cases(config)
    for name in ("tokenizer", "checkpoint"):
        if not SHA.fullmatch(config[name]["revision"]):
            raise ConfigError(f"{name}.revision must be an exact checkpoint commit")
    if config["tokenizer"]["revision"] != config["checkpoint"]["revision"]:
        raise ConfigError("the pilot tokenizer revision must match the checkpoint")
    commands = config.get("validation_commands")
    if not commands or any(not isinstance(command, list) or not command or
                           not all(isinstance(arg, str) for arg in command) for command in commands):
        raise ConfigError("validation_commands must contain the owning family's exact-bundle checks")
    if out.exists() and any(out.iterdir()):
        raise ConfigError("text profile output must be a new or empty directory")
    out.mkdir(parents=True, exist_ok=True)
    repo = environment.path("repo")
    pin = check_aiperf(environment, config["aiperf"])
    bundle = Path(config["server"]["bundle"])
    bundle_hash = digest(bundle)
    diff = subprocess.check_output(["git", "-C", str(repo), "diff", "HEAD", "--", "."])
    provenance = {"repo_commit": git(repo, "rev-parse", "HEAD"),
        "repo_diff_sha256": hashlib.sha256(diff).hexdigest(), "aiperf": pin,
        "bundle": str(bundle.resolve()), "bundle_sha256": bundle_hash,
        "checkpoint": config["checkpoint"], "tokenizer": config["tokenizer"],
        "platform": platform.platform(), "settings": config,
        "timing_scopes": {"nonstreaming": "public_task_call_wall",
                          "streaming": "public_task_stream_wall_including_backpressure"}}
    untracked = git(repo, "ls-files", "--others", "--exclude-standard").splitlines()
    provenance["untracked_sources"] = {name: digest(repo / name) for name in untracked
                                        if (repo / name).is_file() and not (repo / name).is_symlink()}
    with zipfile.ZipFile(out / "source-untracked.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in provenance["untracked_sources"]:
            archive.write(repo / name, name)
    provenance["server_binary_sha256"] = digest(Path(config["server"]["binary"]))
    (out / "source.diff").write_bytes(diff)
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    probe = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv"],
                           capture_output=True, text=True)
    provenance["gpu"] = {"exit_code": probe.returncode, "stdout": probe.stdout, "stderr": probe.stderr}
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    entries = []
    validation_env = build_env(environment)
    if config.get("validation_profile"):
        validation_env.update(TRTMC_E2E_BUNDLE=str(bundle.resolve()),
            TRTMC_E2E_PROFILE=config["validation_profile"],
            TRTMC_E2E_CHECKPOINT_REVISION=config["checkpoint"]["revision"])
    with gpu_exclusive(environment):
        for index, command in enumerate(commands):
            with (out / f"validation-{index}.log").open("w") as log:
                code = _run(command, log, validation_env, config.get("validation_timeout", 7200), cwd=repo)
            (out / f"validation-{index}.json").write_text(json.dumps({"command": command,
                "exit_code": code, "bundle_sha256": bundle_hash}, indent=2) + "\n")
            if code:
                raise ConfigError(f"family validation failed: validation-{index}.log")
        if digest(bundle) != bundle_hash:
            raise ConfigError("validation changed the bundle; validate the final artifact before timing")
        provenance["server_binary_sha256"] = digest(Path(config["server"]["binary"]))
        (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        with text_serving(environment, config["server"], out / "server") as service:
            libraries = json.loads((out / "server/loaded-libraries.json").read_text())
            library_hashes = {name: digest(Path(name)) for name in libraries if Path(name).is_file()}
            (out / "server/runtime-artifacts.json").write_text(json.dumps(library_hashes, indent=2) + "\n")
            if any(case["streaming"] for case in matrix) and service["info"].get("trtmc", {}).get("streaming") != "incremental":
                raise ConfigError("token latency requires incremental family streaming; this model uses buffered SSE")
            for index, case in enumerate(matrix):
                run = run_aiperf(environment, out / f"run-{index:03d}", arguments(service["url"], config, case),
                    model_name=config["server"]["model_name"],
                    command_prefix=[str(environment["aiperf_python"]), "-m", "aiperf"],
                    timeout_s=config.get("profile_timeout", 1800))
                records = [json.loads(line) for line in service["records"].read_text().split("\n") if line.strip()]
                rows = join_records(run, records)
                (run.directory / "joined.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
                incomplete = (len([row for row in rows if row["phase"] == "profiling"]) != config["workload"].get("requests", 20)
                              or summarize(rows)["missing_metric_exports"] > 0)
                entries.append({"case": case, "exit_code": run.exit_code or int(incomplete),
                                "incomplete_export": incomplete, "command": run.command,
                                "summary": summarize(rows), "aiperf_summary": run.summary})
                (out / "report.json").write_text(json.dumps({"runs": entries}, indent=2) + "\n")
                if entries[-1]["exit_code"]:
                    break
        if config.get("reference") and not any(item["exit_code"] for item in entries):
            references, comparisons = reference_profiles(environment, config, out, matrix, entries)
            result = {"runs": entries, "reference_runs": references, "comparisons": comparisons}
            (out / "report.json").write_text(json.dumps(result, indent=2) + "\n")
            return result
    return {"runs": entries}


def response_text(record: dict[str, Any]) -> str | None:
    if record.get("status") != 200 or record.get("error"):
        return None
    try:
        body = json.loads(record["responses"][-1]["text"])
        choice = body["choices"][0]
        return choice["text"] if "text" in choice else choice["message"]["content"]
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def compare_runs(candidate: AiperfRun, reference: AiperfRun,
                 candidate_records: list[dict[str, Any]], reference_records: list[dict[str, Any]]) -> dict[str, Any]:
    """A descriptive native Task ratio only when the complete workload matches."""
    candidate_raw, reference_raw = candidate.raw_records(), reference.raw_records()
    left = {record["metadata"]["session_num"]: record for record in candidate_raw}
    right = {record["metadata"]["session_num"]: record for record in reference_raw}
    if len(left) != len(candidate_raw) or len(right) != len(reference_raw):
        return {"comparable": False, "matched_requests": 0,
                "mismatches": [{"reason": "duplicate session numbers; single-turn comparison required"}],
                "native_task_wall_ratio": None}
    lc = {record["request_id"]: record for record in candidate_records}
    rc = {record["request_id"]: record for record in reference_records}
    candidate_ms, reference_ms = [], []
    mismatches = []
    for index in sorted(set(left) | set(right)):
        a, b = left.get(index), right.get(index)
        if not a or not b:
            mismatches.append({"session_num": index, "reason": "missing request"})
            continue
        a_payload = {key: value for key, value in (a.get("payload") or {}).items() if key != "model"}
        b_payload = {key: value for key, value in (b.get("payload") or {}).items() if key != "model"}
        a_server = lc.get(a["metadata"].get("x_request_id"), {})
        b_server = rc.get(b["metadata"].get("x_request_id"), {})
        text = response_text(a)
        if (not a_payload or a_payload != b_payload or text is None or text != response_text(b)
                or a_server.get("status") != 200 or b_server.get("status") != 200
                or type(a_server.get("completion_tokens")) is not int
                or a_server["completion_tokens"] < 0
                or a_server.get("completion_tokens") != b_server.get("completion_tokens")
                or a_server.get("timing_scope") != "public_task_call_wall"
                or b_server.get("timing_scope") != "public_task_call_wall"
                or "model_call_ms" not in a_server or "model_call_ms" not in b_server):
            mismatches.append({"session_num": index, "reason": "input, output work, error or timing scope differs"})
            continue
        candidate_ms.append(a_server["model_call_ms"])
        reference_ms.append(b_server["model_call_ms"])
    eligible = bool(candidate_ms) and not mismatches
    c50, r50 = percentile(candidate_ms, .5), percentile(reference_ms, .5)
    return {"comparable": eligible, "matched_requests": len(candidate_ms), "mismatches": mismatches,
            "candidate_task_ms": statistics(candidate_ms), "reference_task_ms": statistics(reference_ms),
            "native_task_wall_ratio": r50 / c50 if eligible and c50 and r50 is not None else None,
            "interpretation": "descriptive matched-work ratio; no HTTP-only speedup or qualification verdict"}


def reference_profiles(environment: Environment, config: dict[str, Any], out: Path,
                       matrix: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> tuple[list, list]:
    # Called only after the candidate server has stopped, under the same GPU lock.
    reference = config["reference"]
    checkpoint = config["checkpoint"]
    from .models import _import_repository
    _import_repository(environment.path("repo"))
    from trtmc_benchmark.catalog import ManifestCatalog
    catalog = ManifestCatalog(environment.path("repo") / "families").resolve(reference["profile"])
    if catalog.hf_id != checkpoint["name"] or catalog.precision != checkpoint["precision"]:
        raise ConfigError("reference profile must match the candidate checkpoint and precision")
    model = {"catalog_profile": reference["profile"], "candidate": {"revision": checkpoint["revision"]},
             "reference": {"model": checkpoint["name"], "revision": checkpoint["revision"],
                           "precision": checkpoint["precision"]}}
    entries, comparisons = [], []
    with serving(environment, model, "reference", out / "reference-server", mode=reference.get("mode", "eager"),
                 precision=checkpoint["precision"], python=reference["python"]) as service:
        if service["info"].get("timing_scope") != "task-call-wall":
            raise ConfigError("reference does not declare a matching Task wall timing scope")
        for index, case in enumerate(matrix):
            # Compare single-lane, closed-loop, nonstreaming work only. Different
            # admission or stream policies would confound a backend comparison.
            if case["streaming"] or case["concurrency"] != 1 or case["request_rate"] is not None:
                continue
            run = run_aiperf(environment, out / f"reference-{index:03d}", arguments(service["url"], config, case),
                command_prefix=[str(environment["aiperf_python"]), "-m", "aiperf"],
                timeout_s=config.get("profile_timeout", 1800))
            raw_records = [json.loads(line) for line in service["records"].read_text().split("\n") if line.strip()]
            records = [{**record, **record["timing"], "status": 200,
                        "timing_scope": "public_task_call_wall",
                        "completion_tokens": record["observation"].get("output_tokens"),
                        "completion_token_source": "native_reference"} for record in raw_records]
            rows = join_records(run, records)
            (run.directory / "joined.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            incomplete = (len([row for row in rows if row["phase"] == "profiling"]) != config["workload"].get("requests", 20)
                          or summarize(rows)["missing_metric_exports"] > 0)
            entries.append({"case": case, "exit_code": run.exit_code or int(incomplete),
                            "incomplete_export": incomplete, "summary": summarize(rows),
                            "command": run.command, "aiperf_summary": run.summary})
            candidate = AiperfRun(out / f"run-{index:03d}", candidates[index]["exit_code"], candidates[index]["command"])
            candidate_records = [json.loads(line) for line in (out / "server/records.jsonl").read_text().split("\n") if line.strip()]
            comparison = compare_runs(candidate, run, candidate_records, records)
            if entries[-1]["exit_code"]:
                comparison.update(comparable=False, native_task_wall_ratio=None, incomplete_export=True)
            comparisons.append({"case": case, **comparison})
    return entries, comparisons

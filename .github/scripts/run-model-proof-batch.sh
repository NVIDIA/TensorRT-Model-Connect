#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run isolated model proofs concurrently without oversubscribing the configured
# GPUs. Each long-lived worker owns one GPU and receives a deterministic,
# round-robin subset of the requested models.

set -uo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
proof_runner="${TRTMC_MODEL_PROOF_RUNNER:-$script_dir/run-model-proof.sh}"

models_json=""
expected_count=""
revision=""
suite=""
output_dir=""
state_dir=""
models_file=""
declare -a models=()
declare -a gpu_ids=()
declare -a worker_pids=()

usage() {
  cat <<'EOF'
usage: run-model-proof-batch.sh --models-json JSON --expected-count N \
       --revision SHA --suite premerge|nightly --output-dir DIR

Run every model in the JSON array through run-model-proof.sh. The configured
TRTMC_MODEL_PROOF_GPU_IDS (default: 0,1,2,3) define both the worker pool and
the maximum concurrency.
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --models-json)
      models_json="${2:-}"
      shift 2
      ;;
    --expected-count)
      expected_count="${2:-}"
      shift 2
      ;;
    --revision)
      revision="${2:-}"
      shift 2
      ;;
    --suite)
      suite="${2:-}"
      shift 2
      ;;
    --output-dir)
      output_dir="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      die "unknown argument: $1"
      ;;
  esac
done

[ -n "$models_json" ] || die "--models-json is required"
[[ "$expected_count" =~ ^[1-9][0-9]*$ ]] || \
  die "--expected-count must be a positive integer"
[ -n "$revision" ] || die "--revision is required"
case "$suite" in
  premerge|nightly) ;;
  *) die "--suite must be premerge or nightly" ;;
esac
[ -n "$output_dir" ] || die "--output-dir is required"
[ -f "$proof_runner" ] || die "model proof runner is not a file: $proof_runner"
command -v python3 >/dev/null || die "python3 is required"

configured_gpu_ids="${TRTMC_MODEL_PROOF_GPU_IDS-0,1,2,3}"
[[ "$configured_gpu_ids" =~ ^(0|[1-9][0-9]*)(,(0|[1-9][0-9]*))*$ ]] || \
  die "TRTMC_MODEL_PROOF_GPU_IDS must be a comma-separated list of unique non-negative integers"
IFS=, read -r -a gpu_ids <<< "$configured_gpu_ids"
declare -A seen_gpu_ids=()
for gpu_id in "${gpu_ids[@]}"; do
  [[ -z "${seen_gpu_ids[$gpu_id]+x}" ]] || \
    die "TRTMC_MODEL_PROOF_GPU_IDS contains duplicate GPU ID: $gpu_id"
  seen_gpu_ids[$gpu_id]=1
done

output_dir="$(python3 - "$output_dir" <<'PY'
import sys
from pathlib import Path

print(Path(sys.argv[1]).resolve())
PY
)" || die "could not resolve --output-dir"
[ "$output_dir" != "/" ] || die "refusing to use / as output directory"
mkdir -p -- "$output_dir" || die "could not create output directory: $output_dir"

state_dir="$output_dir/.batch-state"
models_file="$state_dir/models.txt"
rm -rf -- "$state_dir"
mkdir -p -- "$state_dir/results" || die "could not initialize batch state"

python3 - "$models_json" "$expected_count" "$models_file" <<'PY'
import json
import re
import sys
from pathlib import Path

raw, expected_text, output = sys.argv[1:]
try:
    value = json.loads(raw)
except json.JSONDecodeError as exc:
    raise SystemExit(f"invalid --models-json: {exc}") from exc
if not isinstance(value, list):
    raise SystemExit("--models-json must be a JSON array")
if len(value) != int(expected_text):
    raise SystemExit(
        "--expected-count does not match --models-json length: "
        f"expected {expected_text}, found {len(value)}"
    )
safe = re.compile(r"[a-z0-9][a-z0-9._-]*")
seen: set[str] = set()
for index, model in enumerate(value):
    if not isinstance(model, str) or safe.fullmatch(model) is None:
        raise SystemExit(f"unsafe model id at index {index}: {model!r}")
    if model in seen:
        raise SystemExit(f"duplicate model id: {model}")
    seen.add(model)
Path(output).write_text("".join(f"{model}\n" for model in value), encoding="utf-8")
PY
parse_rc=$?
[ "$parse_rc" -eq 0 ] || exit "$parse_rc"
mapfile -t models < "$models_file"

write_batch_outputs() {
  local mode="$1"
  local signal_exit_code="${2:-0}"
  python3 - \
    "$models_file" "$state_dir/results" "$output_dir" "$revision" "$suite" \
    "$configured_gpu_ids" "$expected_count" "$mode" "$signal_exit_code" <<'PY'
import html
import json
import sys
from pathlib import Path

(
    models_path,
    results_path,
    output_path,
    revision,
    suite,
    gpu_text,
    expected_text,
    mode,
    signal_exit_text,
) = sys.argv[1:]
models = models_path and Path(models_path).read_text(encoding="utf-8").splitlines()
results_dir = Path(results_path)
output_dir = Path(output_path)
gpu_ids = gpu_text.split(",")
entries = []

for index, model in enumerate(models):
    result_path = results_dir / f"{index:06d}.json"
    result = None
    if result_path.is_file():
        try:
            loaded = json.loads(result_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                result = loaded
        except (json.JSONDecodeError, OSError):
            result = None
    assigned_gpu = gpu_ids[index % len(gpu_ids)]
    if result is not None:
        status = "passed" if result.get("exit_code") == 0 else "failed"
        exit_code = result.get("exit_code")
        duration = result.get("duration_seconds")
    elif mode == "running":
        status = "queued"
        exit_code = None
        duration = None
    elif mode == "interrupted":
        status = "interrupted"
        exit_code = int(signal_exit_text)
        duration = None
    else:
        status = "failed"
        exit_code = None
        duration = None
    report = f"{model}/artifacts/model-proof-report.html"
    log = f"{model}/batch.log"
    entries.append(
        {
            "model": model,
            "gpu_id": assigned_gpu,
            "status": status,
            "exit_code": exit_code,
            "duration_seconds": duration,
            "output_dir": model,
            "report": report,
            "report_exists": (output_dir / report).is_file(),
            "log": log,
            "log_exists": (output_dir / log).is_file(),
        }
    )

counts = {
    status: sum(entry["status"] == status for entry in entries)
    for status in ("passed", "failed", "interrupted", "queued")
}
if mode == "running":
    outcome = "running"
elif mode == "interrupted":
    outcome = "interrupted"
elif counts["failed"] or counts["interrupted"] or counts["queued"]:
    outcome = "failed"
else:
    outcome = "passed"

payload = {
    "schema_version": 1,
    "report_kind": "model_proof_batch",
    "source_revision": revision,
    "suite": suite,
    "outcome": outcome,
    "expected_count": int(expected_text),
    "model_count": len(models),
    "passed_count": counts["passed"],
    "failed_count": counts["failed"],
    "interrupted_count": counts["interrupted"],
    "queued_count": counts["queued"],
    "gpu_ids": gpu_ids,
    "models": entries,
}
(output_dir / "batch-status.json").write_text(
    json.dumps(payload, indent=2) + "\n", encoding="utf-8"
)

colors = {
    "passed": "#1b5e20",
    "failed": "#b71c1c",
    "interrupted": "#8d4b00",
    "queued": "#455a64",
    "running": "#455a64",
}
rows = []
for entry in entries:
    report_cell = (
        f'<a href="{html.escape(entry["report"], quote=True)}">HTML report</a>'
        if entry["report_exists"]
        else "Unavailable"
    )
    log_cell = (
        f'<a href="{html.escape(entry["log"], quote=True)}">Batch log</a>'
        if entry["log_exists"]
        else "Unavailable"
    )
    exit_cell = "" if entry["exit_code"] is None else str(entry["exit_code"])
    duration_cell = (
        "" if entry["duration_seconds"] is None else f'{entry["duration_seconds"]} s'
    )
    rows.append(
        "<tr>"
        f'<td><code>{html.escape(entry["model"])}</code></td>'
        f'<td><code>{html.escape(entry["gpu_id"])}</code></td>'
        f'<td class="{entry["status"]}">{html.escape(entry["status"])}</td>'
        f"<td>{exit_cell}</td><td>{duration_cell}</td>"
        f"<td>{report_cell}</td><td>{log_cell}</td>"
        "</tr>"
    )
document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Isolated model proof batch</title>
<style>
body{{font:16px system-ui,sans-serif;max-width:1200px;margin:2rem auto;padding:0 1rem;color:#17202a}}
table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccd1d1;padding:.55rem;text-align:left}}
th{{background:#f4f6f7}}.passed{{color:{colors['passed']};font-weight:700}}.failed{{color:{colors['failed']};font-weight:700}}
.interrupted{{color:{colors['interrupted']};font-weight:700}}.queued{{color:{colors['queued']};font-weight:700}}
code{{overflow-wrap:anywhere}}
</style></head><body>
<h1>Isolated model proof batch</h1>
<p>Outcome: <strong class="{outcome}">{html.escape(outcome)}</strong></p>
<dl><dt>Revision</dt><dd><code>{html.escape(revision)}</code></dd>
<dt>Suite</dt><dd>{html.escape(suite)}</dd><dt>Models</dt><dd>{len(models)}</dd>
<dt>GPU workers</dt><dd><code>{html.escape(gpu_text)}</code></dd></dl>
<table><thead><tr><th>Model</th><th>GPU</th><th>Status</th><th>Exit</th><th>Duration</th><th>Evidence</th><th>Log</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
</body></html>"""
(output_dir / "model-proof-index.html").write_text(document, encoding="utf-8")

if mode != "running" and outcome != "passed":
    raise SystemExit(1)
PY
}

write_result() {
  local index="$1"
  local model="$2"
  local gpu_id="$3"
  local exit_code="$4"
  local duration="$5"
  local result_path="$state_dir/results/$(printf '%06d' "$index").json"
  local temporary_path="$result_path.tmp-$BASHPID"
  python3 - "$temporary_path" "$model" "$gpu_id" "$exit_code" "$duration" <<'PY'
import json
import sys
from pathlib import Path

path, model, gpu_id, exit_code, duration = sys.argv[1:]
Path(path).write_text(
    json.dumps(
        {
            "model": model,
            "gpu_id": gpu_id,
            "exit_code": int(exit_code),
            "duration_seconds": int(duration),
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY
  mv -- "$temporary_path" "$result_path"
}

run_worker() {
  local worker_index="$1"
  local gpu_id="$2"
  local current_pid=""
  local pid_file="$state_dir/worker-$worker_index.pid"

  stop_worker() {
    trap - INT TERM HUP
    if [ -n "$current_pid" ]; then
      kill -TERM "$current_pid" 2>/dev/null || true
      wait "$current_pid" 2>/dev/null || true
    fi
    rm -f -- "$pid_file"
    exit 143
  }
  trap stop_worker INT TERM HUP

  local index model model_dir log_path started finished proof_rc missing_evidence evidence
  for ((index = worker_index; index < ${#models[@]}; index += ${#gpu_ids[@]})); do
    model="${models[$index]}"
    model_dir="$output_dir/$model"
    log_path="$model_dir/batch.log"
    mkdir -p -- "$model_dir"
    : > "$log_path"
    started="$(date +%s)"
    echo "Starting isolated proof for $model on GPU $gpu_id"
    TRTMC_GPU_ID="$gpu_id" bash "$proof_runner" \
      --model "$model" \
      --revision "$revision" \
      --suite "$suite" \
      --output-dir "$model_dir" \
      > "$log_path" 2>&1 &
    current_pid=$!
    printf '%s\n' "$current_pid" > "$pid_file"
    wait "$current_pid"
    proof_rc=$?
    current_pid=""
    rm -f -- "$pid_file"
    if [ "$proof_rc" -eq 0 ]; then
      missing_evidence=""
      for evidence in proof.json model-proof-status.json model-proof-report.html; do
        if [ ! -f "$model_dir/artifacts/$evidence" ]; then
          missing_evidence="${missing_evidence}${missing_evidence:+, }$evidence"
        fi
      done
      if [ -n "$missing_evidence" ]; then
        printf 'ERROR: successful model proof did not emit required evidence: %s\n' \
          "$missing_evidence" | tee -a "$log_path" >&2
        proof_rc=1
      fi
    fi
    # A batch can cover every model on one runner.  Preserve the reports and
    # logs, but release each completed scratch build and source projection
    # before this worker starts its next model.  Retaining all of those trees
    # until artifact upload can exhaust the runner disk and create false E2E
    # failures in otherwise independent models.
    if ! rm -rf -- "$model_dir/work" "$model_dir/projection"; then
      printf 'ERROR: could not remove completed scratch state for %s\n' "$model" \
        | tee -a "$log_path" >&2
      proof_rc=1
    fi
    finished="$(date +%s)"
    write_result "$index" "$model" "$gpu_id" "$proof_rc" "$((finished - started))"
    echo "Finished isolated proof for $model on GPU $gpu_id (exit $proof_rc)"
  done
}

handle_signal() {
  local signal_name="$1"
  local signal_exit_code="$2"
  trap - INT TERM HUP
  echo "Received $signal_name; stopping model-proof workers" >&2
  local pid child_pid
  for pid in "${worker_pids[@]}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
  for pid_file in "$state_dir"/worker-*.pid; do
    [ -f "$pid_file" ] || continue
    read -r child_pid < "$pid_file" || true
    if [[ "$child_pid" =~ ^[1-9][0-9]*$ ]]; then
      kill -TERM "$child_pid" 2>/dev/null || true
    fi
  done
  for pid in "${worker_pids[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  write_batch_outputs interrupted "$signal_exit_code" || true
  exit "$signal_exit_code"
}

trap 'handle_signal INT 130' INT
trap 'handle_signal TERM 143' TERM
trap 'handle_signal HUP 129' HUP

# Emit useful machine-readable and human-readable output even while the batch
# is running. A signal or normal completion replaces these queued entries.
write_batch_outputs running || die "could not initialize batch status and HTML index"

worker_count="${#gpu_ids[@]}"
if [ "${#models[@]}" -lt "$worker_count" ]; then
  worker_count="${#models[@]}"
fi
for ((worker_index = 0; worker_index < worker_count; worker_index++)); do
  run_worker "$worker_index" "${gpu_ids[$worker_index]}" &
  worker_pids+=("$!")
done

worker_failure=0
for pid in "${worker_pids[@]}"; do
  wait "$pid" || worker_failure=1
done

write_batch_outputs complete
batch_rc=$?
if [ "$worker_failure" -ne 0 ]; then
  batch_rc=1
fi
exit "$batch_rc"

#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build and validate one model from an allowlisted source projection. The host
# half creates the projection from a Git revision and starts a new container
# that cannot see the checkout. The inner half performs a scratch build and
# emits proof.json only after the model-owned tests and one E2E oracle pass.

set -euo pipefail

proof_container_name=""
proof_artifacts_dir=""
proof_repo_root=""
proof_gpu_id=""
proof_gpu_lease_fd=""
proof_gpu_lease_file=""

die() {
  if [ -n "$proof_artifacts_dir" ]; then
    mkdir -p "$proof_artifacts_dir"
    printf 'ERROR: %s\n' "$*" >> "$proof_artifacts_dir/host-error.log"
  fi
  echo "ERROR: $*" >&2
  exit 1
}

cleanup_proof_container() {
  local rc="$1"
  trap - EXIT INT TERM
  if [ -n "$proof_container_name" ]; then
    docker rm -f "$proof_container_name" >/dev/null 2>&1 || true
  fi
  generate_host_fallback_report "$rc" || true
  if [ -n "$proof_gpu_lease_fd" ]; then
    flock -u "$proof_gpu_lease_fd" || true
  fi
  exit "$rc"
}

usage() {
  cat <<'EOF'
usage: run-model-proof.sh --model ID [--suite premerge|nightly] [--revision SHA] [--output-dir DIR]
       run-model-proof.sh --inner --model ID --suite SUITE --revision SHA --output-dir DIR
EOF
}

model=""
suite="premerge"
revision="HEAD"
output_dir=""
inner=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --model)
      model="${2:-}"
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
    --inner)
      inner=1
      shift
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

[ -n "$model" ] || die "--model is required"
[[ "$model" =~ ^[A-Za-z0-9_.-]+$ ]] || die "unsafe model id: $model"
case "$suite" in
  premerge|nightly) ;;
  *) die "--suite must be premerge or nightly" ;;
esac

python_bin() {
  if [ -x "${TRTMC_HF_PYTHON:-/opt/venv/bin/python}" ]; then
    printf '%s\n' "${TRTMC_HF_PYTHON:-/opt/venv/bin/python}"
  else
    command -v python3
  fi
}

select_proof_gpu() {
  if [[ -v TRTMC_GPU_ID ]]; then
    [[ "$TRTMC_GPU_ID" =~ ^[0-9]+$ ]] || \
      die "TRTMC_GPU_ID must be a non-negative integer"
    proof_gpu_id="$TRTMC_GPU_ID"
    echo "Using explicit model-proof GPU $proof_gpu_id"
    return 0
  fi

  command -v flock >/dev/null || \
    die "flock is required for automatic model-proof GPU leasing"
  local configured_ids="${TRTMC_MODEL_PROOF_GPU_IDS:-0,1,2,3}"
  [[ "$configured_ids" =~ ^(0|[1-9][0-9]*)(,(0|[1-9][0-9]*))*$ ]] || \
    die "TRTMC_MODEL_PROOF_GPU_IDS must be a comma-separated list of unique non-negative integers"
  local timeout="${TRTMC_MODEL_PROOF_GPU_LEASE_TIMEOUT_SECONDS:-60}"
  [[ "$timeout" =~ ^[1-9][0-9]{0,3}$ ]] && [ "$timeout" -le 3600 ] || \
    die "TRTMC_MODEL_PROOF_GPU_LEASE_TIMEOUT_SECONDS must be an integer from 1 to 3600"

  local lock_dir="${TRTMC_MODEL_PROOF_GPU_LOCK_DIR:-/tmp/trtmc-model-proof-gpu-locks}"
  [ -n "$lock_dir" ] || die "TRTMC_MODEL_PROOF_GPU_LOCK_DIR must not be empty"
  mkdir -p -- "$lock_dir" || die "could not create GPU lease directory: $lock_dir"

  local -a gpu_ids=()
  IFS=, read -r -a gpu_ids <<< "$configured_ids"
  local -A seen_ids=()
  local candidate
  for candidate in "${gpu_ids[@]}"; do
    [[ -z "${seen_ids[$candidate]+x}" ]] || \
      die "TRTMC_MODEL_PROOF_GPU_IDS contains duplicate GPU ID: $candidate"
    seen_ids[$candidate]=1
  done

  local deadline=$((SECONDS + timeout))
  local attempted=0
  while true; do
    if [ "$attempted" -eq 1 ] && [ "$SECONDS" -ge "$deadline" ]; then
      die "timed out after ${timeout}s waiting for a model-proof GPU lease from: $configured_ids"
    fi
    attempted=1
    for candidate in "${gpu_ids[@]}"; do
      local candidate_fd
      local lock_file="$lock_dir/gpu-$candidate.lock"
      exec {candidate_fd}>"$lock_file" || die "could not open GPU lease file: $lock_file"
      if flock -n "$candidate_fd"; then
        proof_gpu_id="$candidate"
        proof_gpu_lease_fd="$candidate_fd"
        proof_gpu_lease_file="$lock_file"
        echo "Leased model-proof GPU $proof_gpu_id via $proof_gpu_lease_file"
        return 0
      fi
      exec {candidate_fd}>&-
    done
    sleep 1
  done
}

generate_host_fallback_report() {
  local rc="$1"
  [ -n "$proof_artifacts_dir" ] || return 0
  mkdir -p "$proof_artifacts_dir"
  [ -n "$proof_repo_root" ] || return 0
  python3 "$proof_repo_root/.github/scripts/write-model-proof-fallback-report.py" \
    --artifacts-dir "$proof_artifacts_dir" \
    --model "$model" \
    --revision "$revision" \
    --suite "$suite" \
    --outcome failed \
    --phase host-setup \
    --exit-code "$((rc == 0 ? 1 : rc))" \
    --preserve-rich-report
}

initialize_proof_status() {
  "$(python_bin)" - "$model" "$revision" "$suite" /artifacts/model-proof-status.json <<'PY'
import json
import sys
from pathlib import Path

model, revision, suite, output = sys.argv[1:]
payload = {
    "schema_version": 1,
    "model": model,
    "source_revision": revision,
    "suite": suite,
    "outcome": "running",
    "steps": {
        "projection_validation": {"status": "running", "evidence": "source-projection.json, selection.json"},
        "configure": {"status": "pending", "evidence": "configure.log"},
        "scratch_build": {"status": "pending", "evidence": "build.log"},
        "dso_isolation": {"status": "pending", "evidence": "model-dsos.txt, model-dso.dynamic.txt"},
        "cpp_tests": {"status": "pending", "evidence": "cpp-tests.log"},
        "python_tests": {"status": "pending", "evidence": "python-model-tests.xml"},
        "e2e_reference": {"status": "pending", "evidence": "e2e/junit.xml, e2e/*/result.json"},
        "result_verification": {"status": "pending", "evidence": "e2e-verification.json"},
        "html_report": {"status": "pending", "evidence": "model-proof-report.html"},
    },
}
Path(output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

update_proof_step() {
  local step="$1"
  local status="$2"
  local evidence="${3:-}"
  "$(python_bin)" - /artifacts/model-proof-status.json "$step" "$status" "$evidence" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
step = payload.setdefault("steps", {}).setdefault(sys.argv[2], {})
step["status"] = sys.argv[3]
if sys.argv[4]:
    step["evidence"] = sys.argv[4]
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

update_proof_fact() {
  local key="$1"
  local value="$2"
  "$(python_bin)" - /artifacts/model-proof-status.json "$key" "$value" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
payload[sys.argv[2]] = sys.argv[3]
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

finalize_model_report() {
  local validation_rc="$1"
  trap - EXIT
  set +e

  "$(python_bin)" - /artifacts/model-proof-status.json "$validation_rc" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
rc = int(sys.argv[2])
payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
payload["validation_exit_code"] = rc
payload["outcome"] = "report-validation" if rc == 0 else "failed"
for step in (payload.get("steps") or {}).values():
    if isinstance(step, dict) and step.get("status") == "running":
        step["status"] = "passed" if rc == 0 else "failed"
    elif isinstance(step, dict) and step.get("status") == "pending" and rc != 0:
        step["status"] = "not-run"
report_step = payload.setdefault("steps", {}).setdefault("html_report", {})
report_step["status"] = "running"
report_step["evidence"] = "model-proof-report.html"
evidence = []
for item in sorted(Path("/artifacts").rglob("*")):
    if not item.is_file() or item.name == "model-proof-report.html":
        continue
    rel = str(item.relative_to("/artifacts"))
    if item.parent == Path("/artifacts") or item.suffix in {".json", ".xml", ".log"}:
        evidence.append(rel)
payload["evidence_files"] = evidence[:200]
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

  "$(python_bin)" /src/scripts/generate_e2e_report.py \
    --artifacts-dir /artifacts/e2e \
    --output /artifacts/model-proof-report.html \
    --project-dir /src \
    --title "Isolated Model Proof: $model @ ${revision:0:12}" \
    --proof-status /artifacts/model-proof-status.json \
    --proof-json /artifacts/proof.json \
    --selection-json /artifacts/selection.json \
    --strict-evidence \
    --max-embed-bytes 33554432
  local report_rc="$?"
  "$(python_bin)" - /artifacts/model-proof-status.json "$validation_rc" "$report_rc" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
validation_rc = int(sys.argv[2])
report_rc = int(sys.argv[3])
payload = json.loads(path.read_text(encoding="utf-8"))
report_step = payload.setdefault("steps", {}).setdefault("html_report", {})
report_step["status"] = "passed" if report_rc == 0 else "failed"
report_step["evidence"] = "model-proof-report.html"
payload["report_exit_code"] = report_rc
if validation_rc != 0:
    payload["outcome"] = "failed"
    payload["exit_code"] = validation_rc
elif report_rc != 0:
    payload["outcome"] = "failed"
    payload["exit_code"] = report_rc
else:
    payload["outcome"] = "passed"
    payload["exit_code"] = 0
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
  if [ -f /artifacts/model-proof-report.html ]; then
    echo "Model proof HTML report: /artifacts/model-proof-report.html"
  fi
  if [ "$validation_rc" -eq 0 ] && [ "$report_rc" -ne 0 ]; then
    echo "ERROR: model proof report evidence validation failed (exit $report_rc)" >&2
    exit "$report_rc"
  fi
  exit "$validation_rc"
}

write_model_proof_selection() {
  local source_root="$1"
  local selection_path="$2"
  local config_file="$3"
  "$(python_bin)" - "$model" "$suite" "$revision" "$source_root" "$selection_path" \
    > "$config_file" <<'PY'
import json
import os
import sys
import tomllib
from pathlib import Path

selected = sys.argv[1]
suite = sys.argv[2]
revision = sys.argv[3]
root = Path(sys.argv[4]).resolve()
selection_path = Path(sys.argv[5])
manifest = json.loads((root / ".trtmc-model-projection.json").read_text(encoding="utf-8"))
if manifest.get("revision") != revision:
    raise SystemExit(
        f"projection revision {manifest.get('revision')!r}, expected {revision!r}"
    )
declared = manifest.get("model", manifest.get("selected_model", manifest.get("selected_models")))
if isinstance(declared, list):
    if declared != [selected]:
        raise SystemExit(f"projection selected models {declared!r}, expected {[selected]!r}")
elif declared not in (None, selected):
    raise SystemExit(f"projection selected model {declared!r}, expected {selected!r}")

for path in root.rglob("*"):
    if not path.is_symlink():
        continue
    target = (path.parent / os.readlink(path)).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise SystemExit(f"projection contains escaping symlink: {path} -> {target}") from exc

roots = {
    "python": root / "python/tensorrt_model_connect/families",
    "runtime": root / "src/runtime/models",
    "e2e": root / "tests/e2e/models",
}
owners = {}
for kind, owned_root in roots.items():
    manifests = sorted(owned_root.glob("*/MODEL.toml"))
    minimum = 0 if kind == "python" else 1
    if len(manifests) < minimum or len(manifests) > 1:
        raise SystemExit(
            f"projected {kind} ownership root must contain "
            f"{'at most' if kind == 'python' else 'exactly'} one MODEL.toml; "
            f"found {len(manifests)}"
        )
    if not manifests:
        owners[kind] = None
        continue
    data = tomllib.loads(manifests[0].read_text(encoding="utf-8"))
    owner = str(data.get("id") or "")
    if not owner or owner != manifests[0].parent.name:
        raise SystemExit(f"invalid projected {kind} manifest: {manifests[0]}")
    owners[kind] = owner

runtime_manifest = roots["runtime"] / owners["runtime"] / "MODEL.toml"
if manifest.get("runtime_model") != owners["runtime"]:
    raise SystemExit(
        f"projection runtime model {manifest.get('runtime_model')!r}, "
        f"found {owners['runtime']!r}"
    )
if manifest.get("e2e_family") != owners["e2e"]:
    raise SystemExit(
        f"projection E2E family {manifest.get('e2e_family')!r}, found {owners['e2e']!r}"
    )
runtime_data = tomllib.loads(runtime_manifest.read_text(encoding="utf-8"))
runtime_library = str(
    runtime_data.get("runtime_library") or f"libtrtmc_model_{owners['runtime']}.so"
)
runtime_tests = []
for entry in runtime_data.get("runtime_tests", []):
    fields = str(entry).split("|")
    if len(fields) != 5 or not fields[0]:
        raise SystemExit(f"invalid runtime_tests entry in {runtime_manifest}: {entry!r}")
    runtime_tests.append(fields[0])

e2e_dir = roots["e2e"] / owners["e2e"]
timing_estimates = {}
timing_path = root / "tests/e2e/timing_estimates.json"
if timing_path.is_file():
    timing_payload = json.loads(timing_path.read_text(encoding="utf-8"))
    raw_estimates = timing_payload.get("estimates_s", {})
    if isinstance(raw_estimates, dict):
        timing_estimates = raw_estimates
cases = []
for case_path in sorted((e2e_dir / "manifests").glob("*.json")):
    data = json.loads(case_path.read_text(encoding="utf-8"))
    if data.get("skip_reason") or data.get("skip"):
        continue
    manifest_name = str(data.get("name") or case_path.stem)
    testcases = data.get("testcases")
    if not isinstance(testcases, list) or not testcases:
        raise SystemExit(f"E2E manifest has no testcases: {case_path}")
    for testcase in testcases:
        if not isinstance(testcase, dict):
            raise SystemExit(f"E2E manifest has an invalid testcase: {case_path}")
        if testcase.get("skip_reason") or testcase.get("skip"):
            continue
        name = str(testcase.get("name") or "")
        if not name:
            raise SystemExit(f"E2E manifest has an unnamed testcase: {case_path}")
        tier = str(testcase.get("ci_tier") or data.get("ci_tier") or "")
        if tier == "multi_device":
            continue
        cases.append({
            "name": name,
            "model": manifest_name,
            "manifest": case_path.name,
            "ci_tier": tier,
            "l0_replacement": str(testcase.get("l0_replacement") or ""),
            "estimated_seconds": timing_estimates.get(name),
        })
if not cases:
    raise SystemExit(f"no single-GPU E2E case is available for {owners['e2e']}")
cases.sort(key=lambda case: (case["name"], case["model"], case["manifest"]))
if suite == "premerge":
    eligible = [
        case for case in cases
        if case["ci_tier"] not in {"nightly_only", "multi_device"}
    ]
    if not eligible:
        raise SystemExit(
            f"no premerge E2E case is available for {owners['e2e']}"
        )
    l0_replacements = {
        case["l0_replacement"]
        for case in cases
        if case["ci_tier"] == "nightly_only" and case["l0_replacement"]
    }
    replacements = [case for case in eligible if case["name"] in l0_replacements]
    candidates = replacements or eligible
    priority = {"l0_only": 0, "contract_only": 1, "": 2}
    candidates.sort(
        key=lambda case: (
            priority.get(case["ci_tier"], 2),
            case["estimated_seconds"]
            if isinstance(case["estimated_seconds"], (int, float))
            else float("inf"),
            case["name"],
            case["model"],
        )
    )
    selected_cases = candidates[:1]
else:
    selected_cases = cases

e2e_tests = sorted(e2e_dir.glob("test_*_e2e.py"))
if len(e2e_tests) != 1:
    raise SystemExit(
        f"projected model must have exactly one canonical E2E test; found {len(e2e_tests)}"
    )
python_tests = sorted(
    path for path in e2e_dir.glob("test_*.py")
    if not path.match("test_*_e2e.py")
)

selection = {
    "schema_version": 1,
    "requested_model": selected,
    "owners": owners,
    "runtime_library": runtime_library,
    "runtime_tests": runtime_tests,
    "python_tests": [str(path.relative_to(root)) for path in python_tests],
    "suite": suite,
    "e2e_cases": [
        {
            "name": case["name"],
            "model": case["model"],
            "manifest": case["manifest"],
            "ci_tier": case["ci_tier"],
            "l0_replacement": case["l0_replacement"],
            "estimated_seconds": case["estimated_seconds"],
        }
        for case in selected_cases
    ],
    "e2e_test": str(e2e_tests[0].relative_to(root)),
}
selection_path.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")

print(f"runtime_model={owners['runtime']}")
print(f"runtime_library={runtime_library}")
print(f"e2e_family={owners['e2e']}")
print(f"e2e_test={e2e_tests[0]}")
for model_name in dict.fromkeys(case["model"] for case in selected_cases):
    print(f"e2e_model={model_name}")
for case in selected_cases:
    print(f"e2e_case={case['name']}")
for test in runtime_tests:
    print(f"cpp_test={test}")
PY
}

run_inner() {
  [ "$output_dir" = "/artifacts" ] || die "inner output directory must be /artifacts"
  [ -f /src/.trtmc-model-projection.json ] || \
    die "projection manifest is missing from /src"
  [ ! -e /src/.git ] || die "projected source must not contain Git metadata"
  [ -d /work ] || die "isolated writable work directory is missing"

  mkdir -p /artifacts /artifacts/e2e /work/build /work/engines /work/model-plugins /work/tmp
  trap 'finalize_model_report "$?"' EXIT
  initialize_proof_status
  [[ "${TRTMC_MODEL_PROOF_GPU_ID:-}" =~ ^[0-9]+$ ]] || \
    die "TRTMC_MODEL_PROOF_GPU_ID must be passed as a non-negative integer"
  update_proof_fact gpu_id "$TRTMC_MODEL_PROOF_GPU_ID"
  cp /src/.trtmc-model-projection.json /artifacts/source-projection.json

  local config_file=/work/model-proof-config.txt
  write_model_proof_selection /src /artifacts/selection.json "$config_file"

  update_proof_step projection_validation passed \
    "source-projection.json, selection.json"

  local runtime_model runtime_library e2e_family e2e_test
  runtime_model="$(sed -n 's/^runtime_model=//p' "$config_file")"
  runtime_library="$(sed -n 's/^runtime_library=//p' "$config_file")"
  e2e_family="$(sed -n 's/^e2e_family=//p' "$config_file")"
  e2e_test="$(sed -n 's/^e2e_test=//p' "$config_file")"
  [ -n "$runtime_model" ] || die "could not resolve the runtime model from projection"
  local -a e2e_models=()
  mapfile -t e2e_models < <(sed -n 's/^e2e_model=//p' "$config_file")
  [ "${#e2e_models[@]}" -gt 0 ] || die "could not resolve an E2E model from projection"
  local -a e2e_cases=()
  mapfile -t e2e_cases < <(sed -n 's/^e2e_case=//p' "$config_file")
  [ "${#e2e_cases[@]}" -gt 0 ] || die "could not resolve an E2E case from projection"

  local build_jobs="${TRTMC_MODEL_PROOF_BUILD_JOBS:-8}"
  [[ "$build_jobs" =~ ^[1-9][0-9]*$ ]] || \
    die "TRTMC_MODEL_PROOF_BUILD_JOBS must be a positive integer"

  local cmake_args=(
    -S /src
    -B /work/build
    -DCMAKE_BUILD_TYPE=Release
    -DTRTMC_BUILD_TESTS=ON
    -DTRTMC_BUILD_BENCHMARKS=OFF
    -DTRTMC_ENABLE_LIBTORCH_MULTINOMIAL=OFF
    "-DTRTMC_MODEL_PROOF_MODEL=$runtime_model"
  )
  if command -v ninja >/dev/null 2>&1; then
    cmake_args+=(-G Ninja)
  fi
  update_proof_step configure running "configure.log"
  cmake "${cmake_args[@]}" 2>&1 | tee /artifacts/configure.log
  update_proof_step configure passed "configure.log"

  local targets=(trtmc trtmc_backend_trt "trtmc_model_$runtime_model")
  local cpp_test
  while IFS= read -r cpp_test; do
    [ -n "$cpp_test" ] && targets+=("$cpp_test")
  done < <(sed -n 's/^cpp_test=//p' "$config_file")
  update_proof_step scratch_build running "build.log"
  cmake --build /work/build --parallel "$build_jobs" --target "${targets[@]}" \
    2>&1 | tee /artifacts/build.log
  update_proof_step scratch_build passed "build.log"

  update_proof_step dso_isolation running \
    "model-dsos.txt, model-dso.dynamic.txt"
  local -a built_dsos=()
  mapfile -t built_dsos < <(
    find /work/build/models -type f -name 'libtrtmc_model_*.so' -print | sort
  )
  if [ "${#built_dsos[@]}" -ne 1 ]; then
    printf '%s\n' "${built_dsos[@]}" > /artifacts/model-dsos.txt
    die "scratch build produced ${#built_dsos[@]} model DSOs; expected exactly one"
  fi
  [ "$(basename "${built_dsos[0]}")" = "$runtime_library" ] || \
    die "scratch build produced $(basename "${built_dsos[0]}"), expected $runtime_library"
  printf '%s\n' "${built_dsos[0]}" > /artifacts/model-dsos.txt

  readelf -d "${built_dsos[0]}" > /artifacts/model-dso.dynamic.txt
  local unexpected_model_dependency
  unexpected_model_dependency="$(
    grep -o 'libtrtmc_model_[^] ]*\.so' /artifacts/model-dso.dynamic.txt \
      | grep -v -F "$runtime_library" | head -n 1 || true
  )"
  [ -z "$unexpected_model_dependency" ] || \
    die "model DSO links a sibling model DSO: $unexpected_model_dependency"

  local plugin_dir="/work/model-plugins/$runtime_model"
  mkdir -p "$plugin_dir"
  cp "${built_dsos[0]}" "$plugin_dir/$runtime_library"
  local runtime_library_sha256
  runtime_library_sha256="$("$(python_bin)" - "${built_dsos[0]}" <<'PY'
import hashlib
import sys
from pathlib import Path
print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())
PY
)"
  update_proof_fact runtime_model "$runtime_model"
  update_proof_fact runtime_library "$runtime_library"
  update_proof_fact runtime_library_sha256 "$runtime_library_sha256"
  update_proof_fact sibling_model_count "0"
  update_proof_fact model_dso_count "1"
  update_proof_fact network "disabled"
  update_proof_fact plugin_search "strict"
  update_proof_step dso_isolation passed \
    "exactly one DSO; no sibling model DT_NEEDED"

  local ctest_rc=0
  update_proof_step cpp_tests running "cpp-tests.log"
  : > /artifacts/cpp-tests.log
  while IFS= read -r cpp_test; do
    [ -n "$cpp_test" ] || continue
    set +e
    TRTMC_MODEL_PLUGIN_STRICT=1 \
      TRTMC_MODEL_PLUGIN_DIR=/work/model-plugins \
      ctest --test-dir /work/build --output-on-failure -R "^${cpp_test}$" \
      2>&1 | tee -a /artifacts/cpp-tests.log
    local current_ctest_rc="${PIPESTATUS[0]}"
    set -e
    if [ "$current_ctest_rc" -ne 0 ]; then
      ctest_rc=1
    fi
  done < <(sed -n 's/^cpp_test=//p' "$config_file")
  [ "$ctest_rc" -eq 0 ] || die "one or more model-owned C++ tests failed"
  update_proof_step cpp_tests passed "cpp-tests.log"

  local py
  py="$(python_bin)"
  local python_test_dir="/src/tests/e2e/models/$e2e_family"
  local -a python_tests=()
  mapfile -t python_tests < <(
    find "$python_test_dir" -maxdepth 1 -type f -name 'test_*.py' \
      ! -name 'test_*_e2e.py' -print | sort
  )
  if [ "${#python_tests[@]}" -gt 0 ]; then
    update_proof_step python_tests running \
      "python-model-tests.xml, python-model-tests.log"
    PYTHONPATH=/src/python:/src \
      PYTHONNOUSERSITE=1 \
      PYTHONDONTWRITEBYTECODE=1 \
      "$py" -m pytest "${python_tests[@]}" -v -p no:cacheprovider \
        --rootdir /src -c /src/pyproject.toml \
        --junitxml=/artifacts/python-model-tests.xml \
        2>&1 | tee /artifacts/python-model-tests.log
    update_proof_step python_tests passed \
      "python-model-tests.xml, python-model-tests.log"
  else
    update_proof_step python_tests skipped "no model-owned Python unit tests"
  fi

  local ld_library_path="/work/build:${LD_LIBRARY_PATH:-}"
  local models_file=/work/e2e-models.txt
  printf '%s\n' "${e2e_models[@]}" > "$models_file"
  local -a e2e_filter_args=()
  local e2e_model
  for e2e_model in "${e2e_models[@]}"; do
    e2e_filter_args+=(--e2e-model "$e2e_model")
  done
  local e2e_case
  for e2e_case in "${e2e_cases[@]}"; do
    e2e_filter_args+=(--e2e-testcase "$e2e_case")
  done
  update_proof_step e2e_reference running "e2e/junit.xml, e2e/*/result.json"
  PYTHONPATH=/src/python:/src \
    PYTHONNOUSERSITE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TRTMC_MODEL_PLUGIN_STRICT=1 \
    TRTMC_MODEL_PLUGIN_DIR=/work/model-plugins \
    LD_LIBRARY_PATH="$ld_library_path" \
    "$py" -m pytest "$e2e_test" -v -p no:cacheprovider \
      --rootdir /src -c /src/pyproject.toml \
      "${e2e_filter_args[@]}" \
      --engine-dir /work/engines \
      --trtmc-binary /work/build/trtmc \
      --hf-python "$py" \
      --e2e-artifacts-dir /artifacts/e2e \
      --model-plugin-dir /work/model-plugins \
      --rebuild-engines \
      --junitxml=/artifacts/e2e/junit.xml \
      2>&1 | tee /artifacts/e2e.log
  update_proof_step e2e_reference passed "e2e/junit.xml, e2e/*/result.json"

  update_proof_step result_verification running "e2e-verification.json"
  "$py" /src/tools/model_plugin_isolation.py verify-results \
    --repo-root /src \
    --models-file "$models_file" \
    --artifacts-dir /artifacts/e2e \
    --report /artifacts/e2e-verification.json
  update_proof_step result_verification passed "e2e-verification.json"

  "$py" - "$model" "$revision" "$runtime_model" "$runtime_library" \
    "$TRTMC_MODEL_PROOF_GPU_ID" \
    "${built_dsos[0]}" /artifacts/selection.json /artifacts/proof.json <<'PY'
import hashlib
import json
import sys
from pathlib import Path

(
    model,
    revision,
    runtime_model,
    runtime_library,
    gpu_id,
    dso_path,
    selection_path,
    output,
) = sys.argv[1:]
digest = hashlib.sha256(Path(dso_path).read_bytes()).hexdigest()
selection = json.loads(Path(selection_path).read_text(encoding="utf-8"))
proof = {
    "schema_version": 1,
    "passed": True,
    "model": model,
    "source_revision": revision,
    "runtime_model": runtime_model,
    "runtime_library": runtime_library,
    "runtime_library_sha256": digest,
    "gpu_id": gpu_id,
    "suite": selection["suite"],
    "e2e_cases": [case["name"] for case in selection["e2e_cases"]],
    "sibling_model_count": 0,
    "model_dso_count": 1,
    "network": "disabled",
    "plugin_search": "strict",
}
Path(output).write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
PY
  echo "PASS: isolated model proof completed for $model"
}

run_host() {
  command -v docker >/dev/null || die "docker is required"
  command -v git >/dev/null || die "git is required"

  local repo_root
  repo_root="$(git rev-parse --show-toplevel)"
  proof_repo_root="$repo_root"
  git -C "$repo_root" cat-file -e "${revision}^{commit}" 2>/dev/null || \
    die "revision is not a local commit: $revision"
  revision="$(git -C "$repo_root" rev-parse "${revision}^{commit}")"

  if [ -z "$output_dir" ]; then
    output_dir="${RUNNER_TEMP:-/tmp}/trtmc-model-proof-$model"
  fi
  output_dir="$(python3 - "$output_dir" <<'PY'
import sys
from pathlib import Path
print(Path(sys.argv[1]).resolve())
PY
)"
  [ "$output_dir" != "/" ] || die "refusing to use / as output directory"
  [ "$output_dir" != "$repo_root" ] || die "output directory cannot be the checkout"

  local projection_dir="$output_dir/projection"
  local artifacts_dir="$output_dir/artifacts"
  local work_dir="$output_dir/work"
  mkdir -p "$output_dir"
  rm -rf "$artifacts_dir" "$work_dir"
  mkdir -p "$artifacts_dir" "$work_dir"
  proof_artifacts_dir="$artifacts_dir"
  trap 'cleanup_proof_container "$?"' EXIT
  trap 'cleanup_proof_container 130' INT
  trap 'cleanup_proof_container 143' TERM

  python3 "$repo_root/tools/model_ci.py" project \
    --model "$model" \
    --revision "$revision" \
    --output-dir "$projection_dir" \
    --clean \
    > "$artifacts_dir/projection.json" \
    2> >(tee "$artifacts_dir/projection.stderr.log" >&2)
  [ -f "$projection_dir/.trtmc-model-projection.json" ] || \
    die "model_ci.py did not produce a projection manifest"
  python3 - "$artifacts_dir/projection.json" <<'PY'
import json
import sys

projection = json.load(open(sys.argv[1], encoding="utf-8"))
print(
    "Projection: "
    f"model_files={projection['model_files']} "
    f"platform_files={projection['platform_files']} "
    f"excluded_model_files={projection['excluded_model_files']}"
)
PY

  local host_config_file="$work_dir/model-proof-config-host.txt"
  write_model_proof_selection \
    "$projection_dir" "$artifacts_dir/selection.json" "$host_config_file"
  local -a cache_check_models=()
  mapfile -t cache_check_models < <(sed -n 's/^e2e_model=//p' "$host_config_file")
  [ "${#cache_check_models[@]}" -gt 0 ] || \
    die "could not resolve an E2E model for cache validation"
  printf '%s\n' "${cache_check_models[@]}" > "$artifacts_dir/cache-check-models.txt"

  local image="${TRTMC_CI_IMAGE:-trtmc-dev-gb300:manylinux_2_39}"
  docker image inspect "$image" >/dev/null 2>&1 || die "CI image is not present: $image"
  local build_jobs="${TRTMC_MODEL_PROOF_BUILD_JOBS:-8}"
  [[ "$build_jobs" =~ ^[1-9][0-9]*$ ]] || \
    die "TRTMC_MODEL_PROOF_BUILD_JOBS must be a positive integer"

  local hf_cache_root="${TRTMC_HF_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}}"
  local hf_hub_cache="${TRTMC_HF_HUB_CACHE:-$hf_cache_root/hub}"
  local hf_modules_cache="${TRTMC_HF_MODULES_CACHE:-$hf_cache_root/modules}"
  hf_hub_cache="$(python3 - "$hf_hub_cache" <<'PY'
import sys
from pathlib import Path
print(Path(sys.argv[1]).resolve())
PY
  )"
  hf_modules_cache="$(python3 - "$hf_modules_cache" <<'PY'
import sys
from pathlib import Path
print(Path(sys.argv[1]).resolve())
PY
  )"
  [ "$hf_hub_cache" != "/" ] || die "refusing to use / as the HF Hub cache"
  [ "$hf_hub_cache" != "$repo_root" ] || \
    die "HF Hub cache cannot be the checkout"
  [ "$hf_modules_cache" != "/" ] || \
    die "refusing to use / as the HF modules cache"
  [ "$hf_modules_cache" != "$repo_root" ] || \
    die "HF modules cache cannot be the checkout"

  # Do not preflight these paths as the unprivileged Actions runner. The
  # persistent cache parent can intentionally be non-traversable while the
  # Docker daemon is still authorized to bind its read-only hub/modules
  # children. Docker --mount fails closed when either source is truly absent,
  # and the network-disabled strict cache check below proves readability.

  local container_name="trtmc-model-proof-${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-0}-$model"
  container_name="${container_name//_/-}"
  local cache_check_container_name="${container_name}-cache-check"
  proof_container_name="$cache_check_container_name"
  docker rm -f "$cache_check_container_name" >/dev/null 2>&1 || true

  local -a cache_check_docker_args=(
    run --rm
    --name "$cache_check_container_name"
    --read-only
    --network none
    --cap-drop ALL
    --security-opt no-new-privileges
    --user "$(id -u):$(id -g)"
    --mount "type=bind,src=$projection_dir,dst=/src,readonly"
    --mount "type=bind,src=$artifacts_dir,dst=/artifacts,readonly"
    --mount "type=bind,src=$hf_hub_cache,dst=/hf-cache/hub,readonly"
    --mount "type=bind,src=$hf_modules_cache,dst=/hf-cache/modules,readonly"
    --tmpfs /tmp:rw,exec,nosuid,nodev,size=1g
    --workdir /src
    -e HOME=/tmp
    -e HF_HOME=/hf-cache
    -e HF_HUB_CACHE=/hf-cache/hub
    -e HUGGINGFACE_HUB_CACHE=/hf-cache/hub
    -e HF_MODULES_CACHE=/hf-cache/modules
    -e PYTHONDONTWRITEBYTECODE=1
  )
  set +e
  docker "${cache_check_docker_args[@]}" "$image" \
    /opt/venv/bin/python /src/scripts/warm_hf_cache.py \
      --models-file /artifacts/cache-check-models.txt --local-only --strict \
    2>&1 | tee "$artifacts_dir/cache-check.log"
  local cache_check_rc="${PIPESTATUS[0]}"
  set -e
  [ "$cache_check_rc" -eq 0 ] || \
    die "offline HF cache readiness check failed for $model (exit $cache_check_rc)"

  select_proof_gpu
  local gpu_id="$proof_gpu_id"
  printf '%s\n' "$gpu_id" > "$artifacts_dir/gpu-id.txt"

  proof_container_name="$container_name"
  docker rm -f "$container_name" >/dev/null 2>&1 || true

  local -a docker_args=(
    run --rm
    --name "$container_name"
    --read-only
    --network none
    --cap-drop ALL
    --security-opt no-new-privileges
    --ipc private
    --shm-size "${TRTMC_MODEL_PROOF_SHM_SIZE:-16g}"
    --gpus "device=$gpu_id"
    --user "$(id -u):$(id -g)"
    --mount "type=bind,src=$projection_dir,dst=/src,readonly"
    --mount "type=bind,src=$work_dir,dst=/work"
    --mount "type=bind,src=$artifacts_dir,dst=/artifacts"
    --tmpfs /tmp:rw,exec,nosuid,nodev,size=4g
    -e HOME=/tmp
    -e USER=trtmc-ci
    -e LOGNAME=trtmc-ci
    -e TMPDIR=/work/tmp
    -e TEMP=/work/tmp
    -e TMP=/work/tmp
    -e XDG_CACHE_HOME=/work/cache
    -e TORCHINDUCTOR_CACHE_DIR=/work/torch-cache
    -e HF_HUB_OFFLINE=1
    -e TRANSFORMERS_OFFLINE=1
    -e PYTHONHASHSEED=0
    -e TRTMC_MODEL_PLUGIN_STRICT=1
    -e "TRTMC_MODEL_PROOF_GPU_ID=$gpu_id"
    -e "TRTMC_MODEL_PROOF_BUILD_JOBS=$build_jobs"
  )
  docker_args+=(
    --mount "type=bind,src=$hf_hub_cache,dst=/hf-cache/hub,readonly"
    --mount "type=bind,src=$hf_modules_cache,dst=/hf-cache/modules,readonly"
    -e HF_HOME=/hf-cache
    -e HF_HUB_CACHE=/hf-cache/hub
    -e HUGGINGFACE_HUB_CACHE=/hf-cache/hub
    -e HF_MODULES_CACHE=/hf-cache/modules
  )

  set +e
  docker "${docker_args[@]}" "$image" \
    bash /src/.github/scripts/run-model-proof.sh \
      --inner --model "$model" --suite "$suite" \
      --revision "$revision" --output-dir /artifacts \
    2>&1 | tee "$artifacts_dir/console.log"
  local rc="${PIPESTATUS[0]}"
  set -e
  [ "$rc" -eq 0 ] || die "isolated model proof failed for $model (exit $rc)"
  [ -f "$artifacts_dir/proof.json" ] || die "model proof did not emit proof.json"
  [ -f "$artifacts_dir/model-proof-report.html" ] || \
    die "model proof did not emit model-proof-report.html"
  echo "Model proof artifacts: $artifacts_dir"
}

if [ "$inner" -eq 1 ]; then
  run_inner
else
  run_host
fi

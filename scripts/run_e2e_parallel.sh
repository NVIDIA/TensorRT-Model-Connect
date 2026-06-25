#!/usr/bin/env bash
# Run all E2E tests in parallel across available GPUs.
#
# Each GPU runs multiple concurrent pytest workers (default 4) to maximize
# throughput.  Models are distributed using a size-aware scheduler that
# interleaves large and small models so each GPU runs a balanced mix.
#
# Usage (inside container):
#   ./scripts/run_e2e_parallel.sh --rebuild-engines
#   ./scripts/run_e2e_parallel.sh --engine-dir /path/to/engines --hf-python /opt/venv/bin/python
#   ./scripts/run_e2e_parallel.sh --task-strategy text_generation_causal --rebuild-engines
#
# Usage (from host):
#   docker exec trtmc-dev-gb300 bash -c \
#     "cd /workspace/tensorrt-model-connect && ./scripts/run_e2e_parallel.sh --rebuild-engines"
#
# CLI options (override defaults):
#   --engine-dir PATH        Engine/bundle storage  (default: /workspace/users/yifeif/tensorrt-model-connect/engines)
#   --result-dir PATH        Test output directory  (default: /workspace/users/yifeif/tensorrt-model-connect/test-result)
#   --trtmc-binary PATH       Path to trtmc binary    (default: ./build/trtmc)
#   --hf-python PATH         Python with HF deps    (default: /opt/venv/bin/python)
#   --num-gpus N             Number of GPUs to use  (default: auto-detect)
#   --workers-per-gpu N      Concurrent workers per GPU (default: 4)
#   --task-strategy STR      Filter by task strategy
#   --exclude-ci-tier STR    Exclude manifests with this ci_tier in full mode
#   --tests-file PATH        Pytest node IDs to run (one per line)
#   --progress-interval N    Progress print interval in seconds (default: 30)
#   All other args are passed through to pytest (e.g., --rebuild-engines)
#
# Environment variables (lower priority than CLI):
#   ENGINE_DIR, RESULT_DIR, NUM_GPUS, WORKERS_PER_GPU, TRTMC_BINARY, HF_PYTHON
#   TRTMC_E2E_EXCLUDE_GPU0, TRTMC_E2E_DEPRIORITIZE_GPU0, TRTMC_E2E_MANIFEST_DIR

set -euo pipefail

# --- Configuration -----------------------------------------------------------

ENGINE_DIR="${ENGINE_DIR:-/workspace/users/yifeif/tensorrt-model-connect/engines}"
RESULT_DIR="${RESULT_DIR:-/workspace/users/yifeif/tensorrt-model-connect/test-result}"
TRTMC_BINARY="${TRTMC_BINARY:-./build/trtmc}"
HF_PYTHON="${HF_PYTHON:-/opt/venv/bin/python}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-4}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-30}"
MANIFEST_DIR="${TRTMC_E2E_MANIFEST_DIR:-tests/e2e/models}"

# Auto-detect GPUs if not specified
if [ -z "${NUM_GPUS:-}" ]; then
    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
    if [ "$NUM_GPUS" -eq 0 ]; then
        echo "ERROR: No GPUs detected. Set NUM_GPUS=1 to run on CPU (will likely fail)." >&2
        exit 1
    fi
fi

# Passthrough args (e.g., --rebuild-engines, --task-strategy ...)
EXTRA_ARGS=()
FILTER_ARGS=()
COLLECT_ARGS=()
MODELS_FILE=""
TESTS_FILE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --engine-dir)         ENGINE_DIR="$2"; shift 2 ;;
        --result-dir)         RESULT_DIR="$2"; shift 2 ;;
        --trtmc-binary)        TRTMC_BINARY="$2"; shift 2 ;;
        --hf-python)          HF_PYTHON="$2"; shift 2 ;;
        --num-gpus)           NUM_GPUS="$2"; shift 2 ;;
        --workers-per-gpu)    WORKERS_PER_GPU="$2"; shift 2 ;;
        --progress-interval)  PROGRESS_INTERVAL="$2"; shift 2 ;;
        --task-strategy)
            FILTER_ARGS+=(--e2e-task-strategy "$2")
            shift 2
            ;;
        --exclude-ci-tier)
            COLLECT_ARGS+=(--e2e-exclude-ci-tier "$2")
            shift 2
            ;;
        --models-file)
            MODELS_FILE="$2"
            shift 2
            ;;
        --tests-file)
            TESTS_FILE="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

E2E_SELECTION_ARGS=(--e2e-group-by-bundle)
if [ -n "$MODELS_FILE" ]; then
    E2E_SELECTION_ARGS+=(--e2e-models-file "$MODELS_FILE")
fi

PHYSICAL_GPU_COUNT="$NUM_GPUS"
GPU_IDS=()
for ((gpu_id = 0; gpu_id < PHYSICAL_GPU_COUNT; gpu_id++)); do
    if CUDA_VISIBLE_DEVICES="$gpu_id" "$HF_PYTHON" - <<'PY' >/dev/null 2>&1
import tensorrt as trt

logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
if builder is None:
    raise SystemExit(1)
PY
    then
        GPU_IDS+=("$gpu_id")
    else
        echo "WARN: GPU $gpu_id failed TensorRT builder health check; excluding from E2E schedule." >&2
    fi
done

if [ "${#GPU_IDS[@]}" -eq 0 ]; then
    echo "ERROR: No GPUs passed TensorRT builder health check." >&2
    exit 1
fi

EXCLUDE_GPU0="${TRTMC_E2E_EXCLUDE_GPU0:-}"
if [ -z "$EXCLUDE_GPU0" ]; then
    if [ -n "${GITHUB_RUN_ID:-}" ]; then
        EXCLUDE_GPU0=1
    else
        EXCLUDE_GPU0=0
    fi
fi

if [ "$EXCLUDE_GPU0" != "0" ] && [ "${#GPU_IDS[@]}" -gt 1 ]; then
    FILTERED_GPU_IDS=()
    for gpu_id in "${GPU_IDS[@]}"; do
        if [ "$gpu_id" != "0" ]; then
            FILTERED_GPU_IDS+=("$gpu_id")
        fi
    done
    if [ "${#FILTERED_GPU_IDS[@]}" -gt 0 ] \
        && [ "${#FILTERED_GPU_IDS[@]}" -lt "${#GPU_IDS[@]}" ]; then
        # The shared GitHub GB300 runner can lose communication during large
        # TensorRT compiler builds on physical GPU0. In CI, keep E2E parallel on
        # the remaining GPUs instead of risking a runner-level disconnect.
        GPU_IDS=("${FILTERED_GPU_IDS[@]}")
        echo "INFO: Excluding physical GPU 0 from E2E worker assignment (${GPU_IDS[*]})." \
            >&2
    fi
fi

if [ "${TRTMC_E2E_DEPRIORITIZE_GPU0:-1}" != "0" ] \
    && [ "${#GPU_IDS[@]}" -gt 1 ] \
    && [ "${GPU_IDS[0]}" = "0" ]; then
    # Shared GB300 runners can have physical GPU0 pass a lightweight builder
    # probe but fail the first large TensorRT compiler build. Keep it available,
    # while assigning the scheduler's first exclusive build bucket elsewhere.
    GPU_IDS=("${GPU_IDS[@]:1}" "${GPU_IDS[0]}")
    echo "INFO: Scheduling physical GPU 0 last for E2E worker assignment (${GPU_IDS[*]})." \
        >&2
fi
NUM_GPUS="${#GPU_IDS[@]}"

# --- Setup --------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

mkdir -p "$RESULT_DIR" "$ENGINE_DIR"
rm -f "$RESULT_DIR"/console-gpu*-w*.log \
      "$RESULT_DIR"/junit-gpu*-w*.xml \
      "$RESULT_DIR"/junit.xml

echo "=== E2E Parallel Test Runner ==="
echo "  GPUs:            $NUM_GPUS healthy of $PHYSICAL_GPU_COUNT (${GPU_IDS[*]})"
echo "  Workers/GPU:     $WORKERS_PER_GPU"
echo "  Engines:         $ENGINE_DIR"
echo "  Results:         $RESULT_DIR"
echo "  Binary:          $TRTMC_BINARY"
echo "  HF Python:       $HF_PYTHON"
echo "  Progress every:  ${PROGRESS_INTERVAL}s"
echo "  Extra args:      ${EXTRA_ARGS[*]:-none}"
echo "  Filter:          ${FILTER_ARGS[*]:-all models}"
echo "  Collect args:    ${COLLECT_ARGS[*]:-none}"
echo "  Manifests:       $MANIFEST_DIR"
echo "  Selection args:  ${E2E_SELECTION_ARGS[*]}"
echo "  Models file:     ${MODELS_FILE:-none (collect all)}"
echo "  Tests file:      ${TESTS_FILE:-none}"
echo ""

# --- Collect test IDs ---------------------------------------------------------

if [ -n "$MODELS_FILE" ] && [ ! -f "$MODELS_FILE" ]; then
    echo "ERROR: Models file not found: $MODELS_FILE" >&2
    exit 1
fi

if [ -n "$TESTS_FILE" ] && [ ! -f "$TESTS_FILE" ]; then
    echo "ERROR: Tests file not found: $TESTS_FILE" >&2
    exit 1
fi

if [ -n "$TESTS_FILE" ]; then
    # Selective mode: read pytest node IDs from file (one per line).
    TESTS=$(sed '/^$/d' "$TESTS_FILE" | sort)
else
    # Full/selective model mode: collect only canonical model E2E files. Helper
    # tests and benchmark scripts under tests/e2e/models may require optional
    # deps that are unrelated to the model schedule.
    mapfile -t E2E_COLLECT_FILES < <(
        find tests/e2e/models -mindepth 2 -maxdepth 2 -type f \
            -name 'test_*_e2e.py' | sort
    )
    if [ "${#E2E_COLLECT_FILES[@]}" -eq 0 ]; then
        echo "ERROR: No model E2E collection files found." >&2
        exit 1
    fi
    COLLECTED_TESTS=$(
        "$HF_PYTHON" -m pytest "${E2E_COLLECT_FILES[@]}" --co -q \
            "${FILTER_ARGS[@]}" "${COLLECT_ARGS[@]}" "${E2E_SELECTION_ARGS[@]}"
    )
    TESTS=$(printf "%s\n" "$COLLECTED_TESTS" | grep "test_model_e2e\[" | sort || true)
fi

TOTAL=$(printf "%s\n" "$TESTS" | sed '/^$/d' | wc -l)

if [ "$TOTAL" -eq 0 ]; then
    echo "ERROR: No E2E entries collected. Check --task-strategy filter." >&2
    exit 1
fi

echo "Collected $TOTAL E2E scheduler entries"

# --- Schedule tests across GPUs × workers ------------------------------------

SCHEDULE_JSON="$RESULT_DIR/schedule.json"
SCHEDULE_ARGS=(
    --num-gpus "$NUM_GPUS"
    --workers-per-gpu "$WORKERS_PER_GPU"
    --split-exclusive-phases
    --manifest-dir "$MANIFEST_DIR"
)
echo "$TESTS" | python "$SCRIPT_DIR/schedule_e2e.py" "${SCHEDULE_ARGS[@]}" \
    > "$SCHEDULE_JSON"

echo ""

# --- Helpers ------------------------------------------------------------------

format_duration() {
    local total="$1"
    local h=$(( total / 3600 ))
    local m=$(( (total % 3600) / 60 ))
    local s=$(( total % 60 ))
    if [ "$h" -gt 0 ]; then
        printf "%dh %02dm %02ds" "$h" "$m" "$s"
    else
        printf "%dm %02ds" "$m" "$s"
    fi
}

collect_test_progress() {
    local done=0 pass=0 fail=0 skip=0 xfail=0 xpass=0
    local files=()
    local f status

    for f in "$RESULT_DIR"/console-gpu*-w*.log; do
        [ -f "$f" ] && files+=("$f")
    done

    if [ "${#files[@]}" -eq 0 ]; then
        echo "0 0 0 0 0 0"
        return
    fi

    while IFS= read -r status; do
        [ -z "$status" ] && continue
        done=$((done + 1))
        case "$status" in
            PASSED) pass=$((pass + 1)) ;;
            SKIPPED) skip=$((skip + 1)) ;;
            XFAIL) xfail=$((xfail + 1)) ;;
            XPASS) xpass=$((xpass + 1)) ;;
            FAILED|ERROR) fail=$((fail + 1)) ;;
        esac
    done < <(
        awk '
            /test_(model_)?e2e\[/ {
                node = ""
                status = ""
                for (i = 1; i <= NF; i++) {
                    if ($i ~ /^tests\/.*::test_(model_)?e2e\[/) {
                        node = $i
                    }
                    if ($i ~ /^(PASSED|FAILED|SKIPPED|ERROR|XFAIL|XPASS)$/) {
                        status = $i
                    }
                }
                if (node != "" && status != "") {
                    status_by_node[node] = status
                }
            }
            END {
                for (node in status_by_node) {
                    print status_by_node[node]
                }
            }
        ' "${files[@]}" 2>/dev/null || true
    )

    echo "$done $pass $fail $skip $xfail $xpass"
}

print_progress() {
    local workers_done="$1"
    local workers_running="$2"
    local elapsed now done pass fail skip xfail xpass pct eta
    local eta_str=""

    now=$(date +%s)
    elapsed=$(( now - START_TIME ))
    read -r done pass fail skip xfail xpass < <(collect_test_progress)
    pct=$(awk -v d="$done" -v t="$TOTAL" 'BEGIN { if (t == 0) printf "0.0"; else printf "%.1f", (100.0 * d / t) }')

    if [ "$done" -gt 0 ] && [ "$done" -lt "$TOTAL" ]; then
        eta=$(( elapsed * (TOTAL - done) / done ))
        eta_str=" | ETA $(format_duration "$eta")"
    fi

    echo "[progress $(date +%H:%M:%S)] entries ${done}/${TOTAL} (${pct}%) pass=${pass} fail=${fail} skip=${skip} xfail=${xfail} xpass=${xpass} | ${CURRENT_PHASE} workers ${workers_done}/${TOTAL_WORKERS} done, ${workers_running} running | elapsed $(format_duration "$elapsed")${eta_str}"
}

run_schedule_phase() {
    local phase_idx="$1"
    local phase_name
    phase_name=$(python - "$SCHEDULE_JSON" "$phase_idx" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    data = json.load(f)
print(data["phases"][int(sys.argv[2])]["name"])
PY
)
    CURRENT_PHASE="$phase_name"

    echo "=== E2E phase: $phase_name ==="

    local -a pids=()
    local -a worker_labels=()
    local phase_start
    phase_start=$(date +%s)

    # Parse schedule JSON and launch one pytest per worker slot.
    # jq-free: use Python to emit "gpu_id worker_idx test1 test2 ..." lines.
    while IFS= read -r line; do
        GPU_ID=$(echo "$line" | cut -d' ' -f1)
        WORKER_IDX=$(echo "$line" | cut -d' ' -f2)
        WORKER_TESTS=$(echo "$line" | cut -d' ' -f3-)
        WORKER_COUNT=$(echo "$WORKER_TESTS" | wc -w)

        [ "$WORKER_COUNT" -eq 0 ] && continue

        LABEL="gpu${GPU_ID}-${phase_name}-w${WORKER_IDX}"
        PHYSICAL_GPU_ID="${GPU_IDS[$GPU_ID]}"
        echo "  $LABEL: $WORKER_COUNT entries"

        (
            export CUDA_VISIBLE_DEVICES=$PHYSICAL_GPU_ID
            # shellcheck disable=SC2086
            "$HF_PYTHON" -m pytest $WORKER_TESTS -v \
                --engine-dir "$ENGINE_DIR" \
                --trtmc-binary "$TRTMC_BINARY" \
                --hf-python "$HF_PYTHON" \
                --e2e-artifacts-dir "$RESULT_DIR/artifacts" \
                --junitxml="$RESULT_DIR/junit-${LABEL}.xml" \
                "${E2E_SELECTION_ARGS[@]}" \
                "${EXTRA_ARGS[@]}" \
                > "$RESULT_DIR/console-${LABEL}.log" 2>&1
        ) &
        pids+=($!)
        worker_labels+=("$LABEL")
        ALL_WORKER_LABELS+=("$LABEL")

    done < <(python - "$SCHEDULE_JSON" "$phase_idx" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    data = json.load(f)
schedule = data["phases"][int(sys.argv[2])]["schedule"]
for gpu_id in sorted(schedule, key=int):
    for w_idx, tests in enumerate(schedule[gpu_id]):
        print(f'{gpu_id} {w_idx} {" ".join(tests)}')
PY
)

    TOTAL_WORKERS=${#pids[@]}
    if [ "$TOTAL_WORKERS" -eq 0 ]; then
        echo "No workers in phase $phase_name"
        return 0
    fi

    echo ""
    echo "Workers launched: $TOTAL_WORKERS (PIDs: ${pids[*]})"
    echo "Logs: $RESULT_DIR/console-gpu*-w*.log"
    echo "  (live output suppressed to avoid interleaving; tail -f a log to watch)"
    echo "Waiting for phase $phase_name workers..."
    echo ""

    local phase_failures=0
    local workers_done=0
    local last_progress_ts=0
    local -a worker_finished=()
    local i pid rc running_now now_ts log summary
    for i in "${!pids[@]}"; do
        worker_finished[$i]=0
    done

    while [ "$workers_done" -lt "$TOTAL_WORKERS" ]; do
        running_now=0
        for i in "${!pids[@]}"; do
            if [ "${worker_finished[$i]}" -eq 1 ]; then
                continue
            fi

            pid="${pids[$i]}"
            if kill -0 "$pid" 2>/dev/null; then
                running_now=$((running_now + 1))
                continue
            fi

            if wait "$pid"; then
                rc=0
            else
                rc=$?
            fi

            worker_finished[$i]=1
            workers_done=$((workers_done + 1))
            if [ "$rc" -ne 0 ]; then
                phase_failures=$((phase_failures + 1))
                FAILURES=$((FAILURES + 1))
                echo "  ${worker_labels[$i]}: FAILED (exit code $rc)"
            else
                echo "  ${worker_labels[$i]}: OK"
            fi

            log="$RESULT_DIR/console-${worker_labels[$i]}.log"
            summary=$(grep -E "^=+ .* in .* =+$" "$log" | tail -1 || true)
            [ -n "$summary" ] && echo "    $summary"
        done

        now_ts=$(date +%s)
        if [ "$last_progress_ts" -eq 0 ] || [ $(( now_ts - last_progress_ts )) -ge "$PROGRESS_INTERVAL" ] || [ "$running_now" -eq 0 ]; then
            print_progress "$workers_done" "$running_now"
            last_progress_ts="$now_ts"
        fi

        [ "$workers_done" -lt "$TOTAL_WORKERS" ] && sleep 5
    done

    local phase_end phase_elapsed phase_minutes phase_seconds_rem
    phase_end=$(date +%s)
    phase_elapsed=$(( phase_end - phase_start ))
    phase_minutes=$(( phase_elapsed / 60 ))
    phase_seconds_rem=$(( phase_elapsed % 60 ))

    echo ""
    echo "=== Phase $phase_name finished in ${phase_minutes}m ${phase_seconds_rem}s ==="

    return "$phase_failures"
}

run_pipelined_exclusive_shared() {
    CURRENT_PHASE="pipelined"

    echo "=== E2E pipelined phases: exclusive_gpu -> shared ==="
    echo "Exclusive workers keep whole-GPU isolation; each GPU starts its shared workers as soon as its exclusive queue finishes."
    echo ""

    local -a pids=()
    local -a worker_labels=()
    local -a worker_gpu_ids=()
    local -a worker_phase_names=()
    local -a worker_finished=()
    local -a shared_started=()
    local planned_workers
    planned_workers=$(python - "$SCHEDULE_JSON" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    data = json.load(f)
print(sum(len(workers) for phase in data["phases"] for workers in phase["schedule"].values()))
PY
)
    TOTAL_WORKERS="$planned_workers"

    launch_worker() {
        local phase_name="$1"
        local gpu_id="$2"
        local worker_idx="$3"
        local worker_tests="$4"
        local worker_count label physical_gpu_id

        worker_count=$(echo "$worker_tests" | wc -w)
        [ "$worker_count" -eq 0 ] && return 0

        label="gpu${gpu_id}-${phase_name}-w${worker_idx}"
        physical_gpu_id="${GPU_IDS[$gpu_id]}"
        echo "  $label: $worker_count entries"

        (
            export CUDA_VISIBLE_DEVICES=$physical_gpu_id
            # shellcheck disable=SC2086
            "$HF_PYTHON" -m pytest $worker_tests -v \
                --engine-dir "$ENGINE_DIR" \
                --trtmc-binary "$TRTMC_BINARY" \
                --hf-python "$HF_PYTHON" \
                --e2e-artifacts-dir "$RESULT_DIR/artifacts" \
                --junitxml="$RESULT_DIR/junit-${label}.xml" \
                "${E2E_SELECTION_ARGS[@]}" \
                "${EXTRA_ARGS[@]}" \
                > "$RESULT_DIR/console-${label}.log" 2>&1
        ) &
        pids+=($!)
        worker_labels+=("$label")
        worker_gpu_ids+=("$gpu_id")
        worker_phase_names+=("$phase_name")
        worker_finished+=(0)
        ALL_WORKER_LABELS+=("$label")
    }

    launch_shared_for_gpu() {
        local gpu_id="$1"
        if [ "${shared_started[$gpu_id]:-0}" -eq 1 ]; then
            return 0
        fi
        shared_started[$gpu_id]=1

        while IFS= read -r line; do
            local worker_idx worker_tests
            worker_idx=$(echo "$line" | cut -d' ' -f1)
            worker_tests=$(echo "$line" | cut -d' ' -f2-)
            launch_worker "shared" "$gpu_id" "$worker_idx" "$worker_tests"
        done < <(python - "$SCHEDULE_JSON" "$gpu_id" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    data = json.load(f)
schedule = data["phases"][1]["schedule"]
for w_idx, tests in enumerate(schedule.get(sys.argv[2], [])):
    print(f'{w_idx} {" ".join(tests)}')
PY
)
    }

    local -a exclusive_gpu_ids=()
    while IFS= read -r line; do
        GPU_ID=$(echo "$line" | cut -d' ' -f1)
        WORKER_IDX=$(echo "$line" | cut -d' ' -f2)
        WORKER_TESTS=$(echo "$line" | cut -d' ' -f3-)
        exclusive_gpu_ids+=("$GPU_ID")
        launch_worker "exclusive_gpu" "$GPU_ID" "$WORKER_IDX" "$WORKER_TESTS"
    done < <(python - "$SCHEDULE_JSON" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    data = json.load(f)
schedule = data["phases"][0]["schedule"]
for gpu_id in sorted(schedule, key=int):
    for w_idx, tests in enumerate(schedule[gpu_id]):
        print(f'{gpu_id} {w_idx} {" ".join(tests)}')
PY
)

    while IFS= read -r GPU_ID; do
        local has_exclusive=0
        for exclusive_gpu_id in "${exclusive_gpu_ids[@]}"; do
            if [ "$exclusive_gpu_id" = "$GPU_ID" ]; then
                has_exclusive=1
                break
            fi
        done
        if [ "$has_exclusive" -eq 0 ]; then
            launch_shared_for_gpu "$GPU_ID"
        fi
    done < <(python - "$SCHEDULE_JSON" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    data = json.load(f)
for gpu_id in sorted(data["phases"][1]["schedule"], key=int):
    print(gpu_id)
PY
)

    if [ "${#pids[@]}" -eq 0 ]; then
        echo "No workers in pipelined E2E schedule"
        return 0
    fi

    echo ""
    echo "Workers planned: $planned_workers"
    echo "Initial workers launched: ${#pids[@]} (PIDs: ${pids[*]})"
    echo "Logs: $RESULT_DIR/console-gpu*-w*.log"
    echo "  (live output suppressed to avoid interleaving; tail -f a log to watch)"
    echo "Waiting for pipelined workers..."
    echo ""

    local pipeline_failures=0
    local workers_done=0
    local last_progress_ts=0
    local i pid rc running_now now_ts log summary

    while [ "$workers_done" -lt "$planned_workers" ]; do
        running_now=0
        for i in "${!pids[@]}"; do
            if [ "${worker_finished[$i]}" -eq 1 ]; then
                continue
            fi

            pid="${pids[$i]}"
            if kill -0 "$pid" 2>/dev/null; then
                running_now=$((running_now + 1))
                continue
            fi

            if wait "$pid"; then
                rc=0
            else
                rc=$?
            fi

            worker_finished[$i]=1
            workers_done=$((workers_done + 1))
            if [ "$rc" -ne 0 ]; then
                pipeline_failures=$((pipeline_failures + 1))
                FAILURES=$((FAILURES + 1))
                echo "  ${worker_labels[$i]}: FAILED (exit code $rc)"
            else
                echo "  ${worker_labels[$i]}: OK"
            fi

            log="$RESULT_DIR/console-${worker_labels[$i]}.log"
            summary=$(grep -E "^=+ .* in .* =+$" "$log" | tail -1 || true)
            [ -n "$summary" ] && echo "    $summary"

            if [ "${worker_phase_names[$i]}" = "exclusive_gpu" ]; then
                launch_shared_for_gpu "${worker_gpu_ids[$i]}"
            fi
        done

        now_ts=$(date +%s)
        if [ "$last_progress_ts" -eq 0 ] || [ $(( now_ts - last_progress_ts )) -ge "$PROGRESS_INTERVAL" ] || [ "$running_now" -eq 0 ]; then
            print_progress "$workers_done" "$running_now"
            last_progress_ts="$now_ts"
        fi

        [ "$workers_done" -lt "$planned_workers" ] && sleep 5
    done

    echo ""
    echo "=== Pipelined exclusive/shared schedule finished ==="

    return "$pipeline_failures"
}

# --- Launch workers and collect exit codes ------------------------------------

START_TIME=$(date +%s)
declare -a ALL_WORKER_LABELS=()
FAILURES=0
TOTAL_WORKERS=0
CURRENT_PHASE="startup"

PHASE_COUNT=$(python - "$SCHEDULE_JSON" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    data = json.load(f)
print(len(data["phases"]))
PY
)

if python - "$SCHEDULE_JSON" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    data = json.load(f)
names = [phase["name"] for phase in data["phases"]]
raise SystemExit(0 if names == ["exclusive_gpu", "shared"] else 1)
PY
then
    if ! run_pipelined_exclusive_shared; then
        echo "Stopping after failed pipelined E2E schedule"
    fi
else
    for ((phase_idx = 0; phase_idx < PHASE_COUNT; phase_idx++)); do
        if ! run_schedule_phase "$phase_idx"; then
            echo "Stopping after failed E2E phase: $CURRENT_PHASE"
            break
        fi
    done
fi

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
MINUTES=$(( ELAPSED / 60 ))
SECONDS_REM=$(( ELAPSED % 60 ))

echo ""
echo "=== All E2E phases finished in ${MINUTES}m ${SECONDS_REM}s ==="

# --- Merge JUnit XMLs --------------------------------------------------------

python -c "
from junitparser import JUnitXml
import glob, sys
files = sorted(glob.glob('$RESULT_DIR/junit-gpu*-w*.xml'))
if not files:
    print('No JUnit XML files found to merge.')
    sys.exit(0)
merged = JUnitXml()
for f in files:
    try:
        merged += JUnitXml.from_file(f)
    except Exception as e:
        print(f'Warning: could not parse {f}: {e}')
merged.write('$RESULT_DIR/junit.xml')
t = sum(1 for _ in merged)
print(f'Merged {len(files)} files -> $RESULT_DIR/junit.xml')
print(f'Total: {t} tests')
" 2>/dev/null || echo "(install junitparser to auto-merge: pip install junitparser)"

python3 - "$RESULT_DIR/artifacts" <<'PY' || true
import json
import sys
from pathlib import Path

artifacts_dir = Path(sys.argv[1])
if not artifacts_dir.is_dir():
    raise SystemExit(0)

rows = []
aggregate = {
    "bundle_build_s": 0.0,
    "trt_compile_s": 0.0,
    "trt_load_deserialization_s": 0.0,
    "reference_s": 0.0,
    "inference_s": 0.0,
    "comparison_s": 0.0,
}
for result_path in sorted(artifacts_dir.glob("*/result.json")):
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        continue
    timing = result.get("timing") or {}
    detailed = result.get("detailed_timing") or {}
    row = {
        "case": result.get("case_name") or result_path.parent.name,
        "status": result.get("status") or "",
        "bundle_build_s": float(timing.get("bundle_build_s") or 0.0),
        "trt_compile_s": float(detailed.get("trt_compile_s") or 0.0),
        "trt_load_deserialization_s": float(
            detailed.get("trt_load_deserialization_s") or 0.0),
        "reference_s": float(detailed.get("reference_s") or 0.0),
        "inference_s": float(detailed.get("inference_s") or 0.0),
        "comparison_s": float(detailed.get("comparison_s") or 0.0),
    }
    row["accounted_s"] = (
        row["bundle_build_s"]
        + row["trt_load_deserialization_s"]
        + row["reference_s"]
        + row["inference_s"]
        + row["comparison_s"]
    )
    rows.append(row)
    for key in aggregate:
        aggregate[key] += row[key]

if not rows:
    raise SystemExit(0)

summary_path = artifacts_dir.parent / "timing-summary.json"
summary_path.write_text(
    json.dumps({"aggregate_s": aggregate, "cases": rows}, indent=2, sort_keys=True),
    encoding="utf-8",
)

print("")
print("--- E2E timing bottlenecks ---")
print("Aggregate accounted time:")
for key, value in sorted(aggregate.items(), key=lambda item: -item[1]):
    print(f"  {key}: {value:.1f}s ({value / 60:.1f}m)")
print("Top cases by accounted time:")
for row in sorted(rows, key=lambda item: -item["accounted_s"])[:20]:
    print(
        "  {case:36s} {status:5s} total={total:7.1f}s "
        "build={build:7.1f}s compile={compile:7.1f}s "
        "load={load:6.1f}s ref={ref:6.1f}s infer={infer:5.1f}s".format(
            case=row["case"][:36],
            status=row["status"],
            total=row["accounted_s"],
            build=row["bundle_build_s"],
            compile=row["trt_compile_s"],
            load=row["trt_load_deserialization_s"],
            ref=row["reference_s"],
            infer=row["inference_s"],
        )
    )
print(f"Timing summary: {summary_path}")
PY

# --- Summary ------------------------------------------------------------------

echo ""
echo "Output files:"
echo "  Schedule:      $RESULT_DIR/schedule.json"
echo "  Console logs:  $RESULT_DIR/console-gpu*-w*.log"
echo "  JUnit XML:     $RESULT_DIR/junit-gpu*-w*.xml (merged: $RESULT_DIR/junit.xml)"
echo "  Timing JSON:   $RESULT_DIR/timing-summary.json"
echo "  Artifacts:     $RESULT_DIR/artifacts/"
echo ""

# Per-test results from each worker
echo "--- Per-test results ---"
for LABEL in "${ALL_WORKER_LABELS[@]}"; do
    LOG="$RESULT_DIR/console-${LABEL}.log"
    [ -f "$LOG" ] || continue
    echo ""
    echo "  [$LABEL]"
    grep -E "PASSED|FAILED|SKIPPED|ERROR" "$LOG" \
        | grep -E "^tests/" \
        | sed 's/^/    /' || echo "    (no test results found)"
    SUMMARY=$(grep -E "^=" "$LOG" | tail -1)
    [ -n "$SUMMARY" ] && echo "    $SUMMARY"
done

# Print failure details so CI console shows root causes
if [ "$FAILURES" -gt 0 ]; then
    echo ""
    echo "--- Failure details ---"
    for LABEL in "${ALL_WORKER_LABELS[@]}"; do
        LOG="$RESULT_DIR/console-${LABEL}.log"
        [ -f "$LOG" ] || continue
        python -c "
import re, sys

log = open('$LOG').read()
failed = re.findall(r'tests/.+?::test_(?:model_)?e2e\[(.+?)\] FAILED', log)
if not failed:
    sys.exit(0)

# Extract the FAILURES section
failures_match = re.search(r'=+ FAILURES =+\n(.+?)(?=\n=+ )', log, re.DOTALL)
if not failures_match:
    for name in failed:
        print(f'  [{name}]')
        for line in log.splitlines():
            if 'E2E failed for' in line and name in line:
                print(f'    {line.strip()}')
                break
            if 'Failed:' in line and name in line:
                print(f'    {line.strip()}')
                break
        else:
            print(f'    (no detail found — check console log)')
    sys.exit(0)

failures_text = failures_match.group(1)
blocks = re.split(r'_+ test_(?:model_)?e2e\[(.+?)\] _+\n', failures_text)
for i in range(1, len(blocks), 2):
    name = blocks[i]
    body = blocks[i + 1] if i + 1 < len(blocks) else ''
    print(f'  [{name}]')
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith('E '):
            print(f'    {stripped}')
" 2>/dev/null || true
    done
fi

echo ""
exit "$FAILURES"

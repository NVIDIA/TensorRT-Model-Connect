#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Usage: optimize_supervisor.sh MODEL_ID [MAX_ATTEMPTS] [ACCURACY_THRESHOLD] [AGENT_ID]
#
# Launches a configurable coding agent to optimize MODEL_ID for low-precision inference.
# Restarts the agent up to MAX_ATTEMPTS times until the progress file shows a
# verified passing candidate. The supervisor reads the progress file (written by
# deterministic tools), NOT the agent's self-report.

MODEL_ID="${1:?Usage: optimize_supervisor.sh MODEL_ID [MAX_ATTEMPTS] [ACCURACY] [AGENT_ID]}"
MAX_ATTEMPTS="${2:-5}"
ACCURACY="${3:-0.95}"
AGENT_ID="${4:-agent-3}"

SAFE_NAME="$(echo "$MODEL_ID" | tr '/' '_' | tr '.' '_')"
PROGRESS_FILE="/tmp/optimize_progress_${SAFE_NAME}.json"
CONTAINER="trtmc-dev-gb300-${AGENT_ID}"
REPO_DIR="/workspace/users/yifeif/workspaces/${AGENT_ID}/tensorrt-model-connect"
AGENT_BIN="${TRTMC_AGENT_BIN:-codex}"
read -r -a AGENT_ARGS <<< "${TRTMC_AGENT_ARGS:-exec -s danger-full-access -a never -C {workspace} {prompt}}"

echo "[supervisor] Model:        $MODEL_ID"
echo "[supervisor] Max attempts:  $MAX_ATTEMPTS"
echo "[supervisor] Accuracy:      $ACCURACY"
echo "[supervisor] Progress:      $PROGRESS_FILE"
echo "[supervisor] Container:     $CONTAINER"
echo ""

check_success() {
    python3 -c "
import json, sys
try:
    p = json.load(open('${PROGRESS_FILE}'))
except (FileNotFoundError, json.JSONDecodeError):
    sys.exit(1)
best = p.get('best_passing')
if not best:
    sys.exit(1)
if not best.get('verified', False):
    sys.exit(1)
if best.get('accuracy', 0) < ${ACCURACY}:
    sys.exit(1)
prec = best.get('precision', 'fp32')
quant = best.get('quantize', 'none') or 'none'
if prec == 'fp32' and quant == 'none':
    sys.exit(1)
lat = best.get('latency_ms', 0)
mem = best.get('memory_mb', 0)
acc = best.get('accuracy', 0)
print(f'precision={prec} quantize={quant} accuracy={acc:.3f} latency={lat:.1f}ms memory={mem:.0f}MB')
sys.exit(0)
" 2>/dev/null
}

attempt=0
while [ "$attempt" -lt "$MAX_ATTEMPTS" ]; do
    attempt=$((attempt + 1))
    echo "[supervisor] === Attempt $attempt/$MAX_ATTEMPTS ==="

    prompt="
You are optimizing ${MODEL_ID} for low-precision inference.

Read AGENTS.md first and follow it as the repository ground truth.
Use \$optimize-model-precision. If it is not active, read:
plugins/trtmc-agent-skills/skills/optimize-model-precision/SKILL.md

Progress file: ${PROGRESS_FILE}
Accuracy threshold: ${ACCURACY}
Container: ${CONTAINER}
Repo: ${REPO_DIR}

Read the progress file first. If it exists, resume from where the
previous agent left off. Do not repeat completed attempts.

Your goal: find the best non-FP32 precision config that passes
diff_logits validation. Write every result to the progress file.
"
    rendered_args=()
    has_prompt=false
    for arg in "${AGENT_ARGS[@]}"; do
        case "$arg" in
            "{workspace}")
                rendered_args+=("$REPO_DIR")
                ;;
            "{prompt}")
                rendered_args+=("$prompt")
                has_prompt=true
                ;;
            *)
                arg="${arg//\{workspace\}/$REPO_DIR}"
                if [[ "$arg" == *"{prompt}"* ]]; then
                    arg="${arg//\{prompt\}/$prompt}"
                    has_prompt=true
                fi
                rendered_args+=("$arg")
                ;;
        esac
    done
    if [ "$has_prompt" = false ]; then
        rendered_args+=("$prompt")
    fi
    "$AGENT_BIN" "${rendered_args[@]}" || true

    if check_success; then
        echo ""
        echo "[supervisor] SUCCESS after $attempt attempt(s):"
        check_success
        echo "[supervisor] Progress file: $PROGRESS_FILE"
        exit 0
    fi

    echo "[supervisor] No verified passing candidate yet."
    if [ "$attempt" -lt "$MAX_ATTEMPTS" ]; then
        echo "[supervisor] Restarting agent..."
        echo ""
    fi
done

echo ""
echo "[supervisor] FAILED after $MAX_ATTEMPTS attempts."
echo "[supervisor] Progress file: $PROGRESS_FILE"
exit 1

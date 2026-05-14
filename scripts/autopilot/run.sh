#!/usr/bin/env bash
# One-command autopilot: discover gaps + dispatch agents.
#
# Usage:
#   ./scripts/autopilot/run.sh                          # interactive
#   ./scripts/autopilot/run.sh --mode auto              # fully autonomous
#   ./scripts/autopilot/run.sh --mode dry-run           # preview only
#   ./scripts/autopilot/run.sh --min-downloads 100000   # only very popular models
#
# Prerequisites:
#   - Agent workspaces bootstrapped (agent-1..4)
#   - Agent CLI in PATH. Codex is the default; override with TRTMC_AGENT_BIN/TRTMC_AGENT_ARGS.
#   - HuggingFace Hub accessible (or HF_TOKEN set)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Defaults
MIN_DOWNLOADS=10000
MAX_MODELS=2000
MODE="interactive"
AGENTS=4
DISCOVER_CONTAINER="trtmc-dev-gb300-agent-1"
TASKS_FILE="/tmp/autopilot_tasks.json"

# Parse args — anything not recognized is passed through to dispatch.py
DISPATCH_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --min-downloads)    MIN_DOWNLOADS="$2"; shift 2 ;;
        --max-models)       MAX_MODELS="$2"; shift 2 ;;
        --mode)             MODE="$2"; DISPATCH_ARGS+=("--mode" "$2"); shift 2 ;;
        --agents)           AGENTS="$2"; DISPATCH_ARGS+=("--agents" "$2"); shift 2 ;;
        --discover-in)      DISCOVER_CONTAINER="$2"; shift 2 ;;
        --tasks-file)       TASKS_FILE="$2"; shift 2 ;;
        *)                  DISPATCH_ARGS+=("$1"); shift ;;
    esac
done

echo "=== trtmc Autopilot ==="
echo "  Min downloads:     ${MIN_DOWNLOADS}"
echo "  Max models to scan: ${MAX_MODELS}"
echo "  Mode:              ${MODE}"
echo "  Agents:            ${AGENTS}"
echo ""

# Step 1: Discover gaps (runs inside a container for tensorrt_model_connect imports)
echo "Step 1: Discovering unsupported model families..."
docker exec "$DISCOVER_CONTAINER" python3 scripts/autopilot/discover.py \
    --min-downloads "$MIN_DOWNLOADS" \
    --max-models "$MAX_MODELS" \
    --output "$TASKS_FILE"

TASK_COUNT=$(python3 -c "import json; d=json.load(open('$TASKS_FILE')); print(d['total_gaps'])")

if [[ "$TASK_COUNT" -eq 0 ]]; then
    echo "No gaps found. All popular model families are already supported!"
    exit 0
fi

echo ""
echo "Step 2: Dispatching $TASK_COUNT tasks..."
python3 scripts/autopilot/dispatch.py "$TASKS_FILE" "${DISPATCH_ARGS[@]}"

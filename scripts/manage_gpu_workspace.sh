#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Manage the one-worktree/one-container layout used on shared GPU hosts.
#
# This script deliberately has no delete command. Removing a container, source
# checkout, run directory, or cache requires a separate, explicitly reviewed
# operation.

set -euo pipefail

PROGRAM="$(basename "$0")"
CONTAINER_WORKDIR="/workspace/tensorrt-model-connect"
CONTAINER_RUN_ROOT="/work"
CONTAINER_HF_HOME="/cache/huggingface"
CONTAINER_DATA_ROOT="/mnt/data"
MANAGED_LABEL="com.nvidia.trtmc.managed"
WORKSPACE_LABEL="com.nvidia.trtmc.workspace-id"

usage() {
    cat <<EOF
Usage:
  $PROGRAM start ID
  $PROGRAM stop ID
  $PROGRAM shell ID
  $PROGRAM exec ID -- COMMAND [ARG ...]
  $PROGRAM inspect ID
  $PROGRAM status
  $PROGRAM audit

Required host configuration:
  TRTMC_HOST_ROOT             Canonical storage root shared by all workspaces

Optional host configuration:
  TRTMC_HOST_CONFIG           Shell config file (default: ~/.config/trtmc/host.env)
  TRTMC_DOCKER_IMAGE          Image (default: trtmc-dev-gb300:latest)
  TRTMC_CONTAINER_PREFIX      Name prefix (default: trtmc-dev-gb300)
  TRTMC_GPU_REQUEST           docker --gpus value (default: all)
  TRTMC_RESTART_POLICY        Docker restart policy (default: unless-stopped)

Canonical layout below TRTMC_HOST_ROOT:
  workspaces/ID/repo          Deployed source for exactly one worktree
  runs/ID/engines             Worktree-owned engine bundles
  runs/ID/results             Worktree-owned test and benchmark results
  runs/ID/logs                Worktree-owned logs
  runs/ID/tmp                 Worktree-owned temporary files
  state/ID/workspace.env      Generated ownership/lifecycle metadata
  huggingface/                Shared Hugging Face cache
  data/                       Shared datasets, mounted read-only
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

load_host_config() {
    local config="${TRTMC_HOST_CONFIG:-${XDG_CONFIG_HOME:-$HOME/.config}/trtmc/host.env}"
    local override_container_prefix="${TRTMC_CONTAINER_PREFIX:-}"
    local override_docker_image="${TRTMC_DOCKER_IMAGE:-}"
    local override_gpu_request="${TRTMC_GPU_REQUEST:-}"
    local override_host_root="${TRTMC_HOST_ROOT:-}"
    local override_restart_policy="${TRTMC_RESTART_POLICY:-}"
    if [ -f "$config" ]; then
        # The config is host-owned and may set any TRTMC_* value used below.
        # shellcheck source=/dev/null
        source "$config"
    fi

    [ -z "$override_container_prefix" ] || TRTMC_CONTAINER_PREFIX="$override_container_prefix"
    [ -z "$override_docker_image" ] || TRTMC_DOCKER_IMAGE="$override_docker_image"
    [ -z "$override_gpu_request" ] || TRTMC_GPU_REQUEST="$override_gpu_request"
    [ -z "$override_host_root" ] || TRTMC_HOST_ROOT="$override_host_root"
    [ -z "$override_restart_policy" ] || TRTMC_RESTART_POLICY="$override_restart_policy"

    : "${TRTMC_HOST_ROOT:?Set TRTMC_HOST_ROOT or create the host config file}"
    case "$TRTMC_HOST_ROOT" in
        /*) ;;
        *) die "TRTMC_HOST_ROOT must be an absolute path" ;;
    esac
    [ "$TRTMC_HOST_ROOT" != "/" ] || die "TRTMC_HOST_ROOT cannot be /"

    HOST_ROOT="${TRTMC_HOST_ROOT%/}"
    DOCKER_IMAGE="${TRTMC_DOCKER_IMAGE:-trtmc-dev-gb300:latest}"
    CONTAINER_PREFIX="${TRTMC_CONTAINER_PREFIX:-trtmc-dev-gb300}"
    GPU_REQUEST="${TRTMC_GPU_REQUEST:-all}"
    RESTART_POLICY="${TRTMC_RESTART_POLICY:-unless-stopped}"
}

validate_id() {
    local workspace_id="$1"
    [ "${#workspace_id}" -le 48 ] || die "workspace ID must be at most 48 characters"
    [[ "$workspace_id" =~ ^[a-z0-9][a-z0-9._-]*$ ]] \
        || die "workspace ID must match [a-z0-9][a-z0-9._-]*"
}

set_workspace_paths() {
    WORKSPACE_ID="$1"
    validate_id "$WORKSPACE_ID"
    WORKSPACE_ROOT="$HOST_ROOT/workspaces/$WORKSPACE_ID"
    REPO_DIR="$WORKSPACE_ROOT/repo"
    RUN_ROOT="$HOST_ROOT/runs/$WORKSPACE_ID"
    STATE_ROOT="$HOST_ROOT/state/$WORKSPACE_ID"
    HF_HOME_HOST="$HOST_ROOT/huggingface"
    DATA_ROOT_HOST="$HOST_ROOT/data"
    CONTAINER_NAME="$CONTAINER_PREFIX-$WORKSPACE_ID"
}

container_exists() {
    docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1
}

assert_managed_container() {
    local managed workspace
    managed="$(docker inspect --format "{{index .Config.Labels \"$MANAGED_LABEL\"}}" "$CONTAINER_NAME")"
    workspace="$(docker inspect --format "{{index .Config.Labels \"$WORKSPACE_LABEL\"}}" "$CONTAINER_NAME")"
    [ "$managed" = "true" ] \
        || die "container $CONTAINER_NAME exists but is not managed by $PROGRAM"
    [ "$workspace" = "$WORKSPACE_ID" ] \
        || die "container $CONTAINER_NAME belongs to workspace $workspace"
}

write_state() {
    local git_branch git_sha state_tmp
    git_branch="$(git -C "$REPO_DIR" branch --show-current 2>/dev/null || true)"
    git_sha="$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || true)"
    [ -n "$git_branch" ] || git_branch="unknown"
    [ -n "$git_sha" ] || git_sha="unknown"
    state_tmp="$STATE_ROOT/workspace.env.tmp"

    mkdir -p "$STATE_ROOT"
    {
        printf 'TRTMC_WORKSPACE_ID=%q\n' "$WORKSPACE_ID"
        printf 'TRTMC_CONTAINER_NAME=%q\n' "$CONTAINER_NAME"
        printf 'TRTMC_REPO_DIR=%q\n' "$REPO_DIR"
        printf 'TRTMC_RUN_ROOT=%q\n' "$RUN_ROOT"
        printf 'TRTMC_GIT_BRANCH=%q\n' "$git_branch"
        printf 'TRTMC_GIT_SHA=%q\n' "$git_sha"
        printf 'TRTMC_DOCKER_IMAGE=%q\n' "$DOCKER_IMAGE"
        printf 'TRTMC_OWNER=%q\n' "${USER:-unknown}"
        printf 'TRTMC_UPDATED_AT=%q\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    } > "$state_tmp"
    mv "$state_tmp" "$STATE_ROOT/workspace.env"
}

start_workspace() {
    local git_sha
    [ -d "$REPO_DIR" ] \
        || die "deployed repo is missing: $REPO_DIR"

    if container_exists; then
        assert_managed_container
        write_state
        if [ "$(docker inspect --format '{{.State.Running}}' "$CONTAINER_NAME")" = "true" ]; then
            echo "$CONTAINER_NAME is already running"
        else
            docker start "$CONTAINER_NAME" >/dev/null
            echo "Started existing container $CONTAINER_NAME"
        fi
        return
    fi

    mkdir -p \
        "$RUN_ROOT/engines" \
        "$RUN_ROOT/results" \
        "$RUN_ROOT/logs" \
        "$RUN_ROOT/tmp" \
        "$HF_HOME_HOST/hub" \
        "$HF_HOME_HOST/modules" \
        "$DATA_ROOT_HOST"

    write_state
    git_sha="$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"

    docker run -d \
        --name "$CONTAINER_NAME" \
        --restart "$RESTART_POLICY" \
        --gpus "$GPU_REQUEST" \
        --ipc host \
        --label "$MANAGED_LABEL=true" \
        --label "$WORKSPACE_LABEL=$WORKSPACE_ID" \
        --label "com.nvidia.trtmc.owner=${USER:-unknown}" \
        --label "com.nvidia.trtmc.git-sha=$git_sha" \
        --label "com.nvidia.trtmc.repo-dir=$REPO_DIR" \
        --label "com.nvidia.trtmc.run-root=$RUN_ROOT" \
        -v "$REPO_DIR:$CONTAINER_WORKDIR" \
        -v "$RUN_ROOT:$CONTAINER_RUN_ROOT" \
        -v "$HF_HOME_HOST:$CONTAINER_HF_HOME" \
        -v "$DATA_ROOT_HOST:$CONTAINER_DATA_ROOT:ro" \
        -e "TRTMC_WORKSPACE_ID=$WORKSPACE_ID" \
        -e "TRTMC_STORAGE_ROOT=$CONTAINER_RUN_ROOT" \
        -e "ENGINE_DIR=$CONTAINER_RUN_ROOT/engines" \
        -e "RESULT_DIR=$CONTAINER_RUN_ROOT/results" \
        -e "TMPDIR=$CONTAINER_RUN_ROOT/tmp" \
        -e "HF_HOME=$CONTAINER_HF_HOME" \
        -e "HF_HUB_CACHE=$CONTAINER_HF_HOME/hub" \
        -e "HUGGINGFACE_HUB_CACHE=$CONTAINER_HF_HOME/hub" \
        -e "HF_MODULES_CACHE=$CONTAINER_HF_HOME/modules" \
        -w "$CONTAINER_WORKDIR" \
        "$DOCKER_IMAGE" sleep infinity >/dev/null

    echo "Started $CONTAINER_NAME"
    echo "  repo: $REPO_DIR"
    echo "  run:  $RUN_ROOT"
}

stop_workspace() {
    container_exists || die "container does not exist: $CONTAINER_NAME"
    assert_managed_container
    docker stop "$CONTAINER_NAME" >/dev/null
    echo "Stopped $CONTAINER_NAME (container and files were retained)"
}

shell_workspace() {
    container_exists || die "container does not exist: $CONTAINER_NAME"
    assert_managed_container
    docker exec -it "$CONTAINER_NAME" bash
}

exec_workspace() {
    shift 2
    [ "${1:-}" = "--" ] || die "exec requires -- before the command"
    shift
    [ "$#" -gt 0 ] || die "exec requires a command"
    container_exists || die "container does not exist: $CONTAINER_NAME"
    assert_managed_container
    docker exec "$CONTAINER_NAME" "$@"
}

inspect_workspace() {
    echo "workspace_id=$WORKSPACE_ID"
    echo "repo_dir=$REPO_DIR"
    echo "run_root=$RUN_ROOT"
    if [ -d "$WORKSPACE_ROOT" ]; then
        du -sh "$WORKSPACE_ROOT" "$RUN_ROOT" 2>/dev/null || true
    fi
    if container_exists; then
        assert_managed_container
        docker inspect --format \
            'container={{.Name}} status={{.State.Status}} image={{.Config.Image}} created={{.Created}}' \
            "$CONTAINER_NAME"
    else
        echo "container=missing"
    fi
}

status_workspaces() {
    docker ps -a --size \
        --filter "label=$MANAGED_LABEL=true" \
        --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}\t{{.Size}}'
}

audit_host() {
    echo "Filesystem usage"
    df -hT "$HOST_ROOT" "$(docker info --format '{{.DockerRootDir}}')"
    echo
    echo "Managed directory usage"
    for path in workspaces runs state huggingface data; do
        if [ -e "$HOST_ROOT/$path" ]; then
            du -sh "$HOST_ROOT/$path"
        else
            echo "missing  $HOST_ROOT/$path"
        fi
    done
    echo
    echo "Managed containers"
    status_workspaces
    echo
    echo "Docker usage (reclaimable is advisory; this command does not delete it)"
    docker system df
    echo
    echo "Workspace state"
    find "$HOST_ROOT/state" -mindepth 2 -maxdepth 2 -name workspace.env -print 2>/dev/null \
        | sort || true
}

main() {
    case "${1:-}" in
        -h|--help|help|"")
            usage
            return
            ;;
    esac

    load_host_config
    case "$1" in
        start|stop|shell|exec|inspect)
            [ "$#" -ge 2 ] || die "$1 requires a workspace ID"
            set_workspace_paths "$2"
            ;;
    esac

    case "$1" in
        start) start_workspace ;;
        stop) stop_workspace ;;
        shell) shell_workspace ;;
        exec) exec_workspace "$@" ;;
        inspect) inspect_workspace ;;
        status) status_workspaces ;;
        audit) audit_host ;;
        *) die "unknown command: $1" ;;
    esac
}

main "$@"

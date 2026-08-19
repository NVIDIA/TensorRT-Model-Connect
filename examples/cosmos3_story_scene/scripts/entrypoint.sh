#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -Eeuo pipefail

readonly MODEL_ID="nvidia/Cosmos3-Nano"
readonly MODEL_REVISION="411f42a8fdfb8c5b2583cb8786e0938f49796eaa"
readonly MODEL_PRECISION="bf16"
readonly HF_SECRET_PATH="/run/secrets/hf_token"

TRTMC_BIN="${TRTMC_BIN:-trtmc}"
COSMOS3_BUNDLE="${COSMOS3_BUNDLE:-/models/cosmos3.trtfb}"
COSMOS3_CP_SIZE="${COSMOS3_CP_SIZE:-${CP_SIZE:-1}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/outputs}"

export TRTMC_BIN COSMOS3_BUNDLE COSMOS3_CP_SIZE OUTPUT_ROOT

log() {
    printf '[cosmos3-story-scene] %s\n' "$*" >&2
}

fail() {
    log "ERROR: $*"
    exit 1
}

case "${COSMOS3_CP_SIZE}" in
    1|2|4|8) ;;
    *) fail "COSMOS3_CP_SIZE must be one of 1, 2, 4, or 8" ;;
esac

command -v "${TRTMC_BIN}" >/dev/null 2>&1 || fail "TRTMC_BIN is not executable: ${TRTMC_BIN}"
command -v nvidia-smi >/dev/null 2>&1 || \
    fail "nvidia-smi is unavailable; install NVIDIA Container Toolkit and start with GPU access"

mapfile -t gpu_caps < <(
    nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | sed 's/[[:space:]]//g'
)
mapfile -t gpu_memory < <(
    nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | sed 's/[[:space:]]//g'
)

gpu_count="${#gpu_caps[@]}"
(( gpu_count > 0 )) || fail "no NVIDIA GPU is visible inside the container"
(( gpu_count >= COSMOS3_CP_SIZE )) || \
    fail "COSMOS3_CP_SIZE=${COSMOS3_CP_SIZE} requires at least ${COSMOS3_CP_SIZE} visible GPUs; found ${gpu_count}"

primary_cap="${gpu_caps[0]}"
for ((rank = 1; rank < COSMOS3_CP_SIZE; ++rank)); do
    [[ "${gpu_caps[rank]}" == "${primary_cap}" ]] || \
        fail "context-parallel ranks must use the same GPU architecture; found ${primary_cap} and ${gpu_caps[rank]}"
done

for ((rank = 0; rank < COSMOS3_CP_SIZE; ++rank)); do
    if [[ "${gpu_memory[rank]:-0}" =~ ^[0-9]+$ ]] && (( gpu_memory[rank] < 79000 )); then
        log "WARNING: GPU rank ${rank} reports ${gpu_memory[rank]} MiB; this sample is validated on 80GB-class or larger GPUs"
    fi
done

if (( COSMOS3_CP_SIZE > 1 )); then
    command -v mpirun >/dev/null 2>&1 || fail "mpirun is required when COSMOS3_CP_SIZE is greater than 1"
fi

install -d -m 0755 "$(dirname "${COSMOS3_BUNDLE}")" "${OUTPUT_ROOT}" "${HF_HUB_CACHE:-/root/.cache/huggingface/hub}"

bundle_marker="${COSMOS3_BUNDLE}.build-spec"
build_spec="model=${MODEL_ID};revision=${MODEL_REVISION};precision=${MODEL_PRECISION};cp=${COSMOS3_CP_SIZE};compute_cap=${primary_cap}"
cached_spec=""
if [[ -s "${bundle_marker}" ]]; then
    cached_spec="$(<"${bundle_marker}")"
fi

if [[ -s "${COSMOS3_BUNDLE}" && "${cached_spec}" == "${build_spec}" ]]; then
    log "reusing hardware-specific bundle ${COSMOS3_BUNDLE}"
else
    build_token="${HF_TOKEN:-}"
    if [[ -z "${build_token}" ]]; then
        build_token="${HUGGING_FACE_HUB_TOKEN:-}"
    fi
    if [[ -s "${HF_SECRET_PATH}" ]]; then
        IFS= read -r build_token < "${HF_SECRET_PATH}" || true
        [[ -n "${build_token}" ]] || fail "${HF_SECRET_PATH} is empty"
        log "using Hugging Face token from the mounted Docker secret"
    elif [[ -n "${build_token}" ]]; then
        log "WARNING: using a Hugging Face token from the environment; a Docker secret is safer"
    else
        log "downloading the public checkpoint without a Hugging Face token"
    fi

    if [[ -n "${build_token}" ]]; then
        export HF_TOKEN="${build_token}"
        export HUGGING_FACE_HUB_TOKEN="${build_token}"
    else
        unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
    fi

    temporary_bundle="${COSMOS3_BUNDLE}.tmp.$$"
    temporary_marker="${bundle_marker}.tmp.$$"
    cleanup_temporary_files() {
        rm -f -- "${temporary_bundle}" "${temporary_marker}"
    }
    trap cleanup_temporary_files EXIT INT TERM

    if [[ -e "${COSMOS3_BUNDLE}" ]]; then
        log "the cached bundle does not match revision, topology, or GPU architecture; rebuilding it atomically"
    else
        log "first start: downloading ${MODEL_ID} and compiling its TensorRT bundle"
    fi
    log "this can take hours; follow progress with: docker compose logs -f story-scene"

    "${TRTMC_BIN}" build "${MODEL_ID}" \
        --model-revision "${MODEL_REVISION}" \
        --precision "${MODEL_PRECISION}" \
        --context-parallel-size "${COSMOS3_CP_SIZE}" \
        -o "${temporary_bundle}"

    [[ -s "${temporary_bundle}" ]] || fail "TRTMC completed without producing a bundle"
    printf '%s\n' "${build_spec}" > "${temporary_marker}"
    chmod 0644 "${temporary_bundle}" "${temporary_marker}"
    mv -f -- "${temporary_bundle}" "${COSMOS3_BUNDLE}"
    mv -f -- "${temporary_marker}" "${bundle_marker}"
    trap - EXIT INT TERM
    log "bundle ready: ${COSMOS3_BUNDLE}"
fi

# The web application never needs access to a download credential.
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN

if (( $# == 0 )); then
    set -- python3 -m story_scene
fi

log "starting Story Scene on ${HOST:-0.0.0.0}:${PORT:-8080}"
exec "$@"

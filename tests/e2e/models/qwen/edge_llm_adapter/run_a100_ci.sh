#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

readonly SCRATCH_MARKER_NAME=".trtmc-qwen-edgellm-a100-scratch"
readonly SCRATCH_MARKER_CONTENT="trtmc-qwen-edgellm-a100-scratch-v1"
readonly PATCHELF_VERSION="0.18.0-1.1build1"
readonly PYTHON_VENV_VERSION="3.12.3-1ubuntu0.15"
readonly PIP_VERSION="26.1.2"
readonly BUILD_VERSION="1.5.0"
readonly AUDITWHEEL_VERSION="6.7.0"
readonly PYTEST_VERSION="9.1.1"

fail() {
    echo "$*" >&2
    exit 1
}

validate_image() {
    local candidate_image="$1"
    if [[ ! "$candidate_image" =~ ^.+@sha256:[0-9a-f]{64}$ ]]; then
        fail "TRTMC_QUALIFICATION_IMAGE must be pinned by a lowercase sha256 digest."
    fi
}

qualification_gpu_identity() {
    python3 - "$1" <<'PY'
import sys
import tomllib

with open(sys.argv[1], "rb") as stream:
    target = tomllib.load(stream)["profile_target"]
expected_fields = {
    "os", "architecture", "platform_kind", "gpu_architecture", "gpu_name"
}
if set(target) != expected_fields:
    raise SystemExit("A100 qualification descriptor target has an invalid field set")
architecture = target["gpu_architecture"]
if not isinstance(architecture, str) or not architecture.startswith("sm") or not architecture[2:].isdigit():
    raise SystemExit("A100 qualification descriptor GPU architecture is invalid")
digits = architecture[2:]
compute_capability = f"{int(digits[:-1])}.{digits[-1]}"
print(f'{target["gpu_name"]}, {compute_capability}')
PY
}

run_in_container() {
    set -euxo pipefail

    export DEBIAN_FRONTEND=noninteractive
    export HOME=/workspace/home
    export CMAKE_TOOLCHAIN_FILE=
    export LD_PRELOAD=

    local expected_gpu_identity gpu_identity
    expected_gpu_identity="$(qualification_gpu_identity \
        /workspace/source/tests/e2e/models/qwen/edge_llm_adapter/QUALIFICATION.a100.toml)"
    gpu_identity="$(
        nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader,nounits
    )"
    if [[ "$gpu_identity" != "$expected_gpu_identity" ]]; then
        fail "Qwen EdgeLLM qualification requires exactly ${expected_gpu_identity}; found ${gpu_identity:-<none>}"
    fi
    printf '%s\n' "$gpu_identity" | tee /workspace/artifacts/gpu-identity.txt
    nvidia-smi -q | tee /workspace/artifacts/nvidia-smi.txt

    apt-get update
    apt-get install -y --no-install-recommends \
        "patchelf=${PATCHELF_VERSION}" \
        "python3.12-venv=${PYTHON_VENV_VERSION}"
    dpkg-query -W -f='${binary:Package}=${Version}\n' patchelf python3.12-venv \
        | sort | tee /workspace/artifacts/dpkg-versions.txt

    git config --global --add safe.directory /workspace/source
    git -C /workspace/source rev-parse HEAD | tee /workspace/artifacts/tested-revision.txt

    python3.12 -m venv /workspace/build-tools/venv
    /workspace/build-tools/venv/bin/python -m pip install \
        --disable-pip-version-check --upgrade \
        "pip==${PIP_VERSION}" "build==${BUILD_VERSION}" "auditwheel==${AUDITWHEEL_VERSION}"

    export CONAN_PY_BUILD_PROFILE_AUTODETECT=1
    export TRTMC_CONAN_BUILD_TARGETS="trtmc trtmc_benchmark_worker trtmc_backend_trt trtmc_model_qwen"
    export TRTMC_TRT_INCLUDE_DIR=/usr/include/x86_64-linux-gnu
    export TRTMC_TRT_LIBRARY=/usr/lib/x86_64-linux-gnu/libnvinfer.so
    export TRTMC_CUDA_INCLUDE_DIR=/usr/local/cuda/include
    export TRTMC_CUDART_LIBRARY=/usr/local/cuda/lib64/libcudart.so
    export WHEEL_PYVER=cp312
    export WHEEL_ABI=cp312
    export WHEEL_ARCH=manylinux_2_39_x86_64
    export CMAKE_BUILD_PARALLEL_LEVEL="${TRTMC_QUALIFICATION_BUILD_JOBS:-16}"

    cd /workspace/build-source
    /workspace/build-tools/venv/bin/python -m build \
        --wheel \
        --outdir /workspace/wheels \
        -C build-dir=/workspace/tmp/trtmc-conan \
        . 2>&1 | tee /workspace/artifacts/wheel-build.log

    mapfile -t wheels < <(find /workspace/wheels -maxdepth 1 -type f -name '*.whl' -print)
    test "${#wheels[@]}" -eq 1
    readonly wheel="${wheels[0]}"
    sha256sum "$wheel" | tee /workspace/artifacts/wheel.sha256
    /workspace/build-tools/venv/bin/python -m auditwheel show "$wheel" \
        | tee /workspace/artifacts/auditwheel-show.log
    if unzip -l "$wheel" | grep -E '(^|/)(evidence|artifacts?)/'; then
        fail "The release wheel contains forbidden proof data."
    fi

    python3.12 -m venv /workspace/installed/venv
    /workspace/installed/venv/bin/python -m pip install \
        --disable-pip-version-check --upgrade "pip==${PIP_VERSION}"
    /workspace/installed/venv/bin/python -m pip install \
        --disable-pip-version-check "$wheel" 2>&1 \
        | tee /workspace/artifacts/wheel-install.log
    /workspace/installed/venv/bin/python -m pip install \
        --disable-pip-version-check "pytest==${PYTEST_VERSION}"
    /workspace/installed/venv/bin/python -m pip check
    /workspace/installed/venv/bin/python -m pip freeze \
        | tee /workspace/artifacts/pip-freeze.txt
    {
        /workspace/build-tools/venv/bin/python -m pip --version
        /workspace/build-tools/venv/bin/python -m build --version
        /workspace/build-tools/venv/bin/python -m auditwheel --version
        /workspace/installed/venv/bin/python -m pytest --version
    } | tee /workspace/artifacts/bootstrap-tool-versions.txt
    /workspace/installed/venv/bin/python - <<'PY'
from pathlib import Path
import tomllib

import tensorrt

lock_path = Path(
    "/workspace/source/python/tensorrt_model_connect/families/qwen/"
    "edge_llm_adapter/dependency.lock"
)
with lock_path.open("rb") as stream:
    expected_tensorrt_version = tomllib.load(stream)["tensorrt"]["version"]
assert tensorrt.__version__ == expected_tensorrt_version, (
    tensorrt.__version__,
    expected_tensorrt_version,
)
PY

    export PYTHONNOUSERSITE=1
    export PYTHONDONTWRITEBYTECODE=1
    export PYTHONPATH=
    export PATH=/workspace/installed/venv/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
    export LD_LIBRARY_PATH=/workspace/installed/venv/bin:/workspace/installed/venv/lib/python3.12/site-packages/tensorrt_libs:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64
    export HF_HOME=/workspace/hf-cache
    export HF_HUB_CACHE=/workspace/hf-cache/hub
    export XDG_CACHE_HOME=/workspace/xdg-cache
    export TRTMC_PYTHON_PROFILE_ROOT=/workspace/python-profiles
    export TMPDIR=/workspace/tmp
    export TRTMC_BINARY=/workspace/installed/venv/bin/trtmc
    export TRTMC_CORE_LIBRARY=/workspace/installed/venv/bin/libtrtmc_core.so
    export TRTMC_INCLUDE_DIR=/workspace/installed/venv/lib/python3.12/site-packages/tensorrt_model_connect/runtime_provider/_sdk/include
    export TRTMC_PERF_ARTIFACT_DIR=/workspace/artifacts/performance
    export _TRTMC_INTERNAL_QWEN_EDGE_LLM_BUILD_DIR=/workspace/edge-build

    mkdir -p "$HF_HUB_CACHE" "$TRTMC_PYTHON_PROFILE_ROOT" "$TRTMC_PERF_ARTIFACT_DIR"
    cd /workspace/source
    /workspace/installed/venv/bin/python -m pytest \
        tests/e2e/models/qwen/edge_llm_adapter/test_a100_e2e.py \
        -vv -s --tb=long -p no:cacheprovider \
        --basetemp /workspace/tmp/pytest \
        --junitxml=/workspace/artifacts/a100-e2e.xml 2>&1 \
        | tee /workspace/artifacts/a100-e2e.log
}

if [[ "${TRTMC_QUALIFICATION_IN_CONTAINER:-}" == "1" ]]; then
    if [[ "$#" -ne 0 ]]; then
        fail "The in-container qualification entrypoint does not accept arguments."
    fi
    run_in_container
    exit 0
fi

if [[ "$#" -gt 1 || ( "$#" -eq 1 && "$1" != "--cleanup" ) ]]; then
    fail "usage: $0 [--cleanup]"
fi
readonly cleanup_requested="${1:-}"
readonly script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly repository="$(git -C "$script_dir" rev-parse --show-toplevel)"
readonly resolved_repository="$(realpath "$repository")"
readonly root="${TRTMC_QUALIFICATION_ROOT:?set TRTMC_QUALIFICATION_ROOT outside the source checkout}"
if [[ -L "$root" ]]; then
    fail "TRTMC_QUALIFICATION_ROOT must not be a symlink."
fi
readonly resolved_root="$(realpath -m "$root")"
if [[ "$resolved_root" == "/" ]]; then
    fail "TRTMC_QUALIFICATION_ROOT must not be the filesystem root."
fi
case "$resolved_root" in
    "$resolved_repository" | "$resolved_repository"/*)
        fail "TRTMC_QUALIFICATION_ROOT must be outside the source checkout."
        ;;
esac
case "$resolved_repository" in
    "$resolved_root"/*)
        fail "TRTMC_QUALIFICATION_ROOT must not contain the source checkout."
        ;;
esac

readonly image="${TRTMC_QUALIFICATION_IMAGE:?set TRTMC_QUALIFICATION_IMAGE to an image pinned by sha256 digest}"
validate_image "$image"
readonly scratch_marker="$resolved_root/$SCRATCH_MARKER_NAME"

cleanup_root() {
    if [[ ! -e "$resolved_root" ]]; then
        return 0
    fi
    if [[ ! -d "$resolved_root" || -L "$resolved_root" ]]; then
        fail "Refusing to clean a non-directory qualification root: $resolved_root"
    fi
    if [[ ! -f "$scratch_marker" || -L "$scratch_marker" ]] || \
       [[ "$(<"$scratch_marker")" != "$SCRATCH_MARKER_CONTENT" ]]; then
        fail "Refusing to clean qualification root without its ownership marker: $resolved_root"
    fi
    if ! docker image inspect "$image" >/dev/null 2>&1; then
        docker pull "$image"
    fi
    local cleanup_container
    cleanup_container="trtmc-qwen-edgellm-cleanup-${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-1}-$$"
    trap 'docker rm -f "$cleanup_container" >/dev/null 2>&1 || true' RETURN
    docker run --rm --name "$cleanup_container" \
        -v "$resolved_root:/workspace/qualification-root" \
        "$image" bash -ce '
            marker=/workspace/qualification-root/.trtmc-qwen-edgellm-a100-scratch
            test -f "$marker"
            test ! -L "$marker"
            test "$(cat "$marker")" = trtmc-qwen-edgellm-a100-scratch-v1
            find /workspace/qualification-root -mindepth 1 -delete
        '
    trap - RETURN
    rmdir "$resolved_root"
}

if [[ "$cleanup_requested" == "--cleanup" ]]; then
    cleanup_root
    exit 0
fi

readonly tested_revision="$(git -C "$repository" rev-parse HEAD)"
source_status="$(
    git -C "$repository" status \
        --porcelain=v1 --untracked-files=all --ignored=matching --ignore-submodules=none
)"
if [[ -n "$source_status" ]]; then
    fail "Qualification requires a clean source checkout; git status reported:\n$source_status"
fi
if [[ -d "$resolved_root" ]] && find "$resolved_root" -mindepth 1 -print -quit | grep -q .; then
    fail "Qualification root must be empty: $resolved_root"
fi

readonly gpu_id="${TRTMC_QUALIFICATION_GPU_ID:-0}"
if [[ ! "$gpu_id" =~ ^(0|[1-9][0-9]*)$ ]]; then
    fail "TRTMC_QUALIFICATION_GPU_ID must be one non-negative integer."
fi
readonly expected_gpu_identity="$(qualification_gpu_identity \
    "$script_dir/QUALIFICATION.a100.toml")"
gpu_identity="$(
    nvidia-smi --id="$gpu_id" --query-gpu=name,compute_cap --format=csv,noheader,nounits
)"
if [[ "$gpu_identity" != "$expected_gpu_identity" ]]; then
    fail "Qwen EdgeLLM qualification requires exactly ${expected_gpu_identity}; found ${gpu_identity:-<none>}"
fi

mkdir -p "$resolved_root"
printf '%s\n' "$SCRATCH_MARKER_CONTENT" > "$scratch_marker"
mkdir -p \
    "$resolved_root/artifacts" \
    "$resolved_root/build-source" \
    "$resolved_root/build-tools" \
    "$resolved_root/edge-build" \
    "$resolved_root/hf-cache" \
    "$resolved_root/home" \
    "$resolved_root/installed" \
    "$resolved_root/python-profiles" \
    "$resolved_root/tmp" \
    "$resolved_root/wheels" \
    "$resolved_root/xdg-cache"
printf '%s\n' "$gpu_identity" > "$resolved_root/artifacts/host-gpu-identity.txt"
git -C "$repository" archive --format=tar "$tested_revision" \
    | tar -xf - -C "$resolved_root/build-source"

if ! docker image inspect "$image" >/dev/null 2>&1; then
    docker pull "$image"
fi

readonly container_name="trtmc-qwen-edgellm-a100-${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-1}-$$"
cleanup_container() {
    docker rm -f "$container_name" >/dev/null 2>&1 || true
}
trap cleanup_container EXIT INT TERM

docker_args=(
    docker run --rm --name "$container_name"
    --gpus "device=$gpu_id"
    --ipc=host
    --ulimit memlock=-1:-1
    --ulimit stack=67108864:67108864
)
if [[ -n "${TRTMC_QUALIFICATION_DOCKER_RUNTIME:-}" ]]; then
    docker_args+=(--runtime "$TRTMC_QUALIFICATION_DOCKER_RUNTIME")
fi
docker_args+=(
    -e TRTMC_QUALIFICATION_IN_CONTAINER=1
    -e TRTMC_QUALIFICATION_BUILD_JOBS="${TRTMC_QUALIFICATION_BUILD_JOBS:-16}"
    -e TRTMC_QUALIFICATION_PROFILE_FILES="${TRTMC_QUALIFICATION_PROFILE_FILES:-}"
    -v "$repository:/workspace/source:ro"
    -v "$resolved_root/artifacts:/workspace/artifacts"
    -v "$resolved_root/build-source:/workspace/build-source"
    -v "$resolved_root/build-tools:/workspace/build-tools"
    -v "$resolved_root/edge-build:/workspace/edge-build"
    -v "$resolved_root/hf-cache:/workspace/hf-cache"
    -v "$resolved_root/home:/workspace/home"
    -v "$resolved_root/installed:/workspace/installed"
    -v "$resolved_root/python-profiles:/workspace/python-profiles"
    -v "$resolved_root/tmp:/workspace/tmp"
    -v "$resolved_root/wheels:/workspace/wheels"
    -v "$resolved_root/xdg-cache:/workspace/xdg-cache"
)
docker_args+=(
    "$image"
    bash /workspace/source/tests/e2e/models/qwen/edge_llm_adapter/run_a100_ci.sh
)

"${docker_args[@]}" 2>&1 | tee "$resolved_root/artifacts/a100-ci.log"

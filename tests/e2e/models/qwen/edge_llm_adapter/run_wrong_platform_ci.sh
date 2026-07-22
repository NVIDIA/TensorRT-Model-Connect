#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

readonly SCRATCH_MARKER_NAME=".trtmc-qwen-edgellm-wrong-platform-scratch"
readonly SCRATCH_MARKER_CONTENT="trtmc-qwen-edgellm-wrong-platform-scratch-v1"
readonly PYTHON_VENV_VERSION="3.12.3-1ubuntu0.15"
readonly PIP_VERSION="26.1.2"

fail() {
    echo "$*" >&2
    exit 1
}

validate_image() {
    if [[ ! "$1" =~ ^.+@sha256:[0-9a-f]{64}$ ]]; then
        fail "TRTMC_QUALIFICATION_IMAGE must be pinned by a lowercase sha256 digest."
    fi
}

target_identity() {
    python3 - "$1" <<'PY'
import json
import platform
import sys

target = json.loads(sys.argv[1])
expected = {
    "os": "linux",
    "architecture": "x86_64",
    "platform_kind": "discrete",
    "gpu_architecture": "sm120",
    "gpu_name": "NVIDIA GeForce RTX 5090",
}
if target != expected:
    raise SystemExit(f"consumer target is not the qualified wrong-platform target: {target!r}")
if platform.system().lower() != target["os"] or platform.machine() != target["architecture"]:
    raise SystemExit("consumer host does not match the declared operating-system target")
architecture = target["gpu_architecture"]
compute_capability = f"{int(architecture[2:-1])}.{architecture[-1]}"
print(f'{target["gpu_name"]}, {compute_capability}')
PY
}

run_in_container() {
    set -euxo pipefail

    export DEBIAN_FRONTEND=noninteractive
    export HOME=/workspace/home
    export PYTHONNOUSERSITE=1
    export PYTHONDONTWRITEBYTECODE=1
    export PYTHONPATH=
    export LD_PRELOAD=

    local expected_gpu_identity actual_gpu_identity
    expected_gpu_identity="$(target_identity "$TRTMC_QUALIFICATION_RUNNER_TARGET_JSON")"
    actual_gpu_identity="$(
        nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader,nounits
    )"
    if [[ "$actual_gpu_identity" != "$expected_gpu_identity" ]]; then
        fail "Wrong-platform qualification requires exactly ${expected_gpu_identity}; found ${actual_gpu_identity:-<none>}"
    fi
    printf '%s\n' "$actual_gpu_identity" | tee /workspace/artifacts/gpu-identity.txt
    nvidia-smi -q | tee /workspace/artifacts/nvidia-smi.txt

    mapfile -t input_files < <(find /workspace/input -mindepth 1 -maxdepth 1 -type f -printf '%f\n' | sort)
    expected_files=(SHA256SUMS delegated.trtfb transfer-manifest.json)
    mapfile -t wheels < <(printf '%s\n' "${input_files[@]}" | grep -E '^tensorrt_model_connect-.*\.whl$' || true)
    test "${#wheels[@]}" -eq 1
    expected_files+=("${wheels[0]}")
    mapfile -t expected_files < <(printf '%s\n' "${expected_files[@]}" | sort)
    [[ "${input_files[*]}" == "${expected_files[*]}" ]] || \
        fail "Transfer artifact must contain exactly the bundle, wheel, manifest, and SHA256SUMS."
    find /workspace/input -mindepth 1 -maxdepth 1 -type l -print -quit | grep -q . && \
        fail "Transfer artifact must not contain symlinks."
    (cd /workspace/input && sha256sum --check --strict SHA256SUMS) \
        | tee /workspace/artifacts/transfer-sha256-check.txt

    git config --global --add safe.directory /workspace/source
    local tested_revision
    tested_revision="$(git -C /workspace/source rev-parse HEAD)"
    python3 - /workspace/input/transfer-manifest.json /workspace/input "$tested_revision" \
        /workspace/source/tests/e2e/models/qwen/edge_llm_adapter/QUALIFICATION.rtx5090.toml \
        > /workspace/artifacts/validated-transfer.json <<'PY'
import hashlib
import json
from pathlib import Path
import sys
import tomllib

manifest_path = Path(sys.argv[1])
input_path = Path(sys.argv[2])
revision = sys.argv[3]
descriptor_path = Path(sys.argv[4])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
expected_keys = {
    "schema_version", "source_revision", "producer_id", "model_id",
    "model_revision", "profile_id", "bundle_file", "bundle_sha256",
    "wheel_file", "wheel_sha256",
}
if set(manifest) != expected_keys or manifest["schema_version"] != 1:
    raise SystemExit("transfer manifest does not use the exact schema")
descriptor = tomllib.loads(descriptor_path.read_text(encoding="utf-8"))
if manifest["producer_id"] != descriptor["producer_id"]:
    raise SystemExit("transfer manifest producer does not match the consumer descriptor")
if manifest["source_revision"] != revision:
    raise SystemExit("transfer manifest source revision does not match the checked-out source")
for field in ("model_id", "model_revision", "profile_id"):
    value = manifest[field]
    if not isinstance(value, str) or not value or value != value.strip():
        raise SystemExit(f"transfer manifest {field} is invalid")
for name_field, digest_field in (("bundle_file", "bundle_sha256"), ("wheel_file", "wheel_sha256")):
    name = manifest[name_field]
    if not isinstance(name, str) or Path(name).name != name:
        raise SystemExit(f"transfer manifest {name_field} is invalid")
    path = input_path / name
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"transfer payload is unavailable: {name}")
    digest_builder = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest_builder.update(block)
    digest = digest_builder.hexdigest()
    if digest != manifest[digest_field]:
        raise SystemExit(f"transfer payload digest mismatch: {name}")
if manifest["bundle_file"] != "delegated.trtfb":
    raise SystemExit("transfer manifest must name delegated.trtfb")
if not manifest["wheel_file"].startswith("tensorrt_model_connect-") or not manifest[
    "wheel_file"
].endswith(".whl"):
    raise SystemExit("transfer manifest wheel filename is invalid")
print(json.dumps(manifest, indent=2, sort_keys=True))
PY

    apt-get update
    apt-get install -y --no-install-recommends "python3.12-venv=${PYTHON_VENV_VERSION}"
    dpkg-query -W -f='${binary:Package}=${Version}\n' python3.12-venv \
        | tee /workspace/artifacts/dpkg-versions.txt
    python3.12 -m venv /workspace/installed/venv
    /workspace/installed/venv/bin/python -m pip install \
        --disable-pip-version-check --upgrade "pip==${PIP_VERSION}"
    /workspace/installed/venv/bin/python -m pip install \
        --disable-pip-version-check "/workspace/input/${wheels[0]}" 2>&1 \
        | tee /workspace/artifacts/wheel-install.log
    /workspace/installed/venv/bin/python -m pip check
    /workspace/installed/venv/bin/python -m pip freeze \
        | tee /workspace/artifacts/pip-freeze.txt
    /workspace/installed/venv/bin/python - <<'PY'
from pathlib import Path
import tomllib

import tensorrt

lock = Path(
    "/workspace/source/python/tensorrt_model_connect/families/qwen/"
    "edge_llm_adapter/dependency.lock"
)
with lock.open("rb") as stream:
    expected = tomllib.load(stream)["tensorrt"]["version"]
if tensorrt.__version__ != expected:
    raise SystemExit(f"installed TensorRT {tensorrt.__version__} does not match {expected}")
PY

    export PATH=/workspace/installed/venv/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
    export LD_LIBRARY_PATH=/workspace/installed/venv/bin:/workspace/installed/venv/lib/python3.12/site-packages/tensorrt_libs:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64
    export HF_HOME=/workspace/hf-cache
    export HF_HUB_CACHE=/workspace/hf-cache/hub
    export XDG_CACHE_HOME=/workspace/xdg-cache
    export TRTMC_PYTHON_PROFILE_ROOT=/workspace/python-profiles
    export TMPDIR=/workspace/tmp

    local model_id native_bundle
    model_id="$(python3 -c 'import json; print(json.load(open("/workspace/input/transfer-manifest.json"))["model_id"])')"
    native_bundle=/workspace/tmp/native-current-platform.trtfb
    trtmc build "$model_id" -o "$native_bundle" 2>&1 \
        | tee /workspace/artifacts/native-build.log
    trtmc inspect "$native_bundle" | tee /workspace/artifacts/native-inspect.txt
    if grep -q 'optimized_runtime.json' /workspace/artifacts/native-inspect.txt; then
        fail "The current RTX 5090 build incorrectly selected the A100 EdgeLLM profile."
    fi

    trtmc inspect /workspace/input/delegated.trtfb \
        | tee /workspace/artifacts/delegated-inspect.txt
    grep -q 'optimized_runtime.json' /workspace/artifacts/delegated-inspect.txt
    /workspace/installed/venv/bin/python - \
        /workspace/installed/venv/bin/libtrtmc_core.so \
        /workspace/input/delegated.trtfb /workspace/runtime-cache \
        > /workspace/artifacts/wrong-platform-load.txt 2>&1 <<'PY'
import ctypes
import sys

class Options(ctypes.Structure):
    _fields_ = [
        ("max_new_tokens", ctypes.c_int),
        ("hf_python", ctypes.c_char_p),
        ("image_path", ctypes.c_char_p),
        ("runtime_cache", ctypes.c_char_p),
        ("cuda_graphs", ctypes.c_int),
    ]

library = ctypes.CDLL(sys.argv[1])
create = library.trtmc_create_pipeline_ex
create.argtypes = [ctypes.c_char_p, ctypes.POINTER(Options)]
create.restype = ctypes.c_void_p
last_error = library.trtmc_last_error
last_error.restype = ctypes.c_char_p
options = Options(runtime_cache=sys.argv[3].encode())
pipeline = create(sys.argv[2].encode(), ctypes.byref(options))
error = (last_error() or b"").decode()
expected = "active CUDA device does not match the bundle's qualified deployment target"
if pipeline:
    raise SystemExit("wrong-platform delegated bundle unexpectedly loaded")
if error != expected:
    raise SystemExit(f"unexpected wrong-platform error: {error!r}")
print(error)
PY
    grep -Fx "active CUDA device does not match the bundle's qualified deployment target" \
        /workspace/artifacts/wrong-platform-load.txt
    test ! -e /workspace/tmp/generated.jsonl
}

if [[ "${TRTMC_QUALIFICATION_IN_CONTAINER:-}" == "1" ]]; then
    [[ "$#" -eq 0 ]] || fail "The in-container entrypoint does not accept arguments."
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
readonly input_dir="${TRTMC_QUALIFICATION_INPUT_DIR:?set TRTMC_QUALIFICATION_INPUT_DIR to the producer artifact}"
readonly image="${TRTMC_QUALIFICATION_IMAGE:?set TRTMC_QUALIFICATION_IMAGE from the consumer descriptor}"
readonly runner_target="${TRTMC_QUALIFICATION_RUNNER_TARGET_JSON:?set the descriptor runner target}"
validate_image "$image"

[[ ! -L "$root" ]] || fail "TRTMC_QUALIFICATION_ROOT must not be a symlink."
readonly resolved_root="$(realpath -m "$root")"
[[ "$resolved_root" != "/" ]] || fail "TRTMC_QUALIFICATION_ROOT must not be the filesystem root."
case "$resolved_root" in
    "$resolved_repository" | "$resolved_repository"/*)
        fail "TRTMC_QUALIFICATION_ROOT must be outside the source checkout."
        ;;
esac
[[ -d "$input_dir" && ! -L "$input_dir" ]] || fail "Consumer input must be a real directory."
readonly resolved_input="$(realpath "$input_dir")"
case "$resolved_input" in
    "$resolved_repository" | "$resolved_repository"/*)
        fail "Consumer input must be outside the source checkout."
        ;;
esac
readonly scratch_marker="$resolved_root/$SCRATCH_MARKER_NAME"

cleanup_root() {
    [[ -e "$resolved_root" ]] || return 0
    [[ -d "$resolved_root" && ! -L "$resolved_root" ]] || \
        fail "Refusing to clean a non-directory qualification root: $resolved_root"
    [[ -f "$scratch_marker" && ! -L "$scratch_marker" ]] || \
        fail "Refusing to clean qualification root without its ownership marker: $resolved_root"
    [[ "$(<"$scratch_marker")" == "$SCRATCH_MARKER_CONTENT" ]] || \
        fail "Refusing to clean qualification root with a foreign ownership marker."
    docker image inspect "$image" >/dev/null 2>&1 || docker pull "$image"
    docker run --rm -v "$resolved_root:/workspace/qualification-root" "$image" bash -ce '
        marker=/workspace/qualification-root/.trtmc-qwen-edgellm-wrong-platform-scratch
        test -f "$marker"
        test ! -L "$marker"
        test "$(cat "$marker")" = trtmc-qwen-edgellm-wrong-platform-scratch-v1
        find /workspace/qualification-root -mindepth 1 -delete
    '
    rmdir "$resolved_root"
}

if [[ "$cleanup_requested" == "--cleanup" ]]; then
    cleanup_root
    exit 0
fi

source_status="$(git -C "$repository" status --porcelain=v1 --untracked-files=all --ignored=matching --ignore-submodules=none)"
[[ -z "$source_status" ]] || fail "Qualification requires a clean source checkout; git status reported:\n$source_status"
[[ ! -e "$resolved_root" ]] || fail "Qualification root must not already exist: $resolved_root"

readonly expected_gpu_identity="$(target_identity "$runner_target")"
readonly gpu_id="${TRTMC_QUALIFICATION_GPU_ID:-0}"
[[ "$gpu_id" =~ ^(0|[1-9][0-9]*)$ ]] || fail "TRTMC_QUALIFICATION_GPU_ID must be non-negative."
actual_gpu_identity="$(nvidia-smi --id="$gpu_id" --query-gpu=name,compute_cap --format=csv,noheader,nounits)"
[[ "$actual_gpu_identity" == "$expected_gpu_identity" ]] || \
    fail "Wrong-platform qualification requires exactly ${expected_gpu_identity}; found ${actual_gpu_identity:-<none>}"

mkdir -p "$resolved_root"
printf '%s\n' "$SCRATCH_MARKER_CONTENT" > "$scratch_marker"
mkdir -p "$resolved_root/artifacts" "$resolved_root/hf-cache" "$resolved_root/home" \
    "$resolved_root/installed" "$resolved_root/python-profiles" "$resolved_root/runtime-cache" \
    "$resolved_root/tmp" "$resolved_root/xdg-cache"
printf '%s\n' "$actual_gpu_identity" > "$resolved_root/artifacts/host-gpu-identity.txt"

docker image inspect "$image" >/dev/null 2>&1 || docker pull "$image"
readonly container_name="trtmc-qwen-edgellm-wrong-platform-${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-1}-$$"
cleanup_container() { docker rm -f "$container_name" >/dev/null 2>&1 || true; }
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
    -e TRTMC_QUALIFICATION_RUNNER_TARGET_JSON="$runner_target"
    -v "$repository:/workspace/source:ro"
    -v "$resolved_input:/workspace/input:ro"
    -v "$resolved_root/artifacts:/workspace/artifacts"
    -v "$resolved_root/hf-cache:/workspace/hf-cache"
    -v "$resolved_root/home:/workspace/home"
    -v "$resolved_root/installed:/workspace/installed"
    -v "$resolved_root/python-profiles:/workspace/python-profiles"
    -v "$resolved_root/runtime-cache:/workspace/runtime-cache"
    -v "$resolved_root/tmp:/workspace/tmp"
    -v "$resolved_root/xdg-cache:/workspace/xdg-cache"
    "$image"
    bash /workspace/source/tests/e2e/models/qwen/edge_llm_adapter/run_wrong_platform_ci.sh
)

"${docker_args[@]}" 2>&1 | tee "$resolved_root/artifacts/wrong-platform-ci.log"

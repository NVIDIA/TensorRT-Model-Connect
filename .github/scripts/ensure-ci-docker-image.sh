#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

base_image="${TRTMC_CI_IMAGE:-trtmc-dev-gb300:manylinux_2_39}"
dockerfile="${TRTMC_CI_DOCKERFILE:-Dockerfile}"
lock_file="${TRTMC_CI_IMAGE_LOCK_FILE:-/tmp/trtmc-ci-docker-image.lock}"
lock_timeout="${TRTMC_CI_IMAGE_LOCK_TIMEOUT:-5400}"
fingerprint_label="org.nvidia.trtmc.ci-input-fingerprint"

if ! [[ "$lock_timeout" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: TRTMC_CI_IMAGE_LOCK_TIMEOUT must be a positive integer" >&2
  exit 1
fi
if ! command -v flock >/dev/null 2>&1; then
  echo "ERROR: flock is required to serialize CI image verification" >&2
  exit 1
fi
exec 9>"$lock_file"
if ! flock -w "$lock_timeout" 9; then
  echo "ERROR: Timed out waiting for CI image lock: $lock_file" >&2
  exit 1
fi

docker_input_paths=(
  "$dockerfile"
  .dockerignore
  .github/scripts/build-python-profiles.py
  python/tensorrt_model_connect/__init__.py
  python/tensorrt_model_connect/python_profiles.py
  python/tensorrt_model_connect/families/__init__.py
)

# Family manifests declare profile assets, but most MODEL.toml fields do not
# affect the CI image. Fingerprint the normalized profile registry below so a
# comment or ownership-only model edit cannot force a full image rebuild.
mapfile -t declared_profile_assets < <(
  PYTHONPATH=python python3 - <<'PY'
from pathlib import Path

from tensorrt_model_connect.python_profiles import load_python_profile_registry

package_root = Path("python/tensorrt_model_connect")
assets = set()
for spec in load_python_profile_registry()["profiles"].values():
    if not isinstance(spec, dict):
        continue
    for field in ("requirements", "verification_script_file"):
        value = str(spec.get(field, "") or "").strip()
        if value:
            assets.add(str(package_root / value))
print("\n".join(sorted(assets)))
PY
)
docker_input_paths+=("${declared_profile_assets[@]}")
mapfile -t docker_input_paths < <(
  printf '%s\n' "${docker_input_paths[@]}" | sort -u
)

python_profile_semantic_fingerprint="$({
  PYTHONPATH=python python3 - <<'PY'
import hashlib
import json

from tensorrt_model_connect.python_profiles import load_python_profile_registry

registry = load_python_profile_registry()
profile_image_contract = {
    "version": registry.get("version"),
    "profiles": registry["profiles"],
}
payload = json.dumps(
    profile_image_contract,
    ensure_ascii=True,
    separators=(",", ":"),
    sort_keys=True,
).encode("utf-8")
print(hashlib.sha256(payload).hexdigest())
PY
} | tail -n 1)"
if ! [[ "$python_profile_semantic_fingerprint" =~ ^[0-9a-f]{64}$ ]]; then
  echo "ERROR: Could not fingerprint normalized Python profile declarations" >&2
  exit 1
fi

expected_python_profiles="$({
  PYTHONPATH=python python3 - <<'PY'
from tensorrt_model_connect.python_profiles import (
    DEFAULT_PROFILE,
    load_python_profile_registry,
)

profiles = load_python_profile_registry()["profiles"]
print(",".join(sorted(name for name in profiles if name != DEFAULT_PROFILE)))
PY
} | tail -n 1)"
if [ -z "$expected_python_profiles" ]; then
  echo "ERROR: No family-owned Python execution profiles were declared" >&2
  exit 1
fi

compute_docker_input_fingerprint() {
  local path
  {
    printf 'python-profile-registry\0%s\n' \
      "$python_profile_semantic_fingerprint"
    for path in "${docker_input_paths[@]}"; do
      printf '%s\0' "$path"
      if [ -f "$path" ]; then
        sha256sum "$path" | awk '{ print $1 }'
      else
        printf 'missing\n'
      fi
    done
  } | sha256sum | awk '{ print $1 }'
}

expected_fingerprint="$(compute_docker_input_fingerprint)"
image="${base_image}-${expected_fingerprint:0:12}"
validator_fingerprint="$(sha256sum "$0" | awk '{ print $1 }')"
if ! [[ "$validator_fingerprint" =~ ^[0-9a-f]{64}$ ]]; then
  echo "ERROR: Could not fingerprint the CI image validator" >&2
  exit 1
fi
validation_cache_file="${lock_file}.verified-$(id -u)-${expected_fingerprint}-${validator_fingerprint}"

if [ -n "${GITHUB_ENV:-}" ]; then
  printf 'TRTMC_CI_IMAGE=%s\n' "$image" >> "$GITHUB_ENV"
fi

read_docker_arg() {
  local name="$1"
  awk -F= -v key="$name" '
    $1 == "ARG " key {
      print $2
      exit
    }
  ' "$dockerfile"
}

expected_trt="$(read_docker_arg TENSORRT_VERSION)"
expected_modelopt="$(read_docker_arg MODELOPT_VERSION)"

if [ -z "$expected_trt" ]; then
  echo "ERROR: Could not find ARG TENSORRT_VERSION in $dockerfile" >&2
  exit 1
fi

if [ -z "$expected_modelopt" ]; then
  echo "ERROR: Could not find ARG MODELOPT_VERSION in $dockerfile" >&2
  exit 1
fi

summary() {
  if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
    printf '%s\n' "$*" >> "$GITHUB_STEP_SUMMARY"
  fi
}

query_image_versions() {
  # Validate with an unprivileged identity matching the model-proof security
  # boundary. This catches image-baked profiles that exist for root but cannot
  # be traversed by the proof container.
  docker run --rm --read-only \
    --user 65534:65534 \
    --tmpfs /tmp:rw,exec,nosuid,nodev,size=256m \
    -e HOME=/tmp \
    --entrypoint /bin/bash "$image" -lc '
python3 - <<'"'"'PY'"'"'
import importlib.metadata as metadata
from pathlib import Path

import tensorrt
from nemo.collections.asr.models.rnnt_bpe_models_prompt import (
    EncDecRNNTBPEModelWithPrompt,
)

print(f"TENSORRT_VERSION={tensorrt.__version__}")
print("MODELOPT_VERSION=" + metadata.version("nvidia-modelopt"))
print(
    "NLOHMANN_JSON_HEADER="
    + ("present" if Path("/usr/include/nlohmann/json.hpp").is_file() else "missing")
)
print("NEMO_PROMPT_RNNT=available")

manifest_path = Path("/opt/trtmc-python-profiles/.image-ready.json")
manifest = __import__("json").loads(manifest_path.read_text(encoding="utf-8"))
if Path("/opt/trtmc-profile-source").exists():
    raise SystemExit("profile builder source leaked into the runtime image")
profiles = manifest.get("profiles")
if not isinstance(profiles, dict) or not profiles:
    raise SystemExit("prebuilt Python profile manifest is empty or invalid")
for name, record in profiles.items():
    if not isinstance(record, dict):
        raise SystemExit(f"invalid prebuilt Python profile record: {name}")
    python = Path(str(record.get("python", "")))
    ready = Path(str(record.get("ready", "")))
    if not python.is_file() or not ready.is_file():
        raise SystemExit(f"prebuilt Python profile is incomplete: {name}")
print("PYTHON_PROFILES=" + ",".join(sorted(profiles)))
PY
'
}

query_image_fingerprint() {
  docker image inspect \
    --format "{{ index .Config.Labels \"$fingerprint_label\" }}" \
    "$image"
}

collect_docker_input_changes() {
  local base="${CI_BASE_REF:-}"
  if [ -z "$base" ] || ! git cat-file -e "$base^{commit}" 2>/dev/null; then
    return 0
  fi

  git diff --name-only "$base"...HEAD -- "${docker_input_paths[@]}"
}

rebuild_reasons=()
mapfile -t changed_paths < <(collect_docker_input_changes)

version_file="$(mktemp)"
validation_cache_tmp=""
cleanup_validation_files() {
  rm -f "$version_file"
  if [ -n "$validation_cache_tmp" ]; then
    rm -f "$validation_cache_tmp"
  fi
}
trap cleanup_validation_files EXIT

restore_cached_validation() {
  local expected_image_ref="$1"
  [ -f "$validation_cache_file" ] && \
    [ ! -L "$validation_cache_file" ] && \
    [ "$(stat -c '%u' "$validation_cache_file")" = "$(id -u)" ] || return 1
  [ "$(awk -F= '$1 == "IMAGE_REF" { print $2 }' "$validation_cache_file")" = \
    "$expected_image_ref" ] || return 1
  [ "$(awk -F= '$1 == "TENSORRT_VERSION" { print $2 }' "$validation_cache_file")" = \
    "$expected_trt" ] || return 1
  [ "$(awk -F= '$1 == "MODELOPT_VERSION" { print $2 }' "$validation_cache_file")" = \
    "$expected_modelopt" ] || return 1
  [ "$(awk -F= '$1 == "NLOHMANN_JSON_HEADER" { print $2 }' "$validation_cache_file")" = \
    present ] || return 1
  [ "$(awk -F= '$1 == "NEMO_PROMPT_RNNT" { print $2 }' "$validation_cache_file")" = \
    available ] || return 1
  [ "$(awk -F= '$1 == "PYTHON_PROFILES" { print $2 }' "$validation_cache_file")" = \
    "$expected_python_profiles" ] || return 1
  sed '/^IMAGE_REF=/d' "$validation_cache_file" > "$version_file"
}

persist_validation_cache() {
  local image_ref="$1"
  validation_cache_tmp="$(mktemp "${validation_cache_file}.tmp.XXXXXX")"
  chmod 0600 "$validation_cache_tmp"
  {
    printf 'IMAGE_REF=%s\n' "$image_ref"
    sed '/^IMAGE_REF=/d' "$version_file"
  } > "$validation_cache_tmp"
  mv -f -- "$validation_cache_tmp" "$validation_cache_file"
  validation_cache_tmp=""
}

validation_cache_hit=0
validation_cache_status=miss
image_rebuilt=0

if ! docker image inspect "$image" >/dev/null 2>&1; then
  rebuild_reasons+=("CI Docker image '$image' is missing")
else
  current_image_ref="$(docker image inspect --format '{{.Id}}' "$image" 2>/dev/null || true)"
  if ! [[ "$current_image_ref" =~ ^sha256:[0-9a-f]{64}$ ]]; then
    rebuild_reasons+=("CI Docker image '$image' returned an invalid immutable ID")
  fi
  current_fingerprint="$(query_image_fingerprint 2>/dev/null || true)"
  if [ "$current_fingerprint" != "$expected_fingerprint" ]; then
    rebuild_reasons+=("Docker input fingerprint mismatch: image has '${current_fingerprint:-missing}', source expects '$expected_fingerprint'")
  elif [[ "$current_image_ref" =~ ^sha256:[0-9a-f]{64}$ ]] && \
    restore_cached_validation "$current_image_ref"; then
    validation_cache_hit=1
    validation_cache_status=hit
    echo "Reusing full CI image validation for immutable image '$current_image_ref'"
    summary "Reused full CI image validation for immutable image \`$current_image_ref\`."
  elif ! query_image_versions > "$version_file"; then
    rebuild_reasons+=("CI Docker image '$image' could not report dependency versions")
  else
    current_trt="$(awk -F= '$1 == "TENSORRT_VERSION" { print $2 }' "$version_file")"
    current_modelopt="$(awk -F= '$1 == "MODELOPT_VERSION" { print $2 }' "$version_file")"
    current_nlohmann="$(awk -F= '$1 == "NLOHMANN_JSON_HEADER" { print $2 }' "$version_file")"
    current_nemo_prompt_rnnt="$(awk -F= '$1 == "NEMO_PROMPT_RNNT" { print $2 }' "$version_file")"
    current_python_profiles="$(awk -F= '$1 == "PYTHON_PROFILES" { print $2 }' "$version_file")"

    if [ "$current_trt" != "$expected_trt" ]; then
      rebuild_reasons+=("TensorRT version mismatch: image has '${current_trt:-unknown}', Dockerfile expects '$expected_trt'")
    fi

    if [ "$current_modelopt" != "$expected_modelopt" ]; then
      rebuild_reasons+=("modelopt version mismatch: image has '${current_modelopt:-unknown}', Dockerfile expects '$expected_modelopt'")
    fi

    if [ "$current_nlohmann" != "present" ]; then
      rebuild_reasons+=("nlohmann/json development headers are missing")
    fi

    if [ "$current_nemo_prompt_rnnt" != "available" ]; then
      rebuild_reasons+=("required NeMo prompt RNN-T capability is missing")
    fi

    if [ "$current_python_profiles" != "$expected_python_profiles" ]; then
      rebuild_reasons+=("prebuilt Python profiles differ: image has '${current_python_profiles:-missing}', source expects '$expected_python_profiles'")
    fi
  fi
fi

if [ "${#rebuild_reasons[@]}" -gt 0 ]; then
  echo "Rebuilding CI Docker image '$image' from $dockerfile"
  printf '  reason: %s\n' "${rebuild_reasons[@]}"
  summary "Rebuilding CI Docker image \`$image\` from \`$dockerfile\`."
  for reason in "${rebuild_reasons[@]}"; do
    summary "- $reason"
  done
  if [ "${#changed_paths[@]}" -gt 0 ]; then
    summary ""
    summary "Changed CI Docker image inputs:"
    for path in "${changed_paths[@]}"; do
      summary "- \`$path\`"
    done
  fi

  docker build \
    --label "$fingerprint_label=$expected_fingerprint" \
    -t "$image" \
    -f "$dockerfile" \
    .
  image_rebuilt=1
else
  echo "CI Docker image '$image' already matches $dockerfile"
fi

current_fingerprint="$(query_image_fingerprint 2>/dev/null || true)"
if [ "$current_fingerprint" != "$expected_fingerprint" ]; then
  echo "ERROR: CI Docker image '$image' has input fingerprint '${current_fingerprint:-missing}'; expected '$expected_fingerprint'" >&2
  exit 1
fi

image_ref="$(docker image inspect --format '{{.Id}}' "$image")"
if ! [[ "$image_ref" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "ERROR: CI Docker image '$image' returned an invalid immutable ID: $image_ref" >&2
  exit 1
fi
if [ "$image_rebuilt" -eq 1 ] || [ ! -s "$version_file" ]; then
  query_image_versions > "$version_file"
fi
current_trt="$(awk -F= '$1 == "TENSORRT_VERSION" { print $2 }' "$version_file")"
current_modelopt="$(awk -F= '$1 == "MODELOPT_VERSION" { print $2 }' "$version_file")"
current_nlohmann="$(awk -F= '$1 == "NLOHMANN_JSON_HEADER" { print $2 }' "$version_file")"
current_nemo_prompt_rnnt="$(awk -F= '$1 == "NEMO_PROMPT_RNNT" { print $2 }' "$version_file")"
current_python_profiles="$(awk -F= '$1 == "PYTHON_PROFILES" { print $2 }' "$version_file")"

if [ "$current_trt" != "$expected_trt" ]; then
  echo "ERROR: CI Docker image '$image' has TensorRT '$current_trt'; expected '$expected_trt' from $dockerfile" >&2
  exit 1
fi

if [ "$current_modelopt" != "$expected_modelopt" ]; then
  echo "ERROR: CI Docker image '$image' has modelopt '$current_modelopt'; expected '$expected_modelopt' from $dockerfile" >&2
  exit 1
fi

if [ "$current_nlohmann" != "present" ]; then
  echo "ERROR: CI Docker image '$image' lacks /usr/include/nlohmann/json.hpp" >&2
  exit 1
fi

if [ "$current_nemo_prompt_rnnt" != "available" ]; then
  echo "ERROR: CI Docker image '$image' is missing the required NeMo prompt RNN-T capability" >&2
  exit 1
fi

if [ "$current_python_profiles" != "$expected_python_profiles" ]; then
  echo "ERROR: CI Docker image '$image' has prebuilt Python profiles '${current_python_profiles:-missing}'; expected '$expected_python_profiles' from family metadata" >&2
  exit 1
fi

if [ "$validation_cache_hit" -eq 0 ]; then
  persist_validation_cache "$image_ref"
fi
if [ -n "${GITHUB_OUTPUT:-}" ]; then
  echo "image_ref=$image_ref" >> "$GITHUB_OUTPUT"
  echo "validation_cache=$validation_cache_status" >> "$GITHUB_OUTPUT"
fi

echo "CI image validation cache: $validation_cache_status"
echo "CI Docker image '$image' verified: TensorRT $current_trt, modelopt $current_modelopt, nlohmann/json headers, NeMo prompt RNN-T and prebuilt Python profiles ($current_python_profiles) present, image $image_ref"

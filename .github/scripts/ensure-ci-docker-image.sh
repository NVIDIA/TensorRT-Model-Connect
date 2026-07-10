#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

base_image="${TRTMC_CI_IMAGE:-trtmc-dev-gb300:manylinux_2_39}"
dockerfile="${TRTMC_CI_DOCKERFILE:-Dockerfile}"
lock_file="${TRTMC_CI_IMAGE_LOCK_FILE:-/tmp/trtmc-ci-docker-image.lock}"
lock_timeout="${TRTMC_CI_IMAGE_LOCK_TIMEOUT:-5400}"
verification_dir="${TRTMC_CI_IMAGE_VERIFICATION_DIR:-/tmp/trtmc-ci-image-verifications}"
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

emit_image_ref() {
  local resolved_ref="$1"
  if [ -n "${GITHUB_OUTPUT:-}" ]; then
    echo "image_ref=$resolved_ref" >> "$GITHUB_OUTPUT"
  fi
}

# Every matrix child uses the same immutable merge snapshot and host-local
# Docker daemon. Validate the image once per workflow run, then let siblings
# reuse that proof after a cheap immutable-image-ID check. The global flock
# makes the stamp atomic and also keeps separate runs from rebuilding together.
verification_stamp=""
run_id="${GITHUB_RUN_ID:-}"
run_attempt="${GITHUB_RUN_ATTEMPT:-1}"
if [[ "$run_id" =~ ^[0-9]+$ ]] && [[ "$run_attempt" =~ ^[1-9][0-9]*$ ]]; then
  mkdir -p "$verification_dir"
  verification_stamp="${verification_dir%/}/${run_id}-${run_attempt}-${expected_fingerprint}.verified"
  if [ -f "$verification_stamp" ]; then
    stamped_ref="$(<"$verification_stamp")"
    current_ref="$(docker image inspect --format '{{.Id}}' "$image" 2>/dev/null || true)"
    if [[ "$stamped_ref" =~ ^sha256:[0-9a-f]{64}$ ]] && \
       [ "$current_ref" = "$stamped_ref" ]; then
      emit_image_ref "$current_ref"
      echo "CI Docker image '$image' reused from this workflow run's verified image $current_ref"
      exit 0
    fi
    rm -f -- "$verification_stamp"
  fi
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
image_metadata_valid=false
mapfile -t changed_paths < <(collect_docker_input_changes)

version_file="$(mktemp)"
empty_context=""
cleanup() {
  rm -f "$version_file"
  if [ -n "$empty_context" ]; then
    rm -rf "$empty_context"
  fi
}
trap cleanup EXIT

if ! docker image inspect "$image" >/dev/null 2>&1; then
  rebuild_reasons+=("CI Docker image '$image' is missing")
else
  current_fingerprint="$(query_image_fingerprint 2>/dev/null || true)"
  if [ "$current_fingerprint" != "$expected_fingerprint" ]; then
    rebuild_reasons+=("Docker input fingerprint mismatch: image has '${current_fingerprint:-missing}', source expects '$expected_fingerprint'")
  fi
  if ! query_image_versions > "$version_file"; then
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
    if [ "${#rebuild_reasons[@]}" -eq 0 ]; then
      image_metadata_valid=true
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
  image_metadata_valid=false
else
  echo "CI Docker image '$image' already matches $dockerfile"
fi

current_fingerprint="$(query_image_fingerprint 2>/dev/null || true)"
if [ "$current_fingerprint" != "$expected_fingerprint" ]; then
  echo "ERROR: CI Docker image '$image' has input fingerprint '${current_fingerprint:-missing}'; expected '$expected_fingerprint'" >&2
  exit 1
fi

if [ "$image_metadata_valid" != "true" ]; then
  query_image_versions > "$version_file"
  current_trt="$(awk -F= '$1 == "TENSORRT_VERSION" { print $2 }' "$version_file")"
  current_modelopt="$(awk -F= '$1 == "MODELOPT_VERSION" { print $2 }' "$version_file")"
  current_nlohmann="$(awk -F= '$1 == "NLOHMANN_JSON_HEADER" { print $2 }' "$version_file")"
  current_nemo_prompt_rnnt="$(awk -F= '$1 == "NEMO_PROMPT_RNNT" { print $2 }' "$version_file")"
  current_python_profiles="$(awk -F= '$1 == "PYTHON_PROFILES" { print $2 }' "$version_file")"
fi

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

image_ref="$(docker image inspect --format '{{.Id}}' "$image")"
if ! [[ "$image_ref" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "ERROR: CI Docker image '$image' returned an invalid immutable ID: $image_ref" >&2
  exit 1
fi
emit_image_ref "$image_ref"
if [ -n "$verification_stamp" ]; then
  stamp_tmp="${verification_stamp}.tmp.$$"
  printf '%s\n' "$image_ref" > "$stamp_tmp"
  mv -f -- "$stamp_tmp" "$verification_stamp"
fi

echo "CI Docker image '$image' verified: TensorRT $current_trt, modelopt $current_modelopt, nlohmann/json headers, NeMo prompt RNN-T and prebuilt Python profiles ($current_python_profiles) present, image $image_ref"

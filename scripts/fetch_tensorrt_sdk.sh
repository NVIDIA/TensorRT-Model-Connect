# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Source this file, then call trtmc_stage_tensorrt_sdk with the Dockerfile and
# desired build-context path.

trtmc_read_docker_arg() {
  local dockerfile="$1"
  local name="$2"
  awk -F= -v key="$name" '
    $1 == "ARG " key {
      sub(/^[^=]*=/, "")
      print
      exit
    }
  ' "$dockerfile"
}

trtmc_stage_tensorrt_sdk() {
  local dockerfile="$1"
  local destination="$2"
  local sdk_url sdk_range sdk_sha cache_dir cache_file temporary_file

  sdk_url="$(trtmc_read_docker_arg "$dockerfile" TENSORRT_SDK_URL)"
  sdk_range="$(trtmc_read_docker_arg "$dockerfile" TENSORRT_SDK_RANGE)"
  sdk_sha="$(trtmc_read_docker_arg "$dockerfile" TENSORRT_SDK_SHA256)"
  if [ -z "$sdk_url" ] || [ -z "$sdk_range" ] || [ -z "$sdk_sha" ]; then
    echo "ERROR: Missing pinned TensorRT SDK metadata in $dockerfile" >&2
    return 1
  fi

  cache_dir="${TRTMC_TENSORRT_SDK_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/trtmc}"
  cache_file="$cache_dir/tensorrt-sdk-$sdk_sha.tar.zst"
  mkdir -p "$cache_dir" "$(dirname "$destination")"

  if [ ! -f "$cache_file" ] ||
     ! echo "$sdk_sha  $cache_file" | sha256sum --check --status; then
    trtmc_load_artifactory_credentials
    temporary_file="$cache_file.tmp.$$"
    rm -f "$temporary_file"
    trap 'rm -f "$temporary_file"' RETURN
    curl --fail --location --silent --show-error \
      --user "${TRTMC_ARTIFACTORY_USERNAME}:${TRTMC_ARTIFACTORY_PASSWORD}" \
      --range "$sdk_range" \
      --output "$temporary_file" \
      "$sdk_url"
    echo "$sdk_sha  $temporary_file" | sha256sum --check
    mv "$temporary_file" "$cache_file"
    trap - RETURN
  fi

  rm -f "$destination"
  cp --reflink=auto "$cache_file" "$destination"
  echo "$sdk_sha  $destination" | sha256sum --check --status
}

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Used only by the maintainer-owned TensorRT SDK publishing path. Ordinary
# development and CI builds pull the mirrored SDK from GHCR.
trtmc_load_artifactory_credentials() {
  if [ -n "${TRTMC_ARTIFACTORY_USERNAME:-}" ] &&
     [ -n "${TRTMC_ARTIFACTORY_PASSWORD:-}" ]; then
    export TRTMC_ARTIFACTORY_USERNAME TRTMC_ARTIFACTORY_PASSWORD
    return 0
  fi

  local credential_file="${TRTMC_ARTIFACTORY_CREDENTIAL_FILE:-}"
  if [ -z "$credential_file" ]; then
    echo "ERROR: Set TRTMC_ARTIFACTORY_USERNAME and TRTMC_ARTIFACTORY_PASSWORD," >&2
    echo "       or set TRTMC_ARTIFACTORY_CREDENTIAL_FILE to a two-line credential file." >&2
    return 1
  fi
  if [ ! -r "$credential_file" ]; then
    echo "ERROR: Artifactory credential file is not readable: $credential_file" >&2
    return 1
  fi

  local -a credentials=()
  mapfile -t credentials < "$credential_file"
  if [ "${#credentials[@]}" -lt 2 ] ||
     [ -z "${credentials[0]}" ] || [ -z "${credentials[1]}" ]; then
    echo "ERROR: Artifactory credential file must contain username and password on separate lines." >&2
    return 1
  fi

  TRTMC_ARTIFACTORY_USERNAME="${credentials[0]}"
  TRTMC_ARTIFACTORY_PASSWORD="${credentials[1]}"
  export TRTMC_ARTIFACTORY_USERNAME TRTMC_ARTIFACTORY_PASSWORD
}

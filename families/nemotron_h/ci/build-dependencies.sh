#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
mode=locked
max_jobs=8
image=trtmc-nemotron-h-dependencies:local
base_image=trtmc-community-base:local
from_base=""
oci_source=${TRTMC_OCI_SOURCE:-https://github.com/NVIDIA/TensorRT-Model-Connect}
output=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --bootstrap) mode=bootstrap; shift ;;
    --max-jobs|--output|--image|--base-image|--from-base|--oci-source)
      if [ "$#" -lt 2 ]; then echo "Missing option value" >&2; exit 2; fi
      case "$1" in
        --max-jobs) max_jobs="$2" ;;
        --output) output="$2" ;;
        --image) image="$2" ;;
        --base-image) base_image="$2" ;;
        --from-base)
          if ! [[ "$2" =~ ^sha256:[0-9a-f]{64}$ ]]; then
            echo "--from-base requires an immutable local image ID" >&2; exit 2
          fi
          from_base="$2" ;;
        --oci-source) oci_source="$2" ;;
      esac
      shift 2 ;;
    --help)
      echo 'Usage: build-dependencies.sh [--bootstrap] --output DIRECTORY [--max-jobs N] [--image TAG] [--base-image TAG] [--from-base sha256:LOCAL_IMAGE_ID] [--oci-source HTTPS_URL]'
      exit 0 ;;
    *) echo "Unknown option" >&2; exit 2 ;;
  esac
done
if [ -z "$output" ] || ! [[ "$max_jobs" =~ ^[1-9][0-9]*$ ]] || [ "$max_jobs" -gt 128 ]; then
  echo "Require --output and --max-jobs between 1 and 128" >&2; exit 2
fi
if [ "$(uname -m)" != x86_64 ]; then echo "Use a native Linux x86_64 build host" >&2; exit 2; fi
python3 - "$oci_source" <<'PY'
import sys, urllib.parse
url = urllib.parse.urlsplit(sys.argv[1])
if url.scheme != 'https' or not url.hostname or url.username or url.password or url.query or url.fragment:
    raise SystemExit('OCI source must be an HTTPS repository URL without credentials or query data')
PY
repository=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
family="$repository/families/nemotron_h"
snapshot=20261008T000000Z
if [ "$mode" = locked ] || [ -n "$from_base" ]; then
  python3 "$repository/requirements/image-environment.py" validate-lock --lock "$repository/requirements/community-gpu-linux-amd64.lock"
fi
if [ "$mode" = locked ]; then
  python3 "$repository/requirements/image-environment.py" validate-lock --lock "$family/ci/environment-linux-amd64.lock"
fi
mkdir -p -- "$output"
output=$(cd -- "$output" && pwd)
recipe_context=$(mktemp -d)
base_check_context=""
trap 'rm -rf -- "$recipe_context"; if [ -n "$base_check_context" ]; then rm -rf -- "$base_check_context"; fi' EXIT
cp -- "$family/requirements.txt" "$recipe_context/requirements.txt"
cp -- "$family/ci/Dockerfile.dependencies" "$recipe_context/Dockerfile"
cp -- "$family/ci/constraints-linux-amd64.txt" "$recipe_context/constraints-linux-amd64.txt"
cp -- "$family/ci/environment-linux-amd64.lock" "$recipe_context/environment-linux-amd64.lock"
cp -- "$family/ci/environment-linux-amd64.json" "$recipe_context/environment-linux-amd64.json"
revision=$(git -C "$repository" rev-parse HEAD)
python3 - "$repository" "$recipe_context" "$mode" "$revision" <<'PY'
import hashlib, json, pathlib, sys
repository, output = map(pathlib.Path, sys.argv[1:3])
paths = ['Dockerfile.dev.x86-gpu', 'requirements/community-ci.txt',
         'requirements/image-environment.py', 'requirements/community-gpu-linux-amd64.lock',
         'requirements/community-gpu-linux-amd64.json', 'families/nemotron_h/requirements.txt',
         'families/nemotron_h/ci/Dockerfile.dependencies', 'families/nemotron_h/ci/build-dependencies.sh',
         'families/nemotron_h/ci/constraints-linux-amd64.txt',
         'families/nemotron_h/ci/environment-linux-amd64.lock',
         'families/nemotron_h/ci/environment-linux-amd64.json']
record = {'schema_version': 1, 'source_sha': sys.argv[4], 'mode': sys.argv[3],
          'inputs': {path: hashlib.sha256((repository / path).read_bytes()).hexdigest() for path in paths},
          'native_byok_passed': False, 'family_e2e_passed': False,
          'qualification': 'Build-only environment capture; GPU/model qualification is separate'}
(output / 'build-inputs.json').write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
print(json.dumps({'mode': record['mode'], 'source_sha': record['source_sha'], 'input_files': len(paths)}))
PY
if [ -n "$from_base" ]; then
  identity=$(docker image inspect --format '{{.Id}} {{.Os}} {{.Architecture}}' "$from_base")
  if [ "$identity" != "$from_base linux amd64" ]; then
    echo "The existing base must be the exact local Linux amd64 image" >&2; exit 1
  fi
  base_id="$from_base"
  base_check_context=$(mktemp -d)
  cp -- "$repository/requirements/image-environment.py" "$base_check_context/image-environment.py"
  cp -- "$repository/requirements/community-gpu-linux-amd64.lock" "$base_check_context/environment.lock"
  cp -- "$repository/requirements/community-gpu-linux-amd64.json" "$base_check_context/environment.json"
  # Check the current public package, APT and ABI contract before extending a cached base.
  docker run --rm --network none --volume "$base_check_context:/opt/trtmc-base-verify:ro" \
    --entrypoint /opt/venv/bin/python "$base_id" /opt/trtmc-base-verify/image-environment.py verify \
    --lock /opt/trtmc-base-verify/environment.lock \
    --receipt /opt/trtmc-base-verify/environment.json --snapshot "$snapshot"
else
  docker build --platform linux/amd64 --file "$repository/Dockerfile.dev.x86-gpu" \
    --build-arg "TRTMC_ENV_LOCK_MODE=$mode" --build-arg "UBUNTU_SNAPSHOT=$snapshot" \
    --label "org.opencontainers.image.source=$oci_source" \
    --label "org.opencontainers.image.revision=$revision" --tag "$base_image" "$repository/requirements"
  base_id=$(docker image inspect --format '{{.Id}}' "$base_image")
fi
if ! [[ "$base_id" =~ ^sha256:[0-9a-f]{64}$ ]]; then echo "Base image ID is invalid" >&2; exit 1; fi
frozen_base="trtmc-nemotron-h-base:${base_id#sha256:}"
docker tag "$base_id" "$frozen_base"
docker build --platform linux/amd64 --file "$recipe_context/Dockerfile" \
  --build-arg "BASE_IMAGE=$frozen_base" --build-arg "MAX_JOBS=$max_jobs" \
  --build-arg "TRTMC_ENV_LOCK_MODE=$mode" --build-arg "UBUNTU_SNAPSHOT=$snapshot" \
  --label "org.opencontainers.image.source=$oci_source" \
  --label "org.opencontainers.image.revision=$revision" --tag "$image" "$recipe_context"
for kind in base family; do
  current_image="$image"
  if [ "$kind" = base ]; then current_image="$base_id"; fi
  docker run --rm --network none "$current_image" python -m pip check
  docker run --rm --network none --volume "$output:/output" "$current_image" \
    python /opt/trtmc-ci/image-environment.py capture --snapshot "$snapshot" \
    --lock "/output/$kind-environment.lock" --receipt "/output/$kind-environment.json"
done
cp -- "$recipe_context/build-inputs.json" "$output/build-inputs.json"

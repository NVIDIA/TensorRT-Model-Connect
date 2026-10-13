# FoundationPose Community GPU environment

This layer extends the shared base only when FoundationPose is selected. It
installs the existing `../requirements.txt` using binary packages, downloads the
two public NGC files declared in
`../tests/manifests/foundationpose-ngc-1.0.1.json`, and sets
`TRTMC_FOUNDATIONPOSE_MODEL_DIR=/opt/foundationpose-model` for the existing test.
The image build checks every file's SHA-256, ONNX structure, and CPU ONNX Runtime
loading before the Community GPU coordinator starts the test entry point.

| NGC version | File | SHA-256 |
| --- | --- | --- |
| `1.0.1_onnx` | `refine_model.onnx` | `dcc695a19c4bcfe5e1d909a22d8f652d8ec8bab1e19bd1544c6b45f2d3595cf7` |
| `1.0.1_onnx` | `score_model.onnx` | `0bf1026c0db7320ebf9a548ecf0d3c810c8dbd377948630bd3e5af1d49440503` |

The family layer keeps its NumPy 1.26 requirement and pins the compatible
`ml-dtypes==0.5.4` dependency for ONNX. These changes belong to
the FoundationPose image and do not modify the shared base. Downloaded models
are build outputs, not repository files. Model validation and its thresholds
remain in the existing family tests.

## Reproduce locally

On an x86_64 host, run the following from this repository checkout. The commands
build the shared base from its public recipe and then prepare the same family
context used by the coordinator, including `../requirements.txt`. No private
image access is required.

```bash
(
  set -euo pipefail
  base_revision=14385e12d26a52fb6a07f5d09eacd6bee7cdd98e
  build_tmp="$(mktemp -d)"
  trap 'rm -rf "$build_tmp"' EXIT
  git init --bare --quiet "$build_tmp/source.git"
  git --git-dir="$build_tmp/source.git" fetch --no-tags --depth=1 \
    https://github.com/NVIDIA/TensorRT-Model-Connect.git "$base_revision"
  test "$(git --git-dir="$build_tmp/source.git" rev-parse FETCH_HEAD)" = "$base_revision"
  mkdir "$build_tmp/base" "$build_tmp/family"
  git --git-dir="$build_tmp/source.git" archive FETCH_HEAD -- \
    Dockerfile.dev.x86-gpu \
    requirements/community-ci.txt \
    requirements/image-environment.py \
    requirements/community-gpu-linux-amd64.lock \
    requirements/community-gpu-linux-amd64.json | \
    tar -xf - -C "$build_tmp/base"
  docker build --platform linux/amd64 \
    -f "$build_tmp/base/Dockerfile.dev.x86-gpu" \
    -t trtmc-community-base:local "$build_tmp/base/requirements"
  cp -a families/foundationpose/ci/. "$build_tmp/family/"
  cp families/foundationpose/requirements.txt "$build_tmp/family/requirements.txt"
  docker build --platform linux/amd64 \
    -f "$build_tmp/family/Dockerfile.dependencies" \
    --build-arg BASE_IMAGE=trtmc-community-base:local \
    -t trtmc-foundationpose:local "$build_tmp/family"
)
```

The resulting image contains the model files needed by the offline test
container. A successful build proves environment preparation only; GPU E2E
still requires the native runtime and the selected FoundationPose test case.

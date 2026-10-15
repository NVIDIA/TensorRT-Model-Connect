# Deploy DINOv3 ViT-B/16 at 736x736

PR [#1594](https://github.com/NVIDIA/TensorRT-Model-Connect/pull/1594)
adds native TensorRT image-feature extraction for the public timm DINOv3
ViT-B/16 checkpoint at a fixed 736x736 engine shape. The build goes directly
from the checkpoint to a TensorRT-Model-Connect bundle; it does not use ONNX.

This source deployment path supports Linux x86_64 and aarch64/SBSA hosts with
an NVIDIA GPU supported by the pinned CUDA and TensorRT stack. A100 and H200
are measured configurations. Other supported devices, including GB300, use
the same target-local build flow and must pass the smoke and accuracy checks
before being recorded as qualified.

Build the bundle separately for every deployment GPU model and
TensorRT/runtime cohort. Do not reuse an engine built for one GPU model or
TensorRT version on another. Default TensorRT plans are not portable across
GPU models or TensorRT versions.

Prerequisites are Git, Docker, NVIDIA Container Toolkit, a compatible NVIDIA
driver, and network access for the first container and checkpoint download.
The inference step itself can run without checkpoint or Hugging Face access
after the bundle and matching native runtime are present.

## Deployment contract

| Item | Value |
|---|---|
| Checkpoint | `timm/vit_base_patch16_dinov3_qkvb.lvd1689m` |
| Revision | `f1476ea4e6ff556b5360f09a1921fd721a1b0b68` |
| Task | `image_features` |
| Precision | FP16 network with FP32 attention softmax and FP32 outputs |
| Engine input | Batch 1, RGB, 736x736 |
| Hidden-state output | `[1, 2121, 768]` |
| Pooler output | `[1, 768]`, equal to the CLS token |

The 2,121 output tokens are ordered as one CLS token, four register tokens,
and 2,116 patch tokens from the 46x46 patch grid. The native CLI decodes the
source image as RGB float32 in `[0, 1]`. The DINOv3 runtime resizes it to
736x736 with bilinear filtering and applies the checkpoint normalization.

PR #1594 contains source support only. Until it is merged into a release,
installing an older TensorRT-Model-Connect wheel will not provide this model
builder or its matching DINOv3 runtime library.

## 1. Check out the source

Use a `main` or release revision that contains PR #1594 after the change is
merged. To evaluate the unmerged PR directly:

```bash
git clone https://github.com/NVIDIA/TensorRT-Model-Connect.git
cd TensorRT-Model-Connect
git fetch origin pull/1594/head:pr-1594
git checkout pr-1594
```

Keep the builder and native runtime at the same source revision. Do not build
the bundle from this PR and then load it with an older family library.

## 2. Start the GPU development container

These commands select the repository development image for the host
architecture and expose one selected GPU to the container:

```bash
GPU=0
TRTMC_SM="$(
  nvidia-smi -i "$GPU" \
    --query-gpu=compute_cap \
    --format=csv,noheader,nounits |
  tr -d '.[:space:]'
)"

case "$(uname -m)" in
  x86_64) DOCKERFILE=Dockerfile.dev.x86 ;;
  aarch64) DOCKERFILE=Dockerfile.dev.aarch64 ;;
  *) echo "Unsupported host architecture: $(uname -m)" >&2; exit 1 ;;
esac

DOCKER_ROOT="$(docker info --format '{{.DockerRootDir}}')"
df -h "$DOCKER_ROOT"
docker system df

docker build \
  -f "$DOCKERFILE" \
  -t trtmc-dinov3:pr-1594 \
  requirements

SOURCE_DIR="$(git rev-parse --show-toplevel)"

docker run --rm -it \
  --gpus "device=${GPU}" \
  --ipc=host \
  --mount "type=bind,source=${SOURCE_DIR},target=/src" \
  --workdir /src \
  --env TRTMC_SM="$TRTMC_SM" \
  trtmc-dinov3:pr-1594 \
  bash
```

Run the remaining commands inside this container.

## 3. Build the DINOv3 runtime

```bash
python -m pip install --no-deps -e . -C py-only=true
python -m pip install --requirement families/dinov3/requirements.txt

TRTMC_BUILD_DIR="build-sm${TRTMC_SM}"

cmake -S . -B "$TRTMC_BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="${TRTMC_SM}-real" \
  -DTRTMC_BUILD_BACKEND_RTX=OFF \
  -DTRTMC_BUILD_TESTS=OFF \
  -DTRTMC_BUILD_EXAMPLES=OFF

cmake --build "$TRTMC_BUILD_DIR" --parallel "$(nproc)" --target \
  trtmc \
  trtmc_backend_trt \
  trtmc_model_dinov3

export TRTMC_BINARY="$PWD/$TRTMC_BUILD_DIR/trtmc"
export TRTMC_RUNTIME_ROOT="$PWD/$TRTMC_BUILD_DIR"

test -x "$TRTMC_BINARY"
test -f "$TRTMC_RUNTIME_ROOT/libtrtmc_core.so"
test -f "$TRTMC_RUNTIME_ROOT/libtrtmc_runtime.so"
test -f "$TRTMC_RUNTIME_ROOT/libtrtmc_backend_trt.so"
test -f "$TRTMC_RUNTIME_ROOT/libtrtmc_model_dinov3.so"
```

`TRTMC_RUNTIME_ROOT` must contain the core runtime, TensorRT backend, and
DINOv3 family shared libraries produced by this build.

## 4. Build the TensorRT bundle

The builder downloads the exact public checkpoint revision from Hugging Face
when it is not already cached:

```bash
MODEL_ID="timm/vit_base_patch16_dinov3_qkvb.lvd1689m"
MODEL_REVISION="f1476ea4e6ff556b5360f09a1921fd721a1b0b68"
BUNDLE="$PWD/dinov3-vitb16-736.bundle"

python -m tensorrt_model_connect build "$MODEL_ID" \
  --revision "$MODEL_REVISION" \
  --family dinov3 \
  --task image_features \
  --backend trt \
  --precision fp16 \
  --image-height 736 \
  --image-width 736 \
  --max-sequence-length 1 \
  --max-batch-size 1 \
  --output "$BUNDLE"
```

Inspect the completed bundle before deployment:

```bash
"$TRTMC_BINARY" inspect "$BUNDLE"
```

The bundle header must report family `dinov3`, task `image_features`, and
backend `trt`.

## 5. Extract image features

```bash
IMAGE="$PWD/families/dinov3/tests/data/test_img.jpeg"
OUTPUT_JSON="$PWD/dinov3-features.json"

"$TRTMC_BINARY" extract-features "$BUNDLE" \
  --runtime-root "$TRTMC_RUNTIME_ROOT" \
  --image "$IMAGE" \
  > "$OUTPUT_JSON"
```

Check the output contract without printing the complete feature tensor:

```bash
python - "$OUTPUT_JSON" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    result = json.load(stream)

print("last_hidden_state_shape:", result["last_hidden_state_shape"])
print("pooler_output_shape:", result["pooler_output_shape"])

assert result["last_hidden_state_shape"] == [1, 2121, 768]
assert result["pooler_output_shape"] == [1, 768]
assert len(result["last_hidden_state"]) == 1 * 2121 * 768
assert len(result["pooler_output"]) == 768
assert result["pooler_output"] == result["last_hidden_state"][:768]
PY
```

The repository image above provides a copy-paste smoke input. Replace `IMAGE`
with an image path visible inside the container for application testing.

`extract-features` writes the complete FP32 hidden state and pooler output as
flat row-major JSON arrays, so the output file is large. Use the accompanying
shape fields to reconstruct each tensor. Once the bundle and matching native
runtime are present, inference does not need Python, timm, Hugging Face access,
or the original checkpoint.

Do not pass the CLI `--cuda-graphs` option with the standard TensorRT backend.
The static DINOv3 pipeline enables its supported CUDA Graph path itself; that
CLI option belongs to the TensorRT-RTX runtime path.

PR #1594 does not add an HTTP endpoint for `image_features`; `trtmc-server`
serves text-generation bundles. Use the native `extract-features` command for
this deployment.

## Performance boundary

The latency reported in PR #1594 measures GPU-resident model forward only. The
public CLI additionally performs image decode, resize and normalization,
host-to-device input transfer, device-to-host output transfer, synchronization,
and JSON serialization. Measure those separately when reporting application
latency.

Increasing batch size is not a deployment option for this bundle: the
qualified profile is fixed at batch 1. Other image sizes, precisions, batch
sizes, and checkpoints require a separate build and qualification.

## Troubleshooting

- If family resolution rejects the checkpoint, confirm that the checkout is
  PR #1594 or a later revision containing it, and keep the pinned checkpoint
  revision in the build command.
- If runtime loading reports a missing DSO, verify that
  `TRTMC_RUNTIME_ROOT` is the exact build directory containing
  `libtrtmc_backend_trt.so` and `libtrtmc_model_dinov3.so`.
- If TensorRT cannot deserialize the plan, rebuild the bundle on the deployment
  GPU with the deployment TensorRT version.
- If the image build fails with `no space left on device`, inspect the Docker
  root reported in step 2 and remove only confirmed-unused Docker resources.
- If image decoding fails, use a format supported by the native CLI decoder
  and confirm that the file is readable inside the container.

The checkpoint is distributed separately from this repository and is not
covered by this repository's Apache-2.0 license. Review the
[checkpoint model card](https://huggingface.co/timm/vit_base_patch16_dinov3_qkvb.lvd1689m)
and [DINOv3 license](https://ai.meta.com/resources/models-and-libraries/dinov3-license/)
before downloading or deploying the weights.

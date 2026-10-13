# LANCE image reference environment

This layer extends the shared Community GPU base when LANCE is selected. The
native builder keeps using `/opt/venv`; the existing family reference adapter
uses `TRTMC_REFERENCE_PYTHON=/opt/lance-reference/bin/python`. The second venv
does not inherit system packages, so the reference's Transformers 4.49 and
NumPy 1.26 do not replace the builder's dependencies.

`reference-requirements.lock` contains the complete CPython 3.12, Linux x86_64
wheel closure. Installation requires every recorded SHA-256 and disables both
index resolution and source distributions. A missing wheel or failed import
therefore fails environment preparation instead of starting a long extension
build. `verify_reference_environment.py` checks imports and the Torch/CUDA/C++
ABI combination during the image build; it does not claim GPU or model parity.

The GPU binaries come from the following public releases:

| Package | Binary | SHA-256 |
| --- | --- | --- |
| Torch | `2.10.0+cu130`, CPython 3.12, manylinux 2.28 x86_64 | `858f0cbcc78d726fea9499eb3464faa98392fa093845a3262209bd226b7844d6` |
| Torchvision | `0.25.0+cu130`, CPython 3.12, manylinux 2.28 x86_64 | `c2f5e38f0cd57a2796e4503c0f13365deba01dbc167ef820f0beec7ca96f5f2e` |
| FlashAttention | `2.8.3+cu13torch2.10cxx11abiTRUE`, CPython 3.12, Linux x86_64 | `910d8db9def162de5b7c15474b933e7e2371e93733b980e9d3c07cd3bf2f568e` |

The [Torch CUDA 13 index](https://download.pytorch.org/whl/cu130/torch/),
[Torchvision CUDA 13 index](https://download.pytorch.org/whl/cu130/torchvision/),
and [FlashAttention 2.8.3 release](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.8.3)
publish these binaries. The lock uses their public download URLs. Torchvision's
wheel metadata requires Torch 2.10.0; the FlashAttention wheel names the same
Torch minor, CUDA major, Python ABI, architecture, and C++ ABI.

The remaining dependency versions include the family requirements and the
imports used by the pinned upstream image reference. This recipe does not
install the upstream application's unrelated training, UI, or video dependency
list. The existing image-only reference adapter handles its unused eager video
import; its behavior and all model comparison criteria remain unchanged.

## Reproduce locally

On an x86_64 host, run the following from the repository checkout containing
this family recipe. The shared base uses the public, qualified Dev recipe at
`14385e12d26a52fb6a07f5d09eacd6bee7cdd98e`, including its package locks and
environment receipt. Its five build inputs are extracted into a temporary
directory; the family layer uses this checkout's `families/lance/ci` files.
The commands do not change the checkout or require access to a private image.

```bash
(
  set -euo pipefail
  base_revision=14385e12d26a52fb6a07f5d09eacd6bee7cdd98e
  base_tmp="$(mktemp -d)"
  trap 'rm -rf "$base_tmp"' EXIT
  git init --bare --quiet "$base_tmp/source.git"
  git --git-dir="$base_tmp/source.git" fetch --no-tags --depth=1 \
    https://github.com/NVIDIA/TensorRT-Model-Connect.git "$base_revision"
  test "$(git --git-dir="$base_tmp/source.git" rev-parse FETCH_HEAD)" = "$base_revision"
  mkdir "$base_tmp/context"
  git --git-dir="$base_tmp/source.git" archive FETCH_HEAD -- \
    Dockerfile.dev.x86-gpu \
    requirements/community-ci.txt \
    requirements/image-environment.py \
    requirements/community-gpu-linux-amd64.lock \
    requirements/community-gpu-linux-amd64.json | \
    tar -xf - -C "$base_tmp/context"
  docker build --platform linux/amd64 \
    -f "$base_tmp/context/Dockerfile.dev.x86-gpu" \
    -t trtmc-community-base:local "$base_tmp/context/requirements"
  docker build --platform linux/amd64 -f families/lance/ci/Dockerfile.dependencies \
    --build-arg BASE_IMAGE=trtmc-community-base:local \
    -t trtmc-lance-reference:local families/lance/ci
)
```

The Community GPU coordinator separately stages the exact public source from
`../tests/reference-source.json` and supplies `TRTMC_REFERENCE_SOURCE_DIR`.
To prepare that same checkout for a local run:

```sh
git init /tmp/trtmc-lance-reference
git -C /tmp/trtmc-lance-reference fetch --depth=1 \
  https://github.com/bytedance/Lance.git 4baeee086648996f6ab12e673cbe461b0b149997
git -C /tmp/trtmc-lance-reference checkout --detach FETCH_HEAD
docker run --rm --gpus all \
  -v /tmp/trtmc-lance-reference:/opt/trtmc-reference:ro \
  -e TRTMC_REFERENCE_SOURCE_DIR=/opt/trtmc-reference \
  trtmc-lance-reference:local /opt/lance-reference/bin/python -c \
  'import torch; from flash_attn import flash_attn_varlen_func; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())'
```

That command checks GPU visibility and extension loading. Running the family
E2E still requires its exact cached checkpoint, input assets, native runtime,
and unchanged test selection.

## Updating the lock

Start with `reference-requirements.in` and resolve for CPython 3.12,
`x86_64-manylinux_2_28`, using public PyPI and the official CUDA 13 Torch index.
Select one compatible wheel per resolved package, record its published
SHA-256 and public URL, and add the explicitly published FlashAttention wheel
for that Torch/CUDA/Python/C++ ABI. Verify the full dependency closure, build the
layer, and run GPU E2E before considering a replacement qualified. Do not let an
unavailable binary fall back to source compilation.

# Build the Nemotron-H dependency environment from public source

The private CI builder and external users use this same public entry:

```bash
./families/nemotron_h/ci/build-dependencies.sh \
  --max-jobs 2 --output /tmp/trtmc-nemotron-environment \
  --image trtmc-nemotron-h-dependencies:local
```

Run it from a clean, pinned checkout on Linux x86_64 with Docker access and
adequate build RAM and disk. It first builds the existing
`Dockerfile.dev.x86-gpu`, starting from the public NGC TensorRT SDK pinned by
digest, then builds `Dockerfile.dependencies` from only this family's public
requirements, constraints and environment locks. TensorRT is inherited from the
NGC SDK; no private TensorRT wheel or project prebuilt-image download is needed.
The PyTorch CUDA wheels and remaining packages come from public indexes.

After the locked build, enter that locally built environment with a compatible
NVIDIA driver and container runtime:

```bash
docker run --rm -it --gpus all \
  --volume "$PWD:/workspace/tensorrt-model-connect" \
  --workdir /workspace/tensorrt-model-connect \
  trtmc-nemotron-h-dependencies:local bash
```

Mount the model checkout you intend to test. The embedded manifest's `source_sha`
is the environment recipe SHA; the mounted checkout's HEAD is the model or PR
SHA. CI records these separately because reproducing package/ABI versions does
not prove that another model revision passes the original tests. Checkpoints are
not bundled in this dependency image.

`--base-image` changes the locally built base tag. `--max-jobs` limits native
dependency compilation. `--output` receives the two version locks, two public
environment receipts, and hashes of every consumed recipe input. The entry
performs builds and `pip check` without requiring a GPU. Its receipts explicitly
do **not** claim native BYOK or model E2E qualification; those gates remain
separate before CI image admission. A successful build is not model parity proof.

The identical public input manifest is embedded in the family image at
`/opt/trtmc-ci/build-inputs.json` and copied to the output directory. It binds the
environment source commit, build mode and eleven public input hashes. It contains
no registry URL, credential or private source-label override, and never marks
native or model qualification passed. A later qualifier must verify these
environment inputs separately from the model source commit it tests.

`--oci-source` (or `TRTMC_OCI_SOURCE`) overrides the image's source label with an
HTTPS repository URL without credentials. A private publisher must supply its
own intended repository association before publishing; public environment
receipts do not include that override. This label does not set registry ACLs.

## Refreshing the environment locks

The checked-in snapshot records 83 base-venv packages, 100 family-venv packages,
and 362 installed APT packages from a real Linux x86_64 build. Use the default
locked command to reproduce this software environment. These package receipts
do not imply GPU or model qualification.

Maintainers refreshing dependencies must explicitly request a new capture:

```bash
./families/nemotron_h/ci/build-dependencies.sh --bootstrap \
  --max-jobs 2 --output /tmp/trtmc-nemotron-environment \
  --image trtmc-nemotron-h-bootstrap:local
```

Review and copy `base-environment.lock` and `.json` to
`requirements/community-gpu-linux-amd64.lock` and `.json`. Copy the family pair
to `families/nemotron_h/ci/environment-linux-amd64.lock` and `.json`, commit the
public package-only snapshot, then rerun the default locked command. The recorder
enumerates only the image venv's own distribution directories, so it does not
create conflicting duplicate pins from inherited NGC system packages. Locks
contain exact package versions, not credentials, repository login data or private
wheel coordinates. Compatibility constraints are family-owned and supplement,
rather than replace, the complete resolved lock.

Both Dockerfiles use [Ubuntu archive snapshot](https://snapshot.ubuntu.com/)
`20261008T000000Z` for added build
tools. Locked builds compare the complete installed APT inventory, venv versions
and Python/Torch/CUDA/C++ABI/TensorRT/TVM-FFI metadata with the captured receipts.
The public snapshot service has a retention window; unavailable snapshots or
package versions must fail instead of silently selecting newer inputs.

This aims to reproduce package and ABI versions from identical public inputs.
It does not promise byte-identical images or compiled wheels: compiler output,
build timestamps and image attestations may differ. A changed snapshot or input
requires a new capture, locked rebuild and GPU qualification.

Sources and checkpoints are mounted only during validation, and credentials
remain outside Docker build contexts. The helper never publishes an image or
changes a registry. Distribution of full project CI images remains private.

# Nemotron-H Community dependency image

This protected producer moves native dependency compilation out of PR GPU jobs.
It builds only the Nemotron-H dependency layer; every PR still gets one VM and
runs each selected family in a separate container.

The producer workflow checks the **triggering actor** for maintain/admin access,
accepts only a manual upstream dispatch from `main` or `ci/developer`, and uses
the existing branch-restricted `gpu-ci-dispatch` environment. It checks access
again inside the allocation job, including failed-job reruns.

The trusted Dev commit supplies provisioning, blocking cleanup, the producer
recipe and the GPU coordinator. Authorization freezes the protected `main`
commit separately. That main commit supplies the GPU base Dockerfile,
requirements and model/native test source. PR heads and merge refs are never
inputs to this producer.

The family Dockerfile declares the native dependency imports. The shared CI
controller validates the family directory name and derives its recipe, native
target and selected-family environment mechanically; it contains no Nemotron-H
dependency versions or validation policy.

## Qualification and publication

One AWS `g6.4xlarge` VM with 500 GiB disk performs the expensive x86 build and L4
qualification. The small GitHub-hosted job only controls it. The producer:

1. Builds the exact protected-main GPU base and the unchanged family requirements
   with the full dependency resolver and `pip check`.
2. Imports native dependencies on an actual L4 and records Python, TensorRT,
   Torch/CUDA/C++ABI, final TVM-FFI and the complete resolved package closure.
3. Builds the protected-main native targets with `TRTMC_ENABLE_BYOK=ON`, runs
   the native GPU bridge test, and rejects missing/skipped/failed results.
4. Runs every existing premerge Nemotron-H E2E case through the trusted Dev
   coordinator. Dependencies are already installed; no criteria are changed.
5. Only after qualification, copies a short-lived `GITHUB_TOKEN` to a private
   host file, logs in through stdin with a private temporary Docker config,
   verifies existing GHCR packages are private before pushing, verifies private
   visibility again after each push, and removes both credential locations.
6. Blocks until the owned VM is confirmed deleted, then uploads the candidate
   receipt. An independent job recovers the original lease and confirms
   deletion again without reconstructing a new attempt's allocation name.

The build Docker contexts contain only public base requirements or the family
requirements/recipe. Neither registry nor checkpoint credentials enter a build
context or image. Checkpoint credentials are consumed on the trusted staging
host before family containers. Registry credentials are introduced after those
containers finish. Later PR consumption must use a separate short-lived
`packages:read` credential on the trusted host and remove its Docker config
before launching contributor containers.

The current Mamba pin requires `apache-tvm-ffi<=0.1.9`, whereas the base includes
`0.1.12`. The producer records the final closure and tests the default native
bridge against it. It does not use `--no-deps`, override the package constraint,
or silently disable BYOK. An ABI/build/test failure prevents publication and
requires a family-owned dependency fix; a metadata-only or ARM image is not a
qualified substitute.

## Catalog promotion

A successful workflow creates a reviewable receipt, not an automatic source
commit. Admit it only after the **entire** workflow, including both cleanup
paths, completes successfully. The eventual family-owned lock entry must carry:

- Private qualified-family GHCR `image@sha256:...`, platform `linux/amd64`.
- Immutable local base-image ID and base provenance, without standalone base
  qualification or publication. The ABI/E2E gates apply to the family layer.
- Protected model commit/tree and trusted producer/helper commit.
- GPU-base Dockerfile, base requirements and family requirements SHA256.
- Dependency recipe SHA256, complete resolved closure and canonical closure SHA.
- Observed Python ABI, Torch/CUDA/C++ABI, TensorRT, final TVM-FFI, L4/SM89 and the
  native/family qualification results.

The generic consumer must read the entry from trusted CI code, compare it with
actual PR input hashes, and pull by digest. It must not trust a PR-selected image
or a mutable tag. A missing/mismatched/failed entry is an explicit dependency
qualification failure, never a skipped-green result. No digest is checked in
until a real x86/L4 producer run has established these facts.

## Dev invocation

A new `workflow_dispatch` filename may need default-branch registration before
GitHub accepts direct dispatches. The file also supports `workflow_call`, so the
already registered Community CI entry can call it on Dev without changing main.
The registered Community CI workflow routes this explicit manual mode separately
from PR snapshots, tests, and commit statuses:

```bash
gh workflow run community-ci.yml --repo NVIDIA/TensorRT-Model-Connect \
  --ref ci/developer -f task=dependency-image
```

The caller retains `cancel-in-progress: false` and passes only the named Brev and
Hugging Face secrets. The callee uses its protected job environment and a
short-lived `GITHUB_TOKEN`; it does not require a PAT.

The steps before owner cleanup total at most 300 minutes, leaving at least one
hour of the hosted job's 360 minute limit for cleanup. The producer never replaces a VM for a
package or model failure. Platform force-kill still bounds any hosted job;
the independent cleanup job is the backstop. No cron or shared family image is
introduced.

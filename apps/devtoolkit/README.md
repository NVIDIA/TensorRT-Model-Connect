# TRTMC DevToolkit

`apps/devtoolkit` is the repository-local Python API for preparing a checkout
and composing evidence-producing development environments. It is not a workflow
engine: Python is the composition language, and every operation remains
independently callable.

## Prepare a checkout-owned target

The checkout preparation layer follows the family-owned repository layout. It selects
`Dockerfile.dev.x86` or `Dockerfile.dev.aarch64`, owns one fail-closed persistent
container lifecycle, and installs only the selected family's optional
`families/<family>/requirements.txt`. The local path adopts one explicit existing
interpreter and installs the same family-owned requirements when present.

```python
from pathlib import Path
import sys

repo = Path.cwd()
sys.path.insert(0, str(repo / "apps" / "devtoolkit"))

from trtmc_devtoolkit import DevToolkit, DockerTargetPolicy

toolkit = DevToolkit.from_checkout(repo)
prepared = toolkit.prepare_docker(
    family="qwen",
    gpu="0",
    policy=DockerTargetPolicy.ENSURE,
)
print(" ".join(prepared.command("bash")))
```

A prepared target can be passed directly into the capability layer without
duplicating its container ID or workspace mapping:

```python
from trtmc_devtoolkit import EnvironmentRequest

lock = toolkit.resolve(
    EnvironmentRequest(
        tensorrt="11.1.0.106",
        target=prepared.execution_target(),
    )
)
environment = toolkit.provision(lock)
```

`ENSURE` builds or reuses the checkout image and creates a missing owned
container. `START` starts an existing matching container without rebuilding.
`ADOPT` requires an already-running match and performs no mutation. A foreign
name collision or any image, command, mount, environment, GPU, working-directory,
or IPC drift fails without deleting or replacing the container.

Requested environment values are passed to `docker create` through a temporary
mode-`0600` `--env-file`, removed immediately after the command. Values are
not placed in command arguments or toolkit error messages. `ADOPT` also skips
the selected family's dependency installation.

Local preparation uses one explicit existing system or virtual-environment
interpreter. It does not create an environment, download a toolchain, or change
CUDA, TensorRT, headers, or the driver:

```python
prepared = toolkit.prepare_local(
    python="/opt/trtmc-venv/bin/python",
    family="timm_resnet",
)
print(prepared.python)
```

Both methods return `PreparedEnvironment`. Its `command(...)` method forms a
command for the prepared container or local shell, while
`execution_target(...)` connects it to the capability layer below.

## Runnable build examples

The examples build only the model-agnostic `trtmc` CLI and TensorRT backend,
then execute the evidence-bound binary with `trtmc version`.

Docker preparation installs the repository's generic build dependencies and
uses the TensorRT version pinned by the development Dockerfile:

```bash
python3 apps/devtoolkit/examples/docker_build.py --gpu 0
```

The local example runs directly on the host. It can adopt a matching complete
CUDA/TensorRT installation or materialize a digest-pinned managed toolchain:

```bash
python3 apps/devtoolkit/examples/local_build.py \
  --python /path/to/python3.12 \
  --tensorrt 11.0.0.114
```

Local preparation intentionally does not install generic C++ or checkout
Python build dependencies. The host must provide CMake, a C++ compiler,
Ninja or Make, and nlohmann-json 3.11 or newer. The current top-level CMake
configuration also imports Torch for the SANA-WM family even when the example
does not build that family. Select an interpreter containing Torch when needed:

```bash
python3 apps/devtoolkit/examples/local_build.py \
  --python /usr/bin/python3 \
  --tensorrt 11.0.0.114 \
  --cmake-python /path/to/python-with-torch \
  --cmake-prefix-path /path/to/nlohmann-prefix
```

Both examples accept `--state-root` for receipts and isolated build state.
Run either script with `--help` for the complete set of options.

## Resolve, provision, build, and run

The capability core has four stages:

```text
EnvironmentRequest -> resolve() -> EnvironmentLock
EnvironmentLock    -> provision() -> ProvisionedEnvironment
ProvisionedEnvironment + BuildRecipe -> build() -> BuildResult
ProvisionedEnvironment + CommandSpec -> run() -> CommandResult
```

Attestation and receipts are automatic postconditions of these operations.
There is no cohort admission check in this path.

## Resolve and use the target's existing toolchain

The TensorRT request accepts any exact four-part version. When CUDA is omitted,
resolution first looks for a complete target CUDA toolkit: `nvcc`, headers,
`libcudart`, `libcublas`, and `libcurand` must all be present. If no complete
target CUDA is available, the policy falls back to managed CUDA 13.3.

```python
from pathlib import Path
import sys

repo = Path.cwd()
sys.path.insert(0, str(repo / "apps" / "devtoolkit"))

from trtmc_devtoolkit import (
    DevToolkit,
    EnvironmentRequest,
    ExecutionTarget,
    TrtmcBuildRecipe,
)

toolkit = DevToolkit.from_checkout(repo)
lock = toolkit.resolve(
    EnvironmentRequest(
        tensorrt="11.1.0.106",
        target=ExecutionTarget.local(python="python3.12", gpu="0"),
    )
)
environment = toolkit.provision(lock)
build = toolkit.build(
    environment,
    TrtmcBuildRecipe(
        targets=("trtmc", "trtmc_backend_trt", "trtmc_model_qwen"),
        outputs={"trtmc": "trtmc"},
    ),
)

toolkit.run_trtmc(environment, ("version",), build=build)
```

`build()` itself only snapshots source, serializes identical builds, runs the
selected recipe, hashes outputs, and writes evidence. `TrtmcBuildRecipe` is an
optional sample recipe that configures CMake against the resolved TensorRT
runtime and builds only the requested targets. It does not install the checkout
into or otherwise mutate the locked toolchain Python environment. User recipes
can replace it completely. The sample recipe pins `CMAKE_CUDA_COMPILER` and
`CUDAToolkit_ROOT` to the resolved runtime, disables optional BYOK by default,
queries GPU compute capability through the CUDA Driver API (with `nvidia-smi`
as a fallback), and automatically selects Ninja when available or Unix
Makefiles otherwise. These choices can all be overridden explicitly.

For a user-owned unified CUDA/TensorRT installation, pass
the prefix as toolchain-owned configuration. Resolution checks the prefix
rather than ambient host locations:

```python
request = EnvironmentRequest(
    tensorrt="11.1.0.106",
    target=ExecutionTarget.local(),
    toolchain="prefix",
    toolchain_options={"prefix": "/path/to/toolchain"},
)
```

To combine managed TensorRT artifacts with a caller-owned CUDA prefix, use
`toolchain_options={"cuda_prefix": "/path/to/cuda"}`. Execution target options
never carry toolchain configuration.

## Adopt an existing campaign container

The capability layer's built-in Docker execution provider is adoption-only. It
does not assume an NGC image, `/opt/venv`, `--gpus device=...`, or a checked-in
Dockerfile. It inspects a running container, records its image ID, and probes
its actual Python, CUDA, TensorRT Python package, native library, and headers
before producing the lock.
Docker CLI 20.10 or newer is required so command environment values can use
`docker exec --env-file` without appearing in process arguments.
The lock binds the Docker daemon ID, immutable container ID, and image ID. The
binding is rechecked before provisioning, attestation, builds, and commands, so
a recycled container name or changed Docker context fails closed.

```python
lock = toolkit.resolve(
    EnvironmentRequest(
        tensorrt="11.0.2.2",
        architecture="aarch64",
        target=ExecutionTarget.docker(
            container="jedha-campaign",
            docker_context="default",  # Omit to capture `docker context show`.
            workspace="/workspace/TensorRT-Model-Connect",
        ),
    )
)
environment = toolkit.provision(lock)

toolkit.run_trtmc(
    environment,
    ["build", "qwen3-0.6b", "--precision", "fp8", "--output", "/tmp/q.bundle"],
)
```

The CLI arguments are opaque to DevToolkit. Model-specific flags, validation,
and performance policy stay with the model family or caller recipe.
Command environment values are passed through a short-lived mode-0600 Docker
env file and removed after execution; they are never placed in Docker argv.

## Managed fallback and arbitrary TensorRT

Arbitrary-version support is accept, resolve, install, and attest. Resolution
first tries the target's installed toolchains. If none matches, the built-in
NVIDIA catalog resolves an exact public TensorRT distribution (Python package,
bindings, native libraries, and development headers) to immutable SHA-256
artifacts. With no complete target CUDA, it also resolves the CUDA 13.3 native
build component closure from NVIDIA's redistribution manifest. Provisioning
downloads those pins into a content-addressed cache and installs them under the
environment's isolated state prefix; it does not modify the target's system
Python or `/usr/local/cuda`. The public lock also pins `pip`, `setuptools`, and
`wheel`, so a minimal target whose Python lacks `ensurepip` can bootstrap its
isolated virtual environment without an OS-package install.

Versions unavailable from the public indexes, such as an internal or pre-release
build, can be supplied through a team JSON catalog. This is still automatic
installation—the manifest is discovery metadata, not a cohort allowlist:

```python
from trtmc_devtoolkit import DevToolkit, JsonToolchainCatalog
from trtmc_devtoolkit.spi import ProviderRegistry

registry = ProviderRegistry.with_builtins()
registry.register_catalog(JsonToolchainCatalog((repo / "private-toolchains.json",)))
toolkit = DevToolkit.from_checkout(repo, providers=registry.freeze())

lock = toolkit.resolve(
    EnvironmentRequest(
        tensorrt="11.1.0.106",
        target=ExecutionTarget.local(),
        toolchain_options={"catalog": "json-toolchain-catalog"},
    )
)
```

The corresponding manifest binds the private artifacts by digest and can reuse
the target's complete CUDA toolkit by major version:

```json
{
  "schema_version": 1,
  "toolchains": [
    {
      "id": "gb300-trt-11.1.0.106",
      "tensorrt": "11.1.0.106",
      "python": "3.12",
      "architecture": "x86_64",
      "cuda": {"source": "target", "major": "13"},
      "artifacts": [
        {"name": "tensorrt-bindings", "uri": "https://artifact.example/bindings.whl", "sha256": "<64 lowercase hex>"},
        {"name": "tensorrt-libs", "uri": "https://artifact.example/libs.whl", "sha256": "<64 lowercase hex>"},
        {"name": "tensorrt-headers", "uri": "https://artifact.example/headers.deb", "sha256": "<64 lowercase hex>"}
      ]
    }
  ]
}
```

For an entirely managed CUDA, use `{"source": "managed", "version":
"13.3", "release": "13.3.0", "artifacts": [...]}` and list the named CUDA
component artifacts in the record's common `artifacts` array. A private source
distribution targeting a minimal Python can likewise list digest-pinned
bootstrap wheels in the common array and reference their names through
`python_bootstrap_artifacts`. Relative artifact paths are resolved relative to
the manifest and must be reachable from the execution target. Artifact URI
userinfo cannot contain credentials. Use
pre-authorized URLs, target-visible local paths, or a custom catalog and
materializer when the artifact store requires another transport or
authentication scheme.

If neither a public nor an explicitly registered catalog can supply the exact
version, resolution raises `ArtifactUnavailable`; it never substitutes a nearby
version. Use `CudaPolicy.exact("12.8")`, `CudaPolicy.system_only()`, or
`CudaPolicy.managed("13.3")` to override the default policy.

## Qualification is explicit and source-neutral

DevToolkit does not discover repository qualification records implicitly. A
caller may attach optional qualification evidence through a source adapter;
this never controls which TensorRT version can be attempted.

```python
from trtmc_devtoolkit import JsonQualificationSource

toolkit = DevToolkit.from_checkout(
    repo,
    qualifications=(JsonQualificationSource((Path("my-qualifications"),)),),
)

# Optional provenance, fail closed only because the caller explicitly asks.
request = EnvironmentRequest(
    tensorrt="11.2.1.2",
    target=ExecutionTarget.local(),
    preset="trt112-cu133",
    require_qualification=True,
)
```

A JSON qualification record declares generic facts rather than the historical
cohort shape:

```json
{
  "id": "trt112-cu133",
  "status": "qualified",
  "requirements": {
    "tensorrt": "11.2.1.2",
    "cuda": ["13.3"],
    "architecture": ["x86_64"],
    "execution": ["local", "container"]
  }
}
```

The record's content digest is stored as provenance but does not alter the
identity of an otherwise identical environment.
Malformed qualification metadata is ignored for unrestricted resolution; it
fails closed when the caller requests a preset or requires qualification.

## Identity and evidence

| Identity | Includes | Excludes |
|---|---|---|
| Environment lock | resolved context, effective path mapping, exact Python/CUDA/TRT, provider versions, artifact digests | source revision, GPU SM, preset spelling, private locator |
| Provisioned environment | lock ID, effective execution identity, normalized toolchain runtime, observed file digests | command occurrence |
| Build request | environment ID, source snapshot, SM set, CMake/build inputs | command occurrence |
| Build result | build request ID and output digests | unrelated later runs |
| Command invocation | environment ID, arguments, path scopes, environment-value digest, build/artifact provenance | occurrence ID |

Provisioning writes `environment-lock.json`, `provision-receipt.json`, and an
observed attestation under `.devtoolkit/environments/<lock-id>/`. Builds and
commands write their own v3 receipts below that environment directory. Receipts
do not serialize provider secrets or environment variable values. JSON receipts
are replaced atomically, and provisioning for one environment ID is serialized
across processes to avoid partial or competing terminal state. Identical build
requests are also serialized across processes. A completed managed prefix is
reused after fresh attestation, and a completed build is reused only after its
receipt identity and output digests are revalidated.
Build failures before a build request ID can be computed are recorded below
`builds/preflight/` with the environment ID and failed stage.

## Extension points

There are three provider protocols:

- `ToolchainSource`: discover, materialize, and observe a CUDA/TensorRT toolchain.
- `ToolchainCatalog`: turn version intent into immutable artifacts for a
  registered materializer.
- `ExecutionContext`: resolve/provision a target and execute mapped commands.

Extension contracts live under `trtmc_devtoolkit.spi`. Execution contexts
declare semantic capabilities such as `host-filesystem` or
`container-process`; toolchain adapters select capabilities rather than
provider names. Register adapters explicitly; there is no implicit entry-point
discovery or workflow DAG.

```python
from trtmc_devtoolkit.spi import ProviderRegistry

registry = ProviderRegistry.with_builtins()
registry.register_context(MyRemoteContext())
registry.register_toolchain(MyTensorRTSource())
toolkit = DevToolkit.from_checkout(repo, providers=registry.freeze())
```

## API scope

DevToolkit exposes checkout preparation through `prepare_docker()` and
`prepare_local()`, plus the independent `resolve()`, `provision()`, `build()`,
`run()`, and `run_trtmc()` capabilities. Higher-level development flows belong
in user code or examples composed from those capabilities; DevToolkit does not
define a workflow DAG or a cohort-gated preparation API.

The shared capability layer is model-agnostic. Family-specific dependencies
remain in `families/<family>/requirements.txt`; topology, runtime orchestration,
validation, and target selection remain owned by each family or the caller's
build recipe.

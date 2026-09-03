# TRTMC DevToolkit

`scripts/devToolkit` is a repository-local Python toolkit for composing TRTMC
development environments and commands. It is not a workflow engine: Python is
the composition language, and each operation can be called independently.

The core has four stages:

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
sys.path.insert(0, str(repo / "scripts" / "devToolkit"))

from trtmc_devtoolkit import (
    DevToolkit,
    EnvironmentRequest,
    ExecutionTarget,
    TrtmcBuildRecipe,
)

toolkit = DevToolkit.from_checkout(repo)
lock = toolkit.resolve(
    EnvironmentRequest(
        tensorrt="11.2.0.113",
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
can replace it completely. The sample recipe queries GPU compute capability
through the CUDA Driver API (with `nvidia-smi` as a fallback), and automatically
selects Ninja when available or Unix Makefiles otherwise. Both can be overridden
explicitly.

For a user-owned unified CUDA/TensorRT installation, pass
the prefix as toolchain-owned configuration. Resolution checks the prefix
rather than ambient host locations:

```python
request = EnvironmentRequest(
    tensorrt="11.2.0.113",
    target=ExecutionTarget.local(),
    toolchain="prefix",
    toolchain_options={"prefix": "/path/to/toolchain"},
)
```

To combine managed TensorRT artifacts with a caller-owned CUDA prefix, use
`toolchain_options={"cuda_prefix": "/path/to/cuda"}`. Execution target options
never carry toolchain configuration.

## Adopt an existing campaign container

The built-in Docker provider is adoption-only. It does not assume an NGC image,
`/opt/venv`, `--gpus device=...`, or a checked-in Dockerfile. It inspects a
running container, records its image ID, and probes its actual Python, CUDA,
TensorRT Python package, native library, and headers before producing the lock.
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
        tensorrt="11.2.0.113",
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
      "id": "gb300-trt-11.2.0.113",
      "tensorrt": "11.2.0.113",
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

DevToolkit does not scan `configs/environment-cohorts/` by default. A caller may
attach optional qualification evidence through a source adapter; this never
controls which TensorRT version can be attempted.

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

DevToolkit exposes only environment capabilities: `resolve()`, `provision()`,
`build()`, `run()`, and `run_trtmc()`. Higher-level development flows belong in
user code or examples composed from those capabilities; DevToolkit does not
define a workflow DAG or a second cohort-gated preparation API.

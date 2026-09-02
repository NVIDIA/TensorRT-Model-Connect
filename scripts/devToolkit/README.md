# TRTMC DevToolkit

`scripts/devToolkit` is a repository-local Python toolkit for composing TRTMC
development environments and commands. It is not a workflow engine: Python is
the composition language, and each operation can be called independently.

The core has four stages:

```text
EnvironmentRequest -> resolve() -> EnvironmentLock
EnvironmentLock    -> provision() -> ProvisionedEnvironment
ProvisionedEnvironment + BuildSpec -> build() -> BuildResult
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
    BuildSpec,
    CommandSpec,
    DevToolkit,
    EnvironmentRequest,
    ExecutionTarget,
    repository_path,
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
    BuildSpec(
        targets=("trtmc", "trtmc_backend_trt", "trtmc_model_qwen"),
        outputs={"trtmc": "trtmc"},
    ),
)

toolkit.run(
    environment,
    CommandSpec(
        (build.artifacts[0].path, "version"),
        cwd=repository_path("."),
    ),
)
```

`build()` installs the checkout's Python package editable with `--no-deps`,
configures CMake against the exact observed TensorRT headers and library, and
builds only the requested targets. Dependency installation remains an explicit
environment-composition decision.

For a user-owned unified CUDA/TensorRT installation, pass
`ExecutionTarget.local(prefix="/path/to/toolchain")`. Resolution checks the
prefix rather than ambient host locations. If its CUDA is complete but its
TensorRT does not match, a pinned managed TensorRT request still follows that
prefix CUDA.

## Adopt an existing campaign container

The built-in Docker provider is adoption-only. It does not assume an NGC image,
`/opt/venv`, `--gpus device=...`, or a checked-in Dockerfile. It inspects a
running container, records its image ID, and probes its actual Python, CUDA,
TensorRT Python package, native library, and headers before producing the lock.

```python
lock = toolkit.resolve(
    EnvironmentRequest(
        tensorrt="11.0.2.2",
        architecture="aarch64",
        target=ExecutionTarget.docker(
            container="jedha-campaign",
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

## Managed fallback and arbitrary TensorRT

Arbitrary-version support is accept-and-attempt, not accept-and-download-
unverified. The built-in managed source requires a complete caller-supplied set
of wheel artifacts plus a `tensorrt-headers` Debian artifact. Every artifact
must have an immutable SHA-256 digest. With no complete system CUDA, omitted
CUDA selects managed 13.3:

```python
from trtmc_devtoolkit import ArtifactPin

lock = toolkit.resolve(
    EnvironmentRequest(
        tensorrt="11.0.0.114",
        target=ExecutionTarget.local(),
        artifacts=(
            ArtifactPin(
                name="tensorrt-headers",
                uri="https://artifact.example/libnvinfer-headers.deb",
                sha256="<64 lowercase hex characters>",
                verification="pinned-digest",
            ),
            # Include the complete, mutually compatible wheel closure.
            ArtifactPin(
                name="tensorrt-wheel",
                uri="https://artifact.example/tensorrt.whl",
                sha256="<64 lowercase hex characters>",
                verification="pinned-digest",
            ),
        ),
    )
)
```

If no trusted artifacts or custom `ToolchainSource` can satisfy the request,
resolution raises `ArtifactUnavailable`; it never silently weakens verification.
Use `CudaPolicy.exact("12.8")`, `CudaPolicy.system_only()`, or
`CudaPolicy.managed("13.3")` to override the default policy.

## Cohorts are optional qualification records

Files in `configs/environment-cohorts/` may annotate a resolved environment as
known-qualified. They do not control which TensorRT version can be attempted.
Additional record directories can be supplied with `qualification_roots=`.

```python
toolkit = DevToolkit.from_checkout(repo, qualification_roots=(Path("my-presets"),))

# Optional provenance, fail closed only because the caller explicitly asks.
request = EnvironmentRequest(
    tensorrt="11.2.1.2",
    target=ExecutionTarget.local(),
    preset="trt112-cu133",
    require_qualification=True,
)
```

The record's content digest is stored as provenance but does not alter the
identity of an otherwise identical environment.
Malformed qualification metadata is ignored for unrestricted resolution; it
fails closed when the caller requests a preset or requires qualification.

## Identity and evidence

| Identity | Includes | Excludes |
|---|---|---|
| Environment lock | resolved context, exact Python/CUDA/TRT, provider versions, artifact digests | source revision, GPU SM, preset spelling, container/GPU locator |
| Build request | environment ID, source snapshot, SM set, CMake/build inputs | command occurrence |
| Build result | build request ID and output digests | later runs |
| Command invocation | environment ID, arguments, path scopes, environment-value digest | occurrence ID |

Provisioning writes `environment-lock.json`, `provision-receipt.json`, and an
observed attestation under `.devtoolkit/environments/<lock-id>/`. Builds and
commands write their own v2 receipts below that environment directory. Receipts
do not serialize provider secrets or environment variable values.

## Extension points

There are two provider protocols:

- `ToolchainSource`: discover, materialize, and observe a CUDA/TensorRT toolchain.
- `ExecutionContext`: resolve/provision a target and execute mapped commands.

Register providers explicitly through `ProviderRegistry`; there is no implicit
entry-point discovery or workflow DAG.

```python
registry = ProviderRegistry.with_builtins()
registry.register_context(MyRemoteContext())
registry.register_toolchain(MyTensorRTSource())
toolkit = DevToolkit.from_checkout(repo, providers=registry.freeze())
```

## Legacy recipe

`DevToolkit.plan(PrepareRequest(...))` and `apply()` remain as a compatibility
recipe for the checked-in cohort-based local/Docker setup. It is not the
arbitrary-version capability API. Its old model-smoke field now fails with a
message directing callers to compose the appropriate family CLI through
`run_trtmc()`. Handoff helpers remain under `trtmc_devtoolkit.recipes` with
top-level compatibility imports.

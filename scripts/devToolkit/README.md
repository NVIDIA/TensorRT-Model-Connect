# TRTMC devToolkit

`scripts/devToolkit` is a repository-local Python API for preparing a
TensorRT-Model-Connect build and installation environment. It is not installed
as a console command and does not modify the host NVIDIA driver, CUDA toolkit,
or TensorRT installation.

The first checked-in environment cohort is TensorRT `11.1.0.106` with CUDA
`13.3` on Linux `x86_64` and `aarch64`.

## Docker development environment

```python
from pathlib import Path
import sys

repo = Path.cwd()
sys.path.insert(0, str(repo / "scripts" / "devToolkit"))

from trtmc_devtoolkit import DevToolkit, DockerTarget, PrepareRequest

toolkit = DevToolkit.from_checkout(repo)
plan = toolkit.plan(
    PrepareRequest(
        tensorrt="11.1.0.106",
        cuda="13.3",
        target=DockerTarget(gpu="0"),
        mode="development",
    )
)
result = toolkit.apply(plan)
print(result.environment.activate_command)
```

When a checkout was deployed without Git metadata (for example, by `rsync`),
pass its externally verified commit or content digest through
`source_revision_override`. The value must be 40 or 64 lowercase hexadecimal
characters and is included in the deterministic run ID and receipt.

The development layout installs the Python package editable, builds the CLI,
TensorRT backend, and model DSOs in a checkout-local build directory, and
leaves a labelled container running for later work.

Use `mode="installed"` to build a native wheel, install it into the target
environment, and validate that the installed CLI and native payload do not
fall back to the checkout.

## Local environment

```python
from trtmc_devtoolkit import DevToolkit, LocalTarget, PrepareRequest

request = PrepareRequest(
    tensorrt="11.1.0.106",
    cuda="13.3",
    target=LocalTarget(python="python3.12", gpu="0"),
    mode="development",
)

toolkit = DevToolkit.from_checkout()
result = toolkit.apply(toolkit.plan(request))
print(result.environment.activate_command)
```

Local preparation first requires an already matching CUDA/TensorRT toolchain.
It creates a managed venv under `.devtoolkit/` and never runs `apt`, changes
system library links, or installs drivers.

## Optional model smoke

Attach `ModelRequest` to build, inspect, and run one model after installation.
This is an environment smoke test, not a correctness or performance claim.

Every run writes `plan.json`, `environment.json`, `commands.log`, and either
`receipt.json` or `failure-summary.json` under `.devtoolkit/runs/<run-id>/`.
Existing containers with unrelated ownership labels are never removed or
reused.

## Downstream handoff

The toolkit can generate commands for the existing validation, profiling, and
performance owners without reimplementing their comparison or gating logic:

```python
from trtmc_devtoolkit import validation_handoff

handoff = validation_handoff(
    result,
    model="qwen3-0.6b",
    workload="qwen.generate",
    bundle=result.bundle,
    output=repo / ".devtoolkit" / "validation",
)
print(handoff.command)
```

Docker handoffs run in the prepared container. Local handoffs return the
managed environment variables together with the direct command.

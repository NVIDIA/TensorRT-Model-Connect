# TRTMC DevToolkit

`apps/devtoolkit` prepares only the TensorRT-Model-Connect checkout passed to
it. It has no environment catalog, compatibility matrix, receipt database, or
artifact identity layer.

The shared Docker path is the repository's normal path: it builds the root
`Dockerfile`, whose base image is pinned in its first `FROM`, through
`tools.ci image ensure`. The app then owns the small inspect/create/start
lifecycle for one explicitly named persistent container. It does not calculate
another image, source, dependency, wheel, plan, or bundle digest.

```python
from pathlib import Path
import sys

repo = Path.cwd()
sys.path.insert(0, str(repo / "apps/devtoolkit"))

from trtmc_devtoolkit import DevToolkit, DockerTargetPolicy

environment = DevToolkit.from_checkout(repo).prepare_docker(
    family="timm_resnet",
    gpu="0",
    policy=DockerTargetPolicy.ENSURE,
)
print(" ".join(environment.command("bash")))
```

An existing name is reused only when its checkout, requested image, immutable
image ID, command, working directory, environment, bind mounts, GPU request,
and optional IPC mode still match. A foreign collision or configuration drift
fails closed; the toolkit never removes or replaces a container. `ADOPT`
requires an already-running match, performs no mutation, and reuses any family
dependencies already present. `START` may start a stopped match without
building an image. `ENSURE` and the idempotent-retry spelling `CREATE` run the
checkout's image-ensure command and create a missing target. Other policies
install the selected family's optional requirements after the target is
running.

Requested environment values are passed to `docker create` in a mode-`0600`
temporary `--env-file`, which is removed immediately after the command. Values
are not placed in command arguments or toolkit error messages.

When `family` is present and the policy permits mutation, the toolkit installs
only that owner's optional `families/<family>/requirements.txt`. A family
without that file adds nothing to the shared image. There is no central
dependency registry.

Local preparation uses an explicit existing system or virtual-environment
interpreter. It does not create an environment, download a toolchain, or change
CUDA, TensorRT, headers, or the driver:

```python
environment = DevToolkit.from_checkout(repo).prepare_local(
    python="/opt/trtmc-venv/bin/python",
    family="timm_resnet",
)
print(environment.python)
```

Both methods return `PreparedEnvironment`. Its `command(...)` method forms a
command for the prepared container or the current local shell. Build, test, and
validation behavior remains owned by the public Source CLI and `tools.ci`.

---
sidebar_position: 6
title: Managed GPU host workspaces
---

# Managed GPU host workspaces

Shared GPU hosts use a one-worktree/one-container layout. The same workspace ID
connects the deployed source, generated artifacts, lifecycle metadata, and
Docker container, so owners and cleanup candidates can be resolved without
guessing from directory or container names.

## Host configuration

Create `~/.config/trtmc/host.env` on each managed host. Hostnames, credentials,
storage mount details, and GPU topology belong in this untracked file.

```bash
TRTMC_HOST_ROOT=/srv/trtmc
TRTMC_DOCKER_IMAGE=trtmc-dev-gb300:latest
TRTMC_CONTAINER_PREFIX=trtmc-dev-gb300
TRTMC_GPU_REQUEST=all
TRTMC_RESTART_POLICY=unless-stopped
```

Use the same `TRTMC_HOST_ROOT`, image policy, and container prefix on equivalent
hosts. `TRTMC_GPU_REQUEST` may differ when their physical GPU topology differs.

## Canonical layout

For a workspace ID such as `fix-runtime-reset`, the host tree is:

```text
$TRTMC_HOST_ROOT/
├── workspaces/fix-runtime-reset/repo/
├── runs/fix-runtime-reset/
│   ├── engines/
│   ├── results/
│   ├── logs/
│   └── tmp/
├── state/fix-runtime-reset/workspace.env
├── huggingface/
└── data/
```

The ownership rules are:

- `workspaces/<id>/repo` contains exactly one deployed worktree.
- `runs/<id>` is writable only for that worktree's generated artifacts.
- `state/<id>/workspace.env` records the worktree/container mapping.
- `huggingface` is the shared, reproducible download cache.
- `data` is shared and mounted read-only in containers.
- Existing legacy shared artifact directories remain legacy inputs. New work
  must not add files to them.

Inside every managed container, those paths are stable:

| Host content | Container path | Access |
| --- | --- | --- |
| Deployed worktree | `/workspace/tensorrt-model-connect` | Read/write |
| Workspace run directory | `/work` | Read/write |
| Hugging Face cache | `/cache/huggingface` | Read/write, shared |
| Datasets | `/mnt/data` | Read-only, shared |

`ENGINE_DIR`, `RESULT_DIR`, `TMPDIR`, `HF_HOME`, and the Hugging Face hub/module
cache variables are set to these managed paths by the container launcher.

## Create or start a workspace

Choose a stable, lowercase ID derived from the local worktree or task. Reuse the
same ID on equivalent hosts.

For a pushed branch, bootstrap the canonical checkout and container together:

```bash
./scripts/bootstrap_workspace.sh \
  --id fix-runtime-reset \
  --branch fix/runtime-reset \
  --detach
```

For source deployed by another mechanism, place it at
`$TRTMC_HOST_ROOT/workspaces/<id>/repo` and start only its container:

```bash
./scripts/manage_gpu_workspace.sh start fix-runtime-reset
```

The manager will reuse a stopped container only when its management labels and
workspace ID match. It refuses to adopt an unrelated container with the same
name.

## Work with the matching container

```bash
./scripts/manage_gpu_workspace.sh inspect fix-runtime-reset
./scripts/manage_gpu_workspace.sh shell fix-runtime-reset
./scripts/manage_gpu_workspace.sh exec fix-runtime-reset -- \
  ctest --test-dir build --output-on-failure
./scripts/manage_gpu_workspace.sh stop fix-runtime-reset
```

Stopping retains the container, source, run artifacts, caches, and state
manifest. The manager intentionally provides no delete command.

## Audit and cleanup handoff

Run the read-only audit before discussing cleanup:

```bash
./scripts/manage_gpu_workspace.sh audit
docker system df -v
```

A cleanup proposal must name exact targets and classify them:

1. Retain: running workspaces, open work, reusable shared caches, and evidence
   still needed for an active issue or pull request.
2. Review: stopped workspaces, completed run artifacts, duplicated experiment
   bundles, and caches whose future reuse is unclear.
3. Reclaimable after approval: abandoned workspaces, explicitly archived run
   artifacts, stopped containers, unreferenced images, and build cache.

Docker's `RECLAIMABLE` value is advisory. It is not proof that another user no
longer needs an image, stopped container, or cache. Deletion remains a separate
human-approved operation.

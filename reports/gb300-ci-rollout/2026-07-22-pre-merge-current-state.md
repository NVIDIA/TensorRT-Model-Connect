# GB300 CI Pre-Merge Current-State Snapshot

This report freezes the sanitized state observed immediately before the
28-slot implementation PR is eligible for review. It is rollout evidence, not
a reconstruction of the original pre-change baseline.

## Scope and limitations

- Collection window: `2026-07-22T04:52:58Z` through
  `2026-07-22T04:53:50Z`.
- Repository: `NVIDIA/TensorRT-Model-Connect`.
- GitHub `main`: `2b08f200c5602a369ac592bdeb3960bf4e5e5ce2`.
- Draft implementation PR: [#519](https://github.com/NVIDIA/TensorRT-Model-Connect/pull/519).
- PR head at collection time:
  `07f86c7eda9df1793d314d6d9e524dbfc3a49800`.
- This snapshot was collected after node labels, cache-anchor labels, standby
  labels, and host-local policy had already been installed. It therefore
  cannot prove the original runner labels, service state, or `busy=false`
  state at each earlier mutation.
- No secret value, access token, private key, complete process environment,
  raw GPU UUID, PCI identifier, transient PID, or other user's container name
  is recorded here.
- Raw API and host-command output must be retained in the access-controlled
  rollout artifact store before admission. This source report does not replace
  that raw evidence.

## GitHub runner topology

The runner API returned 33 registrations. All runners were idle at the
snapshot instant.

| Cohort | Registered | Online | Production-labelled | Cache anchors | Standby |
| --- | ---: | ---: | ---: | ---: | ---: |
| Generic compute01 runner | 1 | 1 | 0 | 0 | 0 |
| compute01 proof-00 through proof-15 | 16 | 16 | 0 | 1 | 0 |
| compute02 proof-00 through proof-11 | 12 | 12 | 12 | 1 | 0 |
| compute02 proof-12 through proof-15 | 4 | 0 | 0 | 0 | 4 |
| **Total** | **33** | **29** | **12** | **2** | **4** |

Additional invariants observed:

- Every proof registration had exactly one node label.
- Only proof-00 on each node had `trtmc-cache-anchor`.
- The legacy compute02-specific routing label was absent.
- The common production selector remained `trtmc-gb300-proof`.
- compute01 advertised no production proof capacity.
- compute02 proof-12 through proof-15 were offline, standby-labelled, and had
  no production label.
- The generic compute01 runner remained separate from the proof pool.

## Repository control plane

The repository still had the existing common proof selector and the temporary
global GPU topology variables. The global GPU-ID variable must remain until
the merged workflow has stopped injecting it and every admitted listener has
verified host-local policy.

| Contract | Snapshot result |
| --- | --- |
| Model runner selector | `["trtmc-gb300-proof"]` |
| Slots per GPU | `4` |
| Global GPU IDs | `1,2,3` (not yet removed) |
| Runner discovery App ID variable | **Absent** |
| Runner discovery private-key secret | **Absent** |
| Existing Hugging Face secret metadata | Present; value not queried |

The missing App configuration blocks trusted default-branch discovery. The
App must be repository-scoped, installed only for this repository, and have
read-only Repository Administration permission.

## Host-local policy

Both nodes reported `Linger=yes`, user-owned mode-`0644` policy files, no Hugging
Face token keys in the proof policy, four slots per allowed GPU, and the same
documented cache/reference/lock paths. Every active listener process inherited
the exact node policy. All active listeners on one node resolved the GPU lock
directory to one shared inode; the two nodes used distinct lock namespaces.

Common file hashes:

- `trtmc-github-proof@.service`:
  `dfe03a987b36967c061a459592fafe6178395f461a91c5b2f63655f982415f4a`
- `10-node-env.conf`:
  `8cf0b8df2ff5ef796d458d60dba7d98b03f8d3210c31172bcfe0ad81b7157278`

| Node | Allowed GPU indices | Policy hash | Active/enabled proof units | Lock check | GPU health summary |
| --- | --- | --- | --- | --- | --- |
| compute01 | `0,1,2,3` | `f97f04209a649d806e844b8649c439e784f3dc5456ee06c2823305583a23a3e8` | proof-00 through proof-15 | One shared same-node inode | 4/4 queryable; zero volatile uncorrected ECC; no remap/recovery action |
| compute02 | `1,2,3` | `0b6f5e7e096fb7ef92cedb074bed584d9aa68427940a7e37bf4280736f7fbabb` | proof-00 through proof-11; proof-12 through proof-15 disabled | One shared same-node inode | 4/4 queryable; zero volatile uncorrected ECC; no remap/recovery action |

compute01 proof-01 through proof-15 were already active and enabled, ahead of
the sequencing originally written in Phase 6. Their missing production label
kept this from changing current GitHub CI scheduling. Admission must still
validate each listener immediately before adding the common label.

## Cache and filesystem state

No real `rsync` process was present on either node.

| Node | Hugging Face cache bytes | Ownership exceptions | Unreadable files | Unwritable files | Unwritable directories |
| --- | ---: | ---: | ---: | ---: | ---: |
| compute01 | 1,224,911,183,872 | 0 | 0 | 0 | 0 |
| compute02 | 1,224,912,240,640 | 2,781 | 110 | 1,195 | 661 |

compute01 had approximately 7.2 TB free and representative cached model
snapshots were locally complete. It does not need a manual cache copy or broad
redownload.

compute02's content volume was present, but its measured ownership/access
exceptions block strict warm and replacement of affected entries. Repair must
target only entries not owned by `yifeif`; a blanket recursive ownership
rewrite of the full cache is not acceptable.

## External GPU ownership

- compute01 had no running Docker container and no NVIDIA compute process at
  the snapshot instant. A root GitLab runner remained active, enabled, and
  restartable, so it is a latent all-GPU scheduler conflict until coordinated.
- compute02 had no NVIDIA compute process at the snapshot instant. Existing
  live containers still had Docker device requests covering GPU 2 or all GPUs,
  including a restartable container. They must be coordinated before the
  acceptance window.
- High reported memory on compute02 GPU 3 was inactive file-page cache, not a
  CUDA process or anonymous/mapped/dirty/locked allocation. Admission should
  check active processes and mapped/anonymous state rather than require
  `memory.used == 0`.

## Repository validation evidence

- Final-head Pre-Merge run
  [29890013211](https://github.com/NVIDIA/TensorRT-Model-Connect/actions/runs/29890013211)
  passed on attempt 1 for `07f86c7e`.
- Legal, ownership/impact, source quality, source-only unit tests, five real
  concurrent compute02 model proofs, combined report certification, and the
  final gate passed.
- The five proofs used five distinct compute02 listeners and five distinct
  node-local leases. This proves compatibility with the existing 12-slot pool;
  it does not prove compute01 scheduling or 28-slot capacity.
- Earlier local test-count summaries were not retained as a raw durable log.
  A fresh exact-head focused-test receipt is required after the remaining
  trust-hardening changes are committed.

## Admission blockers at this snapshot

1. Configure and install the read-only runner-discovery GitHub App, then add
   its App ID variable and private-key secret.
2. Repair only the measured compute02 cache ownership/access exceptions and
   rerun the strict audit.
3. Coordinate the compute01 GitLab scheduler and compute02 external Docker GPU
   claims for a declared drained acceptance window.
4. Complete review and final-head CI for the hardened draft PR.
5. Retain the raw pre-admission runner/host snapshot with hashes in the
   access-controlled rollout evidence store.

Until all five blockers are cleared, keep PR #519 in draft, keep compute01
without the production label, and do not merge or dispatch the full hardware
canaries.

## Collection method

The snapshot used read-only GitHub runner, variable, secret-name, branch, PR,
and check APIs. Host checks used read-only `systemctl --user`, `loginctl`,
`sha256sum`, `stat`, `nvidia-smi`, `df`, `du`, `find`, and process queries over
batch SSH. No label, repository setting, service, container, process, file
ownership, cache content, or workflow run was changed during collection.

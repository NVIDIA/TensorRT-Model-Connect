# TRTMC Report Hub

Report Hub turns immutable accuracy and performance reports into an auditable QA
workflow. It indexes `report.json`, preserves links to `report.html`, tracks one
finding across repeated runs, and keeps human decisions separate from generated
evidence.

The UI has four workspaces:

- **QA Findings** — normalize a run, filter failures, assign owner/severity/tags,
  record conclusions, and associate repository work.
- **Test Plan** — prepare a Hub-local plan and review coverage before any future
  DevTest publication.
- **Defect Handoff** — prepare a developer-ready brief and link a GitHub, NVBug,
  or DevTest record after it exists.
- **Trash** — hide a report from active views, restore it during retention, or
  request a purge preflight after retention.

## Evidence and authority

`report.json`, `report.html`, logs, and artifacts remain immutable evidence in
the report store. Report Hub stores only its catalog cache, normalized
observations, findings, triage, drafts, links, lifecycle state, and audit events.

Status values are not mirrored between systems. Hub owns QA triage; DevTest
owns test execution; NVBug owns defect disposition; GitHub owns repository work.
Conflicts should be shown to QA instead of silently overwriting either side.

The service never changes a validation threshold and never writes back into a
report. AI-generated suggestions belong in local drafts and still require a
human decision.

## Run locally

Python 3.10 or newer is sufficient; the service has no third-party runtime
dependencies.

```bash
python3 -m tools.report_hub \
  --dev \
  --database /tmp/report-hub.sqlite3 \
  --evidence-root https://reports.example.test/trtmc
```

Development authentication is accepted only on a loopback bind. Open
`http://127.0.0.1:4180`. Use a disposable database for development because its
identity has the `admin` role.

## Production configuration

Run exactly one application instance behind the company authentication reverse
proxy. The proxy must remove client-supplied identity headers and set trusted
values itself.

| Environment variable | Required | Purpose |
| --- | --- | --- |
| `REPORT_HUB_EVIDENCE_ROOT_URL` | yes | HTTP root containing `benchmark/`, `perf/`, and the catalog |
| `REPORT_HUB_CATALOG_URL` | no | Override the default `report-browser/reports-index.json` location |
| `REPORT_HUB_DATABASE` | yes | Persistent SQLite path |
| `REPORT_HUB_SECRET` | yes | Random secret for per-user mutation tokens |
| `REPORT_HUB_USER_HEADER` | no | Trusted user header; defaults to `X-Forwarded-User` |
| `REPORT_HUB_ROLE_HEADER` | no | Trusted role header; defaults to `X-Report-Hub-Role` |
| `REPORT_HUB_RETENTION_DAYS` | no | Trash retention; defaults to 30 days |
| `REPORT_HUB_MAX_REPORT_BYTES` | no | Maximum JSON response; defaults to 8 MiB |
| `REPORT_HUB_EXTERNAL_HOSTS` | no | Comma-separated HTTPS hosts allowed in manual links |

Accepted roles are `viewer`, `qa`, and `admin`. A `qa` user can sync, analyze,
triage, save drafts, link records, trash, and restore. Only `admin` can schedule
a purge. The default bind is loopback so identity headers cannot be sent directly
from another host.

Build the isolated image from the repository root:

```bash
docker build -f tools/report_hub/Dockerfile -t trtmc-report-hub:local .
docker run --rm \
  --name trtmc-report-hub \
  --network report-hub-internal \
  -v report-hub-data:/var/lib/report-hub \
  -e REPORT_HUB_EVIDENCE_ROOT_URL=https://reports.example.test/trtmc \
  -e REPORT_HUB_SECRET=replace-with-a-random-secret \
  trtmc-report-hub:local
```

Terminate TLS, authentication, body-size limits, request timeouts, and rate
limits at the reverse proxy. Do not publish the application port directly.

## Safe deletion

The state sequence is:

```text
active -> trashed -> purge_scheduled -> purged
            |                ^
            +-- restore -----+
```

Moving to Trash requires a reason, exact folder-name confirmation, a valid QA
identity, a mutation token, and the current record version. It changes Hub
visibility only; evidence bytes are untouched. Catalog refresh never restores a
trashed run.

Purge scheduling additionally requires an admin, retention expiry, the exact
folder name, an irreversibility acknowledgement, no open findings, and no
external references. This release deliberately provides no physical storage
worker, so it cannot permanently delete evidence. A future worker must resolve
the constrained storage target, verify its checksum, use the approved storage
tool, verify absence, and write the permanent tombstone.

## External-system adoption

All adapters fail closed in this release:

1. **Hub only** — local triage, tags, plan/defect drafts, and manual links.
2. **Link and read** — add authenticated read adapters and store snapshots with
   their source revision and last-sync time.
3. **Approved publish** — preview the exact mutation, require explicit human
   approval, reserve a persistent idempotency key, publish once, and read the
   created record back.

`integrations.py` defines the adapter boundary. `Store.prepare_adapter_operation`
rejects reuse of an idempotency key with a different request. A future adapter
must use that reservation before a DevTest, NVBug, or GitHub mutation; enabling
an adapter must be an explicit deployment change, never a fallback to raw REST.

## Operations

- Health check: `GET /api/v1/health` (does not require identity).
- The first QA session syncs the catalog if the database is empty. Later syncs
  are explicit and audited.
- Back up SQLite using its online backup command, not a plain copy while the
  service is writing: `sqlite3 DB_PATH '.backup BACKUP_PATH'`.
- Back up the database before upgrades. Schema migrations fail closed when the
  database is newer than the running service.
- Keep the database volume encrypted and access-controlled; triage notes and
  links may contain internal engineering context.
- Restore tests should validate the database, audit sequence, Trash state, and
  drafts—not only that the file can be opened.

## Validation

```bash
python3 -m pytest -q tests/tools/test_report_hub.py
python3 -m ruff check tools/report_hub tests/tools/test_report_hub.py
```

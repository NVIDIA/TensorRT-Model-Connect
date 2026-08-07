/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

const app = document.querySelector("#app");

const state = {
  session: null,
  integrations: [],
  source: "benchmark",
  runs: [],
  selectedRunId: null,
  findings: [],
  selectedFindingId: null,
  evidence: null,
  filter: "failed",
  query: "",
  planDraft: null,
  defectDraft: null,
  links: [],
  audit: [],
  trash: [],
  busy: false,
  dialog: null,
  toasts: [],
};

const routes = new Set(["findings", "test-plans", "defects", "trash"]);

function route() {
  const value = location.hash.replace(/^#\/?/, "") || "findings";
  return routes.has(value) ? value : "findings";
}

function h(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function can(minimum) {
  const rank = { viewer: 0, qa: 1, admin: 2 };
  return rank[state.session?.role] >= rank[minimum];
}

function selectedRun() {
  return state.runs.find((item) => item.id === state.selectedRunId) || null;
}

function selectedFinding() {
  return state.findings.find((item) => item.id === state.selectedFindingId) || null;
}

async function api(path, options = {}) {
  const headers = { Accept: "application/json" };
  if (options.body !== undefined) {
    headers["Content-Type"] = "application/json";
  }
  if (options.method && options.method !== "GET") {
    headers["X-Report-Hub-CSRF"] = state.session?.csrf_token || "";
  }
  const response = await fetch(path, {
    method: options.method || "GET",
    headers,
    body: options.body === undefined ? undefined : JSON.stringify(options.body),
  });
  let payload = {};
  try {
    payload = await response.json();
  } catch (_error) {
    // The status below still produces a useful transport error.
  }
  if (!response.ok) {
    throw new Error(payload?.error?.message || `Request failed with HTTP ${response.status}`);
  }
  return payload;
}

async function start() {
  try {
    state.session = await api("/api/v1/session");
    const integrations = await api("/api/v1/integrations");
    state.integrations = integrations.integrations || [];
    await loadRuns({ syncIfEmpty: true });
    window.addEventListener("hashchange", async () => {
      await prepareRoute();
      render();
    });
    await prepareRoute();
    render();
  } catch (error) {
    app.innerHTML = `<main class="fatal"><h1>Report Hub unavailable</h1><p>${h(error.message)}</p></main>`;
  }
}

async function loadRuns({ syncIfEmpty = false } = {}) {
  let result = await api(`/api/v1/runs?source=${encodeURIComponent(state.source)}&lifecycle=active`);
  if (!result.runs.length && syncIfEmpty && can("qa")) {
    await api("/api/v1/catalog/sync", { method: "POST", body: {} });
    result = await api(`/api/v1/runs?source=${encodeURIComponent(state.source)}&lifecycle=active`);
  }
  state.runs = result.runs;
  if (!state.runs.some((item) => item.id === state.selectedRunId)) {
    state.selectedRunId = state.runs[0]?.id || null;
  }
  if (state.selectedRunId) await loadFindings({ analyzeIfEmpty: syncIfEmpty });
}

async function loadFindings({ analyzeIfEmpty = false } = {}) {
  if (!state.selectedRunId) {
    state.findings = [];
    return;
  }
  let result = await api(`/api/v1/findings?run_id=${encodeURIComponent(state.selectedRunId)}`);
  if (!result.findings.length && analyzeIfEmpty && can("qa")) {
    const analyzed = await api(`/api/v1/runs/${state.selectedRunId}/analyze`, {
      method: "POST",
      body: {},
    });
    result = { findings: analyzed.findings };
    state.evidence = analyzed.evidence;
  }
  state.findings = result.findings || [];
  if (!state.findings.some((item) => item.id === state.selectedFindingId)) {
    state.selectedFindingId =
      state.findings.find((item) => item.observation.status === "failed")?.id
      || state.findings[0]?.id
      || null;
  }
  await loadFindingContext();
}

async function loadFindingContext() {
  const finding = selectedFinding();
  if (!finding) {
    state.links = [];
    state.audit = [];
    return;
  }
  const [links, audit] = await Promise.all([
    api(`/api/v1/findings/${finding.id}/links`),
    api(`/api/v1/audit?entity_type=finding&entity_id=${finding.id}&limit=12`),
  ]);
  state.links = links.links || [];
  state.audit = audit.events || [];
}

async function prepareRoute() {
  if (route() === "trash") {
    const [trashed, scheduled] = await Promise.all([
      api(`/api/v1/runs?source=${encodeURIComponent(state.source)}&lifecycle=trashed`),
      api(`/api/v1/runs?source=${encodeURIComponent(state.source)}&lifecycle=purge_scheduled`),
    ]);
    state.trash = [...(trashed.runs || []), ...(scheduled.runs || [])];
  }
  if (route() === "test-plans" && state.selectedRunId) {
    const result = await api(`/api/v1/runs/${state.selectedRunId}/test-plan`);
    state.planDraft = result.draft;
  }
  if (route() === "defects" && state.selectedFindingId) {
    const result = await api(`/api/v1/findings/${state.selectedFindingId}/defect-draft`);
    state.defectDraft = result.draft;
    await loadFindingContext();
  }
}

function render() {
  const activeRoute = route();
  app.innerHTML = `
    ${header(activeRoute)}
    ${contextBar(activeRoute)}
    ${activeRoute === "findings" ? findingsPage() : ""}
    ${activeRoute === "test-plans" ? testPlanPage() : ""}
    ${activeRoute === "defects" ? defectPage() : ""}
    ${activeRoute === "trash" ? trashPage() : ""}
    ${dialogMarkup()}
    <div class="toast-stack">${state.toasts.map((toast) => `<div class="toast ${toast.kind}">${h(toast.message)}</div>`).join("")}</div>
  `;
  bindCommon();
  if (activeRoute === "findings") bindFindings();
  if (activeRoute === "test-plans") bindTestPlan();
  if (activeRoute === "defects") bindDefect();
  if (activeRoute === "trash") bindTrash();
  bindDialog();
}

function header(activeRoute) {
  const nav = [
    ["findings", "QA Findings"],
    ["test-plans", "Test Plan"],
    ["defects", "Defect Handoff"],
    ["trash", "Trash"],
  ];
  return `
    <header class="topbar">
      <div class="brand"><span class="brand-mark"></span><div><strong>TRTMC</strong><small>REPORT HUB</small></div></div>
      <nav class="main-nav" aria-label="Primary">
        ${nav.map(([key, label]) => `<a href="#/${key}" class="${activeRoute === key ? "is-active" : ""}">${label}</a>`).join("")}
      </nav>
      <div class="identity"><strong>${h(state.session.user)}</strong><small>${h(state.session.role)}</small></div>
    </header>`;
}

function contextBar(activeRoute) {
  const run = selectedRun();
  return `
    <section class="contextbar">
      <div class="source-switch" aria-label="Evidence source">
        <button data-source="benchmark" class="${state.source === "benchmark" ? "is-active" : ""}">Accuracy</button>
        <button data-source="perf" class="${state.source === "perf" ? "is-active" : ""}">Performance</button>
      </div>
      <div class="run-control">
        <label for="run-select">Evidence run</label>
        <select id="run-select" ${state.runs.length ? "" : "disabled"}>
          ${state.runs.length
            ? state.runs.map((item) => `<option value="${h(item.id)}" ${item.id === state.selectedRunId ? "selected" : ""}>${h(item.date || "undated")} · ${h(item.folder)}</option>`).join("")
            : `<option>No active runs indexed</option>`}
        </select>
      </div>
      <div class="actions">
        <button class="button" id="sync-catalog" ${can("qa") && !state.busy ? "" : "disabled"}>Refresh catalog</button>
        ${activeRoute !== "trash" && run ? `<button class="button" id="open-evidence">Original report</button>` : ""}
        ${activeRoute === "findings" && run ? `<button class="button danger" id="trash-run" ${can("qa") ? "" : "disabled"}>Move to Trash</button>` : ""}
      </div>
    </section>`;
}

function findingsPage() {
  const run = selectedRun();
  const summary = run?.summary || {};
  const visible = filteredFindings();
  const open = state.findings.filter((item) =>
    ["failed", "error"].includes(item.observation.status)
    && !["resolved", "accepted_risk"].includes(item.triage.status)
  ).length;
  return `
    <main class="page">
      <div class="page-title"><div><h1>QA Findings</h1><p>Evidence first. Human conclusions remain separate and auditable.</p></div>
        ${run && can("qa") ? `<button class="button primary" id="analyze-run">Reload evidence</button>` : ""}
      </div>
      <section class="summary-grid">
        ${summaryCard("Cases", summary.total ?? summary.cases ?? state.findings.length)}
        ${summaryCard("Passed", summary.passed ?? 0)}
        ${summaryCard("Failed", summary.failed ?? 0, "failed")}
        ${summaryCard("Open triage", open, "attention")}
      </section>
      <section class="workspace">
        <div class="panel">
          <div class="panel-header"><div class="filters">
            <input id="finding-query" type="search" value="${h(state.query)}" placeholder="Search model or workload" />
            ${["failed", "all", "open", "resolved"].map((key) => `<button class="filter-chip ${state.filter === key ? "is-active" : ""}" data-filter="${key}">${key}</button>`).join("")}
          </div><small>${visible.length} shown</small></div>
          <div class="table-wrap">
            ${visible.length ? findingsTable(visible) : `<div class="empty">${run ? "No findings match this view." : "Refresh the catalog to begin."}</div>`}
          </div>
        </div>
        <aside class="panel">${findingDetail()}</aside>
      </section>
    </main>`;
}

function summaryCard(label, value, className = "") {
  return `<div class="summary-card ${className}"><span>${h(label)}</span><strong>${h(value ?? 0)}</strong></div>`;
}

function filteredFindings() {
  return state.findings.filter((item) => {
    if (state.filter === "failed" && item.observation.status !== "failed") return false;
    if (state.filter === "open" && ["resolved", "accepted_risk"].includes(item.triage.status)) return false;
    if (state.filter === "resolved" && item.triage.status !== "resolved") return false;
    if (!state.query) return true;
    return `${item.model} ${item.workload} ${item.family} ${item.triage.tags.join(" ")}`
      .toLowerCase()
      .includes(state.query.toLowerCase());
  });
}

function findingsTable(findings) {
  return `<table><thead><tr><th>Model / workload</th><th>Result</th><th>Metric</th><th>Triage</th><th>Owner</th></tr></thead><tbody>
    ${findings.map((item) => `<tr data-finding="${h(item.id)}" class="${item.id === state.selectedFindingId ? "is-selected" : ""}">
      <td class="model-cell"><strong>${h(item.model)}</strong><small>${h(item.workload)}</small></td>
      <td><span class="status ${h(item.observation.status)}">${h(item.observation.status)}</span></td>
      <td>${metricValue(item.observation)}</td>
      <td><span class="status ${h(item.triage.status)}">${h(item.triage.status)}</span></td>
      <td>${h(item.triage.owner)}</td>
    </tr>`).join("")}
  </tbody></table>`;
}

function metricValue(observation) {
  if (observation.metric_value === null || observation.metric_value === undefined) return "—";
  const value = Number(observation.metric_value);
  return `<strong>${h(Number.isFinite(value) ? value.toFixed(4) : observation.metric_value)}</strong><br><small>${h(observation.metric_name)}</small>`;
}

function findingDetail() {
  const item = selectedFinding();
  if (!item) return `<div class="detail-empty">Select a finding to review.</div>`;
  const triage = item.triage;
  return `
    <div class="detail-title"><h2>${h(item.model)}</h2><p>${h(item.workload)}</p></div>
    <div class="detail-section">
      <h3>Observation</h3>
      <div class="metric"><span>Result</span><span class="status ${h(item.observation.status)}">${h(item.observation.status)}</span></div>
      <div class="metric"><span>Operation</span><strong>${h(item.observation.operation)}</strong></div>
      <div class="metric"><span>${h(item.observation.metric_name)}</span><strong>${h(item.observation.metric_value ?? "—")}</strong></div>
      ${item.observation.details.error ? `<div class="notice danger">${h(item.observation.details.error)}</div>` : ""}
    </div>
    <form class="detail-section" id="triage-form">
      <h3>QA triage</h3>
      <div class="form-grid">
        ${selectField("triage-status", "Status", ["new", "investigating", "linked", "monitoring", "resolved", "accepted_risk"], triage.status)}
        ${selectField("triage-severity", "Severity", ["unassessed", "blocker", "high", "medium", "low"], triage.severity)}
        ${inputField("triage-owner", "Owner", triage.owner)}
        ${inputField("triage-tags", "Tags", triage.tags.join(", "))}
        <div class="field full"><label for="triage-note">QA note</label><textarea id="triage-note">${h(triage.note)}</textarea></div>
      </div>
      <div class="form-actions"><button class="button primary" ${can("qa") ? "" : "disabled"}>Save triage</button></div>
    </form>
    <div class="detail-section"><h3>Associations</h3>${linksMarkup(state.links)}</div>
    <div class="detail-section"><h3>Recent audit</h3>${auditMarkup()}</div>`;
}

function testPlanPage() {
  const run = selectedRun();
  if (!run) return emptyPage("Test Plan", "Refresh the catalog and choose an evidence run.");
  const data = state.planDraft?.data || {};
  return `
    <main class="page">
      <div class="page-title"><div><h1>Test Plan</h1><p>Prepare locally; DevTest publication is a later approval gate.</p></div></div>
      <div class="notice">This draft does not create or update DevTest records.</div>
      <section class="wide-grid">
        <form class="panel" id="plan-form">
          <div class="panel-header"><h2>Run plan</h2><span class="status">local draft · v${h(state.planDraft?.version || 0)}</span></div>
          <div class="panel-body form-grid">
            ${inputField("plan-name", "Plan name", data.name || `${run.date || "Run"} ${state.source} validation`, true)}
            ${inputField("plan-branch", "Branch / revision", data.branch || "", true)}
            ${inputField("plan-platform", "Platform", data.platform || "GB300")}
            ${inputField("plan-gpu", "GPU", data.gpu || "GB300")}
            ${inputField("plan-os", "OS", data.os || "Linux")}
            ${inputField("plan-driver", "Driver", data.driver || "")}
            <div class="field full"><label for="plan-scope">Scope and coverage</label><textarea id="plan-scope">${h(data.scope || "")}</textarea></div>
            <div class="field full"><label for="plan-exclusions">Exclusions / gates</label><textarea id="plan-exclusions">${h(data.exclusions || "")}</textarea></div>
            <div class="field full"><div class="form-actions"><button class="button primary" ${can("qa") ? "" : "disabled"}>Save local draft</button></div></div>
          </div>
        </form>
        <aside class="panel">
          <div class="panel-header"><h2>Coverage check</h2><span class="status">${state.findings.length} cases</span></div>
          <div class="panel-body">
            <div class="checklist">
              ${checkItem(Boolean(data.name), "Plan identity recorded")}
              ${checkItem(Boolean(data.branch), "Branch or revision recorded")}
              ${checkItem(Boolean(data.platform && data.gpu), "Execution platform recorded")}
              ${checkItem(Boolean(data.scope), "Scope reviewed by QA")}
            </div>
          </div>
          <div class="detail-section"><h3>DevTest adapter</h3><p class="notice">Disabled · local draft remains authoritative in Hub.</p></div>
        </aside>
      </section>
    </main>`;
}

function defectPage() {
  const finding = selectedFinding();
  if (!finding) return emptyPage("Defect Handoff", "Analyze a run and select a finding first.");
  const data = state.defectDraft?.data || {};
  return `
    <main class="page">
      <div class="page-title"><div><h1>Defect Handoff</h1><p>${h(finding.model)} · ${h(finding.workload)}</p></div></div>
      <section class="wide-grid">
        <form class="panel" id="defect-form">
          <div class="panel-header"><h2>Developer-ready brief</h2><span class="status">local draft · v${h(state.defectDraft?.version || 0)}</span></div>
          <div class="panel-body form-grid">
            ${inputField("defect-synopsis", "Synopsis", data.synopsis || `[Validation] ${finding.model}: ${finding.workload}`, true)}
            ${selectField("defect-severity", "Proposed severity", ["unassessed", "blocker", "high", "medium", "low"], data.severity || finding.triage.severity)}
            <div class="field full"><label for="defect-impact">Impact</label><textarea id="defect-impact">${h(data.impact || "")}</textarea></div>
            <div class="field full"><label for="defect-repro">Reproduction</label><textarea id="defect-repro">${h(data.reproduction || "")}</textarea></div>
            <div class="field full"><label for="defect-evidence">Evidence and comparison</label><textarea id="defect-evidence">${h(data.evidence || `${finding.observation.metric_name}: ${finding.observation.metric_value ?? "—"}`)}</textarea></div>
            <div class="field full"><label for="defect-conclusion">QA conclusion / ask</label><textarea id="defect-conclusion">${h(data.conclusion || finding.triage.note)}</textarea></div>
            <div class="field full"><div class="form-actions"><button class="button primary" ${can("qa") ? "" : "disabled"}>Save defect draft</button></div></div>
          </div>
        </form>
        <aside class="panel">
          <div class="panel-header"><h2>Preflight</h2><span class="status ${preflightReady(data, finding) ? "passed" : "other"}">${preflightReady(data, finding) ? "ready" : "incomplete"}</span></div>
          <div class="panel-body checklist">
            ${checkItem(Boolean(data.synopsis), "Clear synopsis")}
            ${checkItem(Boolean(data.impact), "User or release impact")}
            ${checkItem(Boolean(data.reproduction), "Reproduction path")}
            ${checkItem(Boolean(data.evidence), "Comparable evidence")}
            ${checkItem(finding.triage.owner !== "Unassigned", "QA owner")}
          </div>
          <div class="detail-section"><h3>Linked work</h3>${linksMarkup(state.links)}</div>
          <form class="detail-section" id="link-form">
            <h3>Add association</h3>
            <div class="form-grid">
              ${selectField("link-system", "System", ["github", "nvbug", "devtest"], "github")}
              ${inputField("link-type", "Type", "issue")}
              ${inputField("link-id", "ID", "", true)}
              ${inputField("link-url", "HTTPS URL", "", true)}
            </div>
            <div class="form-actions"><button class="button" ${can("qa") ? "" : "disabled"}>Add link</button></div>
          </form>
          <div class="detail-section"><h3>Publication</h3><div class="notice">NVBug creation is disabled. Review here, file through the approved system, then link its ID.</div></div>
        </aside>
      </section>
    </main>`;
}

function trashPage() {
  return `
    <main class="page">
      <div class="page-title"><div><h1>Trash</h1><p>Hidden from active views and recoverable during retention.</p></div></div>
      <div class="notice">Evidence bytes remain untouched. Permanent purge is a separate admin workflow and is not executed by this service.</div>
      <section class="trash-list">
        ${state.trash.length ? state.trash.map((run) => `<article class="trash-row">
          <div><strong>${h(run.folder)}</strong><small>${h(run.source)} · ${run.lifecycle === "purge_scheduled" ? "purge scheduled" : `removed ${h(formatDate(run.trashed_at))} · recoverable until ${h(formatDate(run.purge_after))}`}</small><p>${h(run.purge_reason)}</p></div>
          <div class="actions">${run.lifecycle === "trashed" ? `<button class="button" data-restore="${h(run.id)}" ${can("qa") ? "" : "disabled"}>Restore</button><button class="button danger" data-purge="${h(run.id)}" ${can("admin") ? "" : "disabled"}>Purge preflight</button>` : `<span class="status other">awaiting storage worker</span>`}</div>
        </article>`).join("") : `<div class="panel empty">Trash is empty for this source.</div>`}
      </section>
    </main>`;
}

function emptyPage(title, message) {
  return `<main class="page"><div class="page-title"><div><h1>${h(title)}</h1></div></div><div class="panel empty">${h(message)}</div></main>`;
}

function inputField(id, label, value, full = false) {
  return `<div class="field ${full ? "full" : ""}"><label for="${id}">${h(label)}</label><input id="${id}" value="${h(value)}" /></div>`;
}

function selectField(id, label, options, selected) {
  return `<div class="field"><label for="${id}">${h(label)}</label><select id="${id}">${options.map((value) => `<option value="${h(value)}" ${value === selected ? "selected" : ""}>${h(value)}</option>`).join("")}</select></div>`;
}

function checkItem(ready, label) {
  return `<div class="check ${ready ? "is-ready" : ""}"><b>${ready ? "✓" : ""}</b><span>${h(label)}</span></div>`;
}

function preflightReady(data, finding) {
  return Boolean(data.synopsis && data.impact && data.reproduction && data.evidence && finding.triage.owner !== "Unassigned");
}

function linksMarkup(links) {
  if (!links.length) return `<p class="empty">No linked records.</p>`;
  return `<div class="integration-list">${links.map((link) => `<div class="integration-card"><div><strong>${h(link.system)} · ${h(link.record_type)}</strong><p>${h(link.external_id)}</p></div>${link.url ? `<a class="button" href="${h(link.url)}" target="_blank" rel="noreferrer">Open</a>` : ""}</div>`).join("")}</div>`;
}

function auditMarkup() {
  if (!state.audit.length) return `<p class="empty">No changes recorded.</p>`;
  return state.audit.map((item) => `<div class="metric"><span>${h(item.action)} · ${h(item.actor)}</span><small>${h(formatDate(item.at))}</small></div>`).join("");
}

function dialogMarkup() {
  const run = state.dialog?.run;
  if (!run) return "";
  if (state.dialog.type === "trash") {
    return `<dialog id="action-dialog"><div class="dialog-head"><h2>Move report to Trash</h2></div><div class="dialog-body">
      <div class="notice danger">This hides the run from active QA views. Original evidence is not deleted.</div>
      <p>Type the exact folder name:</p><p><code>${h(run.folder)}</code></p>
      <div class="field"><label for="dialog-reason">Reason</label><textarea id="dialog-reason"></textarea></div>
      <div class="field"><label for="dialog-confirm">Folder name</label><input id="dialog-confirm" autocomplete="off" /></div>
    </div><div class="dialog-actions"><button class="button" data-close-dialog>Cancel</button><button class="button danger" id="confirm-trash" disabled>Move to Trash</button></div></dialog>`;
  }
  return `<dialog id="action-dialog"><div class="dialog-head"><h2>Purge preflight</h2></div><div class="dialog-body">
    <div class="notice danger">Scheduling is allowed only after retention expires and when no open findings or external references remain. This service never deletes NAS bytes.</div>
    <p>Type the exact folder name:</p><p><code>${h(run.folder)}</code></p>
    <div class="field"><label for="dialog-confirm">Folder name</label><input id="dialog-confirm" autocomplete="off" /></div>
    <label class="check"><input type="checkbox" id="dialog-ack" /><span>I understand this schedules an irreversible storage operation.</span></label>
  </div><div class="dialog-actions"><button class="button" data-close-dialog>Cancel</button><button class="button danger" id="confirm-purge" disabled>Schedule purge</button></div></dialog>`;
}

function bindCommon() {
  document.querySelectorAll("[data-source]").forEach((button) => button.addEventListener("click", async () => {
    if (state.source === button.dataset.source) return;
    state.source = button.dataset.source;
    state.selectedRunId = null;
    state.selectedFindingId = null;
    await withBusy(async () => {
      await loadRuns();
      await prepareRoute();
    });
  }));
  document.querySelector("#run-select")?.addEventListener("change", async (event) => {
    state.selectedRunId = event.target.value;
    state.selectedFindingId = null;
    await withBusy(async () => {
      await loadFindings({ analyzeIfEmpty: true });
      await prepareRoute();
    });
  });
  document.querySelector("#sync-catalog")?.addEventListener("click", () => withBusy(async () => {
    const result = await api("/api/v1/catalog/sync", { method: "POST", body: {} });
    await loadRuns();
    toast(`Catalog refreshed: ${result.result.inserted} new, ${result.result.updated} updated.`, "success");
  }));
  document.querySelector("#open-evidence")?.addEventListener("click", async () => {
    try {
      const result = await api(`/api/v1/runs/${state.selectedRunId}`);
      window.open(result.run.evidence?.html_url, "_blank", "noopener,noreferrer");
    } catch (error) { toast(error.message, "error"); }
  });
  document.querySelector("#trash-run")?.addEventListener("click", () => {
    state.dialog = { type: "trash", run: selectedRun() };
    render();
  });
}

function bindFindings() {
  document.querySelector("#analyze-run")?.addEventListener("click", () => withBusy(async () => {
    const result = await api(`/api/v1/runs/${state.selectedRunId}/analyze`, { method: "POST", body: {} });
    state.findings = result.findings;
    state.evidence = result.evidence;
    await loadFindingContext();
    toast(`${result.observations} observations loaded.`, "success");
  }));
  document.querySelector("#finding-query")?.addEventListener("input", (event) => {
    state.query = event.target.value;
    render();
    const input = document.querySelector("#finding-query");
    input?.focus();
    input?.setSelectionRange(state.query.length, state.query.length);
  });
  document.querySelectorAll("[data-filter]").forEach((button) => button.addEventListener("click", () => {
    state.filter = button.dataset.filter;
    render();
  }));
  document.querySelectorAll("[data-finding]").forEach((row) => row.addEventListener("click", async () => {
    state.selectedFindingId = row.dataset.finding;
    await withBusy(loadFindingContext);
  }));
  document.querySelector("#triage-form")?.addEventListener("submit", async (event) => {
    event.preventDefault();
    const finding = selectedFinding();
    await withBusy(async () => {
      const result = await api(`/api/v1/findings/${finding.id}/triage`, {
        method: "PUT",
        body: {
          status: value("triage-status"), severity: value("triage-severity"), owner: value("triage-owner"),
          tags: value("triage-tags").split(",").map((item) => item.trim()).filter(Boolean),
          note: value("triage-note"), expected_version: finding.triage.version,
        },
      });
      finding.triage = result.triage;
      await loadFindingContext();
      toast("Triage saved.", "success");
    });
  });
}

function bindTestPlan() {
  document.querySelector("#plan-form")?.addEventListener("submit", async (event) => {
    event.preventDefault();
    await withBusy(async () => {
      const result = await api(`/api/v1/runs/${state.selectedRunId}/test-plan`, {
        method: "PUT",
        body: { expected_version: state.planDraft?.version || 0, data: {
          name: value("plan-name"), branch: value("plan-branch"), platform: value("plan-platform"),
          gpu: value("plan-gpu"), os: value("plan-os"), driver: value("plan-driver"),
          scope: value("plan-scope"), exclusions: value("plan-exclusions"),
          adoption_mode: "hub_only", selected_finding_ids: state.findings.map((item) => item.id),
        } },
      });
      state.planDraft = result.draft;
      toast("Test plan draft saved locally.", "success");
    });
  });
}

function bindDefect() {
  document.querySelector("#defect-form")?.addEventListener("submit", async (event) => {
    event.preventDefault();
    const finding = selectedFinding();
    await withBusy(async () => {
      const result = await api(`/api/v1/findings/${finding.id}/defect-draft`, {
        method: "PUT",
        body: { expected_version: state.defectDraft?.version || 0, data: {
          synopsis: value("defect-synopsis"), severity: value("defect-severity"), impact: value("defect-impact"),
          reproduction: value("defect-repro"), evidence: value("defect-evidence"), conclusion: value("defect-conclusion"),
          publication_state: "local_draft",
        } },
      });
      state.defectDraft = result.draft;
      toast("Defect draft saved locally.", "success");
    });
  });
  document.querySelector("#link-form")?.addEventListener("submit", async (event) => {
    event.preventDefault();
    const finding = selectedFinding();
    await withBusy(async () => {
      await api(`/api/v1/findings/${finding.id}/links`, { method: "POST", body: {
        system: value("link-system"), record_type: value("link-type"), external_id: value("link-id"), url: value("link-url"),
      } });
      await loadFindingContext();
      toast("Association added.", "success");
    });
  });
}

function bindTrash() {
  document.querySelectorAll("[data-restore]").forEach((button) => button.addEventListener("click", () => withBusy(async () => {
    const run = state.trash.find((item) => item.id === button.dataset.restore);
    await api(`/api/v1/runs/${run.id}/restore`, { method: "POST", body: { expected_version: run.version } });
    await loadRuns();
    await prepareRoute();
    toast("Report restored to active views.", "success");
  })));
  document.querySelectorAll("[data-purge]").forEach((button) => button.addEventListener("click", () => {
    state.dialog = { type: "purge", run: state.trash.find((item) => item.id === button.dataset.purge) };
    render();
  }));
}

function bindDialog() {
  const dialog = document.querySelector("#action-dialog");
  if (!dialog) return;
  dialog.showModal();
  dialog.querySelector("[data-close-dialog]")?.addEventListener("click", () => {
    state.dialog = null;
    render();
  });
  const confirmation = dialog.querySelector("#dialog-confirm");
  if (state.dialog.type === "trash") {
    const reason = dialog.querySelector("#dialog-reason");
    const confirmButton = dialog.querySelector("#confirm-trash");
    const update = () => { confirmButton.disabled = confirmation.value !== state.dialog.run.folder || !reason.value.trim(); };
    confirmation.addEventListener("input", update);
    reason.addEventListener("input", update);
    confirmButton.addEventListener("click", () => withBusy(async () => {
      const run = state.dialog.run;
      await api(`/api/v1/runs/${run.id}/trash`, { method: "POST", body: {
        reason: reason.value, confirmation: confirmation.value, expected_version: run.version,
      } });
      state.dialog = null;
      await loadRuns();
      toast("Report moved to Trash. Evidence was not deleted.", "success");
    }));
  } else {
    const ack = dialog.querySelector("#dialog-ack");
    const confirmButton = dialog.querySelector("#confirm-purge");
    const update = () => { confirmButton.disabled = confirmation.value !== state.dialog.run.folder || !ack.checked; };
    confirmation.addEventListener("input", update);
    ack.addEventListener("change", update);
    confirmButton.addEventListener("click", () => withBusy(async () => {
      const run = state.dialog.run;
      await api(`/api/v1/runs/${run.id}/purge-schedule`, { method: "POST", body: {
        confirmation: confirmation.value, acknowledge_irreversible: ack.checked, expected_version: run.version,
      } });
      state.dialog = null;
      await prepareRoute();
      toast("Purge scheduled for an external storage worker.", "success");
    }));
  }
}

async function withBusy(action) {
  if (state.busy) return;
  state.busy = true;
  render();
  try {
    await action();
  } catch (error) {
    toast(error.message, "error");
  } finally {
    state.busy = false;
    render();
  }
}

function toast(message, kind = "success") {
  const item = { message, kind };
  state.toasts.push(item);
  render();
  setTimeout(() => {
    state.toasts = state.toasts.filter((candidate) => candidate !== item);
    render();
  }, 4200);
}

function value(id) { return document.querySelector(`#${id}`)?.value?.trim() || ""; }
function formatDate(value) { return value ? new Date(value).toLocaleString() : "—"; }

start();

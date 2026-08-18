/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

(() => {
  "use strict";

  const root = document.getElementById("report-root");
  const reportPath = document.body.dataset.report || "report.json";
  const resultOrder = ["white", "red", "yellow", "green"];
  const progressOrder = { running: -2, pending: -1 };
  const priorityOrder = { P0: 0, P1: 1, P2: 2 };
  const names = { green: "Green", yellow: "Yellow", red: "Red", white: "No valid comparison" };

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function add(parent, ...children) {
    children.flat().filter(Boolean).forEach((child) => parent.append(child));
    return parent;
  }

  function signal(result) {
    if (!result) return el("span", "state", "In progress");
    return add(el("span", `signal signal-${result}`), el("span", "signal-light"), el("span", "", names[result]));
  }

  function compareRows(left, right) {
    const leftResult = left.result || left.state;
    const rightResult = right.result || right.state;
    const leftRank = progressOrder[leftResult] ?? resultOrder.indexOf(leftResult);
    const rightRank = progressOrder[rightResult] ?? resultOrder.indexOf(rightResult);
    if (leftRank !== rightRank) return leftRank - rightRank;
    const leftPriority = priorityOrder[left.issue?.priority] ?? 99;
    const rightPriority = priorityOrder[right.issue?.priority] ?? 99;
    if (leftPriority !== rightPriority) return leftPriority - rightPriority;
    return String(left.id || "").localeCompare(String(right.id || ""));
  }

  function number(value, digits = 3) {
    if (value === null || value === undefined || !Number.isFinite(Number(value))) return "—";
    return Number(value).toLocaleString(undefined, { maximumFractionDigits: digits });
  }

  function timestamp(value) {
    if (!value) return "—";
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? String(value) : date.toISOString().replace("T", " ").replace(/\.\d+Z$/, "Z");
  }

  function reportHeader(report) {
    const fragment = document.createDocumentFragment();
    const header = el("header", "report-header");
    add(header, el("p", "report-eyebrow", report.report_kind === "accuracy" ? "Validation report" : "Performance report"), el("h1", "", report.identity.title), el("p", "purpose", report.report_kind === "accuracy" ? "Accuracy and output agreement against the model reference." : "TRTMC and reference latency at aligned public task boundaries."));
    fragment.append(header);
    const run = report.run || {};
    const gpu = (run.gpu_devices || []).map((item) => item.name).filter(Boolean).join(", ") || run.gpu || "—";
    const runMeta = el("p", "meta", `source=${run.source_revision || report.identity.source_revision || "—"} · host=${run.hostname || "—"} · GPU=${gpu} · ${timestamp(run.started_at)} · ${timestamp(run.finished_at)}`);
    if (run.environment_href) {
      const environment = el("a", "environment-link", "Environment snapshot");
      environment.href = run.environment_href;
      environment.target = "_blank";
      environment.rel = "noopener";
      add(runMeta, document.createTextNode(" · "), environment);
    }
    fragment.append(runMeta);
    const disposition = String(report.identity.disposition || "");
    if (disposition && !["passed", "completed", "running"].includes(disposition)) fragment.append(el("div", "disposition", `Run disposition: ${disposition}`));

    const accounting = report.accounting;
    const strip = el("section", "outcome-strip");
    const comparable = el("div", "traffic-summary");
    comparable.append(el("span", "traffic-label", "Comparable results"));
    ["green", "yellow", "red"].forEach((result) => add(comparable, add(el("span", "traffic-item"), signal(result), el("strong", "", accounting.outcomes[result]))));
    const coverage = el("div", "traffic-summary");
    coverage.append(el("span", "traffic-label", "Operational coverage"));
    add(coverage, add(el("span", "traffic-item"), el("span", "", "Valid comparisons"), el("strong", "", `${accounting.comparable} / ${accounting.selected}`)), add(el("span", "traffic-item"), el("span", "", "Coverage"), el("strong", "", `${number(accounting.operational_coverage_percent, 2)}%`)), add(el("span", "traffic-item"), signal("white"), el("strong", "", accounting.outcomes.white)));
    add(strip, comparable, coverage);
    fragment.append(strip);
    if (accounting.progress.pending || accounting.progress.running) {
      const progress = el("div", "traffic-summary progress-summary");
      progress.append(el("span", "traffic-label", "Live progress"));
      add(progress, add(el("span", "traffic-item"), el("span", "", "Pending"), el("strong", "", accounting.progress.pending)), add(el("span", "traffic-item"), el("span", "", "Running"), el("strong", "", accounting.progress.running)), add(el("span", "traffic-item"), el("span", "", "Terminal"), el("strong", "", accounting.progress.terminal)));
      fragment.append(progress);
    }
    return fragment;
  }

  function modelTask(row) {
    const box = el("div");
    add(box, el("code", "", row.model || row.id), el("div", "detail", `${row.task_type || row.operation || "—"} · ${row.workload || row.id || "—"}`));
    return box;
  }

  function precision(row) {
    const value = row.precision || {};
    return add(el("div", ""), add(el("div", "precision-side"), el("span", "", "Reference"), el("strong", "", value.reference || "Not recorded")), add(el("div", "precision-side"), el("span", "", "TRTMC"), el("strong", "", value.candidate || "Not recorded")));
  }

  function samples(row) {
    const planned = Number.isInteger(row.samples?.planned) ? row.samples.planned : null;
    const evaluated = Number.isInteger(row.samples?.evaluated) ? row.samples.evaluated : null;
    if (planned === null && evaluated === null) return el("span", "unavailable", "—");
    if (planned !== null && evaluated !== null && planned !== evaluated) return el("span", "sample-count", `${evaluated} / ${planned}`);
    return el("span", "sample-count", evaluated ?? planned);
  }

  function metricLine(label, value) {
    return add(el("div", "metric"), el("span", "", label), el("strong", "", value));
  }

  function percent(value, signed = false) {
    if (value === null || value === undefined || !Number.isFinite(Number(value))) return "—";
    const amount = Number(value) * 100;
    const sign = signed && amount > 0 ? "+" : "";
    return `${sign}${number(amount, 2)}${signed ? " pp" : "%"}`;
  }

  function agreement(value, metrics) {
    const count = Number(metrics.valid_count ?? metrics.sample_count);
    if (Number.isFinite(Number(value)) && Number.isFinite(count) && count > 0) return `${Math.round(Number(value) * count)} / ${count}`;
    return percent(value);
  }

  function sameMetricValue(left, right) {
    if (!Number.isFinite(Number(left)) || !Number.isFinite(Number(right))) return false;
    return Math.abs(Number(left) - Number(right)) <= 1e-12;
  }

  function accuracyFacts(row) {
    if (row.result === "white") return el("span", "unavailable", "—");
    const comparison = row.comparison || {};
    const metrics = comparison.metrics || {};
    const value = el("div");
    const displayed = new Set();
    const displayedValues = [];
    [["prediction_agreement_rate", "Agreement"], ["correctness_agreement_rate", "Correctness agreement"]].forEach(([key, label]) => { if (metrics[key] !== undefined) { value.append(metricLine(label, agreement(metrics[key], metrics))); displayed.add(key); displayedValues.push(metrics[key]); } });
    [["hf_accuracy", "Reference accuracy"], ["bundle_accuracy", "TRTMC accuracy"]].forEach(([key, label]) => { if (metrics[key] !== undefined) { value.append(metricLine(label, percent(metrics[key]))); displayed.add(key); } });
    if (metrics.accuracy_delta_bundle_minus_hf !== undefined) { value.append(metricLine("Accuracy delta", percent(metrics.accuracy_delta_bundle_minus_hf, true))); displayed.add("accuracy_delta_bundle_minus_hf"); }
    if (comparison.primary_metric && !displayed.has(comparison.primary_metric.name) && !displayedValues.some((item) => sameMetricValue(item, comparison.primary_metric.value))) value.prepend(metricLine(comparison.primary_metric.name, number(comparison.primary_metric.value, 4)));
    const differenceCount = Number(row.sample_differences?.count || 0);
    if (differenceCount > 0) {
      const evaluated = Number.isInteger(row.samples?.evaluated) ? row.samples.evaluated : null;
      const label = row.sample_differences.classification === "failed_samples" ? "Failed samples" : "Sample differences";
      value.append(metricLine(label, evaluated === null ? differenceCount : `${differenceCount} / ${evaluated}`));
    }
    return value.childElementCount ? value : el("span", "unavailable", "See Metrics");
  }

  function outputValidation(row) {
    const validation = row.output_validation || {};
    return add(el("div", ""), el("strong", "", validation.status || "Not completed"), el("div", "detail", validation.contract || "—"));
  }

  function resultCell(row) {
    if (row.state !== "terminal") {
      const progress = row.progress || {};
      const details = [progress.stage, progress.attempt ? `attempt ${progress.attempt}` : null].filter(Boolean).join(" · ");
      return add(el("div", ""), el("span", "state", row.state || "pending"), details ? el("div", "detail", details) : null);
    }
    const value = el("div");
    value.append(signal(row.result));
    if (row.result === "white" && row.issue) value.append(el("div", "detail", `${row.issue.stage} · ${row.issue.code}`));
    return value;
  }

  function details(label, records) {
    const control = el("details", "evidence");
    control.append(el("summary", "", label));
    const body = el("div", "evidence-body");
    records.forEach(([heading, value]) => add(body, el("h4", "", heading), el("pre", "", typeof value === "string" ? value : JSON.stringify(value, null, 2))));
    control.append(body);
    return control;
  }

  function gateEvaluation(value) {
    if (!value || !value.schema_version) return null;
    const control = el("details", "evidence gate-evaluation");
    const sampleText = value.sample_count === null || value.sample_count === undefined ? "sample count unavailable" : `${value.sample_count} valid samples`;
    control.append(el("summary", "", `Gate analysis (shadow) · ${String(value.status || "unknown").toUpperCase()} · ${sampleText}`));
    const body = el("div", "evidence-body");
    (value.checks || []).forEach((check) => {
      const effective = check.effective || {};
      const card = el("section", `gate-check gate-${check.verdict || "unknown"}`);
      const actual = check.actual && typeof check.actual === "object" ? `${number(check.actual.min, 6)}..${number(check.actual.max, 6)}` : number(check.actual, 6);
      add(card, el("h4", "", check.gate || "Unnamed gate"), el("div", "gate-fact", `${check.metric || "—"} · actual ${actual} ${check.operator || ""} required ${number(check.required, 6)}`));
      if (effective.kind === "proportion") add(card, el("div", "gate-fact", `${effective.observed_passes}/${value.sample_count} passed · ${effective.observed_failures} failed · requires ${effective.required_passes}/${value.sample_count} · allows ${effective.allowed_failures} failed`));
      else if (effective.kind === "proportion_drop") add(card, el("div", "gate-fact", `${effective.observed_drop_count} net samples lost · allows ${effective.allowed_drop_count}`));
      else if (effective.kind === "exact") add(card, el("div", "gate-fact", `Observed range ${actual} · ${effective.sample_count ?? value.sample_count ?? "—"} valid samples`));
      else add(card, el("div", "gate-fact", `Continuous metric · ${effective.sample_count ?? value.sample_count ?? "—"} valid samples`));
      card.append(el("strong", "gate-verdict", String(check.verdict || "unknown").toUpperCase()));
      body.append(card);
    });
    (value.issues || []).forEach((issue) => body.append(el("div", "gate-issue", [issue.code, issue.gate, issue.metric].filter(Boolean).join(" · "))));
    control.append(body);
    return control;
  }

  function metrics(row, kind) {
    const records = [];
    if (row.issue) records.push(["Failure", row.issue]);
    if ((row.warnings || []).length) records.push(["Warnings", row.warnings]);
    let gate = null;
    if (kind === "accuracy") {
      const comparison = { ...(row.comparison || {}) };
      gate = gateEvaluation(comparison.gate_evaluation);
      delete comparison.gate_evaluation;
      records.push(["Comparison", comparison], ["Validation", row.validation || {}]);
    }
    else records.push(["Output validation", row.output_validation || {}], ["Reference samples", row.baseline?.samples_ms || []], ["TRTMC samples", row.candidate?.samples_ms || []], ["Comparison", row.comparison || {}]);
    return add(el("div", "metric-evidence"), gate, details("Metrics", records));
  }

  function logs(row) {
    const value = el("div", "log-links");
    (row.debug?.logs || []).forEach((record) => {
      if (!record.href) return;
      const link = el("a", "", record.label || "Open log");
      link.href = record.href; link.target = "_blank"; link.rel = "noopener"; value.append(link);
    });
    if (!value.childElementCount) value.append(el("span", "unavailable", "Unavailable"));
    return value;
  }

  function commandBlock(label, command) {
    if (!command) return null;
    return [label, `$ ${command}`];
  }

  function sampleDifferences(row) {
    const differences = row.sample_differences || {};
    const count = Number(differences.count || 0);
    if (count <= 0) return null;
    const noun = differences.classification === "failed_samples" ? "failed samples" : "sample differences";
    const control = el("details", "evidence sample-differences");
    control.append(el("summary", "", `${count} ${noun} · results and vanilla commands`));
    const body = el("div", "evidence-body");
    (differences.preview || []).forEach((record) => {
      const reason = String(record.reason || "comparison mismatch").replaceAll("_", " ");
      const records = [
        ["Input", record.input || {}],
        ["Reference result", record.reference_result || {}],
        ["TRTMC result", record.trtmc_result || {}],
        ["Comparison", record.comparison || {}],
        commandBlock("Reference vanilla command", record.reproduce?.reference),
        commandBlock("TRTMC vanilla command", record.reproduce?.trtmc),
      ].filter(Boolean);
      body.append(details(`${record.sample_id || "unknown sample"} · ${reason}`, records));
    });
    if (differences.href) {
      const link = el("a", "difference-artifact", "Open complete disagreements.jsonl");
      link.href = differences.href; link.target = "_blank"; link.rel = "noopener"; body.append(link);
    }
    control.append(body);
    return control;
  }

  function vanilla(row) {
    const reproduce = row.reproduce || {};
    const dataset = reproduce.dataset || {};
    const records = [commandBlock(dataset.prepared_input_count === undefined ? "Dataset slice" : `Dataset slice (${dataset.prepared_input_count} samples)`, dataset.command), commandBlock("Reference sample", (reproduce.hf || [])[0]), commandBlock("TRTMC sample", (reproduce.trtmc || [])[0])].filter(Boolean);
    const value = el("div", "reproduction");
    const differences = sampleDifferences(row);
    if (differences) value.append(differences);
    if (records.length) value.append(details(`Dataset · Reference ${(reproduce.commands_shown || {}).hf || 0}/${(reproduce.command_count || {}).hf || 0} · TRTMC ${(reproduce.commands_shown || {}).trtmc || 0}/${(reproduce.command_count || {}).trtmc || 0}`, records));
    if (!value.childElementCount) value.append(el("span", "unavailable", "Unavailable"));
    return value;
  }

  function commands(row) {
    const records = [];
    Object.entries(row.commands || {}).forEach(([name, command]) => {
      if (command?.rendered) records.push([name === "trtmc" ? "TRTMC" : name === "baseline" ? "Reference" : name, `${command.cwd ? `cwd: ${command.cwd}\n` : ""}$ ${command.rendered}`]);
    });
    const value = el("div");
    if (records.length) value.append(details("Commands", records));
    (row.debug?.command_artifacts || []).forEach((record) => {
      if (!record.href) return;
      const link = el("a", "command-artifact", record.label || "Command artifact");
      link.href = record.href;
      link.target = "_blank";
      link.rel = "noopener";
      value.append(link);
    });
    return value.childElementCount ? value : el("span", "unavailable", "Unavailable");
  }

  function table(rows, report, compact = false) {
    const kind = report.report_kind;
    const labels = compact ? (kind === "accuracy" ? ["Result", "Model / task", "Samples", "Accuracy / fidelity", "Metrics", "Logs", "Vanilla reproduction"] : ["Result", "Model / task", "Output validation", "Metrics", "Logs", "Commands"]) : kind === "accuracy" ? ["Model / task", "Samples", "Compute precision", "Accuracy / fidelity", "Result", "Metrics", "Logs", "Vanilla reproduction", "Commands"] : ["Model / task", "Compute precision", "Output validation", "Reference latency", "TRTMC latency", "Result", "Metrics", "Logs", "Commands"];
    const wrap = el("div", `table-wrap${compact ? " failures" : ""}`);
    const value = el("table", compact ? "" : "register-table");
    const head = el("thead"); const headRow = el("tr"); labels.forEach((label) => headRow.append(el("th", "", label))); head.append(headRow);
    const body = el("tbody");
    rows.forEach((row) => {
      const tr = el("tr"); tr.dataset.result = row.result || row.state;
      let cells;
      if (compact) cells = kind === "accuracy" ? [resultCell(row), modelTask(row), samples(row), accuracyFacts(row), metrics(row, kind), logs(row), vanilla(row)] : [resultCell(row), modelTask(row), outputValidation(row), metrics(row, kind), logs(row), commands(row)];
      else if (kind === "accuracy") cells = [modelTask(row), samples(row), precision(row), accuracyFacts(row), resultCell(row), metrics(row, kind), logs(row), vanilla(row), commands(row)];
      else cells = [modelTask(row), precision(row), outputValidation(row), el("span", "timing", row.latency?.reference_ms == null ? "—" : `${number(row.latency.reference_ms)} ms`), el("span", "timing", row.latency?.candidate_ms == null ? "—" : `${number(row.latency.candidate_ms)} ms`), resultCell(row), metrics(row, kind), logs(row), commands(row)];
      cells.forEach((cell) => add(tr, add(el("td", ""), cell))); body.append(tr);
    });
    add(value, head, body); wrap.append(value); return wrap;
  }

  function failures(report) {
    const rows = report.results.filter((row) => row.state === "terminal" && (row.result === "white" || row.result === "red")).sort(compareRows);
    const section = el("section", "failures"); section.append(el("h2", "", "Failures"));
    section.append(rows.length ? table(rows, report, true) : el("div", "empty", "No failures recorded.")); return section;
  }

  function filters(report, state, render, shown) {
    const controls = el("div", "filters");
    const searchLabel = el("label", "", "Search"); const search = el("input"); search.type = "search"; search.value = state.search; search.addEventListener("input", () => { state.search = search.value; render(); }); searchLabel.append(search);
    const statusLabel = el("label", "", "Status"); const status = el("select"); ["", "pending", "running", "green", "yellow", "red", "white"].forEach((name) => { const option = el("option", "", name || "All"); option.value = name; status.append(option); }); status.value = state.status; status.addEventListener("change", () => { state.status = status.value; render(); }); statusLabel.append(status);
    const reset = el("button", "", "Reset"); reset.type = "button"; reset.addEventListener("click", () => { state.search = ""; state.status = ""; render(); });
    add(controls, searchLabel, statusLabel, reset, el("span", "filter-count", `Showing ${shown} of ${report.results.length} selected rows`)); return controls;
  }

  function register(report) {
    const section = el("section", ""); const state = { search: "", status: "" };
    function render() {
      const query = state.search.trim().toLowerCase();
      const rows = report.results.filter((row) => (!state.status || (row.result || row.state) === state.status) && (!query || [row.id, row.model, row.family, row.operation, row.task_type, row.workload].some((value) => String(value || "").toLowerCase().includes(query)))).sort(compareRows);
      section.replaceChildren(el("h2", "", "Complete qualification register"), filters(report, state, render, rows.length), table(rows, report));
    }
    render(); return section;
  }

  function render(report) {
    if (report.schema_version !== "trtmc.qualification-report/v1") throw new Error(`Unsupported schema: ${report.schema_version}`);
    root.replaceChildren(reportHeader(report), failures(report), register(report));
  }

  fetch(reportPath, { cache: "no-store" }).then((response) => { if (!response.ok) throw new Error(`${response.status} ${response.statusText}`); return response.json(); }).then(render).catch((error) => root.replaceChildren(add(el("section", "fatal"), el("h1", "", "Unable to render report.json"), el("pre", "", error.stack || String(error)))));
})();

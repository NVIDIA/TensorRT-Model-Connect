/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// One page, no build step. Every string from the novel or a model is set as text, never as HTML.
// This file is the shell: the header, the activity panel, the project list, a new project and
// Settings. project.js shows one project.

const ui = {
  state: null, list: null, project: null, route: { page: "projects" }, hash: "", step: null, episode: 1,
  dirty: false, presets: {}, lastJobs: "", drawer: false, log: [], dismissed: new Set(), filter: "", renaming: null,
  draft: { source: "idea", idea: "", novel: "", file: null, style: "", custom: false, episodes: 1, seconds: 30, title: "" },
};

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "X-Short-Drama": "1", ...(options.body ? { "Content-Type": "application/json" } : {}) },
    ...options, body: options.body ? JSON.stringify(options.body) : undefined,
  });
  const value = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(value.detail || response.statusText);
  return value;
}

function el(tag, attributes = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attributes)) {
    if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else if (key === "class") node.className = value;
    else if (value !== false && value != null) node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) {
    if (child != null && child !== false) node.append(child instanceof Node ? child : String(child));
  }
  return node;
}

// ---- words and times ----

const STEP_NAMES = { story: "Writing the story", plan: "Planning the drama", revise: "Revising the plan",
  review: "Reviewing the plan", stills: "Drawing reference pictures", shots: "Rendering shots",
  subtitles: "Timing subtitles", assemble: "Assembling the episode" };
const stepName = (job) => (STEP_NAMES[job.step] || `Running ${job.step}`) + (job.episode ? `, episode ${job.episode}` : "");
const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;

function clock(seconds) {
  const s = Math.max(0, Math.round(seconds));
  const pad = (n) => String(n).padStart(2, "0");
  return s >= 3600 ? `${Math.floor(s / 3600)}:${pad(Math.floor(s / 60) % 60)}:${pad(s % 60)}` : `${Math.floor(s / 60)}:${pad(s % 60)}`;
}

const about = (seconds) => (seconds < 90 ? `about ${Math.round(seconds)} s` : `about ${Math.round(seconds / 60)} min`);

function ago(epoch) {
  const seconds = Date.now() / 1000 - epoch;
  if (seconds < 60) return "just now";
  if (seconds < 3600) return `${Math.floor(seconds / 60)} min ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)} h ago`;
  return new Date(epoch * 1000).toLocaleDateString();
}

// Durations come from the server; a local start time keeps them counting between polls.
const since = (seconds) => Date.now() / 1000 - (seconds || 0);
const timer = (start) => el("span", { class: "timer", "data-since": start }, clock(Date.now() / 1000 - start));

function meter(start, estimate) {
  if (!estimate) return el("div", { class: "meter unknown" }, el("span"));
  return el("div", { class: "meter", "data-since": start, "data-estimate": estimate },
    el("span", { style: `width:${Math.min(95, ((Date.now() / 1000 - start) / estimate) * 100)}%` }));
}

setInterval(() => {
  const now = Date.now() / 1000;
  document.querySelectorAll(".timer").forEach((node) => { node.textContent = clock(now - Number(node.dataset.since)); });
  document.querySelectorAll(".meter[data-estimate]").forEach((node) => {
    node.firstChild.style.width = `${Math.min(95, ((now - Number(node.dataset.since)) / Number(node.dataset.estimate)) * 100)}%`;
  });
}, 1000);

function toast(message, kind = "ok") {
  const node = el("div", { class: `toast ${kind}` }, el("span", {}, message),
    el("button", { class: "quiet small", "aria-label": "Dismiss", onclick: () => node.remove() }, "✕"));
  document.getElementById("toasts").append(node);
  if (kind !== "bad") setTimeout(() => node.remove(), 5000);
}

const fail = (error) => toast(error.message || String(error), "bad");

// ---- the header: what the GPU is doing ----

const engineSince = () => since(ui.state.now - (ui.state.engine.since || ui.state.now));
const running = () => ui.state.jobs.find((j) => j.status === "running");
const busy = () => ui.state.jobs.some((j) => j.status === "running" || j.status === "queued");
const projectTitle = (id) => (ui.state.projects.find((p) => p.id === id) || {}).title || id;

function renderStatus() {
  const s = ui.state;
  const job = running();
  const last = s.jobs.find((j) => j.finished);
  const queued = s.jobs.filter((j) => j.status === "queued").length;
  let dot = s.engine.loaded.length ? "ok" : "";
  let text = s.engine.loaded.length ? `GPU idle · ${s.engine.loaded.join(", ")} loaded` : "GPU idle";
  let clockNode = "";
  if (job && s.engine.loading.length) {
    dot = "run";
    text = `Loading ${s.engine.loading.join(", ")} onto the GPU`;
    clockNode = timer(engineSince());
  } else if (job) {
    dot = "run";
    text = `${stepName(job)}${job.message ? ` · ${job.message}` : ""}`;
    clockNode = timer(since(job.phase_seconds ?? job.seconds));
  } else if (last && last.status === "failed" && !ui.dismissed.has(last.id)) {
    dot = "bad";
    text = `${stepName(last)} failed`;
  }
  document.getElementById("status").replaceChildren(el("span", { class: `dot ${dot}` }), el("span", { class: "text" }, text),
    clockNode, queued ? el("span", { class: "badge" }, `+${queued} queued`) : "");
}

// ---- the activity panel: jobs, the GPU, and the model server's log ----

function openDrawer(withLog = false) {
  ui.drawer = true;
  if (withLog) ui.showLog = true;
  document.getElementById("activity").hidden = false;
  renderDrawer();
  loadLog();
}

function closeDrawer() {
  ui.drawer = false;
  document.getElementById("activity").hidden = true;
}

async function loadLog() {
  if (!ui.drawer || !ui.showLog) return;
  try {
    ui.log = (await api("/api/logs/server?lines=300")).lines;
  } catch (error) { ui.log = [`could not read the log: ${error.message}`]; }
  const box = document.getElementById("server-log");
  if (box) {
    const atEnd = box.scrollTop + box.clientHeight >= box.scrollHeight - 8;
    box.textContent = ui.log.join("\n") || "The model server has not written anything yet.";
    if (atEnd) box.scrollTop = box.scrollHeight;
  }
}

function jobItem(job) {
  const dot = { running: "run", queued: "", done: "ok", failed: "bad", cancelled: "" }[job.status];
  const time = job.status === "running" ? timer(since(job.seconds)) : job.seconds != null ? clock(job.seconds) : job.status;
  return el("li", {}, el("span", { class: `dot ${dot}` }), el("span", { class: "what" }, stepName(job)), el("span", { class: "faint" }, time),
    el("span", { class: "sub" }, `${projectTitle(job.project)} · ${job.status}${job.message ? ` · ${job.message}` : ""}`),
    job.error ? el("span", { class: "sub error" }, job.error) : "");
}

function renderDrawer() {
  if (!ui.drawer) return;
  const s = ui.state;
  const active = s.jobs.filter((j) => j.status === "running" || j.status === "queued").reverse();
  const finished = s.jobs.filter((j) => j.finished).slice(0, 15);
  const engine = s.engine;
  const drawer = document.getElementById("activity");
  const scroll = drawer.scrollTop;  // a refresh keeps the reader where they were
  drawer.replaceChildren(
    el("div", { class: "head" }, el("h2", {}, "Activity"),
      busy() ? el("button", { class: "small danger", onclick: cancelAll }, "Cancel all") : "",
      el("button", { class: "quiet", "aria-label": "Close", onclick: closeDrawer }, "✕")),
    el("h3", {}, "GPU"),
    el("p", { class: "note" }, `${s.gpu} · tier ${s.tier.toUpperCase()}`),
    el("div", { class: "models" }, s.models.map((m) => el("span", {
      class: `badge ${engine.loaded.includes(m.name) ? "ok" : engine.loading.includes(m.name) ? "run" : ""}`,
      title: m.bundle,
    }, `${m.name} · ${m.role}${engine.loaded.includes(m.name) ? " · loaded" : engine.loading.includes(m.name) ? " · loading" : ""}`))),
    engine.loading.length ? el("p", { class: "note" }, `Loading ${engine.loading.join(", ")} for `, timer(engineSince()),
      ". The model server reads each bundle into GPU memory before the step starts.") : "",
    engine.failure ? el("p", { class: "error" }, `The model server failed: ${engine.failure}`) : "",
    el("h3", {}, "Now"),
    active.length ? el("ul", { class: "jobs" }, active.map(jobItem)) : el("p", { class: "note" }, "Nothing is running."),
    el("h3", {}, "Recent"),
    finished.length ? el("ul", { class: "jobs" }, finished.map(jobItem)) : el("p", { class: "note" }, "No step has run since the app started."),
    el("h3", {}, "Model server log"),
    ui.showLog ? logBox()
      : el("button", { class: "small", onclick: () => { ui.showLog = true; renderDrawer(); loadLog(); } }, "Show the log"));
  drawer.scrollTop = scroll;
}

// One log box for the page's life, so its scroll position survives every refresh.
function logBox() {
  ui.logBox ||= el("pre", { class: "log", id: "server-log" }, "Loading…");
  return ui.logBox;
}

async function cancelAll() {
  if (!confirm("Cancel every queued step and stop the running one at its next check?")) return;
  await api("/api/jobs/cancel", { method: "POST" }).catch(fail);
  await refresh();
}

// ---- routes: #/ projects, #/new, #/settings, #/p/<id>/<step> ----

function parseRoute() {
  const parts = location.hash.replace(/^#\/?/, "").split("/").filter(Boolean).map(decodeURIComponent);
  if (parts[0] === "p" && parts[1]) return { page: "project", id: parts[1], step: parts[2] || null };
  if (parts[0] === "new") return { page: "new" };
  if (parts[0] === "settings") return { page: "settings" };
  return { page: "projects" };
}

async function navigate() {
  if (ui.dirty && location.hash !== ui.hash && !confirm("Leave without saving your edits?")) {
    history.replaceState(null, "", ui.hash);
    return;
  }
  const previous = ui.route;
  ui.route = parseRoute();
  ui.hash = location.hash;
  ui.dirty = false;
  document.getElementById("nav-projects").setAttribute("aria-current", ui.route.page === "settings" ? "false" : "page");
  document.getElementById("nav-settings").setAttribute("aria-current", ui.route.page === "settings" ? "page" : "false");
  try {
    if (ui.route.page === "project") {
      const same = previous.page === "project" && previous.id === ui.route.id && ui.project;
      if (same) renderProject(); else await openProject(ui.route.id);
    } else if (ui.route.page === "new") renderNew();
    else if (ui.route.page === "settings") renderSettings();
    else { await loadList(); renderHome(); }
  } catch (error) { fail(error); }
  if (previous.page !== ui.route.page || previous.id !== ui.route.id) window.scrollTo(0, 0);
}

window.addEventListener("hashchange", navigate);
window.addEventListener("beforeunload", (event) => { if (ui.dirty) event.preventDefault(); });

// ---- projects ----

async function loadList() {
  ui.list = await api("/api/projects");
}

// The five steps of a listed project: done, current, attention or todo.
function cardSteps(p) {
  const { story, planned, stills, shots, episodes } = p.progress;
  const level = (done, started) => (done ? "done" : started ? "attention" : "todo");
  const steps = [level(story, false), level(planned, false),
    level(stills.needed > 0 && stills.approved === stills.needed, stills.drawn > 0),
    level(shots.total > 0 && shots.rendered === shots.total, shots.rendered > 0),
    level(episodes.total > 0 && episodes.assembled === episodes.total, episodes.assembled > 0)];
  const current = steps.findIndex((s) => s !== "done");
  if (current >= 0 && steps[current] === "todo") steps[current] = "current";
  const where = [
    "Next: write the story from the idea", "Next: plan the drama",
    `Next: reference pictures · ${stills.approved} of ${stills.needed} approved`,
    `Next: shots · ${shots.rendered} of ${shots.total} rendered`,
    `Next: episodes · ${episodes.assembled} of ${episodes.total} assembled`][current] ||
    `Done · ${plural(episodes.total, "episode")} assembled`;
  return { steps, where };
}

function projectCard(p) {
  const { steps, where } = cardSteps(p);
  const job = ui.state.jobs.find((j) => j.project === p.id && (j.status === "running" || j.status === "queued"));
  const last = ui.state.jobs.find((j) => j.project === p.id && j.finished);
  const open = () => { location.hash = `#/p/${encodeURIComponent(p.id)}`; };
  const source = p.idea ? "From an idea" : "From a novel";
  const title = ui.renaming === p.id ? renameBox(p) : el("h3", {}, p.title || p.id);
  return el("article", { class: "card", tabindex: 0, onclick: (e) => { if (!e.target.closest("button, input")) open(); },
    onkeydown: (e) => { if (e.key === "Enter" && e.target.classList.contains("card")) open(); } },
  el("div", { class: "thumb" }, p.progress.picture
    ? el("img", { src: `/api/projects/${encodeURIComponent(p.id)}/files/${p.progress.picture}`, alt: "" })
    : (p.title || "?").slice(0, 1)),
  el("div", {}, title,
    el("div", { class: "what" }, `${source} · ${plural(p.episodes, "episode")} of up to ${p.max_seconds} s`),
    el("div", { class: "segments", "aria-hidden": true }, steps.map((s) => el("span", { class: s }))),
    el("div", { class: "where" },
      job ? [el("span", { class: "dot run" }), `${stepName(job)}${job.status === "queued" ? " (queued)" : ""}`]
        : last?.status === "failed" ? [el("span", { class: "dot bad" }), `${stepName(last)} failed`] : where),
    el("div", { class: "row", style: "align-items:center;margin-top:6px" },
      el("span", { class: "when" }, `Changed ${ago(p.updated)}`),
      el("span", { class: "actions", style: "margin:0;justify-content:flex-end" },
        el("button", { class: "quiet small", onclick: () => { ui.renaming = p.id; renderHome(); } }, "Rename"),
        el("button", { class: "quiet small danger", onclick: () => deleteProject(p) }, "Delete")))));
}

function renameBox(p) {
  const box = el("input", { value: p.title || "", "aria-label": "New title", maxlength: 80 });
  const save = async () => {
    try {
      await api(`/api/projects/${encodeURIComponent(p.id)}/meta`, { method: "PUT", body: { title: box.value } });
      ui.renaming = null;
      await refresh();
      await loadList();
      renderHome();
    } catch (error) { fail(error); }
  };
  box.addEventListener("keydown", (e) => {
    if (e.key === "Enter") save();
    if (e.key === "Escape") { ui.renaming = null; renderHome(); }
  });
  setTimeout(() => box.focus(), 0);
  return el("div", { class: "row", style: "gap:6px;margin-bottom:4px" }, box,
    el("button", { class: "small primary", style: "flex:0 0 auto", onclick: save }, "Save"));
}

async function deleteProject(p) {
  if (!confirm(`Delete “${p.title || p.id}” with every picture and video in it? This cannot be undone.`)) return;
  try {
    await api(`/api/projects/${encodeURIComponent(p.id)}`, { method: "DELETE" });
    toast(`Deleted “${p.title || p.id}”`);
    await refresh();
    await loadList();
    renderHome();
  } catch (error) { fail(error); }
}

function renderHome() {
  if (ui.route.page !== "projects") return;
  const all = ui.list.projects;
  const shown = all.filter((p) => !ui.filter || (p.title || p.id).toLowerCase().includes(ui.filter.toLowerCase()));
  const search = el("input", { type: "search", placeholder: "Find a project", value: ui.filter, "aria-label": "Find a project",
    oninput: (e) => { ui.filter = e.target.value; renderHomeCards(shown); } });
  document.getElementById("main").replaceChildren(
    el("div", { class: "page-head" }, el("div", { class: "grow" }, el("h1", {}, "Projects"),
      el("p", {}, "Turn a novel or an idea into vertical episodes. Every model runs on this GPU through TensorRT Model Connect.")),
    all.length > 6 ? search : "", el("a", { class: "button primary big", href: "#/new" }, "New project")),
    all.length ? el("div", { class: "cards", id: "cards" }, shown.map(projectCard)) : emptyHome());
}

function renderHomeCards() {
  const shown = ui.list.projects.filter((p) => (p.title || p.id).toLowerCase().includes(ui.filter.toLowerCase()));
  document.getElementById("cards").replaceChildren(...shown.map(projectCard));
}

function emptyHome() {
  const start = (source) => { ui.draft.source = source; location.hash = "#/new"; };
  return el("section", { class: "empty-state" }, el("h2", {}, "Start your first drama"),
    el("p", {}, "A language model writes the story and plans the episodes, Qwen-Image draws the cast and places, " +
      "MiniMax H3 films each shot with sound, and Whisper times the subtitles."),
    el("div", { class: "options", style: "max-width:760px;margin:0 auto" }, SOURCES.map(([key, name, text]) =>
      el("button", { class: "option", onclick: () => start(key) }, el("b", {}, name), el("span", {}, text)))));
}

// ---- a new project ----

const SOURCES = [
  ["idea", "An idea", "One or two sentences; the language model writes the story first."],
  ["novel", "My novel", "Paste the text or open a .txt file."],
  ["sample", "The sample novel", "《阿Q正传》 by Lu Xun, public domain."],
];
// The style text goes into every picture and video prompt, so it is English.
const STYLES = [
  ["", "Auto", "The planner picks a style that suits the story."],
  ["realistic live-action drama with natural light", "Live action"],
  ["bright 3D cartoon animation for children, soft rounded shapes, warm pastel colors, gentle soft light", "3D cartoon"],
  ["Japanese anime style with clean line art and soft cel shading", "Anime"],
  ["traditional Chinese ink-wash painting with soft brush strokes and muted colors", "Ink painting"],
];

function styleChips(current, custom, pick) {
  const known = STYLES.some(([text]) => text === current);
  return el("div", { class: "chips", role: "radiogroup", "aria-label": "Visual style" },
    STYLES.map(([text, name]) => el("button", {
      class: `chip${!custom && known && text === current ? " on" : ""}`, role: "radio",
      "aria-checked": String(!custom && known && text === current), onclick: () => pick(text, false),
    }, name)),
    el("button", { class: `chip${custom || !known ? " on" : ""}`, role: "radio", "aria-checked": String(custom || !known),
      onclick: () => pick(current, true) }, "Custom"));
}

function renderNew() {
  const d = ui.draft;
  const estimates = ui.list?.estimates || {};
  const part = (n, title, text, ...body) => el("section", { class: "part" }, el("span", { class: "num" }, n),
    el("div", {}, el("h2", {}, title), text ? el("p", { class: "note" }, text) : "", ...body));
  const rerender = () => renderNew();
  const field = (key, node) => { node.addEventListener("input", () => { d[key] = node.type === "number" ? Number(node.value) : node.value; }); return node; };

  const source = el("div", { class: "options" }, SOURCES.map(([key, name, text]) =>
    el("button", { class: `option${d.source === key ? " on" : ""}`, "aria-pressed": String(d.source === key),
      onclick: () => { d.source = key; if (key !== "idea" && d.episodes === 1 && d.seconds === 30) { d.episodes = 4; d.seconds = 60; } rerender(); } },
    el("b", {}, name), el("span", {}, text))));
  const idea = field("idea", el("textarea", { rows: 3, placeholder: "熊猫宝宝团团在竹林里捡到一颗会发光的种子……" }));
  idea.value = d.idea;
  const novel = field("novel", el("textarea", { rows: 8, placeholder: "第一章 …" }));
  novel.value = d.novel;
  const file = el("input", { type: "file", accept: ".txt,text/plain", onchange: (e) => { d.file = e.target.files[0] || null; } });
  const content = d.source === "idea" ? el("label", {}, "Your idea", idea)
    : d.source === "novel" ? [el("label", {}, "Paste the novel", novel), el("label", {}, "Or open a text file", file)]
      : el("p", { class: "idea" }, "《阿Q正传》 by Lu Xun: nine chapters, about 22,000 characters.");

  const styleText = el("input", { value: d.style, placeholder: "for example: watercolor picture-book illustration, soft daylight",
    oninput: (e) => { d.style = e.target.value; } });
  const pickStyle = (text, custom) => { d.style = text; d.custom = custom; rerender(); };
  const styleNote = STYLES.find(([text]) => text === d.style && !d.custom)?.[2];

  // The planner writes up to one shot per 8 seconds; the estimate follows the numbers as they change.
  const estimate = el("div", { class: "estimate" });
  const estimateText = () => {
    const each = Math.max(2, Math.floor((Number(d.seconds) || 10) / 8));
    const total = each * (Number(d.episodes) || 1);
    estimate.textContent = `About ${plural(total, "shot")} in all (up to ${each} per episode). ` + (estimates.shot
      ? `Rendering them takes ${about(total * estimates.shot)} of GPU time on this machine; you render a sample shot first and decide after.`
      : "Rendering time is measured on your first shot, which you render as a sample before the rest.");
  };
  estimateText();
  const episodes = field("episodes", el("input", { type: "number", min: 1, max: 20, value: d.episodes }));
  const seconds = field("seconds", el("input", { type: "number", min: 10, max: 600, step: 10, value: d.seconds }));
  episodes.addEventListener("input", estimateText);  // after field() has stored the number
  seconds.addEventListener("input", estimateText);
  const title = field("title", el("input", { value: d.title, maxlength: 80, style: "width:100%",
    placeholder: d.source === "sample" ? "阿Q正传" : "Taken from the idea or the novel when empty" }));

  document.getElementById("main").replaceChildren(
    el("div", { class: "crumbs" }, el("a", { href: "#/" }, "Projects"), " › New project"),
    el("div", { class: "page-head" }, el("div", { class: "grow" }, el("h1", {}, "New project"),
      el("p", {}, "Four choices, then the app takes you through the steps one at a time."))),
    el("div", { class: "form" },
      part(1, "Where does the story come from?", "", source, el("div", { style: "margin-top:12px" }, content)),
      part(2, "Visual style", "Every reference picture and shot is drawn in this style. You can change it later, before drawing.",
        styleChips(d.style, d.custom, pickStyle),
        d.custom || !STYLES.some(([text]) => text === d.style) ? el("label", {}, "Describe the style in English", styleText)
          : el("p", { class: "faint" }, styleNote || d.style)),
      part(3, "Length", "", el("div", { class: "row" }, el("label", {}, "Episodes", episodes),
        el("label", {}, "Seconds per episode, at most", seconds)), estimate),
      part(4, "Title", "Optional.", title),
      el("div", { class: "actions", style: "margin-left:50px" },
        el("button", { class: "primary big", id: "create", onclick: createProject }, "Create project"),
        el("a", { class: "button", href: "#/" }, "Cancel"))));
}

async function createProject() {
  const d = ui.draft;
  const button = document.getElementById("create");
  button.disabled = true;
  try {
    const novel = d.source === "novel" ? (d.file ? await d.file.text() : d.novel) : "";
    const idea = d.source === "idea" ? d.idea.trim() : "";
    const created = await api("/api/projects", { method: "POST", body: {
      title: d.title.trim() || (d.source === "sample" ? "阿Q正传" : idea ? idea.slice(0, 16) : "drama"),
      sample: d.source === "sample", novel, idea, style: d.style.trim(),
      episodes: d.episodes, max_seconds: d.seconds,
    } });
    ui.draft = { ...ui.draft, idea: "", novel: "", file: null, title: "" };
    await refresh();
    location.hash = `#/p/${encodeURIComponent(created.id)}`;
    // An idea project starts by writing its story, once a language model is chosen.
    if (idea && ui.state.chat_ready) {
      await api(`/api/projects/${encodeURIComponent(created.id)}/run`, { method: "POST", body: { step: "story" } });
      await refresh();
    }
  } catch (error) {
    fail(error);
    button.disabled = false;
  }
}

// ---- settings: the language model and the models on this GPU ----

function renderSettings() {
  const s = ui.state;
  const chat = s.external_chat;
  const apis = s.free_apis;
  const local = s.models.find((m) => m.role === "llm");
  const saved = Object.keys(apis).find((k) => apis[k].base_url === chat.base_url);
  const choice = el("select", {},
    Object.entries(apis).map(([k, a]) => el("option", { value: k, selected: k === saved }, a.label)),
    el("option", { value: "other", selected: Boolean(chat.base_url && !saved) }, "Another OpenAI-compatible API"));
  const base = el("input", { value: chat.base_url, placeholder: "https://…/v1" });
  const model = el("input", { value: chat.model, list: "chat-models" });
  const models = el("datalist", { id: "chat-models" });
  const key = el("input", { type: "password" });
  const help = el("p", { class: "note" });
  const link = (href, text) => el("a", { href, target: "_blank", rel: "noopener noreferrer" }, text);
  const pick = (fill) => {
    const a = apis[choice.value];
    if (a && fill) { base.value = a.base_url; model.value = a.models[0]; }
    base.disabled = Boolean(a);
    // A saved key belongs to the base URL it was entered with and never follows a new one.
    key.placeholder = chat.has_key && base.value === chat.base_url ? "saved; type to replace" : "paste your key";
    models.replaceChildren(...(a ? a.models : []).map((m) => el("option", { value: m })));
    help.replaceChildren(...(a ? [`${a.note} `, link(a.key_url, "Get a free key"), " · ", link(a.models_url, "current models")]
      : ["Any API that serves /chat/completions the OpenAI way."]));
  };
  choice.addEventListener("change", () => pick(true));
  base.addEventListener("input", () => pick(false));
  pick(!chat.base_url);  // a first visit starts from the first free API
  const save = (body) => api("/api/settings/external-chat", { method: "PUT", body })
    .then(() => toast("Saved")).then(refresh).then(renderSettings).catch(fail);
  const using = chat.base_url ? `Using ${saved ? apis[saved].label : chat.base_url} · ${chat.model}` + (chat.has_key ? " · key saved" : " · no key")
    : local ? `Using ${local.name} on this GPU` : "No language model yet: pick an API, paste your key and save.";
  const engine = s.engine;
  document.getElementById("main").replaceChildren(
    el("div", { class: "page-head" }, el("div", { class: "grow" }, el("h1", {}, "Settings"),
      el("p", {}, "The language model that writes and plans, and the models this GPU serves."))),
    el("section", { class: "panel" },
      el("h2", { style: "margin-top:0" }, "Language model"),
      el("p", { class: chat.base_url || local ? "note" : "error" }, using),
      el("p", { class: "note" }, "It writes stories from ideas, plans the episodes, reviews and revises. " +
        "The story is sent to the API you choose; start the app with --llm local to keep it on this machine. " +
        "The key is stored with mode 600 in the data directory and never shown again."),
      el("div", { class: "row" }, el("label", {}, "API", choice), el("label", {}, "Base URL", base),
        el("label", {}, "Model", model, models), el("label", {}, "API key", key)),
      help,
      el("div", { class: "actions" },
        el("button", { class: "primary", onclick: () => save({ base_url: base.value, model: model.value, api_key: key.value || null }) }, "Save"),
        local ? el("button", { onclick: () => save({ base_url: "", model: "" }) }, `Use ${local.name} on this GPU`) : "")),
    el("section", { class: "panel" },
      el("h2", { style: "margin-top:0" }, "Models on this GPU"),
      el("p", { class: "note" }, `${s.gpu} · tier ${s.tier.toUpperCase()}${s.tier_summary ? ` · ${s.tier_summary}` : ""}. ` +
        "A step loads the models it needs; the page shows the load while it happens."),
      el("table", { class: "edit" }, el("tr", {}, ["Model", "Used for", "Bundle", ""].map((h) => el("th", {}, h))),
        s.models.map((m) => el("tr", {}, el("td", {}, el("b", {}, m.name)), el("td", {}, m.role),
          el("td", { class: "mono" }, m.bundle),
          el("td", {}, engine.loaded.includes(m.name) ? el("span", { class: "badge ok" }, "loaded")
            : engine.loading.includes(m.name) ? el("span", { class: "badge run" }, "loading") : "")))),
      engine.failure ? el("p", { class: "error" }, `Last load failed: ${engine.failure}`) : "",
      el("div", { class: "actions" }, el("button", { onclick: () => openDrawer(true) }, "Show the model server log"))));
}

// ---- data ----

async function refresh() {
  ui.state = await api("/api/state");
  renderStatus();
  renderDrawer();
  loadLog();
  // Pictures and shots save one by one, so their phases count as changes too.
  const jobs = JSON.stringify(ui.state.jobs.map((j) => [j.id, j.status, ["stills", "shots"].includes(j.step) ? j.message : ""]));
  const changed = jobs !== ui.lastJobs;
  ui.lastJobs = jobs;
  if (ui.route.page === "projects" && changed && ui.list && !ui.renaming) {
    await loadList();
    renderHome();
  } else if (ui.route.page === "project" && ui.project) {
    if (changed && !ui.dirty) await reloadProject();
    else renderNextCard();
  }
}

function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  try { localStorage.setItem("short-drama-theme", theme); } catch (_) { /* private mode */ }
  document.getElementById("theme").textContent = theme === "dark" ? "Light" : "Dark";
}

document.getElementById("theme").addEventListener("click", () =>
  setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark"));
document.getElementById("status").addEventListener("click", () => (ui.drawer ? closeDrawer() : openDrawer()));
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && ui.drawer) closeDrawer(); });

// After project.js has run too; fonts and pictures may still be loading.
document.addEventListener("DOMContentLoaded", async () => {
  setTheme(document.documentElement.dataset.theme || "light");
  try {
    await refresh();
    await navigate();
  } catch (error) { fail(error); }
  setInterval(() => refresh().catch(() => {}), 3000);
});

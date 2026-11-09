/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// One page, no build step. Every string from the novel or a model is set as text, never as HTML.
// This file is the shell: the header, the activity panel, the project list, a new project and
// Settings. project.js shows one project; i18n.js holds t() and the translations.

const ui = {
  state: null, list: null, project: null, route: { page: "projects" }, hash: "", step: null, episode: 1,
  dirty: false, presets: {}, lastJobs: "", drawer: false, log: [], dismissed: new Set(), filter: "", renaming: null,
  draft: { source: "idea", idea: "", novel: "", file: null, style: "", custom: false, episodes: 1, seconds: 30, title: "",
    language: "" },
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
  if (tag === "textarea") requestAnimationFrame(() => fit(node));  // after the caller has set its value
  return node;
}

// ---- long text: every textarea grows with its text and opens in a large editor ----

function fit(area) {
  if (!area.isConnected || area.classList.contains("editor-area")) return;
  area.style.height = "auto";
  area.style.height = `${Math.min(area.scrollHeight + 4, Math.round(window.innerHeight * 0.6))}px`;
}

const expander = el("button", { type: "button", class: "expander small", hidden: true,
  onmousedown: (e) => e.preventDefault(),  // keep the focus in the textarea until the click
  onclick: () => expander.target && openEditor(expander.target) }, "⤢");

function placeExpander(area) {
  if (area.classList.contains("editor-area") || area.disabled) return;
  const box = area.getBoundingClientRect();
  expander.textContent = `⤢ ${t("Expand")}`;
  expander.title = t("Open in a large editor");
  expander.hidden = false;
  expander.target = area;
  expander.style.top = `${box.top + window.scrollY + 6}px`;
  expander.style.left = `${box.right + window.scrollX - expander.offsetWidth - 10}px`;
}

function openEditor(area) {
  const label = area.getAttribute("aria-label") || area.closest("label")?.firstChild?.textContent || t("Text");
  const big = el("textarea", { class: "editor-area", "aria-label": label, spellcheck: "false" });
  big.value = area.value;
  const count = el("span", { class: "faint" });
  const counted = () => { count.textContent = t("{n} characters", { n: big.value.length }); };
  big.addEventListener("input", counted);
  counted();
  const dialog = el("dialog", { class: "editor" },
    el("div", { class: "section-title" }, el("h2", {}, label), count),
    big,
    el("div", { class: "actions" },
      el("button", { class: "primary", onclick: () => {
        area.value = big.value;
        area.dispatchEvent(new Event("input", { bubbles: true }));
        area.dispatchEvent(new Event("change", { bubbles: true }));
        dialog.close();
        fit(area);
        area.focus();
      } }, t("Apply")),
      el("button", { onclick: () => dialog.close() }, t("Cancel")),
      el("span", { class: "faint" }, t("Applying puts the text back in the page; save there as usual."))));
  dialog.addEventListener("close", () => dialog.remove());
  document.body.append(dialog);
  dialog.showModal();
  big.focus();
}

const editorOpen = () => Boolean(document.querySelector("dialog.editor[open]"));

document.addEventListener("input", (e) => { if (e.target.matches?.("textarea")) fit(e.target); });
document.addEventListener("focusin", (e) => { if (e.target.matches?.("textarea")) placeExpander(e.target); });
document.addEventListener("mouseover", (e) => {
  const area = e.target.closest?.("textarea");
  if (area) placeExpander(area);
  else if (expander.target && !expander.target.isConnected) expander.hidden = true;
});

// ---- words and times ----

const STEP_NAMES = { story: "Writing the story", plan: "Planning the drama", revise: "Revising the plan",
  review: "Reviewing the plan", stills: "Drawing reference pictures", shots: "Rendering shots",
  subtitles: "Timing subtitles", assemble: "Assembling the episode", analyze: "Analyzing the video",
  apply: "Applying the analysis to the plan" };
const stepName = (job) => (STEP_NAMES[job.step] ? t(STEP_NAMES[job.step]) : t("Running {step}", { step: job.step })) +
  (job.episode ? t(", episode {n}", { n: job.episode }) : "");

function clock(seconds) {
  const s = Math.max(0, Math.round(seconds));
  const pad = (n) => String(n).padStart(2, "0");
  return s >= 3600 ? `${Math.floor(s / 3600)}:${pad(Math.floor(s / 60) % 60)}:${pad(s % 60)}` : `${Math.floor(s / 60)}:${pad(s % 60)}`;
}

const about = (seconds) => (seconds < 90 ? t("about {n} s", { n: Math.round(seconds) }) : t("about {n} min", { n: Math.round(seconds / 60) }));

function ago(epoch) {
  const seconds = Date.now() / 1000 - epoch;
  if (seconds < 60) return t("just now");
  if (seconds < 3600) return t("{n} min ago", { n: Math.floor(seconds / 60) });
  if (seconds < 86400) return t("{n} h ago", { n: Math.floor(seconds / 3600) });
  return new Date(epoch * 1000).toLocaleDateString(LANG === "zh" ? "zh-CN" : undefined);
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
    el("button", { class: "quiet small", "aria-label": t("Dismiss"), onclick: () => node.remove() }, "✕"));
  document.getElementById("toasts").append(node);
  if (kind !== "bad") setTimeout(() => node.remove(), 5000);
}

const fail = (error) => toast(error.message || String(error), "bad");

// ---- the header: what the GPU is doing ----

const engineSince = () => since(ui.state.now - (ui.state.engine.since || ui.state.now));
const running = () => ui.state.jobs.find((j) => j.status === "running");
const busy = () => ui.state.jobs.some((j) => j.status === "running" || j.status === "queued");
const projectTitle = (id) => (ui.state.projects.find((p) => p.id === id) || {}).title || id;
const STATUS = { running: "running", queued: "queued", done: "done", failed: "failed", cancelled: "cancelled" };

function renderStatus() {
  const s = ui.state;
  const job = running();
  const last = s.jobs.find((j) => j.finished);
  const queued = s.jobs.filter((j) => j.status === "queued").length;
  let dot = s.engine.loaded.length ? "ok" : "";
  let text = s.engine.loaded.length ? t("GPU idle · {models} loaded", { models: s.engine.loaded.join(", ") }) : t("GPU idle");
  let clockNode = "";
  if (job && s.engine.loading.length) {
    dot = "run";
    text = t("Loading {models} onto the GPU", { models: s.engine.loading.join(", ") });
    clockNode = timer(engineSince());
  } else if (job) {
    dot = "run";
    text = `${stepName(job)}${job.message ? ` · ${t(job.message)}` : ""}`;
    clockNode = timer(since(job.phase_seconds ?? job.seconds));
  } else if (last && last.status === "failed" && !ui.dismissed.has(last.id)) {
    dot = "bad";
    text = t("{step} failed", { step: stepName(last) });
  }
  document.getElementById("status").replaceChildren(el("span", { class: `dot ${dot}` }), el("span", { class: "text" }, text),
    clockNode, queued ? el("span", { class: "badge" }, t("+{n} queued", { n: queued })) : "");
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
  } catch (error) { ui.log = [t("could not read the log: {error}", { error: error.message })]; }
  const box = document.getElementById("server-log");
  if (box) {
    const atEnd = box.scrollTop + box.clientHeight >= box.scrollHeight - 8;
    box.textContent = ui.log.join("\n") || t("The model server has not written anything yet.");
    if (atEnd) box.scrollTop = box.scrollHeight;
  }
}

function jobItem(job) {
  const dot = { running: "run", queued: "", done: "ok", failed: "bad", cancelled: "" }[job.status];
  const status = t(STATUS[job.status] || job.status);
  const time = job.status === "running" ? timer(since(job.seconds)) : job.seconds != null ? clock(job.seconds) : status;
  return el("li", {}, el("span", { class: `dot ${dot}` }), el("span", { class: "what" }, stepName(job)), el("span", { class: "faint" }, time),
    el("span", { class: "sub" }, `${projectTitle(job.project)} · ${status}${job.message ? ` · ${t(job.message)}` : ""}`),
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
    el("div", { class: "head" }, el("h2", {}, t("Activity")),
      busy() ? el("button", { class: "small danger", onclick: cancelAll }, t("Cancel all")) : "",
      el("button", { class: "quiet", "aria-label": t("Close"), onclick: closeDrawer }, "✕")),
    el("h3", {}, "GPU"),
    el("p", { class: "note" }, t("{gpu} · tier {tier}", { gpu: s.gpu, tier: s.tier.toUpperCase() })),
    el("div", { class: "models" }, s.models.map((m) => el("span", {
      class: `badge ${engine.loaded.includes(m.name) ? "ok" : engine.loading.includes(m.name) ? "run" : ""}`,
      title: m.bundle,
    }, `${m.name} · ${t(m.role)}${engine.loaded.includes(m.name) ? ` · ${t("loaded")}` : engine.loading.includes(m.name) ? ` · ${t("loading")}` : ""}`))),
    engine.loading.length ? el("p", { class: "note" }, t("Loading {models} for", { models: engine.loading.join(", ") }), " ", timer(engineSince()),
      t(". The model server reads each bundle into GPU memory before the step starts.")) : "",
    engine.failure ? el("p", { class: "error" }, t("The model server failed: {error}", { error: engine.failure })) : "",
    el("h3", {}, t("Now")),
    active.length ? el("ul", { class: "jobs" }, active.map(jobItem)) : el("p", { class: "note" }, t("Nothing is running.")),
    el("h3", {}, t("Recent")),
    finished.length ? el("ul", { class: "jobs" }, finished.map(jobItem)) : el("p", { class: "note" }, t("No step has run since the app started.")),
    el("h3", {}, t("Model server log")),
    ui.showLog ? logBox()
      : el("button", { class: "small", onclick: () => { ui.showLog = true; renderDrawer(); loadLog(); } }, t("Show the log")));
  drawer.scrollTop = scroll;
}

// One log box for the page's life, so its scroll position survives every refresh.
function logBox() {
  ui.logBox ||= el("pre", { class: "log", id: "server-log" }, t("Loading…"));
  return ui.logBox;
}

async function cancelAll() {
  if (!confirm(t("Cancel every queued step and stop the running one at its next check?"))) return;
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
  if (ui.dirty && location.hash !== ui.hash && !confirm(t("Leave without saving your edits?"))) {
    history.replaceState(null, "", ui.hash);
    return;
  }
  const previous = ui.route;
  ui.route = parseRoute();
  ui.hash = location.hash;
  ui.dirty = false;
  expander.hidden = true;
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
    t("Next: write the story from the idea"), t("Next: plan the drama"),
    t("Next: reference pictures · {done} of {all} approved", { done: stills.approved, all: stills.needed }),
    t("Next: shots · {done} of {all} rendered", { done: shots.rendered, all: shots.total }),
    t("Next: episodes · {done} of {all} assembled", { done: episodes.assembled, all: episodes.total })][current] ||
    t("Done · {episodes} assembled", { episodes: plural(episodes.total, "episode") });
  return { steps, where };
}

function projectCard(p) {
  const { steps, where } = cardSteps(p);
  const job = ui.state.jobs.find((j) => j.project === p.id && (j.status === "running" || j.status === "queued"));
  const last = ui.state.jobs.find((j) => j.project === p.id && j.finished);
  const open = () => { location.hash = `#/p/${encodeURIComponent(p.id)}`; };
  const source = p.idea ? t("From an idea") : t("From a novel");
  const title = ui.renaming === p.id ? renameBox(p) : el("h3", {}, p.title || p.id);
  return el("article", { class: "card", tabindex: 0, onclick: (e) => { if (!e.target.closest("button, input")) open(); },
    onkeydown: (e) => { if (e.key === "Enter" && e.target.classList.contains("card")) open(); } },
  el("div", { class: "thumb" }, p.progress.picture
    ? el("img", { src: `/api/projects/${encodeURIComponent(p.id)}/files/${p.progress.picture}`, alt: "" })
    : (p.title || "?").slice(0, 1)),
  el("div", {}, title,
    el("div", { class: "what" }, t("{source} · {episodes} of up to {seconds} s · {language}", {
      source, episodes: plural(p.episodes, "episode"), seconds: p.max_seconds, language: LANGUAGES[p.language || "zh"] })),
    el("div", { class: "segments", "aria-hidden": true }, steps.map((s) => el("span", { class: s }))),
    el("div", { class: "where" },
      job ? [el("span", { class: "dot run" }), `${stepName(job)}${job.status === "queued" ? ` (${t("queued")})` : ""}`]
        : last?.status === "failed" ? [el("span", { class: "dot bad" }), t("{step} failed", { step: stepName(last) })] : where),
    el("div", { class: "row", style: "align-items:center;margin-top:6px" },
      el("span", { class: "when" }, t("Changed {when}", { when: ago(p.updated) })),
      el("span", { class: "actions", style: "margin:0;justify-content:flex-end" },
        el("button", { class: "quiet small", onclick: () => { ui.renaming = p.id; renderHome(); } }, t("Rename")),
        el("button", { class: "quiet small danger", onclick: () => deleteProject(p) }, t("Delete"))))));
}

function renameBox(p) {
  const box = el("input", { value: p.title || "", "aria-label": t("New title"), maxlength: 80 });
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
    el("button", { class: "small primary", style: "flex:0 0 auto", onclick: save }, t("Save")));
}

async function deleteProject(p) {
  if (!confirm(t("Delete “{title}” with every picture and video in it? This cannot be undone.", { title: p.title || p.id }))) return;
  try {
    await api(`/api/projects/${encodeURIComponent(p.id)}`, { method: "DELETE" });
    toast(t("Deleted “{title}”", { title: p.title || p.id }));
    await refresh();
    await loadList();
    renderHome();
  } catch (error) { fail(error); }
}

function renderHome() {
  if (ui.route.page !== "projects") return;
  const all = ui.list.projects;
  const shown = all.filter((p) => !ui.filter || (p.title || p.id).toLowerCase().includes(ui.filter.toLowerCase()));
  const search = el("input", { type: "search", placeholder: t("Find a project"), value: ui.filter, "aria-label": t("Find a project"),
    oninput: (e) => { ui.filter = e.target.value; renderHomeCards(shown); } });
  document.getElementById("main").replaceChildren(
    el("div", { class: "page-head" }, el("div", { class: "grow" }, el("h1", {}, t("Projects")),
      el("p", {}, t("Turn a novel or an idea into vertical episodes. Every model runs on this GPU through TensorRT Model Connect."))),
    all.length > 6 ? search : "", el("a", { class: "button primary big", href: "#/new" }, t("New project"))),
    all.length ? el("div", { class: "cards", id: "cards" }, shown.map(projectCard)) : emptyHome());
}

function renderHomeCards() {
  const shown = ui.list.projects.filter((p) => (p.title || p.id).toLowerCase().includes(ui.filter.toLowerCase()));
  document.getElementById("cards").replaceChildren(...shown.map(projectCard));
}

function emptyHome() {
  const start = (source) => { ui.draft.source = source; location.hash = "#/new"; };
  return el("section", { class: "empty-state" }, el("h2", {}, t("Start your first drama")),
    el("p", {}, t("A language model writes the story and plans the episodes, an image model draws the cast and places, " +
      "a video model films each shot with sound, and Whisper times the subtitles.")),
    el("div", { class: "options", style: "max-width:760px;margin:0 auto" }, SOURCES.map(([key, name, text]) =>
      el("button", { class: "option", onclick: () => start(key) }, el("b", {}, t(name)), el("span", {}, t(text))))));
}

// ---- a new project ----

const SOURCES = [
  ["idea", "An idea", "One or two sentences; the language model writes the story first."],
  ["novel", "My novel", "Paste the text or open a .txt file."],
  ["sample", "The sample story", "《团团的星星树》, a children's story written for this app."],
];
const SAMPLE_TITLE = "团团的星星树";
const CARTOON = "bright 3D cartoon animation for children, soft rounded shapes, warm pastel colors, gentle soft light";
// Presets are written in English, which the image and video models follow best; a custom style
// may use any language.
const STYLES = [
  ["", "Auto", "The planner picks a style that suits the story."],
  ["realistic live-action drama with natural light", "Live action"],
  [CARTOON, "3D cartoon"],
  ["Japanese anime style with clean line art and soft cel shading", "Anime"],
  ["traditional Chinese ink-wash painting with soft brush strokes and muted colors", "Ink painting"],
];

function styleChips(current, custom, pick) {
  const known = STYLES.some(([text]) => text === current);
  return el("div", { class: "chips", role: "radiogroup", "aria-label": t("Visual style") },
    STYLES.map(([text, name]) => el("button", {
      class: `chip${!custom && known && text === current ? " on" : ""}`, role: "radio",
      "aria-checked": String(!custom && known && text === current), onclick: () => pick(text, false),
    }, t(name))),
    el("button", { class: `chip${custom || !known ? " on" : ""}`, role: "radio", "aria-checked": String(custom || !known),
      onclick: () => pick(current, true) }, t("Custom")));
}

function languageChips(current, pick) {
  return el("div", { class: "chips", role: "radiogroup", "aria-label": t("Language") },
    Object.entries(LANGUAGES).map(([code, name]) => el("button", {
      class: `chip${code === current ? " on" : ""}`, role: "radio", "aria-checked": String(code === current),
      onclick: () => pick(code),
    }, name)));
}

function renderNew() {
  const d = ui.draft;
  d.language ||= LANG;
  const estimates = ui.list?.estimates || {};
  const part = (n, title, text, ...body) => el("section", { class: "part" }, el("span", { class: "num" }, n),
    el("div", {}, el("h2", {}, title), text ? el("p", { class: "note" }, text) : "", ...body));
  const rerender = () => renderNew();
  const field = (key, node) => { node.addEventListener("input", () => { d[key] = node.type === "number" ? Number(node.value) : node.value; }); return node; };

  const source = el("div", { class: "options" }, SOURCES.map(([key, name, text]) =>
    el("button", { class: `option${d.source === key ? " on" : ""}`, "aria-pressed": String(d.source === key),
      onclick: () => {
        d.source = key;
        if (key === "sample") {  // a one-chapter Chinese cartoon: one 30 s episode
          d.language = "zh";
          if (!d.style) { d.style = CARTOON; d.custom = false; }
          if (d.episodes === 4 && d.seconds === 60) { d.episodes = 1; d.seconds = 30; }
        } else if (key === "novel" && d.episodes === 1 && d.seconds === 30) { d.episodes = 4; d.seconds = 60; }
        rerender();
      } },
    el("b", {}, t(name)), el("span", {}, t(text)))));
  const idea = field("idea", el("textarea", { rows: 3, "aria-label": t("Your idea"),
    placeholder: t("A panda cub named Tuantuan finds a glowing seed in the bamboo forest…") }));
  idea.value = d.idea;
  const novel = field("novel", el("textarea", { rows: 8, "aria-label": t("Paste the novel"), placeholder: t("Chapter 1 …") }));
  novel.value = d.novel;
  const file = el("input", { type: "file", accept: ".txt,text/plain", onchange: (e) => { d.file = e.target.files[0] || null; } });
  const content = d.source === "idea" ? el("label", {}, t("Your idea"), idea)
    : d.source === "novel" ? [el("label", {}, t("Paste the novel"), novel), el("label", {}, t("Or open a text file"), file)]
      : el("p", { class: "idea" }, t("《团团的星星树》: one chapter, about 630 characters, about a panda cub and a rabbit who grow a tree of stars."));

  const styleText = el("input", { value: d.style, placeholder: t("for example: watercolor picture-book illustration, soft daylight"),
    oninput: (e) => { d.style = e.target.value; } });
  const pickStyle = (text, custom) => { d.style = text; d.custom = custom; rerender(); };
  const styleNote = STYLES.find(([text]) => text === d.style && !d.custom)?.[2];

  // The planner writes up to one shot per 8 seconds; the estimate follows the numbers as they change.
  const estimate = el("div", { class: "estimate" });
  const estimateText = () => {
    const each = Math.max(2, Math.floor((Number(d.seconds) || 10) / 8));
    const total = each * (Number(d.episodes) || 1);
    estimate.textContent = t("About {shots} in all (up to {each} per episode). ", { shots: plural(total, "shot"), each }) + (estimates.shot
      ? t("Rendering them takes {time} of GPU time on this machine; you render a sample shot first and decide after.", { time: about(total * estimates.shot) })
      : t("Rendering time is measured on your first shot, which you render as a sample before the rest."));
  };
  estimateText();
  const episodes = field("episodes", el("input", { type: "number", min: 1, max: 20, value: d.episodes }));
  const seconds = field("seconds", el("input", { type: "number", min: 10, max: 600, step: 10, value: d.seconds }));
  episodes.addEventListener("input", estimateText);  // after field() has stored the number
  seconds.addEventListener("input", estimateText);
  const title = field("title", el("input", { value: d.title, maxlength: 80, style: "width:100%",
    placeholder: d.source === "sample" ? SAMPLE_TITLE : t("Taken from the idea or the novel when empty") }));

  document.getElementById("main").replaceChildren(
    el("div", { class: "crumbs" }, el("a", { href: "#/" }, t("Projects")), ` › ${t("New project")}`),
    el("div", { class: "page-head" }, el("div", { class: "grow" }, el("h1", {}, t("New project")),
      el("p", {}, t("Five choices, then the app takes you through the steps one at a time.")))),
    el("div", { class: "form" },
      part(1, t("Where does the story come from?"), "", source, el("div", { style: "margin-top:12px" }, content)),
      part(2, t("Language"), t("The language model writes the story, the plan, the lines and every review in this language, and the characters speak it."),
        languageChips(d.language, (code) => { d.language = code; rerender(); })),
      part(3, t("Visual style"), t("Every reference picture and shot is drawn in this style. You can change it later, before drawing."),
        styleChips(d.style, d.custom, pickStyle),
        d.custom || !STYLES.some(([text]) => text === d.style) ? el("label", {}, t("Describe the style in any language; English works best for the models"), styleText)
          : el("p", { class: "faint" }, styleNote ? t(styleNote) : d.style)),
      part(4, t("Length"), "", el("div", { class: "row" }, el("label", {}, t("Number of episodes"), episodes),
        el("label", {}, t("Seconds per episode, at most"), seconds)), estimate),
      part(5, t("Title"), t("Optional."), title),
      el("div", { class: "actions", style: "margin-left:50px" },
        el("button", { class: "primary big", id: "create", onclick: createProject }, t("Create project")),
        el("a", { class: "button", href: "#/" }, t("Cancel")))));
}

async function createProject() {
  const d = ui.draft;
  const button = document.getElementById("create");
  button.disabled = true;
  try {
    const novel = d.source === "novel" ? (d.file ? await d.file.text() : d.novel) : "";
    const idea = d.source === "idea" ? d.idea.trim() : "";
    const created = await api("/api/projects", { method: "POST", body: {
      title: d.title.trim() || (d.source === "sample" ? SAMPLE_TITLE : idea ? idea.slice(0, 16) : "drama"),
      sample: d.source === "sample", novel, idea, style: d.style.trim(), language: d.language || LANG,
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
    el("option", { value: "other", selected: Boolean(chat.base_url && !saved) }, t("Another OpenAI-compatible API")));
  const base = el("input", { value: chat.base_url, placeholder: "https://…/v1" });
  const model = el("input", { value: chat.model, list: "chat-models" });
  const vision = el("input", { value: chat.vision_model || "", list: "chat-models",
    placeholder: t("empty: measure only; or a model that reads pictures, such as the chat model above") });
  const models = el("datalist", { id: "chat-models" });
  const key = el("input", { type: "password" });
  const help = el("p", { class: "note" });
  const link = (href, text) => el("a", { href, target: "_blank", rel: "noopener noreferrer" }, text);
  const pick = (fill) => {
    const a = apis[choice.value];
    if (a && fill) { base.value = a.base_url; model.value = a.models[0]; }
    base.disabled = Boolean(a);
    // A saved key belongs to the base URL it was entered with and never follows a new one.
    key.placeholder = chat.has_key && base.value === chat.base_url ? t("saved; type to replace") : t("paste your key");
    models.replaceChildren(...(a ? a.models : []).map((m) => el("option", { value: m })));
    help.replaceChildren(...(a ? [`${t(a.note)} `, link(a.key_url, t("Get a free key")), " · ", link(a.models_url, t("current models"))]
      : [t("Any API that serves /chat/completions the OpenAI way.")]));
  };
  choice.addEventListener("change", () => pick(true));
  base.addEventListener("input", () => pick(false));
  pick(!chat.base_url);  // a first visit starts from the first free API
  const save = (body) => api("/api/settings/external-chat", { method: "PUT", body })
    .then(() => toast(t("Saved"))).then(refresh).then(renderSettings).catch(fail);
  const using = chat.base_url ? t("Using {api} · {model}", { api: saved ? apis[saved].label : chat.base_url, model: chat.model }) +
      (chat.has_key ? ` · ${t("key saved")}` : ` · ${t("no key")}`)
    : local ? t("Using {model} on this GPU", { model: local.name }) : t("No language model yet: pick an API, paste your key and save.");
  const engine = s.engine;
  const tiers = s.tiers || [];
  document.getElementById("main").replaceChildren(
    el("div", { class: "page-head" }, el("div", { class: "grow" }, el("h1", {}, t("Settings")),
      el("p", {}, t("The language model that writes and plans, and the models this GPU serves.")))),
    el("section", { class: "panel" },
      el("h2", { style: "margin-top:0" }, t("Language model")),
      el("p", { class: chat.base_url || local ? "note" : "error" }, using),
      el("p", { class: "note" }, t("It writes stories from ideas, plans the episodes, reviews and revises. " +
        "The story is sent to the API you choose; start the app with --llm local to keep it on this machine. " +
        "The key is stored with mode 600 in the data directory and never shown again.")),
      el("div", { class: "row" }, el("label", {}, "API", choice), el("label", {}, t("Base URL"), base),
        el("label", {}, t("Model"), model, models), el("label", {}, t("API key"), key)),
      el("div", { class: "row" }, el("label", {}, t("Vision model for video analysis (optional)"), vision)),
      el("p", { class: "note" }, t("Video analysis sends frames of a clip to a model that reads pictures. " +
        "Without one, it checks the sound, the spoken lines and the continuity of the frames on this machine only.")),
      help,
      el("div", { class: "actions" },
        el("button", { class: "primary", onclick: () => save({ base_url: base.value, model: model.value,
          vision_model: vision.value, api_key: key.value || null }) }, t("Save")),
        local ? el("button", { onclick: () => save({ base_url: "", model: "" }) }, t("Use {model} on this GPU", { model: local.name })) : "")),
    el("section", { class: "panel" },
      el("h2", { style: "margin-top:0" }, t("Models on this GPU")),
      el("p", { class: "note" }, t("{gpu} · tier {tier}", { gpu: s.gpu, tier: s.tier.toUpperCase() }) +
        (s.tier_summary ? ` · ${t(s.tier_summary)}` : "") + ". " +
        t("A step loads the models it needs; the page shows the load while it happens.")),
      el("table", { class: "edit" }, el("tr", {}, [t("Model"), t("Used for"), t("Bundle"), ""].map((h) => el("th", {}, h))),
        s.models.map((m) => el("tr", {}, el("td", {}, el("b", {}, m.name)), el("td", {}, t(m.role)),
          el("td", { class: "mono" }, m.bundle),
          el("td", {}, engine.loaded.includes(m.name) ? el("span", { class: "badge ok" }, t("loaded"))
            : engine.loading.includes(m.name) ? el("span", { class: "badge run" }, t("loading")) : "")))),
      engine.failure ? el("p", { class: "error" }, t("Last load failed: {error}", { error: engine.failure })) : "",
      el("div", { class: "actions" }, el("button", { onclick: () => openDrawer(true) }, t("Show the model server log")))),
    tiers.length ? el("section", { class: "panel" },
      el("h2", { style: "margin-top:0" }, t("Hardware tiers")),
      el("p", { class: "note" }, t("The app picks the tier by GPU memory. Lower tiers keep the same steps with smaller models.")),
      el("table", { class: "edit" }, el("tr", {}, [t("Tier"), t("GPU memory"), t("Shots"), t("Speech")].map((h) => el("th", {}, h))),
        tiers.map((tier) => el("tr", { class: tier.name === s.tier ? "current" : "" },
          el("td", {}, el("b", {}, tier.name.toUpperCase()), tier.name === s.tier ? ` · ${t("this GPU")}` : ""),
          el("td", {}, t("{n} GB or more", { n: tier.min_gb }), el("div", { class: "faint" }, tier.hardware)),
          el("td", {}, t(tier.video)), el("td", {}, t(tier.speech)))))) : "");
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
  if (editorOpen()) return;  // the large editor holds a textarea of the current page
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
  document.getElementById("theme").textContent = theme === "dark" ? t("Light") : t("Dark");
}

// The words the page shell carries in index.html.
function renderShell() {
  document.documentElement.lang = LANG === "zh" ? "zh-CN" : "en";
  document.getElementById("nav-projects").textContent = t("Projects");
  document.getElementById("nav-settings").textContent = t("Settings");
  document.getElementById("status").title = t("What the GPU is doing; click for the activity panel");
  document.querySelector(".brand").setAttribute("aria-label", t("Short Drama projects"));
  document.getElementById("language").textContent = LANG === "zh" ? "EN" : "中文";
  document.getElementById("language").title = t("Switch the page language");
  setTheme(document.documentElement.dataset.theme || "light");
}

document.getElementById("theme").addEventListener("click", () =>
  setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark"));
document.getElementById("language").addEventListener("click", async () => {
  if (ui.dirty && !confirm(t("Leave without saving your edits?"))) return;
  setLanguage(LANG === "zh" ? "en" : "zh");
  renderShell();
  renderStatus();
  renderDrawer();
  ui.route = { page: null };  // render the current page again in the new language
  await navigate();
});
document.getElementById("status").addEventListener("click", () => (ui.drawer ? closeDrawer() : openDrawer()));
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && ui.drawer) closeDrawer(); });

// After project.js has run too; fonts and pictures may still be loading.
document.addEventListener("DOMContentLoaded", async () => {
  document.body.append(expander);
  renderShell();
  try {
    await refresh();
    await navigate();
  } catch (error) { fail(error); }
  setInterval(() => refresh().catch(() => {}), 3000);
});

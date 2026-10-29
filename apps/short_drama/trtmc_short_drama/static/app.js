/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// One page, no build step. Every string from the novel or a model is set as text, never as HTML.

const ui = { state: null, project: null, tab: "plan", episode: 1, lastJobs: "", dirty: false };

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

const file = (relative) => `/api/projects/${ui.project.id}/files/${relative}?v=${encodeURIComponent(relative)}`;
const busy = () => (ui.state?.jobs || []).some((j) => j.status === "running" || j.status === "queued");

async function run(step, extra = {}) {
  try {
    await api(`/api/projects/${ui.project.id}/run`, { method: "POST", body: { step, ...extra } });
    await refresh();
  } catch (error) { alert(error.message); }
}

// ---- top bar and sidebar ----

function renderBar() {
  const s = ui.state;
  document.getElementById("machine").textContent =
    `${s.gpu} · tier ${s.tier.toUpperCase()} · loaded: ${s.loaded.join(", ") || "none"}`;
  const job = s.jobs.find((j) => j.status === "running") || s.jobs[0];
  const activity = document.getElementById("activity");
  activity.replaceChildren();
  if (!job) return;
  const queued = s.jobs.filter((j) => j.status === "queued").length;
  activity.append(
    el("span", { class: job.status }, `${job.step}${job.episode ? " ep" + job.episode : ""}: ${job.status}`),
    el("span", {}, job.error || job.message || ""),
    queued ? el("span", {}, `+${queued} queued`) : "",
    busy() ? el("button", { onclick: () => api("/api/jobs/cancel", { method: "POST" }).then(refresh) }, "Cancel") : "",
  );
}

function renderProjects() {
  const list = document.getElementById("projects");
  list.replaceChildren(...ui.state.projects.map((p) => el("li", {
    class: ui.project?.id === p.id ? "active" : "",
    onclick: () => openProject(p.id),
  }, p.title || p.id, el("small", {}, `${p.episodes} episodes · ${p.max_seconds}s · ${p.flow}`))));
}

// ---- tabs ----

const TABS = [["plan", "Plan"], ["stills", "References"], ["shots", "Shots"], ["episodes", "Episodes"],
  ["flow", "Flow"], ["settings", "Settings"]];

function renderProject() {
  const p = ui.project;
  const main = document.getElementById("main");
  const tabs = el("nav", { class: "tabs" }, TABS.map(([key, label], index) => el("button", {
    class: ui.tab === key ? "on" : "", onclick: () => { ui.tab = key; renderProject(); },
  }, index < 4 ? el("span", { class: "n" }, index + 1) : "", label)));
  const body = { plan: renderPlan, stills: renderStills, shots: renderShots, episodes: renderEpisodes,
    flow: renderFlow, settings: renderSettings }[ui.tab]();
  main.replaceChildren(
    el("div", { class: "head" }, el("h1", {}, p.plan.title || p.meta.title),
      el("span", { class: "meta" }, `${p.novel_chars.toLocaleString()} characters of novel · ` +
        `${p.meta.episodes} episodes of at most ${p.meta.max_seconds}s · flow ${p.meta.flow}`)),
    tabs, body);
  ui.dirty = false;
}

function input(value, onchange, { area = false, rows = 2, type = "text" } = {}) {
  const node = area ? el("textarea", { rows }) : el("input", { type });
  node.value = value ?? "";
  node.addEventListener("input", () => { ui.dirty = true; onchange(node.value); });
  return node;
}

function table(rows, columns) {
  const remove = (row) => { rows.splice(rows.indexOf(row), 1); renderProject(); ui.dirty = true; };
  return el("div", {}, el("table", { class: "edit" },
    el("tr", {}, columns.map(([key, , label]) => el("th", {}, label || key)), el("th", {})),
    rows.map((row) => el("tr", {}, columns.map(([key, width]) => el("td", { style: `width:${width}` },
      input(row[key], (v) => { row[key] = v; }, { area: width === "auto", rows: 2 }))),
      el("td", {}, el("button", { title: "Remove", onclick: () => remove(row) }, "✕"))))),
    el("button", { onclick: () => { rows.push(Object.fromEntries(columns.map(([key]) => [key, ""]))); renderProject(); ui.dirty = true; } }, "Add"));
}

function newShot(episode) {
  const taken = new Set(episode.shots.map((s) => s.id));
  let n = episode.shots.length + 1;
  while (taken.has(`e${episode.number}s${n}`)) n += 1;
  episode.shots.push({ id: `e${episode.number}s${n}`, seconds: 6, location: "", characters: [], props: [],
    speaker: "", line: "", camera: "", action: "", after: "", then: "", sound: "", summary: "" });
  renderProject();
  ui.dirty = true;
}

function renderPlan() {
  const plan = ui.project.plan;
  const actions = el("div", { class: "actions" },
    el("button", { class: "primary", disabled: busy(), onclick: () => run("plan") },
      plan.episodes ? "Plan again from the novel" : "Plan the drama from the novel"),
    plan.episodes ? el("button", { onclick: savePlan }, "Save edits") : "",
    ui.project.flow_steps.filter((s) => !s.core).map((s) => s.installed && s.scope === "project"
      ? el("button", { disabled: busy() || !plan.episodes, onclick: () => run(s.step) }, `Run ${s.step}`)
      : el("span", { class: "error" }, `${s.step}: not installed, skipped`)),
    el("span", { class: "note" }, "The language model reads every chapter, then writes the cast, places and shot lists. Edit anything before rendering."));
  if (!plan.episodes) return el("section", {}, actions);
  const list = (shot, key) => input((shot[key] || []).join(", "), (v) => {
    shot[key] = v.split(",").map((x) => x.trim()).filter(Boolean);
  });
  return el("section", {}, actions,
    el("div", { class: "row" },
      el("label", {}, "Title", input(plan.title, (v) => { plan.title = v; })),
      el("label", {}, "Setting", input(plan.setting, (v) => { plan.setting = v; })),
      el("label", {}, "Visual style", input(plan.style, (v) => { plan.style = v; }))),
    el("h2", {}, "Characters"),
    table(plan.characters, [["id", "90px"], ["name", "90px"], ["role", "auto"], ["look", "auto"], ["voice", "auto"]]),
    el("h2", {}, "Relationships"),
    table(plan.relationships || [], [["from", "120px"], ["to", "120px"], ["relation", "auto"]]),
    el("h2", {}, "Places"), table(plan.locations, [["id", "90px"], ["name", "120px"], ["look", "auto"]]),
    el("h2", {}, "Props"), table(plan.props, [["id", "90px"], ["name", "120px"], ["look", "auto"]]),
    plan.episodes.map((episode) => el("div", {},
      el("h2", {}, `Episode ${episode.number}`),
      el("div", { class: "row" }, el("label", {}, "Title", input(episode.title, (v) => { episode.title = v; })),
        el("label", {}, "Summary", input(episode.summary, (v) => { episode.summary = v; }, { area: true }))),
      el("button", { onclick: () => newShot(episode) }, "Add shot"),
      episode.shots.map((shot) => el("div", { class: "card", style: "margin-bottom:8px" },
        el("div", { class: "title" }, el("h3", {}, shot.id), el("span", { class: "badge" }, `${shot.seconds}s`),
          el("button", { onclick: () => { episode.shots.splice(episode.shots.indexOf(shot), 1); renderProject(); ui.dirty = true; } }, "Remove shot")),
        el("div", { class: "row" },
          el("label", {}, "Seconds", input(shot.seconds, (v) => { shot.seconds = Number(v) || 6; }, { type: "number" })),
          el("label", {}, "Place", input(shot.location, (v) => { shot.location = v; })),
          el("label", {}, "Characters", list(shot, "characters")),
          el("label", {}, "Props", list(shot, "props")),
          el("label", {}, "Speaker", input(shot.speaker, (v) => { shot.speaker = v; }))),
        el("label", {}, "Summary", input(shot.summary, (v) => { shot.summary = v; })),
        el("label", {}, "Line (Chinese)", input(shot.line, (v) => { shot.line = v; })),
        el("div", { class: "row" }, ["camera", "action", "after", "then", "sound"].map((key) =>
          el("label", {}, key, input(shot[key], (v) => { shot[key] = v; }, { area: true, rows: 3 })))))))));
}

async function savePlan() {
  try {
    await api(`/api/projects/${ui.project.id}/plan`, { method: "PUT", body: ui.project.plan });
    ui.dirty = false;
    await openProject(ui.project.id);
  } catch (error) { alert(error.message); }
}

function stillTargets(plan) {
  return [["character", plan.characters || []], ["location", plan.locations || []], ["prop", plan.props || []]]
    .flatMap(([kind, items]) => items.map((item) => ({ key: `${kind}:${item.id}`, kind, item })));
}

function renderStills() {
  const { plan, stills } = ui.project;
  if (!plan.episodes) return el("p", {}, "Plan the drama first.");
  const targets = stillTargets(plan);
  const approved = targets.filter((t) => stills[t.key]?.status === "approved").length;
  const setStill = async (key, body) => {
    await api(`/api/projects/${ui.project.id}/stills/${encodeURIComponent(key)}`, { method: "PUT", body });
    await openProject(ui.project.id);
  };
  return el("section", {},
    el("div", { class: "actions" },
      el("button", { class: "primary", disabled: busy(), onclick: () => run("stills") }, "Draw missing stills"),
      el("span", { class: "note" }, `${approved} of ${targets.length} approved. Every shot uses the approved still of each character, place and prop in it. Jobs run one at a time; buttons queue.`)),
    el("div", { class: "grid" }, targets.map(({ key, kind, item }) => {
      const still = stills[key] || {};
      const prompt = input(still.prompt_override || still.prompt || "", () => {}, { area: true, rows: 5 });
      return el("div", { class: "card" },
        still.file ? el("img", { src: file(still.file), alt: item.name }) : el("div", { class: "note" }, "not drawn yet"),
        el("div", { class: "title" }, el("h3", {}, `${item.name || item.id}`),
          el("span", { class: still.status === "approved" ? "badge ok" : "badge" }, still.status || kind)),
        el("div", { class: "actions" },
          still.file ? el("button", { onclick: () => setStill(key, { status: still.status === "approved" ? "draft" : "approved" }) },
            still.status === "approved" ? "Unapprove" : "Approve") : "",
          el("button", { onclick: () => run("stills", { params: { only: [key] } }) }, still.file ? "Redraw" : "Draw")),
        el("details", {}, el("summary", { class: "note" }, "Prompt"), prompt,
          el("button", { onclick: () => setStill(key, { prompt_override: prompt.value }) }, "Keep this prompt")));
    })));
}

function episodeSelect() {
  const episodes = ui.project.plan.episodes || [];
  return el("select", { onchange: (e) => { ui.episode = Number(e.target.value); renderProject(); } },
    episodes.map((e) => el("option", { value: e.number, selected: e.number === ui.episode }, `Episode ${e.number}: ${e.title}`)));
}

function renderShots() {
  const plan = ui.project.plan;
  if (!plan.episodes) return el("p", {}, "Plan the drama first.");
  const episode = plan.episodes.find((e) => e.number === ui.episode) || plan.episodes[0];
  const state = ui.project.episodes[String(episode.number)] || {};
  const done = episode.shots.filter((s) => state[s.id]?.file).length;
  const previous = state.use_previous_frame ?? state.continues;
  return el("section", {},
    el("div", { class: "actions" }, episodeSelect(),
      el("button", { class: "primary", onclick: () => run("shots", { episode: episode.number }) }, "Render missing shots"),
      el("button", { onclick: () => run("episode", { episode: episode.number }) }, "Render, subtitle and assemble"),
      el("span", { class: "note" }, `${done} of ${episode.shots.length} shots rendered`)),
    episode.number > 1 ? el("label", { class: "check" }, el("input", {
      type: "checkbox", checked: previous, onchange: async (e) => {
        await api(`/api/projects/${ui.project.id}/episodes/${episode.number}`, { method: "PUT", body: { use_previous_frame: e.target.checked } });
        await openProject(ui.project.id);
      },
    }), `Use the last frame of episode ${episode.number - 1} as an extra reference for the first shot` +
      (state.use_previous_frame == null ? ` (default: ${state.continues ? "on, same place" : "off, the place changes"})` : "")) : "",
    episode.shots.map((shot) => {
      const s = state[shot.id] || {};
      const prompt = input(s.prompt_override || s.prompt || ui.project.previews[shot.id] || "", () => {}, { area: true, rows: 12 });
      const speech = s.speech;
      return el("div", { class: "card shot" },
        s.file ? el("video", { src: file(s.file), controls: true, preload: "metadata" }) : el("div", { class: "note" }, "not rendered"),
        el("div", {},
          el("div", { class: "title" }, el("h3", {}, `${shot.id} · ${shot.seconds}s · ${shot.location}`),
            s.take ? el("span", { class: "badge" }, `take ${s.take}`) : ""),
          shot.line ? el("div", { class: "line" }, `${(plan.characters.find((c) => c.id === shot.speaker) || {}).name || shot.speaker}：${shot.line}`)
            : el("div", { class: "note" }, "no line"),
          speech ? el("div", { class: speech.match >= 0.6 ? "note" : "error" },
            `heard “${speech.heard}” · match ${Math.round(speech.match * 100)}% · ${speech.start}–${speech.end}s (${speech.timing})`) : "",
          el("div", { class: "note" }, shot.summary),
          s.references ? el("div", { class: "refs" }, s.references.filter((r) => r.includes("/")).map((r) => el("img", { src: file(r) }))) : "",
          s.sheet ? el("img", { class: "sheet", src: file(s.sheet) }) : "",
          el("div", { class: "actions" },
            el("button", { onclick: () => run("shots", { episode: episode.number, params: { only: [shot.id] } }) }, s.file ? "Render another take" : "Render"),
            (s.takes || []).length > 1 ? s.takes.map((t) => el("button", {
              class: t.take === s.take ? "primary" : "", disabled: busy(),
              onclick: async () => {
                await api(`/api/projects/${ui.project.id}/episodes/${episode.number}`, { method: "PUT", body: { use_take: { [shot.id]: t.take } } });
                await openProject(ui.project.id);
              },
            }, `take ${t.take}`)) : ""),
          el("details", {}, el("summary", { class: "note" }, s.prompt ? "Video prompt of this take" : "Video prompt (preview)"), prompt,
            el("button", { onclick: async () => {
              await api(`/api/projects/${ui.project.id}/episodes/${episode.number}`, { method: "PUT", body: { prompt_overrides: { [shot.id]: prompt.value } } });
              await openProject(ui.project.id);
            } }, "Keep this prompt for the next take"))));
    }));
}

function renderEpisodes() {
  const plan = ui.project.plan;
  if (!plan.episodes) return el("p", {}, "Plan the drama first.");
  return el("section", {}, plan.episodes.map((episode) => {
    const state = ui.project.episodes[String(episode.number)] || {};
    const done = episode.shots.filter((s) => state[s.id]?.file).length;
    const out = `episodes/${String(episode.number).padStart(2, "0")}/episode.mp4`;
    return el("div", { class: "episode" },
      state.assembled ? el("video", { src: file(out), controls: true, preload: "metadata" }) : el("div", { class: "card note" }, "not assembled"),
      el("div", {},
        el("h3", {}, `Episode ${episode.number}: ${episode.title}`),
        el("p", {}, episode.summary),
        el("p", { class: "note" }, `${done} of ${episode.shots.length} shots rendered` +
          (state.assembled ? ` · ${state.assembled.duration}s · ${state.assembled.width}x${state.assembled.height}` : "")),
        el("div", { class: "actions" },
          el("button", { disabled: done < episode.shots.length, onclick: () => run("subtitles", { episode: episode.number }) }, "Time subtitles"),
          el("button", { disabled: done < episode.shots.length, onclick: () => run("assemble", { episode: episode.number }) }, "Assemble"),
          state.assembled ? el("a", { href: file(out), download: `episode-${episode.number}.mp4` }, "Download") : "")));
  }));
}

function renderFlow() {
  const s = ui.state;
  const [name, version] = ui.project.meta.flow.split("@");
  const text = el("textarea", { class: "flow" });
  const pick = el("select", {}, Object.entries(s.flows).flatMap(([flow, versions]) =>
    versions.map((v) => el("option", { value: `${flow}@${v}`, selected: flow === name && String(v) === version }, `${flow}@${v}`))));
  const load = () => api(`/api/flows/${pick.value}`).then((r) => { text.value = JSON.stringify(r.flow, null, 2); });
  pick.addEventListener("change", load);
  load();
  const saveName = el("input", { value: name });
  const note = el("input", { placeholder: "what changed" });
  return el("section", {},
    el("div", { class: "actions" }, pick,
      el("button", { onclick: async () => {
        await api(`/api/projects/${ui.project.id}/flow`, { method: "PUT", body: { flow: pick.value } });
        await openProject(ui.project.id);
      } }, "Use this version for the project")),
    text,
    el("div", { class: "actions" }, el("label", {}, "Flow name", saveName), el("label", {}, "Note", note),
      el("button", { class: "primary", onclick: async () => {
        try {
          const saved = await api(`/api/flows/${saveName.value}`, { method: "POST", body: { flow: JSON.parse(text.value), note: note.value } });
          alert(`Saved ${saved.reference}`);
          await refresh();
          renderProject();
        } catch (error) { alert(error.message); }
      } }, "Save as a new version")),
    el("h2", {}, "Installed steps"),
    el("table", { class: "edit" }, s.steps.map((step) => el("tr", {},
      el("td", {}, el("b", {}, step.name)), el("td", {}, step.scope), el("td", {}, step.role || "–"),
      el("td", {}, step.description), el("td", { class: "note" }, step.source)))),
    Object.entries(s.plugin_errors).map(([plugin, error]) => el("p", { class: "error" }, `${plugin}: ${error}`)),
    el("button", { onclick: () => api("/api/plugins/reload", { method: "POST" }).then(refresh).then(() => openProject(ui.project.id)) }, "Reload plugins"),
    el("p", { class: "note" }, "A plugin is one Python file in the plugins directory that defines STEP. Add its name to a flow version to run it; delete the file and reload to remove it."));
}

function renderSettings() {
  const s = ui.state;
  const base = el("input", { value: s.external_chat.base_url, placeholder: "https://…/v1" });
  const model = el("input", { value: s.external_chat.model });
  const key = el("input", { type: "password", placeholder: s.external_chat.has_key ? "saved; type to replace" : "" });
  const save = (body) => api("/api/settings/external-chat", { method: "PUT", body }).then(refresh).then(renderProject);
  return el("section", {},
    el("h2", {}, "Models on this GPU"),
    el("table", { class: "edit" }, s.models.map((m) => el("tr", {}, el("td", {}, el("b", {}, m.name)), el("td", {}, m.role),
      el("td", { class: "note" }, m.bundle), el("td", {}, s.loaded.includes(m.name) ? el("span", { class: "badge ok" }, "loaded") : "")))),
    el("h2", {}, "Optional external chat API for planning"),
    el("p", { class: "note" }, "Leave empty to plan with the local TRTMC model. The key is stored with mode 600 in the data directory and never shown again."),
    el("div", { class: "row" }, el("label", {}, "Base URL", base), el("label", {}, "Model", model), el("label", {}, "API key", key)),
    el("div", { class: "actions" },
      el("button", { class: "primary", onclick: () => save({ base_url: base.value, model: model.value, api_key: key.value || null }) }, "Save"),
      el("button", { onclick: () => save({ base_url: "", model: "" }) }, "Use the local model")));
}

// ---- data ----

async function openProject(id) {
  ui.project = await api(`/api/projects/${id}`);
  renderProjects();
  renderProject();
}

async function refresh() {
  ui.state = await api("/api/state");
  renderBar();
  renderProjects();
  const jobs = JSON.stringify(ui.state.jobs.map((j) => [j.id, j.status]));
  if (ui.project && jobs !== ui.lastJobs && !ui.dirty) await openProject(ui.project.id);
  ui.lastJobs = jobs;
}

document.getElementById("create").addEventListener("click", async () => {
  const fileInput = document.getElementById("new-file");
  let novel = document.getElementById("new-novel").value;
  if (fileInput.files[0]) novel = await fileInput.files[0].text();
  const sample = document.getElementById("new-sample").checked && !novel.trim();
  try {
    const created = await api("/api/projects", { method: "POST", body: {
      title: document.getElementById("new-title").value, novel, sample,
      episodes: Number(document.getElementById("new-episodes").value),
      max_seconds: Number(document.getElementById("new-seconds").value),
    } });
    await refresh();
    await openProject(created.id);
  } catch (error) { alert(error.message); }
});

refresh().then(() => { if (ui.state.projects.length) openProject(ui.state.projects.at(-1).id); });
setInterval(() => refresh().catch(() => {}), 4000);

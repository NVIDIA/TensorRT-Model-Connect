/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// One page, no build step. Every string from the novel or a model is set as text, never as HTML.

const ui = { state: null, project: null, tab: "plan", episode: 1, lastJobs: "", dirty: false, presets: {} };

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

// Batches ask first and say how long the GPU will be busy, from the times measured on this machine.
function duration(count, each) {
  if (!each) return "the time is unknown until the first one runs on this GPU";
  const total = count * each;
  return total < 90 ? `about ${Math.round(total)} s of GPU time` : `about ${Math.round(total / 60)} min of GPU time`;
}

function confirmRun(what, count, each, start) {
  if (count > 1 && !confirm(`${what}: ${duration(count, each)}. Continue?`)) return;
  start();
}

async function revise(target, instruction) {
  if (ui.dirty) return alert("Save your edits first: the model revises the saved plan.");
  if (!instruction || !instruction.trim()) return;
  delete ui.presets[target.split(":")[1]];
  await run("revise", { params: { target, instruction: instruction.trim() } });
}

function jump(where) {
  const node = document.getElementById(`plan-${where.replace(/ /g, "-")}`);
  if (node) node.scrollIntoView({ behavior: "smooth", block: "center" });
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

// ---- stages and tabs ----

const STAGES = { plan: "Plan", stills: "References", shots: "Shots", episodes: "Episodes" };
const TABS = [["flow", "Flow"], ["settings", "Settings"]];

function renderProject() {
  const p = ui.project;
  const main = document.getElementById("main");
  const open = (key) => () => { ui.tab = key; renderProject(); };
  const nav = el("nav", { class: "tabs" },
    p.stages.map((stage, index) => el("button", {
      class: `stage ${stage.status}${ui.tab === stage.key ? " on" : ""}`, onclick: open(stage.key),
    }, el("span", { class: "label" }, el("span", { class: "n" }, index + 1), STAGES[stage.key]),
    el("small", {}, stage.detail), stage.next ? el("em", {}, stage.next) : "")),
    el("span", { class: "spacer" }),
    TABS.map(([key, label]) => el("button", { class: `plain${ui.tab === key ? " on" : ""}`, onclick: open(key) }, label)));
  const body = { plan: renderPlan, stills: renderStills, shots: renderShots, episodes: renderEpisodes,
    flow: renderFlow, settings: renderSettings }[ui.tab]();
  main.replaceChildren(
    el("div", { class: "head" }, el("h1", {}, p.plan.title || p.meta.title),
      el("span", { class: "meta" }, `${p.novel_chars.toLocaleString()} characters of story · ` +
        `${p.meta.episodes} episodes of at most ${p.meta.max_seconds}s · flow ${p.meta.flow}`)),
    nav, body);
  ui.dirty = false;
}

function input(value, onchange, { area = false, rows = 2, type = "text" } = {}) {
  const node = area ? el("textarea", { rows }) : el("input", { type });
  node.value = value ?? "";
  node.addEventListener("input", () => { ui.dirty = true; onchange(node.value); });
  return node;
}

// An editable table; `kind` adds a button that asks the model to revise one row.
function table(rows, columns, kind = null) {
  const remove = (row) => { rows.splice(rows.indexOf(row), 1); renderProject(); ui.dirty = true; };
  const ask = (row) => revise(`${kind}:${row.id}`, prompt(`What should change about ${row.name || row.id}?`));
  return el("div", {}, el("table", { class: "edit" },
    el("tr", {}, columns.map(([key, , label]) => el("th", {}, label || key)), el("th", {})),
    rows.map((row) => el("tr", {}, columns.map(([key, width]) => el("td", { style: `width:${width}` },
      input(row[key], (v) => { row[key] = v; }, { area: width === "auto", rows: 2 }))),
      el("td", { class: "row-actions" },
        kind ? el("button", { title: "Revise with the model", disabled: busy(), onclick: () => ask(row) }, "✎") : "",
        el("button", { title: "Remove", onclick: () => remove(row) }, "✕"))))),
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

// ---- 1. plan ----

function renderStory() {
  const p = ui.project;
  if (!p.novel_chars) {
    return el("div", { class: "card story" }, el("h3", {}, "Idea"), el("p", { class: "idea-text" }, p.meta.idea || ""),
      el("div", { class: "actions" },
        el("button", { class: "primary", disabled: busy(), onclick: () => run("story") }, "Write the story from this idea"),
        el("span", { class: "note" }, "The language model writes one chapter per episode; read and edit it before planning.")));
  }
  const text = el("textarea", { class: "novel", rows: 14 });
  const box = el("details", {
    class: "story", ontoggle: async () => {
      if (box.open && !text.value) text.value = (await api(`/api/projects/${p.id}/novel`)).novel;
    },
  }, el("summary", {}, `Story · ${p.novel_chars.toLocaleString()} characters` + (p.meta.idea ? " · written from an idea" : "")),
  p.meta.idea ? el("p", { class: "note" }, `Idea: ${p.meta.idea}`) : "", text,
  el("div", { class: "actions" },
    el("button", { onclick: async () => {
      try {
        await api(`/api/projects/${p.id}/novel`, { method: "PUT", body: { novel: text.value } });
        await openProject(p.id);
      } catch (error) { alert(error.message); }
    } }, "Save the story"),
    el("span", { class: "note" }, "Plan again to use an edited story; the current plan is kept under Earlier versions.")));
  if (!p.plan.episodes) box.open = true;
  return box;
}

function renderChecks() {
  const { problems } = ui.project;
  const review = ui.project.model_review?.items ? ui.project.model_review : null;
  const errors = problems.filter((x) => x.level === "error").length;
  return el("section", { class: "checks" },
    el("div", { class: "title" },
      el("h3", {}, problems.length ? `${problems.length} things to check before rendering` +
        (errors ? ` (${errors} must be fixed)` : "") : "The plan checks found nothing to fix"),
      el("button", { disabled: busy(), onclick: () => run("review") }, review ? "Review again with the model" : "Ask the model to review")),
    problems.length ? el("ul", { class: "problems" }, problems.map((x) => el("li", {},
      el("span", { class: `badge ${x.level === "error" ? "bad" : "warn"}` }, x.level),
      el("a", { onclick: () => jump(x.where) }, x.where), ` ${x.message}`,
      x.fix ? el("span", { class: "note" }, ` · ${x.fix}`) : ""))) : "",
    review ? el("div", {},
      el("h4", {}, `Model review: ${review.items.length ? review.items.length + " notes" : "no problems"}` +
        (review.current ? "" : " (written for an earlier version of the shots)")),
      el("ul", { class: "problems" }, review.items.map((x) => el("li", {},
        el("a", { onclick: () => jump(x.shot) }, x.shot), ` ${x.problem}`,
        x.fix ? el("span", { class: "note" }, ` · Fix: ${x.fix} `) : " ",
        el("button", { onclick: () => { ui.presets[x.shot] = x.fix || x.problem; renderProject(); jump(x.shot); } }, "Revise this shot"))))) : "");
}

function renderHistory() {
  const p = ui.project;
  if (!p.history.length) return "";
  return el("details", { class: "history" }, el("summary", { class: "note" }, `Earlier versions of the plan (${p.history.length})`),
    p.history.map((h) => el("div", { class: "version" },
      el("span", {}, `${new Date(h.time * 1000).toLocaleString()} · ${h.note}`),
      el("button", { onclick: async () => {
        if (!confirm("Restore this version? The current plan is kept as a version too.")) return;
        await api(`/api/projects/${p.id}/history/${h.name}/restore`, { method: "POST" });
        await openProject(p.id);
      } }, "Restore"))));
}

function reviseBox(target, preset = "") {
  const note = el("textarea", { rows: 2, placeholder: "What should change? For example: 让他先摸头再开口，台词更短" });
  note.value = preset;
  return el("div", { class: "revise" }, note,
    el("button", { disabled: busy(), onclick: () => revise(target, note.value) }, "Revise with the model"));
}

function renderShotEditor(episode, shot) {
  const p = ui.project;
  const list = (key) => input((shot[key] || []).join(", "), (v) => {
    shot[key] = v.split(",").map((x) => x.trim()).filter(Boolean);
  });
  const issues = p.problems.filter((x) => x.where === shot.id).length;
  const origin = p.origins[shot.id];
  return el("div", { class: "card shot-edit", id: `plan-${shot.id}` },
    el("div", { class: "title" }, el("h3", {}, shot.id),
      el("span", { class: "badges" }, el("span", { class: "badge" }, `${shot.seconds}s`),
        origin ? el("span", { class: `badge ${origin === "novel" ? "ok" : ""}` }, origin === "novel" ? "line from the novel" : "adapted line") : "",
        issues ? el("span", { class: "badge warn" }, `${issues} to check`) : ""),
      el("button", { onclick: () => { episode.shots.splice(episode.shots.indexOf(shot), 1); renderProject(); ui.dirty = true; } }, "Remove shot")),
    el("div", { class: "row" },
      el("label", {}, "Seconds", input(shot.seconds, (v) => { shot.seconds = Number(v) || 6; }, { type: "number" })),
      el("label", {}, "Place", input(shot.location, (v) => { shot.location = v; })),
      el("label", {}, "Characters", list("characters")),
      el("label", {}, "Props", list("props")),
      el("label", {}, "Speaker", input(shot.speaker, (v) => { shot.speaker = v; }))),
    el("label", {}, "Summary", input(shot.summary, (v) => { shot.summary = v; })),
    el("label", {}, "Line (Chinese)", input(shot.line, (v) => { shot.line = v; })),
    el("div", { class: "row" }, ["camera", "action", "after", "then", "sound"].map((key) =>
      el("label", {}, key, input(shot[key], (v) => { shot[key] = v; }, { area: true, rows: 3 })))),
    reviseBox(`shot:${shot.id}`, ui.presets[shot.id] || ""));
}

function renderPlan() {
  const p = ui.project;
  const plan = p.plan;
  const actions = el("div", { class: "actions" },
    el("button", { class: "primary", disabled: busy() || !p.novel_chars, onclick: () => run("plan") },
      plan.episodes ? "Plan again from the story" : "Plan the drama from the story"),
    plan.episodes ? el("button", { onclick: savePlan }, "Save edits") : "",
    p.flow_steps.filter((s) => !s.core).map((s) => s.installed && s.scope === "project"
      ? el("button", { disabled: busy() || !plan.episodes, onclick: () => run(s.step) }, `Run ${s.step}`)
      : el("span", { class: "error" }, `${s.step}: not installed, skipped`)),
    el("span", { class: "note" }, "The language model reads every chapter, then writes the cast, places and shot lists. Edit anything before rendering."));
  if (!plan.episodes) return el("section", {}, renderStory(), actions, renderHistory());
  return el("section", {}, renderStory(), actions, renderChecks(), renderHistory(),
    el("div", { class: "row" },
      el("label", {}, "Title", input(plan.title, (v) => { plan.title = v; })),
      el("label", {}, "Setting", input(plan.setting, (v) => { plan.setting = v; })),
      el("label", {}, "Visual style", input(plan.style, (v) => { plan.style = v; }))),
    el("h2", {}, "Characters"),
    table(plan.characters, [["id", "90px"], ["name", "90px"], ["role", "auto"], ["look", "auto"], ["voice", "auto"]], "character"),
    el("h2", {}, "Relationships"),
    table(plan.relationships || [], [["from", "120px"], ["to", "120px"], ["relation", "auto"]]),
    el("h2", {}, "Places"), table(plan.locations, [["id", "90px"], ["name", "120px"], ["look", "auto"]], "location"),
    el("h2", {}, "Props"), table(plan.props, [["id", "90px"], ["name", "120px"], ["look", "auto"]], "prop"),
    plan.episodes.map((episode) => el("div", {},
      el("h2", { id: `plan-episode-${episode.number}` }, `Episode ${episode.number}`),
      el("div", { class: "row" }, el("label", {}, "Title", input(episode.title, (v) => { episode.title = v; })),
        el("label", {}, "Summary", input(episode.summary, (v) => { episode.summary = v; }, { area: true }))),
      el("button", { onclick: () => newShot(episode) }, "Add shot"),
      episode.shots.map((shot) => renderShotEditor(episode, shot)))));
}

async function savePlan() {
  try {
    await api(`/api/projects/${ui.project.id}/plan`, { method: "PUT", body: ui.project.plan });
    ui.dirty = false;
    await openProject(ui.project.id);
  } catch (error) { alert(error.message); }
}

// ---- 2. references ----

function stillTargets(plan) {
  return [["character", plan.characters || []], ["location", plan.locations || []], ["prop", plan.props || []]]
    .flatMap(([kind, items]) => items.map((item) => ({ key: `${kind}:${item.id}`, kind, item })));
}

function renderStills() {
  const { plan, stills, usage, outdated_stills: outdated, estimates } = ui.project;
  if (!plan.episodes) return el("p", {}, "Plan the drama first.");
  const targets = stillTargets(plan);
  const used = targets.filter((t) => usage[t.key]);
  const unused = targets.filter((t) => !usage[t.key]);
  const missing = used.filter((t) => !stills[t.key]?.file).length;
  const approved = used.filter((t) => stills[t.key]?.status === "approved").length;
  const setStill = async (key, body) => {
    await api(`/api/projects/${ui.project.id}/stills/${encodeURIComponent(key)}`, { method: "PUT", body });
    await openProject(ui.project.id);
  };
  return el("section", {},
    el("div", { class: "actions" },
      el("button", {
        class: "primary", disabled: busy() || !missing,
        onclick: () => confirmRun(`Draw ${missing} stills`, missing, estimates.still, () => run("stills")),
      }, missing ? `Draw ${missing} missing stills` : "Every still is drawn"),
      el("span", { class: "note" }, `${approved} of ${used.length} approved. A shot renders once the stills of everyone and everything in it are approved.`)),
    el("div", { class: "grid" }, used.map(({ key, kind, item }) => {
      const still = stills[key] || {};
      const prompt = input(still.prompt_override || still.prompt || "", () => {}, { area: true, rows: 5 });
      return el("div", { class: "card" },
        still.file ? el("img", { src: file(still.file), alt: item.name }) : el("div", { class: "note placeholder" }, "not drawn yet"),
        el("div", { class: "title" }, el("h3", {}, `${item.name || item.id}`),
          el("span", { class: still.status === "approved" ? "badge ok" : "badge" }, still.status || kind)),
        outdated.includes(key) ? el("div", { class: "error" }, "The description changed after this still was drawn; redraw it.") : "",
        el("div", { class: "note", title: usage[key].join(" · ") }, `In ${usage[key].slice(0, 8).join(" · ")}` +
          (usage[key].length > 8 ? ` and ${usage[key].length - 8} more` : "")),
        el("div", { class: "actions" },
          still.file ? el("button", { onclick: () => setStill(key, { status: still.status === "approved" ? "draft" : "approved" }) },
            still.status === "approved" ? "Unapprove" : "Approve") : "",
          el("button", { onclick: () => run("stills", { params: { only: [key] } }) }, still.file ? "Redraw" : "Draw")),
        el("details", {}, el("summary", { class: "note" }, "Prompt"), prompt,
          el("button", { onclick: () => setStill(key, { prompt_override: prompt.value }) }, "Keep this prompt")));
    })),
    unused.length ? el("p", { class: "note" }, `Not in any shot, so not drawn: ${unused.map((t) => t.item.name || t.item.id).join("、")}.`) : "");
}

// ---- 3. shots ----

function episodeSelect() {
  const episodes = ui.project.plan.episodes || [];
  return el("select", { onchange: (e) => { ui.episode = Number(e.target.value); renderProject(); } },
    episodes.map((e) => el("option", { value: e.number, selected: e.number === ui.episode }, `Episode ${e.number}: ${e.title}`)));
}

const STALE = { references: "a still it used was redrawn or replaced",
  prompt: "its prompt changed since this take (the plan, a kept prompt or the app's template)" };

// A rendered shot shows the first frame of its contact sheet; a missing one shows its place.
function tile(shot, s) {
  const place = ui.project.stills[`location:${shot.location}`];
  const status = !s.file ? "missing" : (s.outdated || []).length ? "outdated" : "done";
  const picture = s.file ? s.sheet : place?.file;
  return el("a", { class: `tile ${status}`, href: `#shot-${shot.id}`, title: shot.summary },
    picture ? el("img", { src: file(picture), alt: "" }) : el("div", { class: "blank" }),
    el("span", { class: "tag" }, `${shot.id} · ${shot.seconds}s`),
    shot.line ? el("span", { class: "say" }, shot.line) : "");
}

function renderShots() {
  const plan = ui.project.plan;
  if (!plan.episodes) return el("p", {}, "Plan the drama first.");
  const episode = plan.episodes.find((e) => e.number === ui.episode) || plan.episodes[0];
  const state = ui.project.episodes[String(episode.number)] || {};
  const each = ui.project.estimates.shot;
  const rendered = episode.shots.filter((s) => state[s.id]?.file);
  const missing = episode.shots.filter((s) => !state[s.id]?.file);
  const stale = rendered.filter((s) => (state[s.id].outdated || []).length);
  const render = (shots, what) => confirmRun(what, shots.length, each,
    () => run("shots", { episode: episode.number, params: { only: shots.map((s) => s.id) } }));
  const previous = state.use_previous_frame ?? state.continues;
  return el("section", {},
    el("div", { class: "actions" }, episodeSelect(),
      !rendered.length && missing.length ? [
        el("button", { class: "primary", disabled: busy(), onclick: () => render(missing.slice(0, 1), "Render a sample") },
          `Render a sample: ${missing[0].id}`),
        el("button", { disabled: busy(), onclick: () => render(missing, `Render ${missing.length} shots`) }, `Render all ${missing.length}`)]
        : missing.length ? el("button", { class: "primary", disabled: busy(), onclick: () => render(missing, `Render ${missing.length} shots`) },
          `Render the other ${missing.length}`) : "",
      stale.length ? el("button", { disabled: busy(), onclick: () => render(stale, `Re-render ${stale.length} outdated shots`) },
        `Re-render ${stale.length} outdated`) : "",
      el("button", { onclick: () => confirmRun(`Render ${missing.length} shots, then subtitle and assemble`, missing.length, each,
        () => run("episode", { episode: episode.number })) }, "Render, subtitle and assemble"),
      el("span", { class: "note" }, `${rendered.length} of ${episode.shots.length} rendered` +
        (each ? ` · about ${Math.round(each / 60)} min of GPU time per shot here` : ""))),
    el("div", { class: "strip" }, episode.shots.map((shot) => tile(shot, state[shot.id] || {}))),
    episode.number > 1 ? el("label", { class: "check" }, el("input", {
      type: "checkbox", checked: previous, onchange: async (e) => {
        await api(`/api/projects/${ui.project.id}/episodes/${episode.number}`, { method: "PUT", body: { use_previous_frame: e.target.checked } });
        await openProject(ui.project.id);
      },
    }), `Use the last frame of episode ${episode.number - 1} as an extra reference for the first shot` +
      (state.use_previous_frame == null ? ` (default: ${state.continues ? "on, same place" : "off, the place changes"})` : "")) : "",
    episode.shots.map((shot) => renderShotCard(episode, shot, state[shot.id] || {})));
}

function renderShotCard(episode, shot, s) {
  const plan = ui.project.plan;
  const prompt = input(s.prompt_override || s.prompt || ui.project.previews[shot.id] || "", () => {}, { area: true, rows: 12 });
  const speech = s.speech;
  const stale = s.outdated || [];
  return el("div", { class: "card shot", id: `shot-${shot.id}` },
    s.file ? el("video", { src: file(s.file), controls: true, preload: "metadata" }) : el("div", { class: "note placeholder" }, "not rendered"),
    el("div", {},
      el("div", { class: "title" }, el("h3", {}, `${shot.id} · ${shot.seconds}s · ${shot.location}`),
        el("span", { class: "badges" }, s.take ? el("span", { class: "badge" }, `take ${s.take}`) : "",
          stale.length ? el("span", { class: "badge warn", title: stale.map((r) => STALE[r]).join("; ") }, "outdated") : "")),
      stale.length ? el("div", { class: "error" }, `Outdated: ${stale.map((r) => STALE[r]).join("; ")}.`) : "",
      shot.line ? el("div", { class: "line" }, `${(plan.characters.find((c) => c.id === shot.speaker) || {}).name || shot.speaker}：${shot.line}`)
        : el("div", { class: "note" }, "no line"),
      speech ? el("div", { class: speech.match >= 0.6 ? "note" : "error" },
        `heard “${speech.heard}” · match ${Math.round(speech.match * 100)}% · ${speech.start}–${speech.end}s (${speech.timing})`) : "",
      el("div", { class: "note" }, shot.summary),
      s.references ? el("div", { class: "refs" }, s.references.filter((r) => r.includes("/")).map((r) => el("img", { src: file(r) }))) : "",
      s.sheet ? el("img", { class: "sheet", src: file(s.sheet) }) : "",
      el("div", { class: "actions" },
        el("button", { disabled: busy(), onclick: () => run("shots", { episode: episode.number, params: { only: [shot.id] } }) },
          s.file ? "Render another take" : "Render"),
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
}

// ---- 4. episodes ----

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

// ---- flow and settings ----

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
      el("td", {}, el("b", {}, step.name)), el("td", {}, step.model_from ? `action · ${step.model_from} model` : step.scope),
      el("td", {}, step.role || "–"), el("td", {}, step.description), el("td", { class: "note" }, step.source)))),
    Object.entries(s.plugin_errors).map(([plugin, error]) => el("p", { class: "error" }, `${plugin}: ${error}`)),
    el("button", { onclick: () => api("/api/plugins/reload", { method: "POST" }).then(refresh).then(() => openProject(ui.project.id)) }, "Reload plugins"),
    el("p", { class: "note" }, "A plugin is one Python file in the plugins directory that defines STEP. Add its name to a flow version to run it; delete the file and reload to remove it. Actions (story, revise, review) run with the model of the flow's plan step."));
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
    el("p", { class: "note" }, "Leave empty to plan, write and revise with the local TRTMC model. The key is stored with mode 600 in the data directory and never shown again."),
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

async function createProject(body) {
  try {
    const created = await api("/api/projects", { method: "POST", body });
    await refresh();
    ui.tab = "plan";
    await openProject(created.id);
    if (body.idea) await run("story");  // an idea project starts by writing its story
  } catch (error) { alert(error.message); }
}

const sampleTitle = document.getElementById("new-title").value;
document.getElementById("new-source").addEventListener("change", () => {
  const source = document.querySelector("input[name=source]:checked").value;
  document.getElementById("source-novel").hidden = source !== "novel";
  document.getElementById("source-idea").hidden = source !== "idea";
  const title = document.getElementById("new-title");
  if (source !== "sample" && title.value === sampleTitle) title.value = "";
});

document.getElementById("create").addEventListener("click", async () => {
  const source = document.querySelector("input[name=source]:checked").value;
  const fileInput = document.getElementById("new-file");
  let novel = document.getElementById("new-novel").value;
  if (fileInput.files[0]) novel = await fileInput.files[0].text();
  const idea = document.getElementById("new-idea").value.trim();
  await createProject({
    title: document.getElementById("new-title").value || (source === "idea" ? idea.slice(0, 16) : "drama"),
    sample: source === "sample", novel: source === "novel" ? novel : "", idea: source === "idea" ? idea : "",
    episodes: Number(document.getElementById("new-episodes").value),
    max_seconds: Number(document.getElementById("new-seconds").value),
  });
});

document.getElementById("idea-go").addEventListener("click", () => {
  const idea = document.getElementById("idea-box").value.trim();
  if (idea) createProject({ title: idea.slice(0, 16), idea, episodes: 4, max_seconds: 60 });
});

refresh().then(() => { if (ui.state.projects.length) openProject(ui.state.projects.at(-1).id); });
setInterval(() => refresh().catch(() => {}), 4000);

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// One project: five steps, the one thing to do next (or what is running, or what failed), and
// the editors of each step. Uses the helpers of app.js.

const STEPS = [["story", "Story"], ["plan", "Plan"], ["pictures", "Reference pictures"], ["shots", "Shots"], ["episodes", "Episodes"]];
const STEP_JOBS = { story: ["story"], plan: ["plan", "revise", "review"], pictures: ["stills"], shots: ["shots"],
  episodes: ["subtitles", "assemble"] };
const KINDS = { character: "Character", location: "Place", prop: "Prop" };
const STALE = { references: "a picture it used was redrawn or replaced",
  prompt: "its prompt changed since this take (the plan, a kept prompt or the app's template)" };

const pid = () => encodeURIComponent(ui.project.id);
const file = (relative) => `/api/projects/${pid()}/files/${relative}?v=${encodeURIComponent(relative)}`;
const projectJobs = () => ui.state.jobs.filter((j) => j.project === ui.project.id);
const projectBusy = () => projectJobs().some((j) => j.status === "running" || j.status === "queued");
const go = (step) => { if (ui.step !== step) location.hash = `#/p/${pid()}/${step}`; };
const capital = (text) => text.charAt(0).toUpperCase() + text.slice(1);
const modelOf = (role) => (ui.state.models.find((m) => m.role === role) || {}).name || `the ${role} model`;

async function openProject(id) {
  try {
    const project = await api(`/api/projects/${encodeURIComponent(id)}`);
    if (ui.project?.id !== project.id) ui.episode = 1;
    ui.project = project;
  } catch (error) {
    fail(error);
    location.hash = "#/";
    return;
  }
  renderProject();
}

async function reloadProject() {
  ui.project = await api(`/api/projects/${pid()}`);
  renderProject();
}

async function run(step, extra = {}) {
  try {
    await api(`/api/projects/${pid()}/run`, { method: "POST", body: { step, ...extra } });
    toast(`Started: ${step === "episode" ? "rendering, subtitles and assembly" : (STEP_NAMES[step] || step).toLowerCase()}`);
    await refresh();
  } catch (error) { fail(error); }
}

function confirmRun(what, count, each, start) {
  const time = each ? `${about(count * each)} of GPU time` : "the time is measured on the first one";
  if (count > 1 && !confirm(`${what}: ${time}. Continue?`)) return;
  start();
}

function markDirty() {
  ui.dirty = true;
  document.getElementById("save-bar")?.removeAttribute("hidden");
}

// ---- the five steps ----

function stepStates() {
  const p = ui.project;
  const stage = Object.fromEntries(p.stages.map((s) => [s.key, s]));
  const planned = Boolean(p.plan.episodes);
  const needed = Object.keys(p.usage);
  const approved = needed.filter((key) => p.stills[key]?.status === "approved").length;
  const rendered = Object.values(p.episodes).some((e) => Object.values(e).some((v) => v && v.file));
  const level = (s) => (s.status === "done" ? "done" : s.status === "attention" ? "attention" : "todo");
  const steps = [
    { key: "story", status: p.novel_chars ? "done" : "todo",
      detail: p.novel_chars ? `${p.novel_chars.toLocaleString()} characters` : "from your idea" },
    { key: "plan", status: !p.novel_chars ? "locked" : level(stage.plan), lock: "Write the story first",
      detail: planned ? `${plural(p.plan.episodes.length, "episode")} · ${plural(Object.keys(p.previews).length, "shot")}` : "not planned yet" },
    { key: "pictures", status: !planned ? "locked" : level(stage.stills), lock: "Plan the drama first", detail: stage.stills.detail },
    { key: "shots", status: !planned || (approved < needed.length && !rendered) ? "locked" : level(stage.shots),
      lock: "Approve the pictures first", detail: stage.shots.detail },
    { key: "episodes", status: !rendered ? "locked" : level(stage.episodes), lock: "Render the shots first",
      detail: stage.episodes.detail },
  ];
  const active = projectJobs().filter((j) => j.status === "running" || j.status === "queued");
  for (const s of steps) s.running = active.some((j) => STEP_JOBS[s.key].includes(j.step));
  const current = steps.find((s) => s.status !== "done" && s.status !== "locked");
  if (current && current.status === "todo") current.status = "current";
  return steps;
}

function stepper(steps, step) {
  return el("nav", { class: "stepper", "aria-label": "Steps" }, steps.map((s, index) => el("a", {
    href: `#/p/${pid()}/${s.key}`, class: s.running ? "running" : s.status,
    "aria-current": s.key === step ? "step" : "false", title: s.status === "locked" ? s.lock : s.detail,
  }, el("span", { class: "circle" }, s.status === "done" && !s.running ? "✓" : index + 1),
  el("span", { class: "label" }, STEPS[index][1]),
  el("small", {}, s.running ? "running…" : s.status === "locked" ? s.lock : s.detail))));
}

function renderProject() {
  if (ui.route.page !== "project" || !ui.project) return;
  const p = ui.project;
  const steps = stepStates();
  const known = STEPS.some(([key]) => key === ui.route.step) || ui.route.step === "flow";
  const step = known ? ui.route.step
    : (steps.find((s) => s.running) || steps.find((s) => s.status === "current" || s.status === "attention") || steps.at(-1)).key;
  ui.step = step;
  const body = { story: storyStep, plan: planStep, pictures: picturesStep, shots: shotsStep, episodes: episodesStep,
    flow: flowStep }[step]();
  const style = p.plan.style || p.meta.style;
  document.getElementById("main").replaceChildren(
    el("div", { class: "crumbs" }, el("a", { href: "#/" }, "Projects"), ` › ${p.meta.title || p.id}`),
    el("div", { class: "page-head" },
      el("div", { class: "grow" }, el("h1", {}, p.meta.title || p.id),
        el("p", {}, `${p.meta.idea ? "From an idea" : "From a novel"} · ${plural(p.meta.episodes, "episode")} of up to ` +
          `${p.meta.max_seconds} s · ${style ? `style: ${style}` : "style chosen when planning"}`)),
      el("button", { class: "small", onclick: renameProject }, "Rename")),
    stepper(steps, step),
    el("div", { class: "subnav" }, el("a", { href: `#/p/${pid()}/flow`, "aria-current": step === "flow" ? "page" : "false" },
      "Flow (advanced)")),
    el("div", { id: "next" }, nextCard()),
    el("div", { id: "step" }, body));
  ui.dirty = false;
}

async function renameProject() {
  const title = prompt("A new title for this project", ui.project.meta.title || "");
  if (title == null || !title.trim()) return;
  try {
    await api(`/api/projects/${pid()}/meta`, { method: "PUT", body: { title } });
    await reloadProject();
  } catch (error) { fail(error); }
}

// ---- the next thing to do, what runs, or what failed ----

function renderNextCard() {
  const box = document.getElementById("next");
  if (box && ui.project) box.replaceChildren(...[nextCard()].flat());
}

function nextCard() {
  const jobs = projectJobs();
  const active = jobs.filter((j) => j.status === "running" || j.status === "queued");
  if (active.length) return runningCard(active);
  const last = jobs.find((j) => j.finished);
  if (last && last.status === "failed" && !ui.dismissed.has(last.id)) return failedCard(last);
  return actionCard(nextAction());
}

function phaseEstimate(job) {
  const estimates = { ...(ui.list?.estimates || {}), ...Object.fromEntries(Object.entries(ui.project.estimates).filter(([, v]) => v)) };
  return { shots: estimates.shot, stills: estimates.still }[job.step] || null;
}

function runningCard(active) {
  const job = active.find((j) => j.status === "running");
  const queued = active.filter((j) => j.status === "queued").reverse();
  const engine = ui.state.engine;
  let title, phase, start = null, estimate = null;
  if (!job) {
    const other = running();
    title = stepName(queued.shift());
    phase = other ? `Waiting for the GPU: “${projectTitle(other.project)}” is ${stepName(other).toLowerCase()} first.` : "Starting…";
  } else if (engine.loading.length) {
    title = stepName(job);
    phase = `Loading ${engine.loading.join(" and ")} onto the GPU. A step first loads the models it needs; this takes a minute or more.`;
    start = engineSince();
  } else {
    title = stepName(job);
    phase = job.message ? capital(job.message) : "Starting…";
    start = since(job.phase_seconds ?? job.seconds);
    estimate = phaseEstimate(job);
  }
  return el("section", { class: "next running" },
    el("div", {}, el("div", { class: "kicker" }, "Running now"), el("h2", {}, title),
      el("p", {}, phase, start != null ? [" · ", timer(start), estimate ? ` (usually ${about(estimate)})` : ""] : "")),
    el("div", { class: "go" }, el("button", { onclick: () => openDrawer(true) }, "Activity and log"),
      el("button", { class: "danger", onclick: cancelAll }, "Cancel")),
    el("div", { class: "detail" }, start != null ? meter(start, estimate) : meter(0, null),
      queued.length ? el("ul", { class: "queue" }, queued.map((q) => el("li", {}, stepName(q)))) : ""));
}

function failedCard(job) {
  const retry = () => run(job.step, { ...(job.episode ? { episode: job.episode } : {}), params: job.params || {} });
  return el("section", { class: "next failed" },
    el("div", {}, el("div", { class: "kicker" }, "Failed"), el("h2", {}, `${stepName(job)} failed`),
      el("p", {}, (job.error || "no reason was given").replace(/^\w+Error: /, ""))),
    el("div", { class: "go" },
      el("button", { class: "primary big", onclick: retry }, "Try again"),
      el("button", { class: "big", onclick: () => openDrawer(true) }, "Model server log"),
      el("button", { class: "quiet", onclick: () => { ui.dismissed.add(job.id); renderNextCard(); renderStatus(); } }, "Dismiss")));
}

function actionCard(action) {
  const index = STEPS.findIndex(([key]) => key === action.step);
  return el("section", { class: "next" },
    el("div", {}, el("div", { class: "kicker" }, action.done ? "All done" : `Next · step ${index + 1} of 5`),
      el("h2", {}, action.title), el("p", {}, action.text)),
    el("div", { class: "go" }, action.buttons));
}

// The first thing the project still needs, with the button that does it.
function nextAction() {
  const p = ui.project;
  const plan = p.plan;
  const act = (step, label, start, primary = true) => el("button", {
    class: primary ? "primary big" : "big", disabled: projectBusy(),
    onclick: () => { go(step); start(); } }, label);
  const look = (step, label) => el("a", { class: "button big", href: `#/p/${pid()}/${step}` }, label);
  if (!ui.state.chat_ready && !plan.episodes) {
    return { step: p.novel_chars ? "plan" : "story", title: "Choose a language model",
      text: "It writes the story and plans the drama. The NVIDIA API catalog and OpenRouter have free tiers: pick one in Settings and paste a key.",
      buttons: [el("a", { class: "button primary big", href: "#/settings" }, "Open Settings")] };
  }
  if (!p.novel_chars) {
    return { step: "story", title: "Write the story from your idea",
      text: "The language model turns the idea into a short story, one chapter per episode. It takes about a minute, and you can edit the story afterwards.",
      buttons: [act("story", "Write the story", () => run("story"))] };
  }
  if (!plan.episodes) {
    return { step: "plan", title: "Plan the drama",
      text: `The language model reads the story (${p.novel_chars.toLocaleString()} characters) and writes the cast, places, props ` +
        "and a shot list for each episode. It takes a few minutes; you can edit every part afterwards.",
      buttons: [act("plan", "Plan the drama", () => run("plan"))] };
  }
  const errors = p.problems.filter((x) => x.level === "error");
  if (errors.length) {
    return { step: "plan", title: `Fix ${plural(errors.length, "problem")} in the plan`,
      text: `${errors[0].where}: ${errors[0].message}${errors[0].fix ? `. ${errors[0].fix}` : ""}. The checks list them all on the Plan step.`,
      buttons: [look("plan", "Show the problems")] };
  }
  const used = stillTargets(plan).filter((t) => p.usage[t.key]);
  const missing = used.filter((t) => !p.stills[t.key]?.file);
  const outdated = used.filter((t) => p.outdated_stills.includes(t.key));
  const unapproved = used.filter((t) => p.stills[t.key]?.file && p.stills[t.key].status !== "approved");
  if (missing.length) {
    return { step: "pictures", title: `Draw ${plural(missing.length, "reference picture")}`,
      text: `The image model (${modelOf("image")}) draws every character, place and prop once in the plan's style; each shot is filmed from these pictures. ` +
        (p.estimates.still ? `${capital(about(missing.length * p.estimates.still))}.` : "The first one also loads the image model onto the GPU."),
      buttons: [act("pictures", `Draw ${plural(missing.length, "picture")}`, () => run("stills"))] };
  }
  if (outdated.length) {
    return { step: "pictures", title: `Redraw ${plural(outdated.length, "outdated picture")}`,
      text: "Their description or the style changed after they were drawn.",
      buttons: [act("pictures", `Redraw ${outdated.length}`, () => run("stills", { params: { only: outdated.map((t) => t.key) } })),
        look("pictures", "Look at them")] };
  }
  if (unapproved.length) {
    return { step: "pictures", title: `Check and approve ${plural(unapproved.length, "picture")}`,
      text: "Shots use approved pictures only. Look at each one: approve it, or redraw it with a new seed or your own prompt.",
      buttons: [act("pictures", `Approve all ${unapproved.length}`, () => approve(unapproved.map((t) => t.key))),
        ui.step === "pictures" ? "" : look("pictures", "Look at them first")] };
  }
  const anyRendered = Object.values(p.episodes).some((e) => Object.values(e).some((v) => v && v.file));
  for (const episode of plan.episodes) {
    const state = p.episodes[String(episode.number)] || {};
    const todo = episode.shots.filter((s) => !state[s.id]?.file);
    const stale = episode.shots.filter((s) => state[s.id]?.file && (state[s.id].outdated || []).length);
    const time = (n) => (p.estimates.shot ? `${capital(about(n * p.estimates.shot))} of GPU time.`
      : "The first shot also loads the video model; then each shot takes several minutes, and this card shows the time as it runs.");
    if (!anyRendered && todo.length) {
      const first = todo[0];
      return { step: "shots", title: "Render a sample shot",
        text: `Render ${first.id} (${first.seconds} s${first.line ? `, “${first.line}”` : ""}) first, to check the look and the voices ` +
          `before the other ${plural(todo.length - 1, "shot")}. ${time(1)}`,
        buttons: [act("shots", "Render the sample", () => { ui.episode = episode.number; run("shots", { episode: episode.number, params: { only: [first.id] } }); })] };
    }
    if (stale.length) {
      return { step: "shots", title: `Re-render ${plural(stale.length, "outdated shot")} of episode ${episode.number}`,
        text: `Something they were rendered from changed. ${time(stale.length)}`,
        buttons: [act("shots", `Re-render ${stale.length}`, () => renderShots(episode, stale, `Re-render ${stale.length} outdated shots`))] };
    }
    if (todo.length) {
      return { step: "shots", title: `Render ${plural(todo.length, "shot")} of episode ${episode.number}`,
        text: `${time(todo.length)} Or render them, time the subtitles and assemble the episode in one go.`,
        buttons: [act("shots", `Render ${plural(todo.length, "shot")}`, () => renderShots(episode, todo, `Render ${todo.length} shots`)),
          act("episodes", "Render, subtitle and assemble", () => runEpisode(episode, todo), false)] };
    }
    if (!state.assembled) {
      return { step: "episodes", title: `Subtitle and assemble episode ${episode.number}`,
        text: `The speech model (${modelOf("asr")}) times each line from the voice in its shot, then the clips are joined into one 9:16 video.`,
        buttons: [act("episodes", "Subtitle and assemble", () => run("episode", { episode: episode.number }))] };
    }
  }
  return { step: "episodes", done: true, title: "The drama is ready",
    text: `${plural(plan.episodes.length, "episode")} assembled. Watch and download them, or re-render any shot and assemble again.`,
    buttons: [ui.step === "episodes" ? "" : look("episodes", "Watch and download")] };
}

function renderShots(episode, shots, what) {
  ui.episode = episode.number;
  confirmRun(what, shots.length, ui.project.estimates.shot,
    () => run("shots", { episode: episode.number, params: { only: shots.map((s) => s.id) } }));
}

function runEpisode(episode, todo) {
  confirmRun(`Render ${todo.length} shots, then subtitle and assemble`, todo.length, ui.project.estimates.shot,
    () => run("episode", { episode: episode.number }));
}

async function approve(keys, status = "approved") {
  try {
    for (const key of keys) {
      await api(`/api/projects/${pid()}/stills/${encodeURIComponent(key)}`, { method: "PUT", body: { status } });
    }
    if (keys.length > 1) toast(`${plural(keys.length, "picture")} approved`);
    await reloadProject();
  } catch (error) { fail(error); }
}

function locked(text, step) {
  const name = STEPS.find(([key]) => key === step)[1];
  return el("section", { class: "panel rail" }, el("p", {}, text),
    el("a", { class: "button", href: `#/p/${pid()}/${step}` }, `Go to ${name}`));
}

// ---- editors ----

function input(value, onchange, { area = false, rows = 2, type = "text" } = {}) {
  const node = area ? el("textarea", { rows }) : el("input", { type });
  node.value = value ?? "";
  node.addEventListener("input", () => { markDirty(); onchange(node.value); });
  return node;
}

// An editable table; `kind` adds a button that asks the model to revise one row.
function table(rows, columns, kind = null) {
  const remove = (row) => { rows.splice(rows.indexOf(row), 1); renderStep(); markDirty(); };
  const ask = (row) => revise(`${kind}:${row.id}`, prompt(`What should change about ${row.name || row.id}?`));
  return el("div", {}, el("table", { class: "edit" },
    el("tr", {}, columns.map(([key, , label]) => el("th", {}, label || key)), el("th", {})),
    rows.map((row) => el("tr", {}, columns.map(([key, width]) => el("td", { style: `width:${width}` },
      input(row[key], (v) => { row[key] = v; }, { area: width === "auto", rows: 2 }))),
    el("td", { class: "row-actions" },
      kind ? el("button", { class: "small", title: "Revise with the model", disabled: projectBusy(), onclick: () => ask(row) }, "Revise") : "",
      el("button", { class: "small danger", title: "Remove", onclick: () => remove(row) }, "✕"))))),
  el("button", { class: "small", style: "margin-top:6px", onclick: () => {
    rows.push(Object.fromEntries(columns.map(([key]) => [key, ""])));
    renderStep();
    markDirty();
  } }, "Add"));
}

// Re-render the open step only, keeping unsaved edits that live in ui.project.plan.
function renderStep() {
  const dirty = ui.dirty;
  const body = { story: storyStep, plan: planStep, pictures: picturesStep, shots: shotsStep, episodes: episodesStep,
    flow: flowStep }[ui.step]();
  document.getElementById("step").replaceChildren(body);
  if (dirty) markDirty();
}

function saveBar() {
  return el("div", { class: "save-bar", id: "save-bar", hidden: true },
    el("span", { class: "note" }, "You have unsaved edits."),
    el("button", { onclick: () => { ui.dirty = false; reloadProject(); } }, "Discard"),
    el("button", { class: "primary", onclick: savePlan }, "Save edits"));
}

async function savePlan() {
  try {
    await api(`/api/projects/${pid()}/plan`, { method: "PUT", body: ui.project.plan });
    ui.dirty = false;
    toast("Plan saved");
    await reloadProject();
  } catch (error) { fail(error); }
}

async function revise(target, instruction) {
  if (ui.dirty) return toast("Save your edits first: the model revises the saved plan.", "bad");
  if (!instruction || !instruction.trim()) return;
  delete ui.presets[target.split(":")[1]];
  await run("revise", { params: { target, instruction: instruction.trim() } });
}

function jump(where) {
  const node = document.getElementById(`plan-${where.replace(/ /g, "-")}`);
  if (node) node.scrollIntoView({ behavior: "smooth", block: "center" });
}

// ---- 1. story ----

function storyStep() {
  const p = ui.project;
  const idea = p.meta.idea ? el("div", { class: "idea" }, el("div", { class: "faint" }, "Your idea"), p.meta.idea) : "";
  if (!p.novel_chars) return el("section", {}, idea, el("p", { class: "note" }, "The story appears here once it is written."));
  const text = el("textarea", { class: "story-text", "aria-label": "The story" });
  const cached = ui.novel?.id === p.id && ui.novel.chars === p.novel_chars;
  text.value = cached ? ui.novel.text : "Loading…";
  if (!cached) {
    api(`/api/projects/${pid()}/novel`).then((r) => {
      ui.novel = { id: p.id, chars: p.novel_chars, text: r.novel };
      if (!ui.dirty) text.value = r.novel;
    }).catch(fail);
  }
  text.addEventListener("input", () => { ui.dirty = true; });
  const save = async () => {
    try {
      await api(`/api/projects/${pid()}/novel`, { method: "PUT", body: { novel: text.value } });
      ui.dirty = false;
      toast("Story saved");
      await reloadProject();
    } catch (error) { fail(error); }
  };
  return el("section", {}, idea,
    el("div", { class: "section-title" }, el("h2", {}, "The story"), el("span", { class: "faint" }, `${p.novel_chars.toLocaleString()} characters`)),
    text,
    el("div", { class: "actions" }, el("button", { class: "primary", onclick: save }, "Save the story"),
      el("span", { class: "note" }, p.plan.episodes ? "To use an edited story, plan again on the Plan step; the current plan is kept under Earlier versions."
        : "Edit freely; the plan is written from the saved story.")));
}

// ---- 2. plan ----

function renderChecks() {
  const { problems } = ui.project;
  const review = ui.project.model_review?.items ? ui.project.model_review : null;
  const errors = problems.filter((x) => x.level === "error").length;
  return el("section", { class: `panel checks${problems.length ? "" : " clean"}` },
    el("div", { class: "title" },
      el("h3", {}, problems.length ? `${plural(problems.length, "thing")} to check before rendering` +
        (errors ? ` (${errors} must be fixed)` : "") : "The plan checks found nothing to fix"),
      el("button", { class: "small", disabled: projectBusy(), onclick: () => run("review") },
        review ? "Review again with the model" : "Ask the model to review")),
    problems.length ? el("ul", { class: "problems" }, problems.map((x) => el("li", {},
      el("span", { class: `badge ${x.level === "error" ? "bad" : "warn"}` }, x.level), " ",
      el("a", { onclick: () => jump(x.where) }, x.where), ` ${x.message}`,
      x.fix ? el("span", { class: "note" }, ` · ${x.fix}`) : ""))) : "",
    review ? el("div", {},
      el("h3", { style: "margin-top:12px" }, `Model review: ${review.items.length ? plural(review.items.length, "note") : "no problems"}` +
        (review.current ? "" : " (written for an earlier version of the shots)")),
      el("ul", { class: "problems" }, review.items.map((x) => el("li", {},
        el("a", { onclick: () => jump(x.shot) }, x.shot), ` ${x.problem}`,
        x.fix ? el("span", { class: "note" }, ` · Fix: ${x.fix} `) : " ",
        el("button", { class: "small", onclick: () => { ui.presets[x.shot] = x.fix || x.problem; renderStep(); jump(x.shot); } },
          "Revise this shot"))))) : "");
}

function renderHistory() {
  const p = ui.project;
  if (!p.history.length) return "";
  return el("details", { style: "margin:12px 0" }, el("summary", {}, `Earlier versions of the plan (${p.history.length})`),
    p.history.map((h) => el("div", { class: "actions", style: "margin:4px 0" },
      el("span", { class: "note" }, `${new Date(h.time * 1000).toLocaleString()} · ${h.note}`),
      el("button", { class: "small", onclick: async () => {
        if (!confirm("Restore this version? The current plan is kept as a version too.")) return;
        await api(`/api/projects/${pid()}/history/${h.name}/restore`, { method: "POST" }).catch(fail);
        await reloadProject();
      } }, "Restore"))));
}

function reviseBox(target, preset = "") {
  const note = el("textarea", { rows: 2, placeholder: "What should change? For example: 让他先摸头再开口，台词更短" });
  note.value = preset;
  return el("div", { class: "revise" }, note,
    el("button", { disabled: projectBusy(), onclick: () => revise(target, note.value) }, "Revise with the model"));
}

function newShot(episode) {
  const taken = new Set(episode.shots.map((s) => s.id));
  let n = episode.shots.length + 1;
  while (taken.has(`e${episode.number}s${n}`)) n += 1;
  episode.shots.push({ id: `e${episode.number}s${n}`, seconds: 6, location: "", characters: [], props: [],
    speaker: "", line: "", camera: "", action: "", after: "", then: "", sound: "", summary: "" });
  renderStep();
  markDirty();
}

function renderShotEditor(episode, shot) {
  const p = ui.project;
  const list = (key) => input((shot[key] || []).join(", "), (v) => {
    shot[key] = v.split(",").map((x) => x.trim()).filter(Boolean);
  });
  const issues = p.problems.filter((x) => x.where === shot.id).length;
  const origin = p.origins[shot.id];
  return el("div", { class: "panel shot-edit", id: `plan-${shot.id}` },
    el("div", { class: "title" }, el("h3", {}, shot.id),
      el("span", { class: "badges" }, el("span", { class: "badge" }, `${shot.seconds} s`),
        origin ? el("span", { class: `badge ${origin === "novel" ? "ok" : ""}` }, origin === "novel" ? "line from the novel" : "adapted line") : "",
        issues ? el("span", { class: "badge warn" }, `${issues} to check`) : ""),
      el("span", { class: "spacer", style: "flex:1" }),
      el("button", { class: "small danger", onclick: () => { episode.shots.splice(episode.shots.indexOf(shot), 1); renderStep(); markDirty(); } },
        "Remove shot")),
    el("div", { class: "row" },
      el("label", {}, "Seconds", input(shot.seconds, (v) => { shot.seconds = Number(v) || 6; }, { type: "number" })),
      el("label", {}, "Place", input(shot.location, (v) => { shot.location = v; })),
      el("label", {}, "Characters", list("characters")),
      el("label", {}, "Props", list("props")),
      el("label", {}, "Speaker", input(shot.speaker, (v) => { shot.speaker = v; }))),
    el("label", {}, "Summary", input(shot.summary, (v) => { shot.summary = v; })),
    el("label", {}, "Line (Chinese)", input(shot.line, (v) => { shot.line = v; })),
    el("div", { class: "row" }, ["camera", "action", "after", "then", "sound"].map((key) =>
      el("label", {}, capital(key), input(shot[key], (v) => { shot[key] = v; }, { area: true, rows: 3 })))),
    reviseBox(`shot:${shot.id}`, ui.presets[shot.id] || ""));
}

function planStep() {
  const p = ui.project;
  const plan = p.plan;
  if (!p.novel_chars) return locked("The plan is written from the story; write the story first.", "story");
  const again = el("button", { disabled: projectBusy() || !ui.state.chat_ready, onclick: () => {
    if (!plan.episodes || confirm("Plan again from the story? The current plan is kept under Earlier versions.")) run("plan");
  } }, plan.episodes ? "Plan again from the story" : "Plan the drama");
  const extra = p.flow_steps.filter((s) => !s.core).map((s) => (s.installed && s.scope === "project"
    ? el("button", { disabled: projectBusy() || !plan.episodes, onclick: () => run(s.step) }, `Run ${s.step}`)
    : el("span", { class: "error" }, `${s.step}: not installed, skipped`)));
  if (!plan.episodes) return el("section", {}, el("div", { class: "actions" }, again, extra), renderHistory());
  return el("section", {},
    el("div", { class: "actions" }, again, extra,
      el("span", { class: "note" }, "Edit anything below, then save; the checks run again on every save.")),
    renderChecks(), renderHistory(),
    el("h2", {}, "Title and setting"),
    el("div", { class: "row" },
      el("label", {}, "Title", input(plan.title, (v) => { plan.title = v; })),
      el("label", { style: "flex-grow:3" }, "Setting (English)", input(plan.setting, (v) => { plan.setting = v; }))),
    el("p", { class: "faint" }, "The visual style is set on the Reference pictures step."),
    el("h2", {}, "Characters"),
    el("p", { class: "note" }, "Each character's look and voice go into the picture and video prompts, in English."),
    table(plan.characters, [["id", "90px"], ["name", "100px"], ["role", "auto"], ["look", "auto"], ["voice", "auto"]], "character"),
    el("h2", {}, "Relationships"),
    table(plan.relationships || [], [["from", "120px"], ["to", "120px"], ["relation", "auto"]]),
    el("h2", {}, "Places"), table(plan.locations, [["id", "90px"], ["name", "120px"], ["look", "auto"]], "location"),
    el("h2", {}, "Props"), table(plan.props, [["id", "90px"], ["name", "120px"], ["look", "auto"]], "prop"),
    plan.episodes.map((episode) => el("div", {},
      el("div", { class: "section-title", id: `plan-episode-${episode.number}` }, el("h2", {}, `Episode ${episode.number}`),
        el("span", { class: "faint" }, `${plural(episode.shots.length, "shot")} · ${episode.shots.reduce((t, s) => t + Number(s.seconds || 0), 0)} s`)),
      el("div", { class: "row" }, el("label", {}, "Title", input(episode.title, (v) => { episode.title = v; })),
        el("label", { style: "flex-grow:3" }, "Summary", input(episode.summary, (v) => { episode.summary = v; }, { area: true }))),
      episode.shots.map((shot) => renderShotEditor(episode, shot)),
      el("button", { style: "margin-top:10px", onclick: () => newShot(episode) }, "Add shot"))),
    saveBar());
}

// ---- 3. reference pictures ----

function stillTargets(plan) {
  return [["character", plan.characters || []], ["location", plan.locations || []], ["prop", plan.props || []]]
    .flatMap(([kind, items]) => items.map((item) => ({ key: `${kind}:${item.id}`, kind, item })));
}

function styleBox() {
  const plan = ui.project.plan;
  const current = plan.style || "";
  const text = el("input", { value: current, "aria-label": "Visual style, in English" });
  const save = async (style) => {
    if (!style || style === current) return;
    try {
      await api(`/api/projects/${pid()}/plan`, { method: "PUT", body: { ...plan, style } });
      toast("Style saved. Pictures drawn in the old style are marked outdated.");
      await reloadProject();
    } catch (error) { fail(error); }
  };
  return el("section", { class: "panel rail style-box" },
    el("h3", {}, "Visual style"),
    el("p", { class: "note", style: "margin:0" }, "Every picture and every shot prompt starts from this style. Pick one, or describe your own; " +
      "English works best, since the prompts around it are English."),
    el("div", { class: "chips" }, STYLES.filter(([style]) => style).map(([style, name]) =>
      el("button", { class: `chip${style === current ? " on" : ""}`, disabled: projectBusy(), onclick: () => save(style) }, name))),
    el("div", { class: "row", style: "align-items:flex-end" },
      el("label", { style: "flex:4 1 320px;margin:0" }, "Style", text),
      el("button", { style: "flex:0 0 auto", disabled: projectBusy(), onclick: () => save(text.value.trim()) }, "Save style")));
}

function pictureCard({ key, kind, item }) {
  const p = ui.project;
  const still = p.stills[key] || {};
  const job = running();
  const drawing = job && job.project === p.id && job.step === "stills" && (job.message || "").endsWith(`: ${item.name || item.id}`);
  const outdated = p.outdated_stills.includes(key);
  const status = !still.file ? "" : outdated ? "outdated" : still.status === "approved" ? "approved" : "";
  const prompt = el("textarea", { rows: 6 });
  prompt.value = still.prompt_override || still.prompt || "";
  const set = async (body) => {
    try {
      await api(`/api/projects/${pid()}/stills/${encodeURIComponent(key)}`, { method: "PUT", body });
      await reloadProject();
    } catch (error) { fail(error); }
  };
  const uses = p.usage[key] || [];
  return el("article", { class: `picture ${status}` },
    el("div", { class: "frame" }, still.file ? el("img", { src: file(still.file), alt: item.name || item.id, loading: "lazy" })
      : drawing ? el("span", {}, el("span", { class: "dot run" }), " Drawing…") : "Not drawn yet"),
    el("div", { class: "body" },
      el("div", { class: "title" }, el("h3", { style: "margin:0" }, item.name || item.id),
        el("span", { class: `badge ${status === "approved" ? "ok" : status === "outdated" ? "warn" : ""}` },
          !still.file ? KINDS[kind] : outdated ? "outdated" : still.status === "approved" ? "approved" : "to check")),
      el("div", { class: "faint", title: uses.join(" · ") }, `${KINDS[kind]} · in ${uses.slice(0, 6).join(", ")}` +
        (uses.length > 6 ? ` and ${uses.length - 6} more` : "")),
      outdated ? el("div", { class: "error", style: "font-size:13px" }, "Drawn before its description or the style changed; redraw it.") : "",
      el("div", { class: "actions" },
        still.file ? el("button", { class: `small${still.status === "approved" ? "" : " primary"}`,
          onclick: () => set({ status: still.status === "approved" ? "draft" : "approved" }) },
        still.status === "approved" ? "Unapprove" : "Approve") : "",
        el("button", { class: "small", disabled: projectBusy(), onclick: () => run("stills", { params: { only: [key] } }) },
          still.file ? "Redraw" : "Draw")),
      el("details", {}, el("summary", {}, "Prompt"), prompt,
        el("button", { class: "small", style: "margin-top:6px", onclick: () => set({ prompt_override: prompt.value }) },
          "Keep this prompt for the next drawing"))));
}

function picturesStep() {
  const p = ui.project;
  const plan = p.plan;
  if (!plan.episodes) return locked("Pictures are drawn from the plan's cast, places and props; plan the drama first.", "plan");
  const targets = stillTargets(plan);
  const used = targets.filter((t) => p.usage[t.key]);
  const unused = targets.filter((t) => !p.usage[t.key]);
  const missing = used.filter((t) => !p.stills[t.key]?.file);
  const waiting = used.filter((t) => p.stills[t.key]?.file && p.stills[t.key].status !== "approved");
  const approved = used.length - missing.length - waiting.length;
  return el("section", {}, styleBox(),
    el("div", { class: "section-title" }, el("h2", {}, "Pictures"),
      el("span", { class: "faint" }, `${approved} of ${used.length} approved`)),
    el("div", { class: "actions", style: "margin-top:0" },
      missing.length ? el("button", { class: "primary", disabled: projectBusy(),
        onclick: () => confirmRun(`Draw ${missing.length} pictures`, missing.length, p.estimates.still, () => run("stills")) },
      `Draw ${plural(missing.length, "missing picture")}`) : "",
      waiting.length ? el("button", { onclick: () => approve(waiting.map((t) => t.key)) }, `Approve all ${waiting.length}`) : "",
      el("span", { class: "note" }, "Only what appears in a shot gets a picture. A shot renders once every picture it uses is approved.")),
    el("div", { class: "pictures" }, used.map(pictureCard)),
    unused.length ? el("p", { class: "note", style: "margin-top:14px" },
      `Not in any shot, so not drawn: ${unused.map((t) => t.item.name || t.item.id).join("、")}.`) : "");
}

// ---- 4. shots ----

// A rendered shot shows the first frame of its contact sheet; a missing one shows its place.
function tile(shot, s, live) {
  const place = ui.project.stills[`location:${shot.location}`];
  const status = live ? "rendering" : !s.file ? "missing" : (s.outdated || []).length ? "outdated" : "done";
  const picture = s.file ? s.sheet : place?.file;
  return el("a", { class: `tile ${status}`, href: `#shot-${shot.id}`, title: shot.summary,
    onclick: (e) => { e.preventDefault(); document.getElementById(`shot-${shot.id}`)?.scrollIntoView({ behavior: "smooth" }); } },
  picture ? el("img", { src: file(picture), alt: "" }) : el("div", { class: "blank" }),
  el("span", { class: "tag" }, `${shot.id} · ${shot.seconds}s`),
  shot.line ? el("span", { class: "say" }, shot.line) : "",
  live ? el("span", { class: "live" }, "rendering") : "");
}

function liveShot() {
  const job = running();
  if (!job || job.project !== ui.project.id || job.step !== "shots") return null;
  return ((job.message || "").match(/\(([a-z0-9_]+)\)$/) || [])[1] || null;
}

function shotsStep() {
  const p = ui.project;
  const plan = p.plan;
  if (!plan.episodes) return locked("Shots come from the plan; plan the drama first.", "plan");
  const episode = plan.episodes.find((e) => e.number === ui.episode) || plan.episodes[0];
  ui.episode = episode.number;
  const state = p.episodes[String(episode.number)] || {};
  const each = p.estimates.shot;
  const rendered = episode.shots.filter((s) => state[s.id]?.file);
  const missing = episode.shots.filter((s) => !state[s.id]?.file);
  const stale = rendered.filter((s) => (state[s.id].outdated || []).length);
  const needed = Object.keys(p.usage);
  const unapproved = needed.filter((key) => p.stills[key]?.status !== "approved").length;
  const ready = (shot) => needed.every((key) => !p.usage[key].includes(shot.id) || p.stills[key]?.status === "approved");
  const allReady = (shots) => shots.every(ready);
  const live = liveShot();
  const previous = state.use_previous_frame ?? state.continues;
  return el("section", {},
    unapproved ? el("p", { class: "panel rail" }, `${plural(unapproved, "picture")} still to approve: a shot renders once every picture it uses is approved. `,
      el("a", { href: `#/p/${pid()}/pictures` }, "Go to Reference pictures")) : "",
    plan.episodes.length > 1 ? el("div", { class: "episode-tabs", role: "tablist" }, plan.episodes.map((e) => el("button", {
      class: `small${e.number === episode.number ? " primary" : ""}`, role: "tab", "aria-selected": String(e.number === episode.number),
      onclick: () => { ui.episode = e.number; renderStep(); } }, `Episode ${e.number}`))) : "",
    el("div", { class: "section-title" }, el("h2", {}, `Episode ${episode.number}: ${episode.title}`),
      el("span", { class: "faint" }, `${rendered.length} of ${episode.shots.length} rendered` +
        (each ? ` · ${about(each)} of GPU time per shot here` : ""))),
    el("div", { class: "actions", style: "margin-top:0" },
      !rendered.length && missing.length ? el("button", { class: "primary", disabled: projectBusy() || !ready(missing[0]),
        onclick: () => run("shots", { episode: episode.number, params: { only: [missing[0].id] } }) }, `Render a sample: ${missing[0].id}`) : "",
      missing.length ? el("button", { class: rendered.length ? "primary" : "", disabled: projectBusy() || !allReady(missing),
        onclick: () => renderShots(episode, missing, `Render ${missing.length} shots`) },
      rendered.length ? `Render the other ${missing.length}` : `Render all ${missing.length}`) : "",
      stale.length ? el("button", { disabled: projectBusy() || !allReady(stale), onclick: () => renderShots(episode, stale, `Re-render ${stale.length} outdated shots`) },
        `Re-render ${stale.length} outdated`) : "",
      el("button", { disabled: projectBusy() || !allReady(missing), onclick: () => runEpisode(episode, missing) }, "Render, subtitle and assemble")),
    el("div", { class: "strip" }, episode.shots.map((shot) => tile(shot, state[shot.id] || {}, live === shot.id))),
    episode.number > 1 ? el("label", { class: "check" }, el("input", {
      type: "checkbox", checked: previous, onchange: async (e) => {
        await api(`/api/projects/${pid()}/episodes/${episode.number}`, { method: "PUT", body: { use_previous_frame: e.target.checked } }).catch(fail);
        await reloadProject();
      },
    }), `Use the last frame of episode ${episode.number - 1} as an extra reference for the first shot` +
      (state.use_previous_frame == null ? ` (default: ${state.continues ? "on, same place" : "off, the place changes"})` : "")) : "",
    episode.shots.map((shot) => shotCard(episode, shot, state[shot.id] || {}, live === shot.id, ready(shot))));
}

function shotCard(episode, shot, s, live, ready) {
  const p = ui.project;
  const plan = p.plan;
  const prompt = el("textarea", { rows: 12 });
  prompt.value = s.prompt_override || s.prompt || p.previews[shot.id] || "";
  const speech = s.speech;
  const stale = s.outdated || [];
  const keep = async (body) => {
    await api(`/api/projects/${pid()}/episodes/${episode.number}`, { method: "PUT", body }).catch(fail);
    await reloadProject();
  };
  const speaker = (plan.characters.find((c) => c.id === shot.speaker) || {}).name || shot.speaker;
  return el("article", { class: "panel shot", id: `shot-${shot.id}` },
    s.file ? el("video", { src: file(s.file), controls: true, preload: "metadata" })
      : el("div", { class: "placeholder" }, live ? [el("span", { class: "dot run" }), " Rendering…"] : "Not rendered yet"),
    el("div", {},
      el("div", { class: "title" }, el("h3", {}, `${shot.id} · ${shot.seconds} s · ${shot.location}`),
        el("span", { class: "badges" }, s.take ? el("span", { class: "badge" }, `take ${s.take}`) : "",
          stale.length ? el("span", { class: "badge warn", title: stale.map((r) => STALE[r]).join("; ") }, "outdated") : "")),
      stale.length ? el("div", { class: "error" }, `Outdated: ${stale.map((r) => STALE[r]).join("; ")}.`) : "",
      shot.line ? el("div", { class: "line" }, `${speaker}：${shot.line}`) : el("div", { class: "note" }, "no line"),
      speech ? el("div", { class: speech.match >= 0.6 ? "note" : "error" },
        `heard “${speech.heard}” · match ${Math.round(speech.match * 100)}% · ${speech.start}–${speech.end} s (${speech.timing})`) : "",
      el("div", { class: "note" }, shot.summary),
      s.references ? el("div", { class: "refs" }, s.references.filter((r) => r.includes("/")).map((r) => el("img", { src: file(r), alt: "" }))) : "",
      s.sheet ? el("img", { class: "sheet", src: file(s.sheet), alt: `Frames of ${shot.id}` }) : "",
      el("div", { class: "actions" },
        el("button", { class: "small", disabled: projectBusy() || !ready, title: ready ? "" : "Approve the pictures this shot uses first",
          onclick: () => run("shots", { episode: episode.number, params: { only: [shot.id] } }) }, s.file ? "Render another take" : "Render"),
        (s.takes || []).length > 1 ? s.takes.map((t) => el("button", {
          class: `small${t.take === s.take ? " primary" : ""}`, disabled: projectBusy(),
          onclick: () => keep({ use_take: { [shot.id]: t.take } }),
        }, `take ${t.take}`)) : ""),
      el("details", {}, el("summary", {}, s.prompt ? "Video prompt of this take" : "Video prompt (preview)"), prompt,
        el("button", { class: "small", style: "margin-top:6px", onclick: () => keep({ prompt_overrides: { [shot.id]: prompt.value } }) },
          "Keep this prompt for the next take"))));
}

// ---- 5. episodes ----

function episodesStep() {
  const plan = ui.project.plan;
  if (!plan.episodes) return locked("Episodes are assembled from rendered shots; plan the drama first.", "plan");
  return el("section", {}, plan.episodes.map((episode) => {
    const state = ui.project.episodes[String(episode.number)] || {};
    const done = episode.shots.filter((s) => state[s.id]?.file).length;
    const complete = done === episode.shots.length;
    const out = `episodes/${String(episode.number).padStart(2, "0")}/episode.mp4`;
    return el("article", { class: "panel episode" },
      state.assembled ? el("video", { src: file(out), controls: true, preload: "metadata" })
        : el("div", { class: "placeholder" }, complete ? "Ready to assemble" : `${done} of ${episode.shots.length} shots rendered`),
      el("div", {},
        el("h2", { style: "margin-top:0" }, `Episode ${episode.number}: ${episode.title}`),
        el("p", {}, episode.summary),
        el("p", { class: "note" }, `${done} of ${episode.shots.length} shots rendered` +
          (state.assembled ? ` · ${state.assembled.duration} s · ${state.assembled.width}×${state.assembled.height}` : "")),
        el("div", { class: "actions" },
          state.assembled ? el("a", { class: "button primary", href: file(out), download: `episode-${episode.number}.mp4` }, "Download") : "",
          el("button", { class: state.assembled || !complete ? "" : "primary", disabled: !complete || projectBusy(),
            onclick: () => run("episode", { episode: episode.number }) }, state.assembled ? "Subtitle and assemble again" : "Subtitle and assemble"),
          el("button", { class: "small", disabled: !complete || projectBusy(), onclick: () => run("subtitles", { episode: episode.number }) },
            "Time subtitles only"),
          complete ? "" : el("a", { class: "button small", href: `#/p/${pid()}/shots`, onclick: () => { ui.episode = episode.number; } },
            "Render its shots"))));
  }));
}

// ---- flow (advanced) ----

function flowStep() {
  const s = ui.state;
  const [name, version] = ui.project.meta.flow.split("@");
  const text = el("textarea", { class: "flow" });
  const pick = el("select", {}, Object.entries(s.flows).flatMap(([flow, versions]) =>
    versions.map((v) => el("option", { value: `${flow}@${v}`, selected: flow === name && String(v) === version }, `${flow}@${v}`))));
  const load = () => api(`/api/flows/${pick.value}`).then((r) => { text.value = JSON.stringify(r.flow, null, 2); }).catch(fail);
  pick.addEventListener("change", load);
  load();
  const saveName = el("input", { value: name });
  const note = el("input", { placeholder: "what changed" });
  return el("section", {},
    el("p", { class: "note" }, "A flow lists the steps, the model of each and its parameters, as numbered versions. " +
      "Most projects never need to change it."),
    el("div", { class: "actions" }, pick,
      el("button", { onclick: async () => {
        await api(`/api/projects/${pid()}/flow`, { method: "PUT", body: { flow: pick.value } }).catch(fail);
        await reloadProject();
      } }, "Use this version for the project")),
    text,
    el("div", { class: "row", style: "align-items:flex-end" }, el("label", {}, "Flow name", saveName), el("label", {}, "Note", note),
      el("button", { class: "primary", style: "flex:0 0 auto;margin:8px 0", onclick: async () => {
        try {
          const saved = await api(`/api/flows/${saveName.value}`, { method: "POST", body: { flow: JSON.parse(text.value), note: note.value } });
          toast(`Saved ${saved.reference}`);
          await refresh();
          renderProject();
        } catch (error) { fail(error); }
      } }, "Save as a new version")),
    el("h2", {}, "Installed steps"),
    el("table", { class: "edit" }, s.steps.map((step) => el("tr", {},
      el("td", {}, el("b", {}, step.name)), el("td", {}, step.model_from ? `action · ${step.model_from} model` : step.scope),
      el("td", {}, step.role || "–"), el("td", {}, step.description), el("td", { class: "faint" }, step.source)))),
    Object.entries(s.plugin_errors).map(([plugin, error]) => el("p", { class: "error" }, `${plugin}: ${error}`)),
    el("div", { class: "actions" },
      el("button", { onclick: () => api("/api/plugins/reload", { method: "POST" }).then(refresh).then(reloadProject).catch(fail) }, "Reload plugins")),
    el("p", { class: "note" }, "A plugin is one Python file in the plugins directory that defines STEP. Add its name to a flow version to run it; " +
      "delete the file and reload to remove it. Actions (story, revise, review) run with the model of the flow's plan step."));
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// One project: five steps, the one thing to do next (or what is running, or what failed), and
// the editors of each step. Uses the helpers of app.js.

const STEPS = [["story", "Story"], ["plan", "Plan"], ["pictures", "Reference pictures"], ["shots", "Shots"], ["episodes", "Episodes"]];
const STEP_JOBS = { story: ["story"], plan: ["plan", "revise", "review"], pictures: ["stills"], shots: ["shots", "analyze"],
  episodes: ["subtitles", "assemble"] };
const KINDS = { character: "Character", location: "Place", prop: "Prop" };
const STALE = { references: "a picture it used was redrawn or replaced",
  previous: "the shot before it has a new take, so it no longer opens where that one ends",
  prompt: "its prompt changed since this take (the plan, a kept prompt or the app's template)" };
const CATEGORY = { sound: "Sound", speech: "Speech", picture: "Picture", content: "Content", text: "On-screen text",
  continuity: "Continuity", consistency: "Consistency", reference: "Reference picture" };
const ACTION = { revise: "Revise the plan", retake: "Render another take", redraw: "Redraw the picture", none: "Ignore" };

const pid = () => encodeURIComponent(ui.project.id);
const file = (relative) => `/api/projects/${pid()}/files/${relative}?v=${encodeURIComponent(relative)}`;
const projectJobs = () => ui.state.jobs.filter((j) => j.project === ui.project.id);
const projectBusy = () => projectJobs().some((j) => j.status === "running" || j.status === "queued");
const go = (step) => { if (ui.step !== step) location.hash = `#/p/${pid()}/${step}`; };
const capital = (text) => text.charAt(0).toUpperCase() + text.slice(1);
const modelOf = (role) => (ui.state.models.find((m) => m.role === role) || {}).name || t("the {role} model", { role: t(role) });
const stepLabel = (key) => t(STEPS.find(([k]) => k === key)[1]);

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
    toast(t("Started: {what}", { what: step === "episode" ? t("rendering, subtitles and assembly") : (STEP_NAMES[step] ? t(STEP_NAMES[step]) : step) }));
    await refresh();
  } catch (error) { fail(error); }
}

function confirmRun(what, count, each, start) {
  const time = each ? t("{time} of GPU time", { time: about(count * each) }) : t("the time is measured on the first one");
  if (count > 1 && !confirm(t("{what}: {time}. Continue?", { what, time }))) return;
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
      detail: p.novel_chars ? t("{n} characters", { n: p.novel_chars.toLocaleString() }) : t("from your idea") },
    { key: "plan", status: !p.novel_chars ? "locked" : level(stage.plan), lock: t("Write the story first"),
      detail: planned ? `${plural(p.plan.episodes.length, "episode")} · ${plural(Object.keys(p.previews).length, "shot")}` : t("not planned yet") },
    { key: "pictures", status: !planned ? "locked" : level(stage.stills), lock: t("Plan the drama first"), detail: t(stage.stills.detail) },
    { key: "shots", status: !planned || (approved < needed.length && !rendered) ? "locked" : level(stage.shots),
      lock: t("Approve the pictures first"), detail: t(stage.shots.detail) },
    { key: "episodes", status: !rendered ? "locked" : level(stage.episodes), lock: t("Render the shots first"),
      detail: t(stage.episodes.detail) },
  ];
  const active = projectJobs().filter((j) => j.status === "running" || j.status === "queued");
  for (const s of steps) s.running = active.some((j) => STEP_JOBS[s.key].includes(j.step));
  const current = steps.find((s) => s.status !== "done" && s.status !== "locked");
  if (current && current.status === "todo") current.status = "current";
  return steps;
}

function stepper(steps, step) {
  return el("nav", { class: "stepper", "aria-label": t("Steps") }, steps.map((s, index) => el("a", {
    href: `#/p/${pid()}/${s.key}`, class: s.running ? "running" : s.status,
    "aria-current": s.key === step ? "step" : "false", title: s.status === "locked" ? s.lock : s.detail,
  }, el("span", { class: "circle" }, s.status === "done" && !s.running ? "✓" : index + 1),
  el("span", { class: "label" }, t(STEPS[index][1])),
  el("small", {}, s.running ? t("running…") : s.status === "locked" ? s.lock : s.detail))));
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
    el("div", { class: "crumbs" }, el("a", { href: "#/" }, t("Projects")), ` › ${p.meta.title || p.id}`),
    el("div", { class: "page-head" },
      el("div", { class: "grow" }, el("h1", {}, p.meta.title || p.id),
        el("p", {}, t("{source} · {episodes} of up to {seconds} s · written in {language} · {style}", {
          source: p.meta.idea ? t("From an idea") : t("From a novel"), episodes: plural(p.meta.episodes, "episode"),
          seconds: p.meta.max_seconds, language: LANGUAGES[p.meta.language || "zh"],
          style: style ? t("style: {style}", { style }) : t("style chosen when planning") }))),
      el("button", { class: "small", onclick: renameProject }, t("Rename"))),
    stepper(steps, step),
    el("div", { class: "subnav" }, el("a", { href: `#/p/${pid()}/flow`, "aria-current": step === "flow" ? "page" : "false" },
      t("Flow (advanced)"))),
    el("div", { id: "next" }, nextCard()),
    el("div", { id: "step" }, body));
  ui.dirty = false;
}

async function renameProject() {
  const title = prompt(t("A new title for this project"), ui.project.meta.title || "");
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
    phase = other ? t("Waiting for the GPU: “{title}” is busy with {step} first.", { title: projectTitle(other.project), step: stepName(other) })
      : t("Starting…");
  } else if (engine.loading.length) {
    title = stepName(job);
    phase = t("Loading {models} onto the GPU. A step first loads the models it needs; this takes a minute or more.",
      { models: engine.loading.join(" · ") });
    start = engineSince();
  } else {
    title = stepName(job);
    phase = job.message ? capital(t(job.message)) : t("Starting…");
    start = since(job.phase_seconds ?? job.seconds);
    estimate = phaseEstimate(job);
  }
  return el("section", { class: "next running" },
    el("div", {}, el("div", { class: "kicker" }, t("Running now")), el("h2", {}, title),
      el("p", {}, phase, start != null ? [" · ", timer(start), estimate ? ` (${t("usually {time}", { time: about(estimate) })})` : ""] : "")),
    el("div", { class: "go" }, el("button", { onclick: () => openDrawer(true) }, t("Activity and log")),
      el("button", { class: "danger", onclick: cancelAll }, t("Cancel"))),
    el("div", { class: "detail" }, start != null ? meter(start, estimate) : meter(0, null),
      queued.length ? el("ul", { class: "queue" }, queued.map((q) => el("li", {}, stepName(q)))) : ""));
}

function failedCard(job) {
  const retry = () => run(job.step, { ...(job.episode ? { episode: job.episode } : {}), params: job.params || {} });
  return el("section", { class: "next failed" },
    el("div", {}, el("div", { class: "kicker" }, t("Failed")), el("h2", {}, t("{step} failed", { step: stepName(job) })),
      el("p", {}, (job.error || t("no reason was given")).replace(/^\w+Error: /, ""))),
    el("div", { class: "go" },
      el("button", { class: "primary big", onclick: retry }, t("Try again")),
      el("button", { class: "big", onclick: () => openDrawer(true) }, t("Model server log")),
      el("button", { class: "quiet", onclick: () => { ui.dismissed.add(job.id); renderNextCard(); renderStatus(); } }, t("Dismiss"))));
}

function actionCard(action) {
  const index = STEPS.findIndex(([key]) => key === action.step);
  return el("section", { class: "next" },
    el("div", {}, el("div", { class: "kicker" }, action.done ? t("All done") : t("Next · step {n} of 5", { n: index + 1 })),
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
    return { step: p.novel_chars ? "plan" : "story", title: t("Choose a language model"),
      text: t("It writes the story and plans the drama. The NVIDIA API catalog and OpenRouter have free tiers: pick one in Settings and paste a key."),
      buttons: [el("a", { class: "button primary big", href: "#/settings" }, t("Open Settings"))] };
  }
  if (!p.novel_chars) {
    return { step: "story", title: t("Write the story from your idea"),
      text: t("The language model turns the idea into a short story, one chapter per episode. It takes about a minute, and you can edit the story afterwards."),
      buttons: [act("story", t("Write the story"), () => run("story"))] };
  }
  if (!plan.episodes) {
    return { step: "plan", title: t("Plan the drama"),
      text: t("The language model reads the story ({n} characters) and writes the cast, places, props and a shot list for each episode. It takes a few minutes; you can edit every part afterwards.",
        { n: p.novel_chars.toLocaleString() }),
      buttons: [act("plan", t("Plan the drama"), () => run("plan"))] };
  }
  const errors = p.problems.filter((x) => x.level === "error");
  if (errors.length) {
    return { step: "plan", title: t("Fix {problems} in the plan", { problems: plural(errors.length, "problem") }),
      text: `${errors[0].where}: ${errors[0].message}${errors[0].fix ? `. ${errors[0].fix}` : ""}. ${t("The checks list them all on the Plan step.")}`,
      buttons: [look("plan", t("Show the problems"))] };
  }
  const used = stillTargets(plan).filter((x) => p.usage[x.key]);
  const missing = used.filter((x) => !p.stills[x.key]?.file);
  const outdated = used.filter((x) => p.outdated_stills.includes(x.key));
  const unapproved = used.filter((x) => p.stills[x.key]?.file && p.stills[x.key].status !== "approved");
  if (missing.length) {
    return { step: "pictures", title: t("Draw {pictures}", { pictures: plural(missing.length, "reference picture") }),
      text: t("The image model ({model}) draws every character, place and prop once in the plan's style; each shot is filmed from these pictures.", { model: modelOf("image") }) +
        " " + (p.estimates.still ? `${capital(about(missing.length * p.estimates.still))}.` : t("The first one also loads the image model onto the GPU.")),
      buttons: [act("pictures", t("Draw {pictures}", { pictures: plural(missing.length, "picture") }), () => run("stills"))] };
  }
  if (outdated.length) {
    return { step: "pictures", title: t("Redraw {pictures}", { pictures: plural(outdated.length, "outdated picture") }),
      text: t("Their description or the style changed after they were drawn."),
      buttons: [act("pictures", t("Redraw {n}", { n: outdated.length }), () => run("stills", { params: { only: outdated.map((x) => x.key) } })),
        look("pictures", t("Look at them"))] };
  }
  if (unapproved.length) {
    return { step: "pictures", title: t("Check and approve {pictures}", { pictures: plural(unapproved.length, "picture") }),
      text: t("Shots use approved pictures only. Look at each one: approve it, or redraw it with a new seed or your own prompt."),
      buttons: [act("pictures", t("Approve all {n}", { n: unapproved.length }), () => approve(unapproved.map((x) => x.key))),
        ui.step === "pictures" ? "" : look("pictures", t("Look at them first"))] };
  }
  const anyRendered = Object.values(p.episodes).some((e) => Object.values(e).some((v) => v && v.file));
  for (const episode of plan.episodes) {
    const state = p.episodes[String(episode.number)] || {};
    const todo = episode.shots.filter((s) => !state[s.id]?.file);
    const stale = episode.shots.filter((s) => state[s.id]?.file && (state[s.id].outdated || []).length);
    const time = (n) => (p.estimates.shot ? t("{time} of GPU time.", { time: capital(about(n * p.estimates.shot)) })
      : t("The first shot also loads the video model; then each shot takes several minutes, and this card shows the time as it runs."));
    if (!anyRendered && todo.length) {
      const first = todo[0];
      return { step: "shots", title: t("Render a sample shot"),
        text: t("Render {shot} ({seconds} s{line}) first, to check the look and the voices before the other {others}.",
          { shot: first.id, seconds: first.seconds, line: first.line ? `, “${first.line}”` : "", others: plural(todo.length - 1, "shot") }) + ` ${time(1)}`,
        buttons: [act("shots", t("Render the sample"), () => { ui.episode = episode.number; run("shots", { episode: episode.number, params: { only: [first.id] } }); })] };
    }
    if (stale.length) {
      return { step: "shots", title: t("Re-render {shots} of episode {n}", { shots: plural(stale.length, "outdated shot"), n: episode.number }),
        text: `${t("Something they were rendered from changed.")} ${time(stale.length)}`,
        buttons: [act("shots", t("Re-render {n}", { n: stale.length }), () => renderShots(episode, stale, t("Re-render {n} outdated shots", { n: stale.length })))] };
    }
    if (todo.length) {
      return { step: "shots", title: t("Render {shots} of episode {n}", { shots: plural(todo.length, "shot"), n: episode.number }),
        text: `${time(todo.length)} ${t("Or render them, time the subtitles and assemble the episode in one go.")}`,
        buttons: [act("shots", t("Render {shots}", { shots: plural(todo.length, "shot") }), () => renderShots(episode, todo, t("Render {n} shots", { n: todo.length }))),
          act("episodes", t("Render, subtitle and assemble"), () => runEpisode(episode, todo), false)] };
    }
    if (!state.assembled) {
      return { step: "episodes", title: t("Subtitle and assemble episode {n}", { n: episode.number }),
        text: t("The speech model ({model}) times each line from the voice in its shot, then the clips are joined into one 9:16 video.", { model: modelOf("asr") }),
        buttons: [act("episodes", t("Subtitle and assemble"), () => run("episode", { episode: episode.number }))] };
    }
  }
  return { step: "episodes", done: true, title: t("The drama is ready"),
    text: t("{episodes} assembled. Watch and download them, analyze the video and send the findings back into the plan, or re-render any shot and assemble again.",
      { episodes: plural(plan.episodes.length, "episode") }),
    buttons: [ui.step === "episodes" ? "" : look("episodes", t("Watch and download"))] };
}

function renderShots(episode, shots, what) {
  ui.episode = episode.number;
  confirmRun(what, shots.length, ui.project.estimates.shot,
    () => run("shots", { episode: episode.number, params: { only: shots.map((s) => s.id) } }));
}

function runEpisode(episode, todo) {
  confirmRun(t("Render {n} shots, then subtitle and assemble", { n: todo.length }), todo.length, ui.project.estimates.shot,
    () => run("episode", { episode: episode.number }));
}

async function approve(keys, status = "approved") {
  try {
    for (const key of keys) {
      await api(`/api/projects/${pid()}/stills/${encodeURIComponent(key)}`, { method: "PUT", body: { status } });
    }
    if (keys.length > 1) toast(t("{pictures} approved", { pictures: plural(keys.length, "picture") }));
    await reloadProject();
  } catch (error) { fail(error); }
}

function locked(text, step) {
  return el("section", { class: "panel rail" }, el("p", {}, text),
    el("a", { class: "button", href: `#/p/${pid()}/${step}` }, t("Go to {step}", { step: stepLabel(step) })));
}

// ---- editors ----

function input(value, onchange, { area = false, rows = 2, type = "text", label = "" } = {}) {
  const node = area ? el("textarea", { rows, "aria-label": label || null }) : el("input", { type });
  node.value = value ?? "";
  node.addEventListener("input", () => { markDirty(); onchange(node.value); });
  return node;
}

// An editable table; `kind` adds a button that asks the model to revise one row.
function table(rows, columns, kind = null) {
  const remove = (row) => { rows.splice(rows.indexOf(row), 1); renderStep(); markDirty(); };
  const ask = (row) => revise(`${kind}:${row.id}`, prompt(t("What should change about {name}?", { name: row.name || row.id })));
  return el("div", {}, el("table", { class: "edit" },
    el("tr", {}, columns.map(([key, , label]) => el("th", {}, t(label || key))), el("th", {})),
    rows.map((row) => el("tr", {}, columns.map(([key, width]) => el("td", { style: `width:${width}` },
      input(row[key], (v) => { row[key] = v; }, { area: width === "auto", rows: 2, label: `${row.name || row.id || ""} · ${t(key)}` }))),
    el("td", { class: "row-actions" },
      kind ? el("button", { class: "small", title: t("Revise with the model"), disabled: projectBusy(), onclick: () => ask(row) }, t("Revise")) : "",
      el("button", { class: "small danger", title: t("Remove"), onclick: () => remove(row) }, "✕"))))),
  el("button", { class: "small", style: "margin-top:6px", onclick: () => {
    rows.push(Object.fromEntries(columns.map(([key]) => [key, ""])));
    renderStep();
    markDirty();
  } }, t("Add")));
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
    el("span", { class: "note" }, t("You have unsaved edits.")),
    el("button", { onclick: () => { ui.dirty = false; reloadProject(); } }, t("Discard")),
    el("button", { class: "primary", onclick: savePlan }, t("Save edits")));
}

async function savePlan() {
  try {
    await api(`/api/projects/${pid()}/plan`, { method: "PUT", body: ui.project.plan });
    ui.dirty = false;
    toast(t("Plan saved"));
    await reloadProject();
  } catch (error) { fail(error); }
}

async function revise(target, instruction) {
  if (ui.dirty) return toast(t("Save your edits first: the model revises the saved plan."), "bad");
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
  const idea = p.meta.idea ? el("div", { class: "idea" }, el("div", { class: "faint" }, t("Your idea")), p.meta.idea) : "";
  if (!p.novel_chars) return el("section", {}, idea, el("p", { class: "note" }, t("The story appears here once it is written.")));
  const text = el("textarea", { class: "story-text", "aria-label": t("The story") });
  const cached = ui.novel?.id === p.id && ui.novel.chars === p.novel_chars;
  text.value = cached ? ui.novel.text : t("Loading…");
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
      toast(t("Story saved"));
      await reloadProject();
    } catch (error) { fail(error); }
  };
  return el("section", {}, idea,
    el("div", { class: "section-title" }, el("h2", {}, t("The story")), el("span", { class: "faint" }, t("{n} characters", { n: p.novel_chars.toLocaleString() }))),
    text,
    el("div", { class: "actions" }, el("button", { class: "primary", onclick: save }, t("Save the story")),
      el("span", { class: "note" }, p.plan.episodes ? t("To use an edited story, plan again on the Plan step; the current plan is kept under Earlier versions.")
        : t("Edit freely; the plan is written from the saved story."))));
}

// ---- 2. plan ----

function renderChecks() {
  const { problems } = ui.project;
  const review = ui.project.model_review?.items ? ui.project.model_review : null;
  const errors = problems.filter((x) => x.level === "error").length;
  return el("section", { class: `panel checks${problems.length ? "" : " clean"}` },
    el("div", { class: "title" },
      el("h3", {}, problems.length ? t("{things} to check before rendering", { things: plural(problems.length, "thing") }) +
        (errors ? ` (${t("{n} must be fixed", { n: errors })})` : "") : t("The plan checks found nothing to fix")),
      el("button", { class: "small", disabled: projectBusy(), onclick: () => run("review") },
        review ? t("Review again with the model") : t("Ask the model to review"))),
    problems.length ? el("ul", { class: "problems" }, problems.map((x) => el("li", {},
      el("span", { class: `badge ${x.level === "error" ? "bad" : "warn"}` }, t(x.level)), " ",
      el("a", { onclick: () => jump(x.where) }, x.where), ` ${x.message}`,
      x.fix ? el("span", { class: "note" }, ` · ${x.fix}`) : ""))) : "",
    review ? el("div", {},
      el("h3", { style: "margin-top:12px" }, t("Model review: {notes}", { notes: review.items.length ? plural(review.items.length, "note") : t("no problems") }) +
        (review.current ? "" : ` (${t("written for an earlier version of the shots")})`)),
      el("ul", { class: "problems" }, review.items.map((x) => el("li", {},
        el("a", { onclick: () => jump(x.shot) }, x.shot), ` ${x.problem}`,
        x.fix ? el("span", { class: "note" }, ` · ${t("Fix")}: ${x.fix} `) : " ",
        el("button", { class: "small", onclick: () => { ui.presets[x.shot] = x.fix || x.problem; renderStep(); jump(x.shot); } },
          t("Revise this shot")))))) : "");
}

function renderHistory() {
  const p = ui.project;
  if (!p.history.length) return "";
  return el("details", { style: "margin:12px 0" }, el("summary", {}, t("Earlier versions of the plan ({n})", { n: p.history.length })),
    p.history.map((h) => el("div", { class: "actions", style: "margin:4px 0" },
      el("span", { class: "note" }, `${new Date(h.time * 1000).toLocaleString(LANG === "zh" ? "zh-CN" : undefined)} · ${h.note}`),
      el("button", { class: "small", onclick: async () => {
        if (!confirm(t("Restore this version? The current plan is kept as a version too."))) return;
        await api(`/api/projects/${pid()}/history/${h.name}/restore`, { method: "POST" }).catch(fail);
        await reloadProject();
      } }, t("Restore")))));
}

function reviseBox(target, preset = "") {
  const note = el("textarea", { rows: 2, "aria-label": t("Note for the model"),
    placeholder: t("What should change? For example: let him touch his head before he speaks, and keep the line shorter") });
  note.value = preset;
  return el("div", { class: "revise" }, note,
    el("button", { disabled: projectBusy(), onclick: () => revise(target, note.value) }, t("Revise with the model")));
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

const SHOT_FIELDS = { camera: "Camera", action: "Action", after: "After", then: "Then", sound: "Sound" };

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
        origin ? el("span", { class: `badge ${origin === "novel" ? "ok" : ""}` }, origin === "novel" ? t("line from the novel") : t("adapted line")) : "",
        issues ? el("span", { class: "badge warn" }, t("{n} to check", { n: issues })) : ""),
      el("span", { class: "spacer", style: "flex:1" }),
      el("button", { class: "small danger", onclick: () => { episode.shots.splice(episode.shots.indexOf(shot), 1); renderStep(); markDirty(); } },
        t("Remove shot"))),
    el("div", { class: "row" },
      el("label", {}, t("Seconds"), input(shot.seconds, (v) => { shot.seconds = Number(v) || 6; }, { type: "number" })),
      el("label", {}, t("Place"), input(shot.location, (v) => { shot.location = v; })),
      el("label", {}, t("Characters"), list("characters")),
      el("label", {}, t("Props"), list("props")),
      el("label", {}, t("Speaker"), input(shot.speaker, (v) => { shot.speaker = v; }))),
    el("label", {}, t("Summary"), input(shot.summary, (v) => { shot.summary = v; })),
    el("label", {}, t("Line"), input(shot.line, (v) => { shot.line = v; })),
    el("div", { class: "row" }, Object.entries(SHOT_FIELDS).map(([key, name]) =>
      el("label", {}, t(name), input(shot[key], (v) => { shot[key] = v; }, { area: true, rows: 3, label: `${shot.id} · ${t(name)}` })))),
    reviseBox(`shot:${shot.id}`, ui.presets[shot.id] || ""));
}

function planStep() {
  const p = ui.project;
  const plan = p.plan;
  if (!p.novel_chars) return locked(t("The plan is written from the story; write the story first."), "story");
  const again = el("button", { disabled: projectBusy() || !ui.state.chat_ready, onclick: () => {
    if (!plan.episodes || confirm(t("Plan again from the story? The current plan is kept under Earlier versions."))) run("plan");
  } }, plan.episodes ? t("Plan again from the story") : t("Plan the drama"));
  const extra = p.flow_steps.filter((s) => !s.core).map((s) => (s.installed && s.scope === "project"
    ? el("button", { disabled: projectBusy() || !plan.episodes, onclick: () => run(s.step) }, t("Run {step}", { step: s.step }))
    : el("span", { class: "error" }, t("{step}: not installed, skipped", { step: s.step }))));
  if (!plan.episodes) return el("section", {}, el("div", { class: "actions" }, again, extra), renderHistory());
  return el("section", {},
    el("div", { class: "actions" }, again, extra,
      el("span", { class: "note" }, t("Edit anything below, in any language, then save; the checks run again on every save."))),
    renderChecks(), renderHistory(),
    el("h2", {}, t("Title and setting")),
    el("div", { class: "row" },
      el("label", {}, t("Title"), input(plan.title, (v) => { plan.title = v; })),
      el("label", { style: "flex-grow:3" }, t("Setting"), input(plan.setting, (v) => { plan.setting = v; }))),
    el("p", { class: "faint" }, t("The visual style is set on the Reference pictures step.")),
    el("h2", {}, t("Characters")),
    el("p", { class: "note" }, t("Each character's look and voice go into the picture and video prompts.")),
    table(plan.characters, [["id", "90px"], ["name", "100px"], ["role", "auto"], ["look", "auto"], ["voice", "auto"]], "character"),
    el("h2", {}, t("Relationships")),
    table(plan.relationships || [], [["from", "120px"], ["to", "120px"], ["relation", "auto"]]),
    el("h2", {}, t("Places")), table(plan.locations, [["id", "90px"], ["name", "120px"], ["look", "auto"]], "location"),
    el("h2", {}, t("Props")), table(plan.props, [["id", "90px"], ["name", "120px"], ["look", "auto"]], "prop"),
    plan.episodes.map((episode) => el("div", {},
      el("div", { class: "section-title", id: `plan-episode-${episode.number}` }, el("h2", {}, t("Episode {n}", { n: episode.number })),
        el("span", { class: "faint" }, `${plural(episode.shots.length, "shot")} · ${episode.shots.reduce((sum, s) => sum + Number(s.seconds || 0), 0)} s`)),
      el("div", { class: "row" }, el("label", {}, t("Title"), input(episode.title, (v) => { episode.title = v; })),
        el("label", { style: "flex-grow:3" }, t("Summary"), input(episode.summary, (v) => { episode.summary = v; }, { area: true, label: t("Summary") }))),
      episode.shots.map((shot) => renderShotEditor(episode, shot)),
      el("button", { style: "margin-top:10px", onclick: () => newShot(episode) }, t("Add shot")))),
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
  const text = el("input", { value: current, "aria-label": t("Visual style") });
  const save = async (style) => {
    if (!style || style === current) return;
    try {
      await api(`/api/projects/${pid()}/plan`, { method: "PUT", body: { ...plan, style } });
      toast(t("Style saved. Pictures drawn in the old style are marked outdated."));
      await reloadProject();
    } catch (error) { fail(error); }
  };
  return el("section", { class: "panel rail style-box" },
    el("h3", {}, t("Visual style")),
    el("p", { class: "note", style: "margin:0" }, t("Every picture and every shot prompt starts from this style. Pick one, or describe your own in any language; the presets are English, which the models follow best.")),
    el("div", { class: "chips" }, STYLES.filter(([style]) => style).map(([style, name]) =>
      el("button", { class: `chip${style === current ? " on" : ""}`, disabled: projectBusy(), onclick: () => save(style) }, t(name)))),
    el("div", { class: "row", style: "align-items:flex-end" },
      el("label", { style: "flex:4 1 320px;margin:0" }, t("Style"), text),
      el("button", { style: "flex:0 0 auto", disabled: projectBusy(), onclick: () => save(text.value.trim()) }, t("Save style"))));
}

function pictureCard({ key, kind, item }) {
  const p = ui.project;
  const still = p.stills[key] || {};
  const job = running();
  const drawing = job && job.project === p.id && job.step === "stills" && (job.message || "").endsWith(`: ${item.name || item.id}`);
  const outdated = p.outdated_stills.includes(key);
  const status = !still.file ? "" : outdated ? "outdated" : still.status === "approved" ? "approved" : "";
  const promptBox = el("textarea", { rows: 6, "aria-label": t("Prompt of {name}", { name: item.name || item.id }) });
  promptBox.value = still.prompt_override || still.prompt || "";
  const set = async (body) => {
    try {
      await api(`/api/projects/${pid()}/stills/${encodeURIComponent(key)}`, { method: "PUT", body });
      await reloadProject();
    } catch (error) { fail(error); }
  };
  const uses = p.usage[key] || [];
  return el("article", { class: `picture ${status}` },
    el("div", { class: "frame" }, still.file ? el("img", { src: file(still.file), alt: item.name || item.id, loading: "lazy" })
      : drawing ? el("span", {}, el("span", { class: "dot run" }), ` ${t("Drawing…")}`) : t("Not drawn yet")),
    el("div", { class: "body" },
      el("div", { class: "title" }, el("h3", { style: "margin:0" }, item.name || item.id),
        el("span", { class: `badge ${status === "approved" ? "ok" : status === "outdated" ? "warn" : ""}` },
          !still.file ? t(KINDS[kind]) : outdated ? t("outdated") : still.status === "approved" ? t("approved") : t("to check"))),
      el("div", { class: "faint", title: uses.join(" · ") }, t("{kind} · in {shots}", { kind: t(KINDS[kind]), shots: uses.slice(0, 6).join(", ") }) +
        (uses.length > 6 ? ` ${t("and {n} more", { n: uses.length - 6 })}` : "")),
      outdated ? el("div", { class: "error", style: "font-size:13px" }, t("Drawn before its description or the style changed; redraw it.")) : "",
      el("div", { class: "actions" },
        still.file ? el("button", { class: `small${still.status === "approved" ? "" : " primary"}`,
          onclick: () => set({ status: still.status === "approved" ? "draft" : "approved" }) },
        still.status === "approved" ? t("Unapprove") : t("Approve")) : "",
        el("button", { class: "small", disabled: projectBusy(), onclick: () => run("stills", { params: { only: [key] } }) },
          still.file ? t("Redraw") : t("Draw"))),
      el("details", {}, el("summary", {}, t("Prompt")), promptBox,
        el("button", { class: "small", style: "margin-top:6px", onclick: () => set({ prompt_override: promptBox.value }) },
          t("Keep this prompt for the next drawing")))));
}

function picturesStep() {
  const p = ui.project;
  const plan = p.plan;
  if (!plan.episodes) return locked(t("Pictures are drawn from the plan's cast, places and props; plan the drama first."), "plan");
  const targets = stillTargets(plan);
  const used = targets.filter((x) => p.usage[x.key]);
  const unused = targets.filter((x) => !p.usage[x.key]);
  const missing = used.filter((x) => !p.stills[x.key]?.file);
  const waiting = used.filter((x) => p.stills[x.key]?.file && p.stills[x.key].status !== "approved");
  const approved = used.length - missing.length - waiting.length;
  return el("section", {}, styleBox(),
    el("div", { class: "section-title" }, el("h2", {}, t("Pictures")),
      el("span", { class: "faint" }, t("{done} of {all} approved", { done: approved, all: used.length }))),
    el("div", { class: "actions", style: "margin-top:0" },
      missing.length ? el("button", { class: "primary", disabled: projectBusy(),
        onclick: () => confirmRun(t("Draw {n} pictures", { n: missing.length }), missing.length, p.estimates.still, () => run("stills")) },
      t("Draw {pictures}", { pictures: plural(missing.length, "missing picture") })) : "",
      waiting.length ? el("button", { onclick: () => approve(waiting.map((x) => x.key)) }, t("Approve all {n}", { n: waiting.length })) : "",
      el("span", { class: "note" }, t("Only what appears in a shot gets a picture. A shot renders once every picture it uses is approved."))),
    el("div", { class: "pictures" }, used.map(pictureCard)),
    unused.length ? el("p", { class: "note", style: "margin-top:14px" },
      t("Not in any shot, so not drawn: {names}.", { names: unused.map((x) => x.item.name || x.item.id).join("、") })) : "");
}

// ---- 4. shots ----

// A rendered shot shows the first frame of its contact sheet; a missing one shows its place.
function tile(shot, s, live, count) {
  const place = ui.project.stills[`location:${shot.location}`];
  const status = live ? "rendering" : !s.file ? "missing" : (s.outdated || []).length ? "outdated" : "done";
  const picture = s.file ? s.sheet : place?.file;
  return el("a", { class: `tile ${status}`, href: `#shot-${shot.id}`, title: shot.summary,
    onclick: (e) => { e.preventDefault(); document.getElementById(`shot-${shot.id}`)?.scrollIntoView({ behavior: "smooth" }); } },
  picture ? el("img", { src: file(picture), alt: "" }) : el("div", { class: "blank" }),
  el("span", { class: "tag" }, `${shot.id} · ${shot.seconds}s`),
  shot.line ? el("span", { class: "say" }, shot.line) : "",
  count ? el("span", { class: "findings" }, `⚑ ${count}`) : "",
  live ? el("span", { class: "live" }, t("rendering")) : "");
}

function liveShot() {
  const job = running();
  if (!job || job.project !== ui.project.id || job.step !== "shots") return null;
  return ((job.message || "").match(/\(([a-z0-9_]+)\)$/) || [])[1] || null;
}

function shotsStep() {
  const p = ui.project;
  const plan = p.plan;
  if (!plan.episodes) return locked(t("Shots come from the plan; plan the drama first."), "plan");
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
  const counts = {};
  for (const f of state.analysis?.findings || []) counts[f.shot] = (counts[f.shot] || 0) + 1;
  return el("section", {},
    unapproved ? el("p", { class: "panel rail" }, t("{pictures} still to approve: a shot renders once every picture it uses is approved.", { pictures: plural(unapproved, "picture") }), " ",
      el("a", { href: `#/p/${pid()}/pictures` }, t("Go to {step}", { step: stepLabel("pictures") }))) : "",
    plan.episodes.length > 1 ? el("div", { class: "episode-tabs", role: "tablist" }, plan.episodes.map((e) => el("button", {
      class: `small${e.number === episode.number ? " primary" : ""}`, role: "tab", "aria-selected": String(e.number === episode.number),
      onclick: () => { ui.episode = e.number; renderStep(); } }, t("Episode {n}", { n: e.number })))) : "",
    el("div", { class: "section-title" }, el("h2", {}, t("Episode {n}: {title}", { n: episode.number, title: episode.title })),
      el("span", { class: "faint" }, t("{done} of {all} rendered", { done: rendered.length, all: episode.shots.length }) +
        (each ? ` · ${t("{time} of GPU time per shot here", { time: about(each) })}` : ""))),
    el("div", { class: "actions", style: "margin-top:0" },
      !rendered.length && missing.length ? el("button", { class: "primary", disabled: projectBusy() || !ready(missing[0]),
        onclick: () => run("shots", { episode: episode.number, params: { only: [missing[0].id] } }) }, t("Render a sample: {shot}", { shot: missing[0].id })) : "",
      missing.length ? el("button", { class: rendered.length ? "primary" : "", disabled: projectBusy() || !allReady(missing),
        onclick: () => renderShots(episode, missing, t("Render {n} shots", { n: missing.length })) },
      rendered.length ? t("Render the other {n}", { n: missing.length }) : t("Render all {n}", { n: missing.length })) : "",
      stale.length ? el("button", { disabled: projectBusy() || !allReady(stale), onclick: () => renderShots(episode, stale, t("Re-render {n} outdated shots", { n: stale.length })) },
        t("Re-render {n} outdated", { n: stale.length })) : "",
      el("button", { disabled: projectBusy() || !allReady(missing), onclick: () => runEpisode(episode, missing) }, t("Render, subtitle and assemble"))),
    el("div", { class: "strip" }, episode.shots.map((shot) => tile(shot, state[shot.id] || {}, live === shot.id, counts[shot.id]))),
    episode.number > 1 ? el("label", { class: "check" }, el("input", {
      type: "checkbox", checked: previous, onchange: async (e) => {
        await api(`/api/projects/${pid()}/episodes/${episode.number}`, { method: "PUT", body: { use_previous_frame: e.target.checked } }).catch(fail);
        await reloadProject();
      },
    }), t("Use the last frame of episode {n} as an extra reference for the first shot", { n: episode.number - 1 }) +
      (state.use_previous_frame == null ? ` (${state.continues ? t("default: on, same place") : t("default: off, the place changes")})` : "")) : "",
    el("label", { class: "check" }, el("input", {
      type: "checkbox", checked: state.chained !== false, onchange: async (e) => {
        await api(`/api/projects/${pid()}/episodes/${episode.number}`, { method: "PUT", body: { chain_shots: e.target.checked } }).catch(fail);
        await reloadProject();
      },
    }), t("Open each shot where the shot before it ends (same place): its last frame becomes an extra reference. Render in order; a new take of a shot marks the next one outdated.")),
    analysisPanel(episode, state, rendered),
    episode.shots.map((shot) => shotCard(episode, shot, state[shot.id] || {}, live === shot.id, ready(shot))));
}

function shotCard(episode, shot, s, live, ready) {
  const p = ui.project;
  const plan = p.plan;
  const promptBox = el("textarea", { rows: 12, "aria-label": t("Video prompt of {shot}", { shot: shot.id }) });
  promptBox.value = s.prompt_override || s.prompt || p.previews[shot.id] || "";
  const speech = s.speech;
  const stale = s.outdated || [];
  const keep = async (body) => {
    await api(`/api/projects/${pid()}/episodes/${episode.number}`, { method: "PUT", body }).catch(fail);
    await reloadProject();
  };
  const speaker = (plan.characters.find((c) => c.id === shot.speaker) || {}).name || shot.speaker;
  return el("article", { class: "panel shot", id: `shot-${shot.id}` },
    s.file ? el("video", { src: file(s.file), controls: true, preload: "metadata" })
      : el("div", { class: "placeholder" }, live ? [el("span", { class: "dot run" }), ` ${t("Rendering…")}`] : t("Not rendered yet")),
    el("div", {},
      el("div", { class: "title" }, el("h3", {}, `${shot.id} · ${shot.seconds} s · ${shot.location}`),
        el("span", { class: "badges" }, s.take ? el("span", { class: "badge" }, t("take {n}", { n: s.take })) : "",
          stale.length ? el("span", { class: "badge warn", title: stale.map((r) => t(STALE[r])).join("; ") }, t("outdated")) : "")),
      stale.length ? el("div", { class: "error" }, t("Outdated: {why}.", { why: stale.map((r) => t(STALE[r])).join("; ") })) : "",
      shot.line ? el("div", { class: "line" }, `${speaker}：${shot.line}`) : el("div", { class: "note" }, t("no line")),
      speech ? el("div", { class: speech.match >= 0.6 ? "note" : "error" },
        t("heard “{heard}” · match {match}% · {start}–{end} s ({timing})", { heard: speech.heard, match: Math.round(speech.match * 100),
          start: speech.start, end: speech.end, timing: t(speech.timing) })) : "",
      el("div", { class: "note" }, shot.summary),
      s.references ? el("div", { class: "refs" }, s.references.filter((r) => r.includes("/")).map((r) => el("img", { src: file(r), alt: "" }))) : "",
      s.sheet ? el("img", { class: "sheet", src: file(s.sheet), alt: t("Frames of {shot}", { shot: shot.id }) }) : "",
      el("div", { class: "actions" },
        el("button", { class: "small", disabled: projectBusy() || !ready, title: ready ? "" : t("Approve the pictures this shot uses first"),
          onclick: () => run("shots", { episode: episode.number, params: { only: [shot.id] } }) }, s.file ? t("Render another take") : t("Render")),
        (s.takes || []).length > 1 ? s.takes.map((x) => el("button", {
          class: `small${x.take === s.take ? " primary" : ""}`, disabled: projectBusy(),
          onclick: () => keep({ use_take: { [shot.id]: x.take } }),
        }, t("take {n}", { n: x.take }))) : ""),
      el("details", {}, el("summary", {}, s.prompt ? t("Video prompt of this take") : t("Video prompt (preview)")), promptBox,
        el("button", { class: "small", style: "margin-top:6px", onclick: () => keep({ prompt_overrides: { [shot.id]: promptBox.value } }) },
          t("Keep this prompt for the next take")))));
}

// ---- video analysis: findings that go back into the plan ----

function targetName(target) {
  const [kind, id] = target.split(":");
  if (kind === "shot") return id;
  const group = { character: "characters", location: "locations", prop: "props" }[kind];
  const item = (ui.project.plan[group] || []).find((x) => x.id === id);
  return `${t(KINDS[kind])} · ${item?.name || id}`;
}

// The plan items a note can change: the episode's shots, then the cast, places and props.
function targetOptions(episode) {
  const plan = ui.project.plan;
  return [...episode.shots.map((s) => `shot:${s.id}`),
    ...(plan.characters || []).map((c) => `character:${c.id}`),
    ...(plan.locations || []).map((x) => `location:${x.id}`),
    ...(plan.props || []).map((x) => `prop:${x.id}`)];
}

function analysisPanel(episode, state, rendered) {
  const result = state.analysis;
  const analyzing = projectJobs().some((j) => j.step === "analyze" && j.episode === episode.number && (j.status === "running" || j.status === "queued"));
  const vision = ui.state.external_chat?.vision_model;
  const head = el("div", { class: "title" },
    el("h3", {}, t("Video analysis")),
    el("span", { class: "spacer", style: "flex:1" }),
    el("button", { class: result ? "small" : "small primary", disabled: projectBusy() || !rendered.length,
      onclick: () => run("analyze", { episode: episode.number }) }, result ? t("Analyze again") : t("Analyze the video")));
  const intro = el("p", { class: "note" }, t("Checks the sound, the spoken lines, the picture and the cuts of every rendered shot. ") +
    (vision ? t("{model} also looks at the frames, the reference pictures and the previous shot's last frame.", { model: vision })
      : t("Set a vision model in Settings to also check content, on-screen text, continuity and whether people match their pictures.")));
  if (!result) return el("section", { class: "panel analysis" }, head, intro, analyzing ? el("p", {}, el("span", { class: "dot run" }), ` ${t("Analyzing…")}`) : "");
  const old = rendered.some((s) => result.shots?.[s.id] && result.shots[s.id].take !== state[s.id].take);
  const selection = ui.analysisPicks?.[episode.number] || {};
  const items = result.findings.map((f) => {
    const pick = selection[f.id] || { use: f.action !== "none" && f.severity !== "low", action: f.action, note: f.suggestion, target: f.target };
    selection[f.id] = pick;
    const box = el("input", { type: "checkbox", checked: pick.use, onchange: (e) => { pick.use = e.target.checked; } });
    const action = el("select", { onchange: (e) => { pick.action = e.target.value; } },
      ["revise", "retake", "redraw", "none"].filter((a) => a !== "redraw" || !pick.target.startsWith("shot:"))
        .map((a) => el("option", { value: a, selected: a === pick.action }, t(ACTION[a]))));
    const note = el("textarea", { rows: 2, "aria-label": t("Suggestion for {target}", { target: targetName(pick.target) }) });
    note.value = pick.note;
    note.addEventListener("input", () => { pick.note = note.value; });
    return el("li", { class: `finding ${f.severity}` },
      el("label", { class: "pick" }, box),
      el("div", { class: "grow" },
        el("div", { class: "what" },
          el("span", { class: `badge ${f.severity === "high" ? "bad" : f.severity === "medium" ? "warn" : ""}` }, t(CATEGORY[f.category] || f.category)),
          " ", el("b", {}, targetName(f.target)), ` · ${f.finding}`,
          f.source === "vision" ? el("span", { class: "faint" }, ` · ${t("seen by {model}", { model: result.vision })}`) : ""),
        el("div", { class: "row", style: "align-items:flex-start" }, el("label", { style: "flex:0 0 200px" }, t("Action"), action),
          el("label", { style: "flex:1 1 320px" }, t("Suggestion (edit freely, in any language)"), note))));
  });
  ui.analysisPicks = { ...(ui.analysisPicks || {}), [episode.number]: selection };
  const own = { target: `shot:${episode.shots[0].id}`, note: "" };
  const ownTarget = el("select", { onchange: (e) => { own.target = e.target.value; } },
    targetOptions(episode).map((x) => el("option", { value: x }, targetName(x))));
  const ownNote = el("textarea", { rows: 2, "aria-label": t("Your own note"), placeholder: t("Your own note for the model, in any language") });
  ownNote.addEventListener("input", () => { own.note = ownNote.value; });
  const cuts = (result.cuts || []).map((c) => `${c.from}→${c.to} ${c.jump}${c.same_place ? "" : ` (${t("new place")})`}`).join(" · ");
  return el("section", { class: "panel analysis" }, head, intro,
    el("p", { class: "faint" }, t("Analyzed {when} · {findings}", { when: ago(result.time), findings: plural(result.findings.length, "finding") }) +
      (result.vision ? ` · ${t("vision model: {model}", { model: result.vision })}` : ` · ${t("measured on this machine only")}`) +
      (cuts ? ` · ${t("picture change at the cuts: {cuts}", { cuts })}` : "")),
    old ? el("p", { class: "error" }, t("Some shots have a newer take than the one analyzed; analyze again.")) : "",
    (result.vision_errors || []).length ? el("p", { class: "error" }, t("The vision model failed on: {errors}", { errors: result.vision_errors.join("; ") })) : "",
    items.length ? el("ul", { class: "findings" }, items) : el("p", { class: "note" }, t("No problems found.")),
    el("div", { class: "row", style: "align-items:flex-start;margin-top:8px" },
      el("label", { style: "flex:0 0 200px" }, t("Change"), ownTarget),
      el("label", { style: "flex:1 1 320px" }, t("Your own note"), ownNote)),
    el("div", { class: "actions" },
      el("button", { class: "primary", disabled: projectBusy(), onclick: () => applyFindings(episode, selection, own) }, t("Apply the selected changes")),
      el("span", { class: "note" }, t("Notes revise the plan with the language model; the shots and pictures they touch become outdated, and the next card offers to render them again."))));
}

async function applyFindings(episode, selection, own) {
  const chosen = Object.values(selection).filter((x) => x.use && x.action !== "none");
  if (own.note.trim()) chosen.push({ action: "revise", target: own.target, note: own.note.trim() });
  if (!chosen.length) return toast(t("Select at least one change, or write your own note."), "bad");
  if (ui.dirty) return toast(t("Save your edits first: the model revises the saved plan."), "bad");
  const requests = [];
  const retakes = new Set();
  for (const x of chosen) {
    const [kind, id] = x.target.split(":");
    if (x.action === "revise" && x.note.trim()) requests.push({ step: "revise", params: { target: x.target, instruction: x.note.trim().slice(0, 600) } });
    else if (x.action === "retake" && kind === "shot") retakes.add(id);
    else if (x.action === "redraw" && kind !== "shot") requests.push({ step: "stills", params: { only: [x.target] } });
  }
  if (retakes.size) requests.push({ step: "shots", episode: episode.number, params: { only: [...retakes] } });
  try {
    for (const body of requests) await api(`/api/projects/${pid()}/run`, { method: "POST", body });
    toast(t("Queued {n} changes. Revised shots and pictures become outdated; render them again from the next card.", { n: requests.length }));
    delete ui.analysisPicks[episode.number];
    await refresh();
  } catch (error) { fail(error); }
}

// ---- 5. episodes ----

function episodesStep() {
  const plan = ui.project.plan;
  if (!plan.episodes) return locked(t("Episodes are assembled from rendered shots; plan the drama first."), "plan");
  return el("section", {}, plan.episodes.map((episode) => {
    const state = ui.project.episodes[String(episode.number)] || {};
    const done = episode.shots.filter((s) => state[s.id]?.file).length;
    const complete = done === episode.shots.length;
    const out = `episodes/${String(episode.number).padStart(2, "0")}/episode.mp4`;
    const findings = state.analysis?.findings?.length;
    return el("article", { class: "panel episode" },
      state.assembled ? el("video", { src: file(out), controls: true, preload: "metadata" })
        : el("div", { class: "placeholder" }, complete ? t("Ready to assemble") : t("{done} of {all} shots rendered", { done, all: episode.shots.length })),
      el("div", {},
        el("h2", { style: "margin-top:0" }, t("Episode {n}: {title}", { n: episode.number, title: episode.title })),
        el("p", {}, episode.summary),
        el("p", { class: "note" }, t("{done} of {all} shots rendered", { done, all: episode.shots.length }) +
          (state.assembled ? ` · ${state.assembled.duration} s · ${state.assembled.width}×${state.assembled.height}` : "") +
          (findings != null ? ` · ${t("analysis: {findings}", { findings: plural(findings, "finding") })}` : "")),
        el("div", { class: "actions" },
          state.assembled ? el("a", { class: "button primary", href: file(out), download: `episode-${episode.number}.mp4` }, t("Download")) : "",
          el("button", { class: state.assembled || !complete ? "" : "primary", disabled: !complete || projectBusy(),
            onclick: () => run("episode", { episode: episode.number }) }, state.assembled ? t("Subtitle and assemble again") : t("Subtitle and assemble")),
          el("button", { class: "small", disabled: !complete || projectBusy(), onclick: () => run("subtitles", { episode: episode.number }) },
            t("Time subtitles only")),
          el("button", { class: "small", disabled: !done || projectBusy(), onclick: () => { ui.episode = episode.number; go("shots");
            run("analyze", { episode: episode.number }); } }, t("Analyze the video")),
          complete ? "" : el("a", { class: "button small", href: `#/p/${pid()}/shots`, onclick: () => { ui.episode = episode.number; } },
            t("Render its shots")))));
  }));
}

// ---- flow (advanced) ----

function flowStep() {
  const s = ui.state;
  const [name, version] = ui.project.meta.flow.split("@");
  const text = el("textarea", { class: "flow", "aria-label": t("Flow") });
  const pick = el("select", {}, Object.entries(s.flows).flatMap(([flow, versions]) =>
    versions.map((v) => el("option", { value: `${flow}@${v}`, selected: flow === name && String(v) === version }, `${flow}@${v}`))));
  const load = () => api(`/api/flows/${pick.value}`).then((r) => { text.value = JSON.stringify(r.flow, null, 2); fit(text); }).catch(fail);
  pick.addEventListener("change", load);
  load();
  const saveName = el("input", { value: name });
  const note = el("input", { placeholder: t("what changed") });
  return el("section", {},
    el("p", { class: "note" }, t("A flow lists the steps, the model of each and its parameters, as numbered versions. Most projects never need to change it.")),
    el("div", { class: "actions" }, pick,
      el("button", { onclick: async () => {
        await api(`/api/projects/${pid()}/flow`, { method: "PUT", body: { flow: pick.value } }).catch(fail);
        await reloadProject();
      } }, t("Use this version for the project"))),
    text,
    el("div", { class: "row", style: "align-items:flex-end" }, el("label", {}, t("Flow name"), saveName), el("label", {}, t("Note"), note),
      el("button", { class: "primary", style: "flex:0 0 auto;margin:8px 0", onclick: async () => {
        try {
          const saved = await api(`/api/flows/${saveName.value}`, { method: "POST", body: { flow: JSON.parse(text.value), note: note.value } });
          toast(t("Saved {flow}", { flow: saved.reference }));
          await refresh();
          renderProject();
        } catch (error) { fail(error); }
      } }, t("Save as a new version"))),
    el("h2", {}, t("Installed steps")),
    el("table", { class: "edit" }, s.steps.map((step) => el("tr", {},
      el("td", {}, el("b", {}, step.name)), el("td", {}, step.model_from ? t("action · {step} model", { step: step.model_from }) : t(step.scope)),
      el("td", {}, step.role ? t(step.role) : "–"), el("td", {}, t(step.description)), el("td", { class: "faint" }, step.source)))),
    Object.entries(s.plugin_errors).map(([plugin, error]) => el("p", { class: "error" }, `${plugin}: ${error}`)),
    el("div", { class: "actions" },
      el("button", { onclick: () => api("/api/plugins/reload", { method: "POST" }).then(refresh).then(reloadProject).catch(fail) }, t("Reload plugins"))),
    el("p", { class: "note" }, t("A plugin is one Python file in the plugins directory that defines STEP. Add its name to a flow version to run it; delete the file and reload to remove it. Actions (story, revise, review, analyze) run with the model of a flow step.")));
}

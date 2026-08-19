/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

"use strict";

const DRAFT_STORAGE_KEY = "cosmos-story-scene:draft:v1";
const POLL_INTERVAL_MS = 2000;
const MAX_CONSECUTIVE_POLL_ERRORS = 3;

const PRESETS = Object.freeze({
  "impossible-asmr": {
    label: "Impossible ASMR",
    subject: "A translucent mechanical strawberry with tiny glass seeds",
    hook: "Extreme macro: a crystal blade hovers, then makes the first perfect slice",
    visual_twist:
      "Each paper-thin slice rings like a tuning fork, ripples into liquid chrome, and snaps back into one untouched strawberry on the final beat.",
    camera: "Macro push-in",
    lighting: "Hard flash editorial",
    cta: "Sound on: satisfying or cursed?",
  },
  "pocket-universe": {
    label: "Pocket Universe",
    subject: "A weathered ceramic coffee mug resting on a quiet kitchen table",
    hook: "The camera peers over the rim—and the coffee has become a living spiral galaxy",
    visual_twist:
      "A tiny comet loops out of the mug, circles the handle, then splashes back into ordinary coffee as a hand reaches into frame.",
    camera: "Slow orbit",
    lighting: "Soft window daylight",
    cta: "What would be hiding in your cup?",
  },
  "product-metamorphosis": {
    label: "Product Metamorphosis",
    subject: "A sculptural matte-white sneaker on a seamless pedestal",
    hook: "The laces pull themselves tight as the sole begins to breathe",
    visual_twist:
      "The sneaker folds through a smooth one-shot transformation into a chrome hummingbird, hovers for one beat, then lands as the original product.",
    camera: "Locked symmetrical frame",
    lighting: "Iridescent studio glow",
    cta: "Built to move. Would you wear it?",
  },
  "plot-twist": {
    label: "Plot Twist",
    subject: "A miniature birthday cake with one flickering candle",
    hook: "A tiny figure walks toward the candle to make a wish",
    visual_twist:
      "The flame bends sideways and reveals that the cake is a full-sized neon city block seen from above; the tiny figure is actually a commuter rushing for a train.",
    camera: "Handheld discovery",
    lighting: "Neon noir contrast",
    cta: "Did you catch the scale switch?",
  },
  "nature-glitch": {
    label: "Nature Glitch",
    subject: "A lone dandelion in a dew-covered meadow before sunrise",
    hook: "The first seed lifts off in a perfectly still world",
    visual_twist:
      "The floating seeds duplicate into geometric constellations, the meadow briefly pixel-sorts like a living screen, then one seed lands and blooms instantly.",
    camera: "Macro push-in",
    lighting: "Golden-hour haze",
    cta: "Nature.exe is evolving.",
  },
});

const STATUS_ALIASES = Object.freeze({
  created: "queued",
  pending: "queued",
  waiting: "queued",
  queued: "queued",
  submitted: "queued",
  running: "generating",
  processing: "generating",
  inference: "generating",
  generating: "generating",
  postprocessing: "packaging",
  post_processing: "packaging",
  rendering: "packaging",
  encoding: "packaging",
  packaging: "packaging",
  done: "completed",
  complete: "completed",
  completed: "completed",
  success: "completed",
  succeeded: "completed",
  cancelled: "failed",
  canceled: "failed",
  error: "failed",
  failed: "failed",
});

const STATUS_COPY = Object.freeze({
  queued: {
    title: "Queued",
    detail: "Waiting for the generation worker.",
  },
  generating: {
    title: "Generating",
    detail: "Cosmos 3 is rendering the 1280×720 source.",
  },
  packaging: {
    title: "Packaging",
    detail: "Encoding the master and deriving the vertical edit.",
  },
  completed: {
    title: "Ready",
    detail: "Both deliverables are ready to preview.",
  },
  failed: {
    title: "Generation stopped",
    detail: "The server could not finish this clip.",
  },
});

const form = document.querySelector("#story-form");
const presetButtons = Array.from(document.querySelectorAll("[data-preset]"));
const selectedPresetLabel = document.querySelector("#selected-preset-label");
const clearDraftButton = document.querySelector("#clear-draft");
const draftIndicator = document.querySelector("#draft-indicator");
const compiledPrompt = document.querySelector("#compiled-prompt");
const copyPromptButton = document.querySelector("#copy-prompt");
const formError = document.querySelector("#form-error");
const storyButton = document.querySelector("#story-button");
const storyButtonLabel = storyButton.querySelector(".story-button-label");
const jobBadge = document.querySelector("#job-badge");
const previewStage = document.querySelector("#preview-stage");
const previewEmpty = document.querySelector("#preview-empty");
const videoPreview = document.querySelector("#video-preview");
const frameLabels = document.querySelector("#frame-labels");
const previewFormatLabel = document.querySelector("#preview-format-label");
const progressPanel = document.querySelector("#progress-panel");
const progressTitle = document.querySelector("#progress-title");
const progressDetail = document.querySelector("#progress-detail");
const jobProgress = document.querySelector("#job-progress");
const stageItems = Array.from(document.querySelectorAll("[data-stage]"));
const retryStatusButton = document.querySelector("#retry-status");
const resultPanel = document.querySelector("#result-panel");
const outputTabs = Array.from(document.querySelectorAll("[data-output]"));
const socialDownload = document.querySelector("#download-social");
const cleanDownload = document.querySelector("#download-clean");
const toast = document.querySelector("#toast");

let selectedPreset = "custom";
let activeJobId = null;
let activeOutput = "social";
let outputUrls = { social: null, clean: null };
let pollTimer = null;
let pollController = null;
let consecutivePollErrors = 0;
let draftTimer = null;
let toastTimer = null;

function formField(name) {
  return form.elements.namedItem(name);
}

function setFieldValue(name, value) {
  const field = formField(name);
  if (field) {
    field.value = value ?? "";
  }
}

function getFieldValue(name) {
  const field = formField(name);
  return field ? field.value.trim() : "";
}

function updateCharacterCounts() {
  document.querySelectorAll("[data-count-for]").forEach((counter) => {
    const field = document.getElementById(counter.dataset.countFor);
    counter.textContent = field ? String(field.value.length) : "0";
  });
}

function selectPreset(presetId, applyValues = true) {
  const preset = PRESETS[presetId];
  selectedPreset = preset ? presetId : "custom";

  presetButtons.forEach((button) => {
    button.setAttribute("aria-pressed", String(button.dataset.preset === selectedPreset));
  });

  selectedPresetLabel.textContent = preset ? preset.label : "Custom direction";

  if (preset && applyValues) {
    Object.entries(preset).forEach(([fieldName, value]) => {
      if (fieldName !== "label") {
        setFieldValue(fieldName, value);
      }
    });
    updateCharacterCounts();
    updateCompiledPrompt();
    scheduleDraftSave();
    showToast(`${preset.label} loaded. Shape it into your own.`);
  }
}

function collectDraft() {
  return {
    preset: selectedPreset,
    subject: getFieldValue("subject"),
    hook: getFieldValue("hook"),
    visual_twist: getFieldValue("visual_twist"),
    camera: getFieldValue("camera"),
    lighting: getFieldValue("lighting"),
    cta: getFieldValue("cta"),
    seed: getFieldValue("seed"),
  };
}

function saveDraft() {
  window.clearTimeout(draftTimer);
  draftIndicator.classList.add("is-saving");

  try {
    window.localStorage.setItem(DRAFT_STORAGE_KEY, JSON.stringify(collectDraft()));
    draftIndicator.innerHTML = '<span aria-hidden="true">✓</span> Draft saved in this browser';
  } catch (_error) {
    draftIndicator.innerHTML = '<span aria-hidden="true">·</span> Local draft unavailable';
  } finally {
    draftIndicator.classList.remove("is-saving");
  }
}

function scheduleDraftSave() {
  window.clearTimeout(draftTimer);
  draftIndicator.classList.add("is-saving");
  draftIndicator.innerHTML = '<span aria-hidden="true">·</span> Saving draft';
  draftTimer = window.setTimeout(saveDraft, 350);
}

function restoreDraft() {
  let draft;

  try {
    draft = JSON.parse(window.localStorage.getItem(DRAFT_STORAGE_KEY) || "null");
  } catch (_error) {
    draft = null;
  }

  if (!draft || typeof draft !== "object") {
    updateCharacterCounts();
    return;
  }

  ["subject", "hook", "visual_twist", "camera", "lighting", "cta", "seed"].forEach((name) => {
    if (typeof draft[name] === "string") {
      setFieldValue(name, draft[name]);
    }
  });

  selectPreset(typeof draft.preset === "string" ? draft.preset : "custom", false);
  updateCharacterCounts();
  updateCompiledPrompt();
  draftIndicator.innerHTML = '<span aria-hidden="true">✓</span> Local draft restored';
}

function clearDraft() {
  window.clearTimeout(draftTimer);
  form.reset();
  selectPreset("custom", false);
  hideFormError();
  updateCharacterCounts();
  updateCompiledPrompt();

  try {
    window.localStorage.removeItem(DRAFT_STORAGE_KEY);
    draftIndicator.innerHTML = '<span aria-hidden="true">✓</span> Draft cleared';
  } catch (_error) {
    draftIndicator.innerHTML = '<span aria-hidden="true">·</span> Local draft unavailable';
  }

  showToast("Creative brief cleared.");
}

function compileLocalPrompt() {
  const values = collectDraft();
  if (!values.subject && !values.hook && !values.visual_twist) {
    return "";
  }

  const sentences = [];
  if (values.subject) {
    sentences.push(`Subject: ${values.subject}.`);
  }
  if (values.hook) {
    sentences.push(`Open with: ${values.hook}.`);
  }
  if (values.visual_twist) {
    sentences.push(`Then reveal: ${values.visual_twist}.`);
  }
  if (values.camera) {
    sentences.push(`Camera: ${values.camera}.`);
  }
  if (values.lighting) {
    sentences.push(`Lighting: ${values.lighting}.`);
  }
  if (values.cta) {
    sentences.push(`Final on-screen CTA: “${values.cta}”`);
  }

  sentences.push(
    "Compose for a fixed 7.9-second, 1280×720 landscape source with the hero action kept center-safe for a derived 9:16 reframe."
  );
  return sentences.join(" ");
}

function displayPrompt(prompt) {
  const hasPrompt = typeof prompt === "string" && prompt.trim().length > 0;
  compiledPrompt.textContent = hasPrompt
    ? prompt.trim()
    : "Your production-ready prompt will appear here.";
  compiledPrompt.classList.toggle("has-prompt", hasPrompt);
  copyPromptButton.disabled = !hasPrompt;
}

function updateCompiledPrompt() {
  displayPrompt(compileLocalPrompt());
}

function validateForm() {
  let valid = true;
  let firstInvalid = null;

  ["subject", "hook", "visual_twist"].forEach((name) => {
    const field = formField(name);
    const fieldIsValid = Boolean(field && field.value.trim().length >= 3 && field.checkValidity());
    field.setAttribute("aria-invalid", String(!fieldIsValid));
    if (!fieldIsValid && !firstInvalid) {
      firstInvalid = field;
    }
    valid = valid && fieldIsValid;
  });

  const seedField = formField("seed");
  const seedIsValid = seedField.value === "" || seedField.checkValidity();
  seedField.setAttribute("aria-invalid", String(!seedIsValid));
  if (!seedIsValid && !firstInvalid) {
    firstInvalid = seedField;
  }
  valid = valid && seedIsValid;

  if (!valid) {
    showFormError(
      "Add a subject, opening hook, and visual twist (at least 3 characters each). The seed must be a whole number from 0 to 2,147,483,647."
    );
    firstInvalid?.focus();
  } else {
    hideFormError();
  }

  return valid;
}

function buildRequestBody() {
  const seedValue = getFieldValue("seed");
  const ctaValue = getFieldValue("cta");
  const payload = {
    preset: selectedPreset,
    subject: getFieldValue("subject"),
    hook: getFieldValue("hook"),
    visual_twist: getFieldValue("visual_twist"),
    camera: getFieldValue("camera"),
    lighting: getFieldValue("lighting"),
  };

  if (ctaValue) {
    payload.cta = ctaValue;
  }
  if (seedValue) {
    payload.seed = Number.parseInt(seedValue, 10);
  }
  return payload;
}

function showFormError(message) {
  formError.textContent = message;
  formError.hidden = false;
}

function hideFormError() {
  formError.textContent = "";
  formError.hidden = true;
}

function setStoryBusy(isBusy) {
  storyButton.disabled = isBusy;
  storyButtonLabel.textContent = isBusy ? "Generating scene…" : "Generate my scene";
}

function showToast(message) {
  window.clearTimeout(toastTimer);
  toast.textContent = message;
  toast.hidden = false;
  toastTimer = window.setTimeout(() => {
    toast.hidden = true;
  }, 3200);
}

function normalizeStatus(value) {
  const normalized = String(value || "queued")
    .trim()
    .toLowerCase()
    .replace(/[\s-]+/g, "_");
  return STATUS_ALIASES[normalized] || "queued";
}

function normalizeProgress(value, status) {
  if (status === "completed") {
    return 100;
  }

  const numeric = Number(value);
  if (!Number.isFinite(numeric) || numeric < 0) {
    return null;
  }

  const percent = numeric > 0 && numeric <= 1 ? numeric * 100 : numeric;
  return Math.min(100, Math.max(0, Math.round(percent)));
}

function renderStatus(statusValue, progressValue) {
  const status = normalizeStatus(statusValue);
  const copy = STATUS_COPY[status];
  const progress = normalizeProgress(progressValue, status);
  const stageOrder = ["queued", "generating", "packaging"];
  const currentStage = status === "completed" ? stageOrder.length : stageOrder.indexOf(status);

  progressPanel.hidden = false;
  progressTitle.textContent = copy.title;
  progressDetail.textContent = copy.detail;
  jobBadge.textContent = copy.title;
  jobBadge.className = "job-badge";
  jobBadge.classList.add(
    status === "completed" ? "is-complete" : status === "failed" ? "is-error" : "is-active"
  );

  if (progress === null) {
    jobProgress.removeAttribute("value");
    jobProgress.textContent = "In progress";
  } else {
    jobProgress.value = progress;
    jobProgress.textContent = `${progress}%`;
  }

  stageItems.forEach((item, index) => {
    item.classList.toggle("is-done", status === "completed" || index < currentStage);
    item.classList.toggle("is-current", index === currentStage);
  });

  if (status === "failed") {
    stageItems.forEach((item) => item.classList.remove("is-current"));
  }

  return status;
}

function safeMediaUrl(value) {
  const candidate = typeof value === "object" && value !== null ? value.url : value;
  if (typeof candidate !== "string" || candidate.trim() === "") {
    return null;
  }

  try {
    const parsed = new URL(candidate, window.location.href);
    return parsed.origin === window.location.origin ? parsed.href : null;
  } catch (_error) {
    return null;
  }
}

function firstSafeUrl(...candidates) {
  for (const candidate of candidates) {
    const safeUrl = safeMediaUrl(candidate);
    if (safeUrl) {
      return safeUrl;
    }
  }
  return null;
}

function extractOutputUrls(payload) {
  const outputs = payload.outputs || payload.result?.outputs || {};
  const result = payload.result || {};
  const artifacts = payload.artifacts || {};

  return {
    social: firstSafeUrl(
      payload.social_video_url,
      payload.vertical_video_url,
      payload.social_url,
      payload.vertical_url,
      outputs.social,
      outputs.vertical,
      result.social_video_url,
      result.social,
      artifacts.social
    ),
    clean: firstSafeUrl(
      payload.clean_video_url,
      payload.horizontal_video_url,
      payload.clean_url,
      payload.output_url,
      outputs.clean,
      outputs.horizontal,
      result.clean_video_url,
      result.clean,
      artifacts.clean
    ),
  };
}

function setDownloadLink(link, url) {
  if (url) {
    link.href = url;
    link.hidden = false;
  } else {
    link.removeAttribute("href");
    link.hidden = true;
  }
}

function switchOutput(outputName) {
  if (!outputUrls[outputName]) {
    return;
  }

  activeOutput = outputName;
  outputTabs.forEach((tab) => {
    const active = tab.dataset.output === activeOutput;
    tab.classList.toggle("is-active", active);
    tab.setAttribute("aria-pressed", String(active));
  });

  videoPreview.pause();
  videoPreview.src = outputUrls[activeOutput];
  videoPreview.load();
  videoPreview.hidden = false;
  previewEmpty.hidden = true;
  frameLabels.hidden = false;
  previewStage.classList.toggle("is-social", activeOutput === "social");
  previewFormatLabel.textContent =
    activeOutput === "social" ? "Social · derived 9:16" : "Clean · 1280×720 source";
}

function renderOutputs(payload) {
  outputUrls = extractOutputUrls(payload);
  setDownloadLink(socialDownload, outputUrls.social);
  setDownloadLink(cleanDownload, outputUrls.clean);

  outputTabs.forEach((tab) => {
    tab.disabled = !outputUrls[tab.dataset.output];
  });

  const preferredOutput = outputUrls.social ? "social" : outputUrls.clean ? "clean" : null;
  if (!preferredOutput) {
    resultPanel.hidden = true;
    showFormError(
      "The job completed, but the server did not advertise a playable output. Retry the status check or inspect the server logs."
    );
    retryStatusButton.hidden = false;
    return false;
  }

  resultPanel.hidden = false;
  switchOutput(preferredOutput);
  return true;
}

function applyJobPayload(payload) {
  const serverPrompt = payload.compiled_prompt || payload.result?.compiled_prompt;
  if (typeof serverPrompt === "string" && serverPrompt.trim()) {
    displayPrompt(serverPrompt);
  }

  const status = renderStatus(payload.status || payload.state, payload.progress);

  if (status === "completed") {
    const hasOutputs = renderOutputs(payload);
    setStoryBusy(false);
    retryStatusButton.hidden = hasOutputs;
    if (hasOutputs) {
      hideFormError();
      showToast("Your clip is ready. Pick a cut and ship it.");
    }
  } else if (status === "failed") {
    setStoryBusy(false);
    retryStatusButton.hidden = true;
    showFormError(
      "Generation stopped before the clip was finished. Adjust the creative brief and try again; server credentials remain server-side."
    );
  }

  return status;
}

function cancelPolling() {
  window.clearTimeout(pollTimer);
  pollTimer = null;
  if (pollController) {
    pollController.abort();
    pollController = null;
  }
}

function schedulePoll(delay = POLL_INTERVAL_MS) {
  window.clearTimeout(pollTimer);
  pollTimer = window.setTimeout(pollJob, delay);
}

async function parseJsonResponse(response) {
  const contentType = response.headers.get("content-type") || "";
  if (!contentType.includes("application/json")) {
    return {};
  }

  try {
    return await response.json();
  } catch (_error) {
    return {};
  }
}

async function pollJob() {
  if (!activeJobId) {
    return;
  }

  pollController = new AbortController();

  try {
    const response = await fetch(`/api/jobs/${encodeURIComponent(activeJobId)}`, {
      method: "GET",
      cache: "no-store",
      credentials: "same-origin",
      headers: { Accept: "application/json" },
      signal: pollController.signal,
    });
    const payload = await parseJsonResponse(response);

    if (!response.ok) {
      throw new Error(`Status request failed with HTTP ${response.status}`);
    }

    consecutivePollErrors = 0;
    retryStatusButton.hidden = true;
    const status = applyJobPayload(payload);
    if (!["completed", "failed"].includes(status)) {
      schedulePoll(document.hidden ? 5000 : POLL_INTERVAL_MS);
    }
  } catch (error) {
    if (error.name === "AbortError") {
      return;
    }

    consecutivePollErrors += 1;
    if (consecutivePollErrors < MAX_CONSECUTIVE_POLL_ERRORS) {
      progressTitle.textContent = "Reconnecting";
      progressDetail.textContent = "The status check was interrupted; trying again.";
      schedulePoll(POLL_INTERVAL_MS * consecutivePollErrors);
    } else {
      setStoryBusy(false);
      retryStatusButton.hidden = false;
      jobBadge.textContent = "Connection lost";
      jobBadge.className = "job-badge is-error";
      showFormError(
        "The job may still be running, but this page cannot reach its status endpoint. Retry the status check before starting a duplicate request."
      );
    }
  } finally {
    pollController = null;
  }
}

async function submitJob(event) {
  event.preventDefault();
  if (!validateForm()) {
    return;
  }

  cancelPolling();
  activeJobId = null;
  consecutivePollErrors = 0;
  outputUrls = { social: null, clean: null };
  setStoryBusy(true);
  hideFormError();
  retryStatusButton.hidden = true;
  resultPanel.hidden = true;
  progressPanel.hidden = false;
  previewEmpty.hidden = false;
  videoPreview.hidden = true;
  frameLabels.hidden = true;
  previewStage.classList.remove("is-social");
  videoPreview.removeAttribute("src");
  videoPreview.load();
  displayPrompt(compileLocalPrompt());
  renderStatus("queued", 0);
  progressDetail.textContent = "Submitting the creative brief to the server.";

  try {
    const response = await fetch("/api/jobs", {
      method: "POST",
      credentials: "same-origin",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      body: JSON.stringify(buildRequestBody()),
    });
    const payload = await parseJsonResponse(response);

    if (!response.ok) {
      throw new Error(`Generation request failed with HTTP ${response.status}`);
    }

    activeJobId = payload.job_id || payload.id || payload.uuid || null;
    const status = applyJobPayload(payload);
    saveDraft();

    if (!["completed", "failed"].includes(status)) {
      if (!activeJobId) {
        throw new Error("The server accepted the request but returned no job identifier");
      }
      schedulePoll(POLL_INTERVAL_MS);
    }
  } catch (_error) {
    setStoryBusy(false);
    jobBadge.textContent = "Request failed";
    jobBadge.className = "job-badge is-error";
    progressTitle.textContent = "Could not start";
    progressDetail.textContent = "The server did not accept the generation request.";
    jobProgress.value = 0;
    showFormError(
      "Story Scene could not start this job. Confirm that the app server and generation worker are running, then try again."
    );
  }
}

async function copyPromptToClipboard() {
  const prompt = compiledPrompt.textContent.trim();
  if (!prompt || copyPromptButton.disabled) {
    return;
  }

  try {
    await navigator.clipboard.writeText(prompt);
  } catch (_error) {
    const helper = document.createElement("textarea");
    helper.value = prompt;
    helper.setAttribute("readonly", "");
    helper.className = "clipboard-helper";
    document.body.appendChild(helper);
    helper.select();
    document.execCommand("copy");
    helper.remove();
  }

  showToast("Compiled prompt copied.");
}

presetButtons.forEach((button) => {
  button.addEventListener("click", () => selectPreset(button.dataset.preset, true));
});

form.addEventListener("input", (event) => {
  if (event.target.matches("input, textarea, select")) {
    event.target.removeAttribute("aria-invalid");
    updateCharacterCounts();
    updateCompiledPrompt();
    scheduleDraftSave();
  }
});

form.addEventListener("change", (event) => {
  if (event.target.matches("select")) {
    updateCompiledPrompt();
    scheduleDraftSave();
  }
});

form.addEventListener("submit", submitJob);
clearDraftButton.addEventListener("click", clearDraft);
copyPromptButton.addEventListener("click", copyPromptToClipboard);
retryStatusButton.addEventListener("click", () => {
  if (!activeJobId) {
    return;
  }
  consecutivePollErrors = 0;
  retryStatusButton.hidden = true;
  hideFormError();
  setStoryBusy(true);
  pollJob();
});

outputTabs.forEach((tab) => {
  tab.addEventListener("click", () => switchOutput(tab.dataset.output));
});

videoPreview.addEventListener("error", () => {
  showToast("Preview unavailable in this browser. The download may still work.");
});

document.addEventListener("visibilitychange", () => {
  if (!document.hidden && activeJobId && pollTimer) {
    schedulePoll(250);
  }
});

window.addEventListener("beforeunload", cancelPolling);

restoreDraft();

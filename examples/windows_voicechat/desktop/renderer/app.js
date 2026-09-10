'use strict';

(() => {
  const $ = id => document.getElementById(id);
  const api = window.voiceLab;
  const dom = Object.fromEntries([
    'connectButton', 'connectLabel', 'muteButton', 'interruptButton', 'settingsDialog', 'bundlePath',
    'bridgePath', 'systemPrompt', 'sessionState', 'sessionHint', 'stateDot',
    'rehearsalBadge', 'rehearsalButton', 'transcriptScroll', 'transcriptEmpty',
    'transcriptCount', 'transcriptMode', 'transcriptDot', 'transcriptFootnote',
    'sessionTimer', 'modelDetail', 'modelStatusDot', 'backendStatus', 'backendDot',
    'pipelineMic', 'pipelineModel', 'pipelineOutput', 'inputLevel', 'outputLevel',
    'latencyMetric', 'gpuMetric', 'streamButton', 'streamExit', 'streamSessionCaption',
    'toast', 'orbCanvas', 'visualizerWrap',
  ].map(id => [id, $(id)]));

  let config = {bundlePath: '', bridgePath: '', systemPrompt: ''};
  let state = 'disconnected';
  let connected = false;
  let starting = false;
  let awaitingInitialBridge = false;
  let generation = 0;
  let muted = false;
  let interruptPending = false;
  let rehearsal = false;
  let streamMode = false;
  let fullscreen = false;
  let lastError = '';
  let startedAt = null;
  let elapsedSeconds = 0;
  let microphone = null;
  let audioContext = null;
  let captureNode = null;
  let outputAnalyser = null;
  let outputData = null;
  let inputEnergy = 0;
  let outputEnergy = 0;
  let outputCursor = 0;
  let toastTimeout;
  let rehearsalTimeouts = [];
  let visualPhase = 0;
  const playback = new Set();
  // Two 80 ms model frames absorb short inference/IPC jitter. Refill this
  // cushion only when starting or after underrun; queued packets stay contiguous.
  const playbackPrebufferSeconds = 0.16;
  const playbackSchedulingMarginSeconds = 0.005;
  const activeEntries = new Map();
  let transcriptEntries = 0;
  let assistantTranscriptEpoch = null;
  let activeBundlePath = '';
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');

  const baseName = value => String(value || '').split(/[\\/]/).pop();
  const elapsedLabel = seconds => `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`;

  function showToast(message, isError = false) {
    clearTimeout(toastTimeout);
    dom.toast.textContent = message;
    dom.toast.className = `toast ${isError ? 'error' : 'info'}`;
    dom.toast.hidden = false;
    toastTimeout = setTimeout(() => { dom.toast.hidden = true; }, isError ? 14000 : 6000);
  }

  function updateTimer() {
    if (startedAt !== null) elapsedSeconds = Math.floor((performance.now() - startedAt) / 1000);
    dom.sessionTimer.textContent = elapsedLabel(elapsedSeconds);
  }

  function updateUI(message) {
    const active = connected || rehearsal;
    document.body.classList.toggle('session-active', active);
    document.body.dataset.sessionState = state;
    dom.connectButton.classList.toggle('disconnect', active || starting);
    dom.connectLabel.textContent = rehearsal ? 'End rehearsal' : starting || state === 'loading' ? 'Cancel connection' : connected ? 'End conversation' : 'Start conversation';
    dom.muteButton.disabled = !connected || starting || rehearsal;
    dom.interruptButton.disabled = !connected || starting || rehearsal || interruptPending || !(playback.size || state === 'speaking' || state === 'thinking');
    dom.muteButton.setAttribute('aria-pressed', String(muted));
    dom.muteButton.setAttribute('aria-label', muted ? 'Unmute microphone' : 'Mute microphone');
    dom.muteButton.title = `${muted ? 'Unmute' : 'Mute'} microphone (Space)`;
    dom.muteButton.querySelector('use').setAttribute('href', muted ? '#i-muted' : '#i-mic');
    dom.stateDot.className = `state-dot${state === 'loading' ? ' loading' : lastError ? ' error' : active ? ' active' : ''}`;
    dom.rehearsalBadge.hidden = !rehearsal;
    dom.rehearsalButton.hidden = active || starting;
    dom.transcriptMode.textContent = rehearsal ? 'SAMPLE SCRIPT · NO INFERENCE' : active ? 'LIVE · FULL DUPLEX' : 'AWAITING CONNECTION';
    dom.transcriptDot.classList.toggle('active', active && !rehearsal);
    dom.transcriptFootnote.textContent = rehearsal ? 'Simulated visuals. No model or audio inference.' : 'Speak naturally. Press I to stop a response.';
    dom.streamSessionCaption.textContent = rehearsal ? 'VISUAL REHEARSAL · NO LIVE INFERENCE' : connected ? 'FULL-DUPLEX VOICE · LOCAL RTX INFERENCE' : 'NEMOTRON VOICECHAT · TENSORRT-RTX';
    dom.modelStatusDot.classList.toggle('active', connected && state !== 'loading');
    dom.backendDot.classList.toggle('active', connected && state !== 'loading');
    dom.backendStatus.textContent = rehearsal ? 'Visual rehearsal' : state === 'loading' ? 'Loading engines' : connected ? 'Connected locally' : 'Not connected';
    dom.pipelineMic.classList.toggle('active', connected && !muted && state !== 'loading');
    dom.pipelineModel.classList.toggle('active', connected && state !== 'loading');
    dom.pipelineOutput.classList.toggle('active', connected && state === 'speaking');
    dom.modelDetail.textContent = rehearsal ? 'Visual rehearsal · model disconnected' : connected ? baseName(activeBundlePath) : config.bundlePath ? baseName(config.bundlePath) : 'Select your TensorRT-RTX bundle';
    dom.modelDetail.title = rehearsal ? 'No inference is running' : connected ? activeBundlePath : config.bundlePath;
    const labels = {
      disconnected: ['Ready when you are', 'Connect a local model bundle to start the conversation.'],
      loading: ['Waking up Nemotron', 'Initializing the local engines. This can take a moment.'],
      listening: ['I’m listening', 'Say what’s on your mind. Press I to stop a response.'],
      thinking: ['A thought is taking shape', 'Nemotron is generating a response on your RTX GPU.'],
      speaking: ['Let’s think out loud', 'Speak naturally, or press I to stop this response.'],
    };
    let [label, hint] = labels[state] || labels.disconnected;
    if (muted && connected && state === 'listening') {
      label = 'A moment of quiet';
      hint = 'Your microphone is muted. Press Space when you’re ready.';
    }
    if (interruptPending) { label = 'Stopping the response'; hint = muted ? 'Your microphone is still muted. Press Space to speak.' : 'Keep speaking. Your microphone is still on.'; }
    if (lastError) { label = 'Let’s get connected'; hint = lastError; }
    if (rehearsal) hint = 'Visual rehearsal with a sample script. No live model inference.';
    dom.sessionState.textContent = label;
    dom.sessionHint.textContent = message || hint;
    dom.sessionHint.title = message || hint;
  }

  function clearTranscript() {
    for (const child of [...dom.transcriptScroll.children]) {
      if (child !== dom.transcriptEmpty) child.remove();
    }
    dom.transcriptEmpty.hidden = false;
    transcriptEntries = 0;
    activeEntries.clear();
    assistantTranscriptEpoch = null;
    dom.transcriptCount.textContent = '00';
  }

  function addNotice(message) {
    dom.transcriptEmpty.hidden = true;
    const note = document.createElement('p');
    note.className = 'transcript-notice';
    note.textContent = message;
    dom.transcriptScroll.append(note);
    pruneTranscript();
    dom.transcriptScroll.scrollTop = dom.transcriptScroll.scrollHeight;
  }

  function pruneTranscript() {
    // Notices share the same history budget as speech, so context refreshes
    // cannot grow the DOM for the lifetime of a long conversation.
    const rows = dom.transcriptScroll.querySelectorAll('.transcript-entry, .transcript-notice');
    for (let index = 0; index < rows.length - 300; index += 1) {
      const row = rows[index];
      for (const [key, entry] of activeEntries) {
        if (entry.article === row) activeEntries.delete(key);
      }
      row.remove();
    }
  }

  function finishTranscripts(role) {
    for (const [key, entry] of activeEntries) {
      if (role && entry.role !== role) continue;
      entry.text.classList.remove('partial');
      activeEntries.delete(key);
    }
  }

  function appendTranscript(event) {
    if (typeof event.text !== 'string' || !event.text) return;
    const role = event.role === 'user' ? 'user' : 'assistant';
    const key = event.id ? `${role}:${event.id}` : role;
    // Output epochs delimit agent responses. A single user utterance may span
    // agent start/finish or barge-in, so keep its row until its own final snapshot.
    if (role === 'assistant' && event.epoch !== undefined) {
      if (assistantTranscriptEpoch !== null && event.epoch !== assistantTranscriptEpoch) finishTranscripts('assistant');
      assistantTranscriptEpoch = event.epoch;
    }
    const nearBottom = dom.transcriptScroll.scrollHeight - dom.transcriptScroll.scrollTop - dom.transcriptScroll.clientHeight < 90;
    let entry = activeEntries.get(key);
    if (!entry) {
      dom.transcriptEmpty.hidden = true;
      const article = document.createElement('article');
      article.className = `transcript-entry ${role}`;
      const header = document.createElement('div');
      header.className = 'entry-header';
      const avatar = document.createElement('span');
      avatar.className = 'entry-avatar';
      avatar.textContent = role === 'user' ? 'Y' : 'N';
      avatar.setAttribute('aria-hidden', 'true');
      const speaker = document.createElement('span');
      speaker.textContent = role === 'user' ? 'YOU' : 'NEMOTRON';
      const time = document.createElement('time');
      time.textContent = dom.sessionTimer.textContent;
      header.append(avatar, speaker, time);
      const text = document.createElement('p');
      text.className = 'entry-text';
      const divider = document.createElement('div');
      divider.className = 'entry-divider';
      divider.setAttribute('aria-hidden', 'true');
      article.append(header, text, divider);
      dom.transcriptScroll.append(article);
      entry = {article, text, value: '', role};
      activeEntries.set(key, entry);
      transcriptEntries += 1;
      dom.transcriptCount.textContent = String(transcriptEntries).padStart(2, '0');
    }
    entry.value = event.delta === true ? entry.value + event.text : event.text;
    entry.text.textContent = entry.value;
    entry.text.classList.toggle('partial', event.final === false);
    if (event.final !== false) activeEntries.delete(key);
    // Keep a long session bounded without stealing the scroll position when reading.
    pruneTranscript();
    if (nearBottom) dom.transcriptScroll.scrollTop = dom.transcriptScroll.scrollHeight;
  }

  async function setupAudio(token) {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true},
      video: false,
    });
    if (token !== generation) { stream.getTracks().forEach(track => track.stop()); return false; }
    microphone = stream;
    audioContext = new AudioContext({sampleRate: 48000, latencyHint: 'interactive'});
    const context = audioContext;
    await context.resume();
    await context.audioWorklet.addModule('capture-worklet.js');
    if (token !== generation) return false;
    const source = context.createMediaStreamSource(stream);
    // Low-pass before 16 kHz downsampling so higher frequencies cannot alias.
    const lowpassA = context.createBiquadFilter();
    const lowpassB = context.createBiquadFilter();
    for (const filter of [lowpassA, lowpassB]) {
      filter.type = 'lowpass'; filter.frequency.value = 7200; filter.Q.value = Math.SQRT1_2;
    }
    captureNode = new AudioWorkletNode(context, 'voice-capture', {numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1]});
    source.connect(lowpassA).connect(lowpassB).connect(captureNode).connect(context.destination);
    outputAnalyser = context.createAnalyser();
    outputAnalyser.fftSize = 512;
    outputData = new Float32Array(outputAnalyser.fftSize);
    outputAnalyser.connect(context.destination);
    const ratio = context.sampleRate / 16000;
    let carry = 0;
    let previousSample = 0;
    // Send 20 ms packets so capture does not race the native 80 ms idle clock.
    const inputPacketSamples = 320;
    let frame = new Float32Array(inputPacketSamples);
    let frameOffset = 0;
    captureNode.port.onmessage = ({data}) => {
      if (token !== generation) return;
      const samples = data;
      let sum = 0;
      for (let i = 0; i < samples.length; i += 1) sum += samples[i] * samples[i];
      inputEnergy = muted ? 0 : Math.min(1, Math.sqrt(sum / samples.length) * 7);
      const combined = new Float32Array(samples.length + 1);
      combined[0] = previousSample;
      combined.set(samples, 1);
      for (; carry + 1 < combined.length; carry += ratio) {
        const left = Math.floor(carry);
        const fraction = carry - left;
        frame[frameOffset++] = muted ? 0 : Math.max(-1, Math.min(1, combined[left] * (1 - fraction) + combined[left + 1] * fraction));
        if (frameOffset === frame.length) {
          if (connected && state !== 'loading') api.sendAudio({samples: Array.from(frame), sampleRate: 16000});
          frame = new Float32Array(inputPacketSamples);
          frameOffset = 0;
        }
      }
      carry -= samples.length;
      previousSample = samples[samples.length - 1];
    };
    for (const track of stream.getAudioTracks()) track.addEventListener('ended', () => {
      if (token === generation && connected) {
        lastError = 'Your microphone disconnected. Reconnect it and start a new conversation.';
        showToast(lastError, true);
        void disconnectSession();
      }
    });
    return true;
  }

  function flushPlayback() {
    for (const source of playback) {
      source.onended = null;
      try { source.stop(); } catch {}
      source.disconnect();
    }
    playback.clear();
    outputCursor = 0;
    outputEnergy = 0;
  }

  async function stopAudio() {
    interruptPending = false;
    flushPlayback();
    if (captureNode) { captureNode.port.onmessage = null; captureNode.disconnect(); captureNode = null; }
    if (microphone) { microphone.getTracks().forEach(track => track.stop()); microphone = null; }
    const oldContext = audioContext;
    audioContext = null;
    outputAnalyser = null;
    outputData = null;
    inputEnergy = 0;
    if (oldContext && oldContext.state !== 'closed') await oldContext.close().catch(() => {});
  }

  function playAudio(event) {
    if (!audioContext || !outputAnalyser || !connected || rehearsal) return;
    const samples = event.samples;
    const rate = Number(event.sampleRate);
    if (!samples?.length || !Number.isFinite(rate) || rate < 8000 || rate > 192000) return;
    const context = audioContext;
    // A runaway output queue makes turn-taking unusable. Stop instead of drifting.
    if (outputCursor - context.currentTime > 15) {
      lastError = 'More than 15 seconds of audio queued for playback. Start a new conversation to reset the stream.';
      showToast(lastError, true);
      void disconnectSession();
      return;
    }
    const buffer = context.createBuffer(1, samples.length, rate);
    buffer.copyToChannel(Float32Array.from(samples), 0);
    const source = context.createBufferSource();
    source.buffer = buffer;
    source.connect(outputAnalyser);
    playback.add(source);
    const needsPrebuffer = !playback.size || outputCursor <= context.currentTime + playbackSchedulingMarginSeconds;
    const when = needsPrebuffer ? context.currentTime + playbackPrebufferSeconds : outputCursor;
    source.start(when);
    outputCursor = when + buffer.duration;
    source.onended = () => {
      playback.delete(source);
      source.disconnect();
      if (!playback.size && connected && state === 'speaking') {
        state = 'listening';
        updateUI();
      }
    };
    if (state !== 'speaking') { state = 'speaking'; updateUI(); }
  }

  async function startSession() {
    if (rehearsal) endRehearsal();
    if (!api) {
      showToast('Open the Windows desktop app to connect a local model. Visual rehearsal is available here.', true);
      return;
    }
    if (!config.bundlePath || !config.bridgePath) { openSettings(); return; }
    const token = ++generation;
    const sessionConfig = {...config};
    activeBundlePath = sessionConfig.bundlePath;
    starting = true;
    awaitingInitialBridge = true;
    lastError = '';
    muted = false;
    interruptPending = false;
    state = 'loading';
    updateUI('Connecting your microphone and loading the local model…');
    try {
      if (!await setupAudio(token) || token !== generation) return;
      clearTranscript();
      elapsedSeconds = 0;
      startedAt = null;
      updateTimer();
      await api.connect(sessionConfig);
      if (token !== generation) return;
      starting = false;
      awaitingInitialBridge = false;
      connected = true;
      updateUI();
    } catch (error) {
      if (token !== generation) return;
      generation += 1;
      starting = false;
      awaitingInitialBridge = false;
      connected = false;
      state = 'disconnected';
      lastError = error.name === 'NotAllowedError' ? 'Microphone access was denied. Allow microphone access in Windows privacy settings.' : error.message || String(error);
      await stopAudio();
      showToast(lastError, true);
      updateUI();
    }
  }

  async function disconnectSession() {
    if (rehearsal) { endRehearsal(); return; }
    generation += 1;
    starting = false;
    awaitingInitialBridge = false;
    connected = false;
    interruptPending = false;
    state = 'disconnected';
    updateTimer();
    startedAt = null;
    finishTranscripts();
    updateUI();
    // Dispatch native shutdown before awaiting device cleanup. A quick reconnect
    // must never be cancelled by an older disconnect sent after context.close().
    const stoppingBridge = Promise.resolve(api?.disconnect()).catch(error => { showToast(error.message || String(error), true); });
    await Promise.all([stopAudio(), stoppingBridge]);
  }

  function handleEvent(event) {
    if (!event || typeof event !== 'object') return;
    if (event.type === 'metrics') {
      if (Number.isFinite(event.latencyMs)) {
        dom.latencyMetric.querySelector('strong').textContent = `${Math.round(event.latencyMs)} ms`;
        dom.latencyMetric.title = 'Latency reported by the native runtime';
      }
      if (event.gpuName) {
        const shortName = event.gpuName.replace(/^NVIDIA\s+/, '').replace(/^GeForce\s+/, '');
        dom.gpuMetric.querySelector('strong').textContent = shortName;
        dom.gpuMetric.title = `${event.gpuName}${Number.isFinite(event.gpuMemoryMb) ? ` · ${(event.gpuMemoryMb / 1024).toFixed(1)} GB used` : ''}${Number.isFinite(event.gpuTotalMemoryMb) ? ` / ${(event.gpuTotalMemoryMb / 1024).toFixed(1)} GB total` : ''}${Number.isFinite(event.gpuUtilization) ? ` · ${event.gpuUtilization}% GPU utilization` : ''}`;
      }
      return;
    }
    if (rehearsal) return;
    // Native output already in the IPC pipe can arrive after a local stop click.
    // Keep it silent until the bridge's interruption barrier acknowledges it.
    if (interruptPending && (event.type === 'audio' || (event.type === 'transcript' && event.role === 'assistant'))) return;
    if (event.type === 'state') {
      // connect() first closes an old bridge. That initial event is not a failure.
      if (event.state === 'disconnected' && awaitingInitialBridge) return;
      if (event.state === 'loading') awaitingInitialBridge = false;
      state = event.state === 'ready' ? 'listening' : event.state;
      if (state === 'disconnected') {
        generation += 1;
        connected = false;
        starting = false;
        updateTimer();
        startedAt = null;
        finishTranscripts();
        void stopAudio();
      } else if (state !== 'loading') {
        connected = true;
        starting = false;
        if (startedAt === null) startedAt = performance.now();
      }
      // The runtime may finish generating while queued audio is still playing.
      if (state === 'listening' && playback.size) state = 'speaking';
      updateUI(lastError ? undefined : event.message);
    } else if (event.type === 'audio') {
      playAudio(event);
    } else if (event.type === 'transcript') {
      appendTranscript(event);
    } else if (event.type === 'flush') {
      if (['interrupt', 'reset', 'cancelled'].includes(event.reason)) interruptPending = false;
      flushPlayback();
      // Yield/interrupt cancels output while the same user utterance continues.
      const clearsInput = event.reason === 'reset' || event.reason === 'cancelled';
      finishTranscripts(clearsInput ? undefined : 'assistant');
      if (event.reason === 'reset') addNotice('Conversation refreshed. Earlier history was cleared.');
      if (connected && state !== 'loading') state = 'listening';
      updateUI();
    } else if (event.type === 'error') {
      interruptPending = false;
      lastError = event.message || 'The local runtime reported an error.';
      showToast(lastError, true);
      updateUI();
      if (event.fatal === true) void disconnectSession();
    } else if (event.type === 'context_rolled') {
      finishTranscripts();
      addNotice('Conversation refreshed. Earlier history was cleared.');
    }
  }

  function openSettings() {
    dom.bundlePath.value = config.bundlePath || '';
    dom.bridgePath.value = config.bridgePath || '';
    dom.systemPrompt.value = config.systemPrompt || '';
    dom.settingsDialog.showModal();
  }

  async function choosePath(kind) {
    const method = kind === 'bundlePath' ? 'chooseBundle' : 'chooseBridge';
    if (!api?.[method]) { showToast('Folder and file selection is available in the Windows desktop app.'); return; }
    try {
      const result = await api[method]();
      if (typeof result === 'string') dom[kind].value = result;
      else if (result && typeof result[kind] === 'string') dom[kind].value = result[kind];
    } catch (error) { showToast(error.message || String(error), true); }
  }

  function scheduleRehearsal(fn, delay) { rehearsalTimeouts.push(setTimeout(() => { if (rehearsal) fn(); }, delay)); }

  function sampleUtterance(role, text, startDelay, duration) {
    const words = text.split(' ');
    for (let i = 1; i <= words.length; i += 1) {
      scheduleRehearsal(() => {
        state = role === 'assistant' ? 'speaking' : 'listening';
        appendTranscript({role, text: words.slice(0, i).join(' '), final: i === words.length});
        updateUI();
      }, startDelay + duration * i / words.length);
    }
  }

  function startRehearsal() {
    if (connected || starting) {
      showToast('End the current conversation before starting a visual rehearsal.');
      return;
    }
    if (rehearsal) return;
    lastError = '';
    rehearsal = true;
    state = 'listening';
    clearTranscript();
    startedAt = performance.now();
    elapsedSeconds = 0;
    updateTimer();
    addNotice('VISUAL REHEARSAL · Sample conversation. No microphone capture, generated audio, or model inference.');
    dom.settingsDialog.close();
    updateUI();
    const runScript = () => {
      sampleUtterance('user', 'What could we build with a voice that runs locally?', 500, 2500);
      scheduleRehearsal(() => { state = 'thinking'; updateUI(); }, 3100);
      sampleUtterance('assistant', 'Imagine a creative copilot that listens, thinks, and speaks right on your RTX PC. A natural conversation, with your ideas at the center.', 4000, 7000);
      scheduleRehearsal(() => { state = 'listening'; updateUI(); }, 11500);
      sampleUtterance('user', 'And I can just jump in with a new idea?', 14500, 2400);
      scheduleRehearsal(() => { state = 'thinking'; updateUI(); }, 17000);
      sampleUtterance('assistant', 'Exactly. Speak freely. Interrupt, explore a tangent, or build on a thought. What do you want to create?', 18000, 5600);
      scheduleRehearsal(() => { state = 'listening'; updateUI(); }, 24000);
    };
    runScript();
  }

  function endRehearsal() {
    rehearsal = false;
    for (const timeout of rehearsalTimeouts) clearTimeout(timeout);
    rehearsalTimeouts = [];
    inputEnergy = 0;
    outputEnergy = 0;
    updateTimer();
    startedAt = null;
    state = 'disconnected';
    finishTranscripts();
    updateUI();
    dom.transcriptMode.textContent = 'SAMPLE SCRIPT · REHEARSAL ENDED';
    dom.transcriptFootnote.textContent = 'Sample transcript from visual rehearsal.';
  }

  function toggleMute() {
    if (!connected || rehearsal || starting) return;
    muted = !muted;
    inputEnergy = 0;
    // Continue sending silence while muted so the full-duplex clock stays aligned.
    updateUI();
  }

  async function interruptResponse() {
    if (dom.interruptButton.disabled || !api?.interrupt) return;
    const token = generation;
    interruptPending = true;
    // Cut both audible and scheduled PCM immediately; capture remains connected.
    flushPlayback();
    finishTranscripts('assistant');
    state = 'listening';
    updateUI();
    try {
      const result = await api.interrupt();
      if (token !== generation) return;
      // No native response may remain when its final PCM is still queued locally.
      if (result?.accepted === false) { interruptPending = false; updateUI(); }
    } catch (error) {
      if (token !== generation) return;
      interruptPending = false;
      showToast(error.message || String(error), true);
      updateUI();
    }
  }

  function setStreamMode(value) {
    streamMode = value;
    document.body.classList.toggle('stream-mode', value);
    dom.streamButton.setAttribute('aria-pressed', String(value));
    dom.streamExit.hidden = !value;
    if (!value) { dom.streamButton.focus(); dom.toast.hidden = true; }
    else showToast('Stream mode is on. Press I to stop a response; Escape brings back the controls.');
  }

  async function toggleFullscreen() {
    try {
      if (api?.setFullscreen) {
        fullscreen = !fullscreen;
        await api.setFullscreen(fullscreen);
      } else if (document.fullscreenElement) await document.exitFullscreen();
      else await document.documentElement.requestFullscreen();
    } catch (error) { showToast(error.message || String(error), true); }
  }

  $('settingsButton').addEventListener('click', openSettings);
  $('railSettings').addEventListener('click', openSettings);
  $('closeSettings').addEventListener('click', () => dom.settingsDialog.close());
  $('browseBundle').addEventListener('click', () => { void choosePath('bundlePath'); });
  $('browseBridge').addEventListener('click', () => { void choosePath('bridgePath'); });
  $('settingsForm').addEventListener('submit', event => {
    event.preventDefault();
    config = {...config, bundlePath: dom.bundlePath.value.trim(), bridgePath: dom.bridgePath.value.trim(), systemPrompt: dom.systemPrompt.value.trim()};
    try { localStorage.setItem('voiceLabSettings', JSON.stringify(config)); } catch {}
    dom.settingsDialog.close();
    updateUI();
    showToast(connected ? 'Settings saved for your next conversation.' : 'Settings saved. Start a conversation when you’re ready.');
  });
  dom.connectButton.addEventListener('click', () => {
    if (connected || starting || rehearsal || state === 'loading') void disconnectSession();
    else void startSession();
  });
  dom.muteButton.addEventListener('click', toggleMute);
  dom.interruptButton.addEventListener('click', () => { void interruptResponse(); });
  dom.rehearsalButton.addEventListener('click', startRehearsal);
  $('settingsRehearsal').addEventListener('click', startRehearsal);
  $('clearTranscript').addEventListener('click', () => {
    clearTranscript();
    if (rehearsal) addNotice('VISUAL REHEARSAL · Sample script. No live model inference.');
    showToast('Displayed transcript cleared. The current model context is unchanged.');
  });
  dom.streamButton.addEventListener('click', () => setStreamMode(!streamMode));
  dom.streamExit.addEventListener('click', () => setStreamMode(false));
  $('fullscreenButton').addEventListener('click', () => { void toggleFullscreen(); });
  dom.toast.addEventListener('click', () => { dom.toast.hidden = true; });
  document.querySelector('.brand-mark').addEventListener('click', event => event.preventDefault());
  document.addEventListener('keydown', event => {
    const typing = event.target.matches('input,textarea,select,[contenteditable="true"]');
    if (event.code === 'KeyI' && !typing && !dom.settingsDialog.open && !event.ctrlKey && !event.altKey && !event.metaKey) {
      event.preventDefault();
      if (!event.repeat) void interruptResponse();
    }
    if (event.code === 'Space' && !typing && !dom.settingsDialog.open && !event.target.closest('button,a') && connected) {
      event.preventDefault();
      if (!event.repeat) toggleMute();
    }
    if (event.key === 'Escape' && streamMode && !dom.settingsDialog.open) setStreamMode(false);
  });
  window.addEventListener('beforeunload', () => {
    if (microphone) microphone.getTracks().forEach(track => track.stop());
    flushPlayback();
  });

  // This particle field is drawn locally; no remote fonts, images, or animation APIs.
  const canvas = dom.orbCanvas;
  const ctx = canvas.getContext('2d', {alpha: true});
  let width = 1, height = 1;
  let smoothInput = 0, smoothOutput = 0;
  let previousFrame = 0;
  let labelFrame = 0;
  const points = [];
  let randomSeed = 771;
  const random = () => { randomSeed = (randomSeed * 16807) % 2147483647; return (randomSeed - 1) / 2147483646; };
  for (let i = 0; i < 2100; i += 1) points.push({a: random() * Math.PI * 2, b: random() * Math.PI * 2, thickness: random(), alpha: .15 + random() * .75, size: .35 + random() * 1.2, drift: random()});
  const stars = Array.from({length: 65}, () => ({x: random(), y: random(), alpha: random(), size: random()}));
  const sprite = document.createElement('canvas');
  sprite.width = sprite.height = 32;
  const spriteCtx = sprite.getContext('2d');
  const spriteGradient = spriteCtx.createRadialGradient(16, 16, 0, 16, 16, 16);
  spriteGradient.addColorStop(0, '#eaffc9'); spriteGradient.addColorStop(.15, '#b8ff60'); spriteGradient.addColorStop(.4, '#8ced3260'); spriteGradient.addColorStop(1, '#8ced3200');
  spriteCtx.fillStyle = spriteGradient; spriteCtx.fillRect(0, 0, 32, 32);
  new ResizeObserver(entries => {
    const rect = entries[0].contentRect;
    width = rect.width; height = rect.height;
    const scale = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(width * scale); canvas.height = Math.round(height * scale);
    ctx.setTransform(scale, 0, 0, scale, 0, 0);
  }).observe(dom.visualizerWrap);

  function drawFrame(timestamp) {
    requestAnimationFrame(drawFrame);
    if (document.hidden || (reducedMotion.matches && timestamp - previousFrame < 160)) return;
    const dt = Math.min(.06, (timestamp - previousFrame) / 1000 || .016);
    previousFrame = timestamp;
    visualPhase += reducedMotion.matches ? 0 : dt;
    const t = visualPhase;
    if (rehearsal) {
      inputEnergy = state === 'listening' ? .13 + .25 * Math.pow(Math.sin(t * 3.2), 2) : .015;
      outputEnergy = state === 'speaking' ? .25 + .35 * Math.pow(Math.sin(t * 4.1), 2) : 0;
    } else if (outputAnalyser && outputData) {
      outputAnalyser.getFloatTimeDomainData(outputData);
      let energy = 0;
      for (const value of outputData) energy += value * value;
      outputEnergy = Math.min(1, Math.sqrt(energy / outputData.length) * 5);
    } else outputEnergy = 0;
    smoothInput += (inputEnergy - smoothInput) * Math.min(1, dt * 8);
    smoothOutput += (outputEnergy - smoothOutput) * Math.min(1, dt * 8);
    const energy = Math.max(smoothInput, smoothOutput);
    const active = connected || rehearsal;
    const thinking = state === 'thinking' || state === 'loading';
    const pulse = Math.sin(t * 1.2) * .014;
    const radius = Math.min(width * .28, height * .405, 224) * (1 + pulse + energy * .095);
    const cx = width / 2, cy = height / 2;
    ctx.clearRect(0, 0, width, height);
    const atmosphere = ctx.createRadialGradient(cx, cy, radius * .15, cx, cy, radius * 1.8);
    atmosphere.addColorStop(0, '#81d14200');
    atmosphere.addColorStop(.42, active ? '#73d91b0a' : '#73d91b05');
    atmosphere.addColorStop(.58, active ? '#8af43610' : '#8af4360a');
    atmosphere.addColorStop(1, '#73d91b00');
    ctx.fillStyle = atmosphere; ctx.fillRect(0, 0, width, height);
    for (const star of stars) {
      ctx.fillStyle = `rgba(168,218,121,${(.06 + star.alpha * .15) * (.6 + Math.sin(t * .3 + star.alpha * 7) * .4)})`;
      ctx.fillRect(star.x * width, star.y * height, star.size > .8 ? 1.5 : .8, star.size > .8 ? 1.5 : .8);
    }
    // Instrument markings stay fine and restrained so the living ring leads.
    ctx.strokeStyle = '#77985417'; ctx.lineWidth = .65;
    ctx.beginPath(); ctx.arc(cx, cy, radius * 1.26, 0, Math.PI * 2); ctx.stroke();
    for (let i = 0; i < 80; i += 1) {
      const angle = i / 80 * Math.PI * 2;
      const tick = i % 10 === 0 ? 5 : 2;
      ctx.strokeStyle = i % 10 === 0 ? '#9abb7935' : '#7798541c';
      ctx.beginPath(); ctx.moveTo(cx + Math.cos(angle) * (radius * 1.26 - tick), cy + Math.sin(angle) * (radius * 1.26 - tick));
      ctx.lineTo(cx + Math.cos(angle) * radius * 1.26, cy + Math.sin(angle) * radius * 1.26); ctx.stroke();
    }
    const outerGlow = ctx.createRadialGradient(cx, cy, radius * .72, cx, cy, radius * 1.19);
    outerGlow.addColorStop(0, '#93ff3600'); outerGlow.addColorStop(.45, '#9afd4215'); outerGlow.addColorStop(.64, '#a8ff5220'); outerGlow.addColorStop(1, '#93ff3600');
    ctx.fillStyle = outerGlow; ctx.beginPath(); ctx.arc(cx, cy, radius * 1.2, 0, Math.PI * 2); ctx.fill();
    ctx.globalCompositeOperation = 'lighter';
    for (let ring = 0; ring < 9; ring += 1) {
      ctx.beginPath();
      for (let step = 0; step <= 230; step += 1) {
        const angle = step / 230 * Math.PI * 2;
        const ripple = Math.sin(angle * 4 + t * .65 + ring * .53) * .023 + Math.sin(angle * 9 - t * .4 + ring) * .012;
        const voice = energy * .065 * Math.sin(angle * 13 + t * 6 + ring);
        const r = radius * (.91 + ring * .012 + ripple + voice);
        const x = cx + Math.cos(angle) * r;
        const y = cy + Math.sin(angle) * r * (.92 + Math.sin(t * .12) * .04);
        if (!step) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = `rgba(${140 + ring * 8},245,${65 + ring * 9},${.04 + ring * .009 + energy * .045})`;
      ctx.lineWidth = ring === 5 ? 1.25 : .6; ctx.stroke();
    }
    const rotation = t * (thinking ? .3 : .085);
    for (const point of points) {
      const a = point.a + rotation * (.5 + point.drift * .5);
      const b = point.b + t * .19;
      const ripple = Math.sin(a * 3 + t * .55) * .034 + Math.cos(a * 7 - t * .35) * .018;
      const thickness = .075 + point.thickness * .065 + energy * .1;
      const r = radius * (1 + Math.cos(b) * thickness + ripple);
      const x = cx + Math.cos(a) * r;
      const y = cy + Math.sin(a) * r * .94 + Math.sin(b) * radius * .057;
      const depth = (Math.sin(b) + 1) / 2;
      const bright = .24 + depth * .7;
      const size = point.size * (depth * .6 + .6) * (1 + energy * .5);
      ctx.globalAlpha = point.alpha * bright * (active ? .95 : .75);
      if (point.size > 1.25 && depth > .65) ctx.drawImage(sprite, x - size * 3, y - size * 3, size * 6, size * 6);
      else { ctx.fillStyle = depth > .7 ? '#d0ffa3' : '#87dc38'; ctx.fillRect(x, y, size, size); }
    }
    ctx.globalAlpha = 1;
    // Two travelling sparks trace the outer instrument ring.
    for (let i = 0; i < 2; i += 1) {
      const angle = t * .13 + i * Math.PI + .7;
      const x = cx + Math.cos(angle) * radius * 1.26, y = cy + Math.sin(angle) * radius * 1.26;
      ctx.drawImage(sprite, x - 6, y - 6, 12, 12);
    }
    ctx.globalCompositeOperation = 'source-over';
    const centerShade = ctx.createRadialGradient(cx, cy, 0, cx, cy, radius * .8);
    centerShade.addColorStop(0, '#090e09bd'); centerShade.addColorStop(.6, '#0a0e0990'); centerShade.addColorStop(1, '#0a0e0900');
    ctx.fillStyle = centerShade; ctx.beginPath(); ctx.arc(cx, cy, radius * .8, 0, Math.PI * 2); ctx.fill();
    if (timestamp - labelFrame > 100) {
      labelFrame = timestamp;
      dom.inputLevel.textContent = rehearsal ? 'SIMULATED' : !connected ? 'STANDBY' : muted ? 'MUTED' : smoothInput > .065 ? 'RECEIVING' : 'LISTENING';
      dom.outputLevel.textContent = rehearsal ? 'SIMULATED' : !connected ? 'STANDBY' : playback.size ? 'SPEAKING' : 'READY';
      const bars = document.querySelectorAll('.orb-monogram i');
      bars.forEach((bar, index) => {
        const base = [13, 24, 38, 24, 13][index];
        bar.style.height = `${base + energy * 30 * (.3 + Math.abs(Math.sin(t * 7 + index * 1.1)))}px`;
      });
    }
  }
  requestAnimationFrame(drawFrame);
  setInterval(updateTimer, 1000);
  updateUI();

  async function initialize() {
    try {
      let saved = {};
      try { saved = JSON.parse(localStorage.getItem('voiceLabSettings') || '{}'); } catch {}
      config = {...config, ...saved};
      if (api) {
        api.onEvent(handleEvent);
        config = {...config, ...await api.getConfig(), ...saved};
        const status = await api.getStatus();
        if (status?.state && status.state !== 'disconnected') {
          // Reloading the renderer loses the capture clock; reopen a clean session.
          await api.disconnect();
        }
      }
      updateUI();
    } catch (error) { showToast(`Could not load settings: ${error.message || error}`, true); }
  }
  void initialize();
})();

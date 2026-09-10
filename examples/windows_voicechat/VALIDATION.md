# Windows validation record

This is a source example, with locally built application evidence. No third-party binaries, model weights, test audio, screenshots, or private runtime receipts are included in its distribution.

## Tested environment

- Windows x64, RTX 5090 (32 GB VRAM), 128 GB system RAM, NVIDIA driver 591.86.
- MSVC 19.44, CUDA 13.4, TensorRT-RTX 1.6.1.120, CPython 3.12, Electron 44.3.0.
- Runtime implementation based on PR #1218 revision `7458623963a038fa1ae1f1bac28eee6a5c792514` plus the Windows example and recovery changes in this PR.
- Checkpoint revision `359ada7b1c60851e40ff08065f9b0340244f27e0`; tokenizer revision `6533e8de2c68e4536bf7c411d7a3ce5734111476`.
- Checkpoint size 44,382,749,892 bytes, SHA256 `d553750c29434a6bb524377e17634c6cafdbf621892e643a77f406e51570354b`. Locally compiled RTX bundle: 18,005,370,305 bytes.

## Source and deployment checks

- Desktop unit suite: 23 passed, including relative installation paths, fresh source discovery, relocation, bounded diagnostics, protocol validation, lifecycle races, and interruption.
- Model-free Electron audio integration: passed using downloaded Electron as Node and as the desktop executable. Real WebAudio/preload/IPC delivered 320-sample 16 kHz packets every 20 ms; reset acknowledgments, late-output suppression, microphone continuity, playback flush, mute, and resource cleanup passed.
- Transcript integration: passed 1,034 events with exactly 300 retained rows/notices, correct active partial updates, and recovery after row eviction.
- Python source audit: prospective and index checks reject binaries, models, archives, vendored dependencies, encoded media, oversized files, and force-added ignored artifacts. Negative cases were verified in isolated temporary Git repositories.
- Windows PowerShell scripts are parsed separately from execution. Setup's Python stage was exercised in an isolated directory: verified CPython download, extraction, venv creation, and SSL/venv imports. Existing-environment plan, local app assembly, and launcher path checks also passed.

## Native and model checks

- MSVC build of core, runtime, TensorRT-RTX backend, Nemotron family, and bridge completed. Ten CTests passed, including real RTX dynamic-input inference, DLL/bundle loading, session state, conversation memory, streaming mel continuity, codec reconstruction, and PCM protocol.
- Four native startup/help/rejection checks passed with PATH restricted to Windows system directories, including Unicode paths and rejection of a standard TensorRT bundle.
- Eighteen focused family build-policy/quantization tests passed. Actual RTX W8A8 graph output matched an independent NumPy quantized reference with maximum absolute error `1.7881393432617188e-7`; silence was exactly zero.
- The full bundle was compiled locally through TensorRT-RTX and used by the native bridge and desktop tests. A clean recorded question produced the correct answer about Paris through renderer, production IPC, native inference, and WebAudio playback.

## Continuous conversation and recovery

The measured final session ran **540.028 seconds of continuous capture**, with **12 topic checks**, **six actual context refreshes**, one spoken interruption, and one button interruption. Two replies were deliberately interrupted after establishing their topic; ten completed. Five refreshes were age-based and one recovered speech at a response boundary. The test used one continuously connected model process and never called offline `finish_input()` to drain audio.

| Measurement | Result |
| --- | --- |
| Spoken interruption to old audible speech stopping | 803 ms |
| Stop button to active playback stopping | 26 ms after click dispatch |
| Stop button to completed native reset acknowledgment | 60 ms |
| Maximum scheduled audio ahead of playback | 0.941333 s; did not accumulate |
| Microphone packets | 27,008, each 320 samples at 16 kHz |
| Packet gap p99 / maximum | 23 ms / 55 ms |
| Native private memory after warmup | Approximately 16.786–16.823 GiB |
| Runtime / renderer errors | Zero |

The replacement arithmetic request and subsequent unrelated topics received new answers. The old unwanted story did not return after topic changes or context refresh. A separate focused Stop recovery test passed after 45 seconds of accumulated context, with correct follow-up answers about Egypt and weekdays, acknowledgment in 61 ms, and maximum playback lead 0.898667 s.

Three fixes underpin this result: 20 ms capture avoids racing the native 80 ms missing-input clock; refresh stops reinjecting old assistant replies; and a family-owned conversation-reset barrier clears dialogue/generation while preserving queued microphone PCM and continuous acoustic state. General full-reset/cancel APIs keep their existing behavior. The frontend regression checks 1,200 rebases over 100 model frames against uninterrupted processing with bitwise equality and bounded buffers. Repetition detection keeps a bounded history and permits only one automatic retry per request.

## Reproduce optional integration checks

Run from this directory after native setup. A separate Node.js installation is convenient for optional harnesses. Install Playwright locally (for example, `npm install --prefix <outside-source-test-directory> playwright`) and point `PLAYWRIGHT_MODULE` at its installed module. It is a test dependency, not part of the application. Tests launch downloaded Electron, so Playwright browser downloads are unnecessary.

```powershell
$env:VOICE_LAB_WORKSPACE = $workspace
$env:ELECTRON_EXECUTABLE = "$workspace\dependencies\electron\electron.exe"
$env:PLAYWRIGHT_MODULE = '<outside-source-test-directory>\node_modules\playwright'
node desktop/tests/electron-audio.integration.cjs
node desktop/tests/electron-transcript.integration.cjs

# Synthesize test microphone input with the local Windows speech service.
.\native\New-SoakFixtures.ps1 -OutputDirectory "$workspace\logs\voice-soak-fixtures"
node desktop/tests/electron-cancel-recovery.integration.cjs
node desktop/tests/electron-conversation-soak.integration.cjs
```

The latter two tests run real GPU inference. Stop other model sessions first and leave sufficient GPU capacity. Only microphone input is synthesized; every response comes from the model through production IPC. Harnesses retain strict interruption, semantic topic, duration, context-refresh, and two-second playback-lead criteria. Relative audio paths resolve beside the fixture manifest; `VOICE_LAB_SOAK_MANIFEST` can select another manifest. Receipts are written under the workspace's ignored `logs` directory.

For the shorter Paris question harness, set `VOICE_LAB_AUDIO_FIXTURE` to a locally recorded or synthesized WAV asking for France's capital, then run `node desktop/tests/electron-real-model.integration.cjs`. `VOICE_LAB_FIXTURE_REPEATS=3` repeats it 28 seconds apart to check playback lead across turns. These audio fixtures are generated or supplied locally, not shipped.

## Limits

The full five-stage bootstrap has not been repeated on a freshly installed Windows machine. Python provisioning and source/layout checks are distinct from complete clean-machine, toolchain, dependency-resolution, and model-build qualification. The new CI workflow checks source/JavaScript/PowerShell behavior without a GPU; it does not replace native or model testing.

Nine minutes is the measured duration, not empirical proof of unlimited operation. Bounded state permits further refreshes, but model answer accuracy, acoustic echo cancellation, lower-memory GPUs, other drivers/architectures, and heavy simultaneous gaming or livestream workloads remain unqualified. Earlier attempts with another model process or a busy game filled the native input queue; the final passing run had available GPU capacity and exactly one model runtime. An overloaded GPU cannot be made real-time by increasing queues.

Rehearsal mode and fake-microphone audio tests establish UI/transport behavior only. The full-model results above were obtained before source-only packaging and path portability refinements; those refinements received separate unit and model-free integration checks. No new full-model qualification is implied for unrelated environments.

# Nemotron Voice Lab for Windows

A local desktop voice application using **TensorRT-RTX** for model compilation and inference. It demonstrates an animated audio-reactive particle halo, streaming transcripts, microphone controls, interruption, fullscreen, GPU telemetry, and a stream mode for live presentations. Labeled visual rehearsal works before downloading the model.

This example builds on [PR #1218](https://github.com/NVIDIA/TensorRT-Model-Connect/pull/1218). Use the complete example PR checkout while that dependency is unmerged; copying this directory into an older checkout omits required runtime changes.

## Source-only distribution

The example contains application source, the C++ bridge, setup scripts, tests, and documentation. **It ships no third-party software or model weights.** There are no checked-in Electron binaries, NVIDIA or Microsoft DLLs, Python environments, npm packages, SDK headers, bundles, or downloaded media. Visuals use canvas, CSS, and inline SVG.

Setup downloads and installs dependencies on the user's machine, outside the source checkout. `Package-App.ps1` assembles a local installation using downloaded Electron; its output is not a source artifact or a redistributable release ZIP. Do not upload the workspace, `dependencies`, `runtime`, `models`, or the assembled application with this example. The upstream repository's existing third-party files are outside this example's distribution scope.

## Requirements

- Windows x64 with Windows PowerShell 5.1 or later, `curl.exe`, and `tar.exe`.
- An NVIDIA RTX GPU and driver compatible with TensorRT-RTX 1.6.1 / CUDA 13.4. The tested configuration is RTX 5090, 32 GB VRAM, driver 591.86. Smaller GPUs and other architectures have not been qualified by this example.
- Substantial system RAM and disk space: 128 GB RAM was tested. Reserve at least 120 GB free disk space for downloads and builds. The checkpoint is 44.4 GB and the compiled bundle approximately 18 GB; peak requirements vary.
- Internet for setup, a microphone, and headphones. Inference runs locally after setup. Leave GPU capacity available for the model during streaming.
- Administrator PowerShell for native setup if Microsoft C++ Build Tools and the Windows SDK are missing. An existing toolchain is reused.

This is a Windows native workflow. The repository's Linux/ALSA Docker example is a separate application and does not build this desktop interface.

## Build and run

Clone or extract the **complete repository at the example PR revision**. Run these commands from the repository root. No preinstalled Python or Node.js is required. Choose a writable output directory outside the checkout:

```powershell
$workspace = Join-Path $env:LOCALAPPDATA 'TRTMC-VoiceLab'

# Inspect stages without changing the machine.
.\examples\windows_voicechat\Setup.ps1 -WorkspaceRoot $workspace -Plan

# Download dependencies, compile the native runtime/model, and install locally.
.\examples\windows_voicechat\Setup.ps1 -WorkspaceRoot $workspace

# Start the desktop application.
.\examples\windows_voicechat\Run.ps1 -WorkspaceRoot $workspace
```

If execution policy blocks a reviewed downloaded script, invoke it with `powershell.exe -NoProfile -ExecutionPolicy Bypass -File <script-path>` and the same arguments. This changes policy only for that process. Initial downloads and compilation can take considerable time; setup reports the active stage. Install the NVIDIA display driver before setup.

The five stages run in dependency order: `Python`, `Dependencies`, `Native`, `Model`, and `App`. Reruns reuse the virtual environment, verified downloads, and existing bundle. Use `-RebuildBundle` after changing graph code or TensorRT-RTX version. The bridge rejects bundles using the standard TensorRT backend.

```powershell
# Preview visuals without the model or native build tools.
.\examples\windows_voicechat\Setup.ps1 -WorkspaceRoot $workspace -Stage App
.\examples\windows_voicechat\Run.ps1 -WorkspaceRoot $workspace -Rehearsal

# Rebuild native code, retaining installed Python packages.
.\examples\windows_voicechat\Setup.ps1 -WorkspaceRoot $workspace -Stage Native

# Recompile the model and refresh desktop source.
.\examples\windows_voicechat\Setup.ps1 -WorkspaceRoot $workspace -Stage Model,App -RebuildBundle

# Check installation paths without launching.
.\examples\windows_voicechat\Run.ps1 -WorkspaceRoot $workspace -ValidateOnly
```

`-Rehearsal` requires only the assembled app; select **Visual rehearsal** after it opens. Combine `-Rehearsal -ValidateOnly` to check this preview installation without launching. Voice conversation requires all five stages. Pass `-Python <python.exe>` to use existing CPython 3.12 x64 when creating the virtual environment. `-UsePortableWindowsSdk` supports an existing C++ toolchain needing the separately downloaded SDK. Native tests run by default; `-SkipTests` omits them and does not constitute a validated build.

The bundle command uses `--backend trt_rtx --precision fp32 --quantization int8 --max-sequence-length 512`. The `fp32` flag preserves the graph's tensor contract; the family's explicit mixed precision policy uses W8A8 for selected Thinker matrices, higher precision for sensitive projections, and FP16 TTS linear layers.

## Use the application

1. Connect a microphone and headphones. Enable Windows microphone access.
2. Open settings to adjust the system prompt or select a local RTX bundle. Default bridge and bundle paths are filled automatically.
3. Select **Start conversation**. The loading state remains visible until the native session is ready; first load includes RTX specialization.
4. Speak naturally. **Stop speaking**, or **I**, flushes playback and clears conversation context while microphone capture continues. Spoken barge-in follows the model's yield decision.
5. Select **Stream mode** and capture the window in your streaming software. **Esc** restores controls. **Space** toggles mute outside text fields.

Context refresh forgets earlier dialogue and carries only the latest unanswered request when needed. Repetition recovery is bounded and waits for fresh speech if its one retry fails. The transcript retains at most 300 rows and refresh notices. Model state, audio queues, history, and diagnostics have fixed bounds. This permits continued conversation through refreshes; the measured continuous test is nine minutes, not proof of unlimited conversational accuracy.

Capture uses 20 ms packets to stay ahead of the native 80 ms missing-input clock. Playback starts with a 160 ms cushion. Headphones help avoid feedback; browser echo cancellation is requested, but acoustic echo cancellation has not been qualified.

## Local files and dependency provenance

```text
<workspace>/
  dependencies/                  Downloaded Python, SDKs, build tools, Electron
  runtime/                       Locally built bridge and installed runtime DLLs
  models/
    Nemotron-VoiceChat-11B/       Downloaded pinned checkpoint
    huggingface/                 Downloaded pinned tokenizer cache
    nemotron-voicechat-rtx.bundle Locally compiled multi-engine RTX bundle
  Nemotron Voice Lab/            Locally assembled desktop application
  logs/                          Diagnostics and optional test receipts
  voice-lab-config.json           Local settings
  Start Voice Lab.ps1             Launcher relative to this workspace
```

Internal paths are saved relative to the workspace so settings follow a moved installation. External model paths remain absolute. Build environments and engine caches are not portable deployment artifacts; rebuild or requalify on the target machine. `VOICE_LAB_WORKSPACE` explicitly selects an installation when launching desktop source.

| Component | Installed from | Version / verification |
| --- | --- | --- |
| CPython | Astral python-build-standalone GitHub release | 3.12.14, SHA256 pinned |
| Python build packages | PyPI | Versions in `requirements-windows.txt` |
| MSVC / Windows SDK | Microsoft | Existing installation or signed VS 2022 bootstrapper; optional SDK hashes pinned |
| CUDA components | NVIDIA redistributable service | 13.4.1 manifest SHA256 pinned; component SHA256 checked against the manifest |
| TensorRT-RTX SDK | NVIDIA | 1.6.1.120, archive SHA256 pinned |
| nlohmann JSON | Upstream GitHub release | 3.12.0, archive SHA256 pinned |
| Electron | Upstream GitHub release | 44.3.0, archive SHA256 pinned |
| Model / tokenizers | Hugging Face NVIDIA repositories | Immutable revisions in `SOURCE.json`; checkpoint SHA256 checked |

Review upstream terms for downloaded components. Local runtime assembly preserves vendor notices. MSVC's signed bootstrapper selects current supported components, and pip resolves transitive Python dependencies. This is not a bit-for-bit reproducible lockfile or a third-party redistribution license grant.

Transcript text stays in memory. Diagnostics store lifecycle events, refresh reasons, and session-keyed text fingerprints without audio or transcript content, bounded to two 2 MB files. Settings and Chromium state persist locally.

## Verify and contribute

With Node.js installed, run `node --test examples/windows_voicechat/desktop/tests/*.test.js` from the repository root. Alternatively, use downloaded Electron:

```powershell
$savedElectronMode = $env:ELECTRON_RUN_AS_NODE
try {
    $env:ELECTRON_RUN_AS_NODE = '1'
    & "$workspace\dependencies\electron\electron.exe" --test './examples/windows_voicechat/desktop/tests/*.test.js' | Out-Host
    if ($LASTEXITCODE -ne 0) { throw 'Desktop unit tests failed.' }
} finally { $env:ELECTRON_RUN_AS_NODE = $savedElectronMode }

# Inspect actual Git index bytes before committing or archiving source.
& "$workspace\dependencies\python\Scripts\python.exe" examples/windows_voicechat/audit_source.py --staged
```

The audit checks example files for binary payloads, dependency directories, model/media files, symlinks, and oversized files, including force-added ignored files. Export only audited tracked example source when preparing an example-only ZIP; a whole upstream repository archive also includes its existing third-party files. The CI source check runs desktop unit tests and parses PowerShell scripts; it does not qualify GPU inference or fresh installation.

See [VALIDATION.md](VALIDATION.md) for optional integration tests, full-model measurements, and remaining gaps. The renderer has no Node access and uses a sandboxed preload and local resources. The bridge uses bounded NDJSON with 16 kHz microphone input and 48 kHz output; see [native/PROTOCOL.md](native/PROTOCOL.md). Model-specific recovery stays in `families/nemotron_voicechat`; shared Windows loader changes remain model-agnostic.

---
title: Windows FastVideo MiniMax H3 VSA
description: Install and run the pinned single-GPU FastVideo VSA reproduction profile on Windows.
---

This is an experimental reproduction helper for the public
[FastVideo](https://github.com/hao-ai-lab/FastVideo) MiniMax H3 VSA path on
Windows 11 Arm64 with x64 process emulation. It uses PyTorch, Triton-Windows,
and the public FastH3 VSA adapter. It is separate from the
[native TensorRT-RTX MiniMax H3 path](windows-native-h3.md) and does not add a
VSA backend to the TensorRT Model Connect runtime.

The helper deliberately fixes one narrow cohort:

- one Windows 11 Arm64 computer with an x64 PowerShell/Python process and one
  compute-capability 12.1 NVIDIA GPU;
- Python 3.13, CUDA 13 PyTorch, Triton-Windows, and the pinned CUDA 12.9.86
  `ptxas` required to assemble SM121 Triton kernels;
- the public MiniMax H3 base checkpoint and FastH3 VSA/Data-Free adapter at
  exact revisions;
- 1344x768 output at 24 fps, with either 124 or 345 frames;
- five scheduler grid points, which perform exactly four DiT forwards;
- 90% VSA sparsity with tile size 64 and the Triton sparse kernel;
- full MiniMax H3 video-VAE decoding; and
- one generation, with no warmup and no repeated request.

The exact public revisions, package versions, hashes, and workload are in
`tests/e2e/models/minimax_h3/fastvideo_windows_vsa_profile.json`. The scripts
read that profile rather than maintaining a second set of defaults.

:::caution External assets and licenses

The installer downloads source code and Python packages, but it does not
download checkpoints or accept licenses. Before running generation, review and
accept the
[MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/48d93ede732756e404a3b1b2f3b3a9b5a22f6cfc/LICENSE)
and the
[FastH3 adapter license](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA/blob/bcf40ca6f457ed66f8badf13514943e390205fca/LICENSE)
through the providers' normal access flow.

This repository does not redistribute model or adapter weights, virtual
environments, generated media, or authentication credentials.

:::

## Prerequisites

Use an up-to-date NVIDIA display driver with CUDA 13 support. Use Windows 11
Arm64 with an x64 PowerShell process. The pinned x64 Python, Triton-Windows,
CUDA toolchain, and helper programs run through Windows' x64 emulation layer;
native Arm64 Python is outside this reproduction cohort. Install these public
tools before starting:

- Windows 11 Arm64;
- Git for Windows;
- an x64 Python 3.13.5 launcher (`py -3.13`); and
- enough local disk and system commit for the MiniMax H3 snapshot, the 5.34 GB
  adapter, compiled-kernel caches, and generated media.

Close other GPU and unified-memory workloads before a measured run. The
installer is per-user, requires no administrator elevation, and does not add
the environment to the system `PATH`.

## Install

For the double-click flow, download or clone this repository and open:

```text
scripts\install_windows_h3_fastvideo_vsa.cmd
```

The window remains open so an installation error is visible. For an auditable
PowerShell invocation from the repository root, run:

```powershell
& .\scripts\install_windows_h3_fastvideo_vsa.ps1
```

To keep the environment on another volume, pass a directory outside the source
checkout:

```powershell
$InstallRoot = '<per-user-install-directory>'
& .\scripts\install_windows_h3_fastvideo_vsa.ps1 -InstallRoot $InstallRoot
```

The installer performs these checks before changing the selected directory:

1. reads the checked-in reproduction profile;
2. verifies the checked-in FastVideo patch SHA-256;
3. fetches only the exact public FastVideo commit;
4. checks and applies the patch against that commit;
5. verifies that the patch changed only its declared FastVideo paths;
6. downloads and verifies NVIDIA's CUDA 12.9.1 NVCC redistributable;
7. verifies the extracted `ptxas.exe` size, SHA-256, build version, and the
   exact path selected by Triton;
8. creates a Python virtual environment; and
9. installs the pinned inference dependencies and imports the checked-in
   FastVideo Triton VSA and index kernels on CUDA.

Windows uses the pure-Python Triton kernels from the pinned FastVideo source
checkout. It does not install FastVideo's Linux CUDA-extension wheel, FA4, or a
locally built binary kernel package.

It records a sanitized receipt inside the installation. If an incomplete or
different installation already occupies that directory, it fails rather than
resetting or overwriting it.

## Authorize Hugging Face access

Accept both model licenses in the browser, then authenticate the installed
Hugging Face client. Tokens are handled by the Hugging Face client and are not
accepted as script arguments or written to benchmark summaries.

```powershell
if (-not $InstallRoot) {
    $InstallRoot = Join-Path `
        ([Environment]::GetFolderPath('LocalApplicationData')) `
        'TensorRT-Model-Connect\minimax-h3-fastvideo-vsa-sm121-v1'
}
$Hf = Join-Path $InstallRoot '.venv\Scripts\hf.exe'
& $Hf auth login
```

## Generate one 124-frame video

Choose an output directory outside the repository. The runner resolves the
exact base-model snapshot and one adapter file through the authenticated client,
then verifies the adapter's public byte size and SHA-256 before model loading.

```powershell
$OutputDirectory = '<output-directory-outside-checkout>'

& .\scripts\run_windows_h3_fastvideo_vsa.ps1 `
    -InstallRoot $InstallRoot `
    -OutputDirectory $OutputDirectory `
    -NumFrames 124
```

The checked-in public prompt is used by default. Pass another JSON file with a
non-empty `prompt` field through `-PromptFile` when needed. A custom prompt does
not change the fixed model, scheduler, VSA, or decoder profile.

## Generate the long 345-frame profile

MiniMax H3's released latent grid accepts frame counts of the form `17n + 5`.
The checked-in long profile therefore uses 345 frames, the largest supported
aligned value below the 15-second boundary. It does not use 360 or 362 frames.

```powershell
& .\scripts\run_windows_h3_fastvideo_vsa.ps1 `
    -InstallRoot $InstallRoot `
    -OutputDirectory $OutputDirectory `
    -NumFrames 345
```

The patch uses a no-grad in-place gate merge for inference. This reuses the
fresh sparse-attention output instead of allocating two additional full-length
345-frame tensors. Grad-enabled training keeps the original out-of-place merge
so saved tensors remain valid for backward. The runner also pins Triton to the
verified CUDA 12.9.86 `ptxas`; the older assembler bundled with the validated
Triton-Windows wheel cannot target SM121.

The runner uses the platform-default CUDA allocator. The validation run had
requested `expandable_segments:True`, but PyTorch reported that expandable
segments are unsupported on this Windows platform, so the option was not
enabled and is not part of the reproduction profile.

## Timing and evidence boundary

The launcher makes exactly one call to FastVideo's public generation entrypoint
with `--no-warmup --repeats 1`. FastVideo reports its internal end-to-end time
for that one generation. The wrapper also records process wall time, which
includes Python startup, model construction, compilation, and teardown but not
the preceding Hugging Face downloads.

The output directory receives the generated MP4 and
`fastvideo-vsa-summary.json`. The summary records public revisions and hashes,
shape, frame count, the five-grid-point/four-forward contract, decoder choice,
exit status, and timing. It does not contain tokens, host names, GPU UUIDs, or
resolved local paths.

The checked-in sanitized validation record is
`tests/e2e/models/minimax_h3/fastvideo_windows_vsa_benchmark.json`. It pins the
hardware/software cohort, execution flags, timing boundary, stage timings, and
media probe without recording a host name, local path, GPU UUID, or token.

A completed run is evidence for that exact machine, source revision, patch,
asset revision, and dependency cohort. It is not a portable latency guarantee,
a TensorRT runtime result, or a replacement for playable-video and visual
quality review.

## Validated Windows SM121 result

The 345-frame profile was run once on an NVIDIA RTX Spark N1X (63,424 MiB,
compute capability 12.1, driver 616.67) on Windows 11 Arm64 build 28000 using
x64 Python emulation. The request used seed 0, no warmup, regional Triton VSA
compile, the `all` fusion profile, serial full-H3 VAE decode, and a replicated
single-GPU DiT. These are cold request-stage measurements; model downloads and
pre-request generator construction occurred before the measured request.

| Boundary | Seconds |
| --- | ---: |
| Conditioning stage | 104.646 |
| Denoising stage | 987.096 |
| Full H3 video-decoding stage | 252.946 |
| Audio-decoding stage | 3.436 |
| Request wall time, including save | 1358.751 |

The output passed media validation as 345 H.264 frames at 1344x768 and 24 fps
(14.375 seconds), with a 32 kHz stereo AAC stream. Its size was 9,386,238 bytes
and its SHA-256 was
`4e48850904ce696e97ec9ed5b05f9d6d6ac2628e29018d2d34a6ddee061670e8`.

This result is a 22.65-minute measured request E2E, excluding downloads and
pre-request generator construction. Denoising alone was 16.45 minutes, so the
native 345-frame, full-VAE workload did not reach 10 minutes on this single RTX
Spark N1X. The result must not be represented as a 10-minute single-GPU
guarantee.

## Troubleshooting

- A patch hash or path failure means the source checkout is incomplete or has
  been modified. Restore the reviewed repository files; do not bypass the
  check.
- A Hugging Face download failure usually means the license has not been
  accepted or the client has not been authenticated.
- A compute-capability failure means the machine is outside this reproduction
  cohort. The helper intentionally does not label another GPU as equivalent.
- A `ptxas` version or hash failure means the pinned NVIDIA CUDA redistributable
  is missing or incomplete. Do not fall back to the older bundled assembler.
- If installation was interrupted, choose a new empty install directory. The
  installer does not destructively repair a partial environment.
- Keep generated videos and summaries outside the source checkout so they
  cannot be included accidentally in a commit.

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Short drama (`trtmc_short_drama`)

Turn a Chinese novel into vertical (9:16) short-drama episodes with sound and subtitles, on
your own GPU. Every model step runs on TensorRT Model Connect through `trtmc-server`:

| Step | What it does | Model (tier A) |
| --- | --- | --- |
| Plan | reads every chapter, writes the cast, relationships, places, props, episodes and a shot list per episode | `Qwen/Qwen3.6-27B` |
| References | draws a still of each character, place and prop for you to approve or redraw | `Qwen/Qwen-Image` |
| Shots | films each shot from its approved stills, with dialogue and sound | `FastVideo/FastVideo-FastH3-Omni-8-Step-V1` (MiniMax H3 reference-to-video) |
| Subtitles | checks every line by speech recognition and times its subtitle | `openai/whisper-large-v3-turbo` |
| Assemble | joins the shots, burns in subtitles, exports 1080x1920 MP4 | FFmpeg |

Everything the models write can be edited in the app before anything is rendered.

On one GB300, planning the bundled 9-chapter novel into 4 episodes takes about 9 minutes, a
reference still about 10 seconds, and each 5 to 9 second shot about 10 minutes (one H3 call
renders 226 frames, 9.4 s, with audio).

## Requirements

- An NVIDIA GPU and driver. Tier A needs about 140 GB of GPU memory (MiniMax H3 peaks near
  110 GB at 768x1344 and 226 frames); it was run on a GB300.
  Tier B (24 to 48 GB) is planned; see [Tiers](#tiers).
- A TRTMC build with the server, bring-your-own kernels and the H3 sparse-attention kernel:

  ```bash
  cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DTRTMC_BUILD_SERVER=ON \
    -DTRTMC_ENABLE_BYOK=ON -DTRTMC_BUILD_MINIMAX_H3_OMNIREF_VSA=ON \
    -DTRTMC_BUILD_MINIMAX_H3_OMNIREF_VSA_SM100A=ON   # Blackwell sm100/sm103 fast path
  ninja -C build trtmc-server trtmc_backend_trt trtmc_byok_tvm_ffi trtmc_minimax_h3_omniref_vsa \
    trtmc_model_minimax_h3 trtmc_model_qwen3_8 trtmc_model_qwen_image trtmc_model_whisper
  ```

- Python 3.10+ with `pip install -r apps/short_drama/requirements.txt` and the server extras
  (`pip install -e '.[serve]'`).
- FFmpeg built with libass, and a CJK font such as Noto Sans CJK SC for the subtitles.
- Disk: about 250 GB while the H3 bundles build (130 GB each), 110 GB for the 27B language
  model, a few GB for the others, plus the checkpoints.

## Run

From the repository root:

```bash
PYTHONPATH=apps/short_drama python -m trtmc_short_drama up
```

`up` picks the GPU with the most memory and the tier that fits it, builds every missing bundle
into `short-drama-data/bundles/` with `python -m tensorrt_model_connect build`, and serves the
app at <http://127.0.0.1:8700>. A second run reuses the bundles. `--dry-run` prints the plan
and the build commands without running them.

| Option | Meaning |
| --- | --- |
| `--data DIR` | projects, bundles, settings and logs (default `./short-drama-data`) |
| `--tier a\|b` | force a tier instead of choosing by GPU memory |
| `--gpu N` | GPU index as `nvidia-smi` numbers it |
| `--bundle ROLE=PATH` | use an existing bundle for `llm`, `image`, `video`, `asr` or `tts` |
| `--runtime-root DIR` | directory with `trtmc-server` and the runtime libraries (default `build/`) |
| `--host`, `--port`, `--server-port` | where the app (8700) and `trtmc-server` (8701) listen; any host other than loopback prints a URL with a one-time `?token=` that the browser needs once |

The app starts one `trtmc-server` and loads only the bundles the current step needs, so the
models never have to fit in memory together. Speech recognition stays loaded next to the video
model because it is small.

The Whisper family fixes the decoder language when the bundle is built and falls back to
English, which turns Chinese speech into English text. `up` therefore builds Whisper from a
local view of the checkpoint (`short-drama-data/checkpoints/`, weights linked, not copied) whose
generation config forces `<|zh|>`; the bundle is named `whisper-zh-sm<arch>.bundle`.

### The H3 bundle

MiniMax H3 reference-to-video is an explicit experimental build in the `minimax_h3` family
(see `families/minimax_h3/README.md`). `up` runs the family command
`python -m tensorrt_model_connect minimax_h3 build-omniref`: the original `MiniMaxAI/MiniMax-H3`
two-window bundle first (kept in `bundles/scratch/`), then the FastH3 OmniRef 8-step student
that reuses six of its eight plans. Plan for about 250 GB of
disk and 100 GB of host memory while it builds, or pass an existing bundle with
`--bundle video=PATH`.

## Use

1. **Create a project.** Paste a novel, open a text file, or use the bundled sample,
   《阿Q正传》 by Lu Xun (public domain). Set the number of episodes and their maximum length.
2. **Plan.** One click. Edit the cast (each character's `look` and `voice` feed the image and
   video prompts), places, props, episodes and every shot: place, characters, seconds, the
   line, and what happens before, during and after it.
3. **References.** Draw the stills, then approve each one or redraw it, optionally with your
   own prompt. Shots only use approved stills.
4. **Shots.** Render one shot, all missing shots, or the whole episode. The last frame of an
   episode is an extra reference for the first shot of the next one; switch it off per episode.
   Every take keeps its raw clip, a contact sheet and the exact prompt, which you can edit for
   the next take.
5. **Episodes.** Time the subtitles and assemble; play or download the MP4.

### How shots are prompted

The language model only fills short fields per shot. The app writes the H3 prompt from them:
subject definitions that bind each `<Picture N>` to a character, place or prop; an action that
runs through the whole shot; the line in the first seconds inside `<d>[Chinese] …</d>`; a light
camera move; and a second-window prompt that keeps the action going without new dialogue.
Text in double quotes is drawn on screen by H3, so the app removes quotes from descriptions.

## Customize

**Flow.** The Flow tab shows the steps, their order, the model of each step and its parameters
as JSON. Save edits as a new version (`default@2`, …) and select the version a project uses;
earlier versions never change.

**Plugins.** A step is a Python file in `apps/short_drama/plugins/` (or `--plugins DIR`) that
defines `STEP = Step(name, run, role=..., scope="project"|"episode")`. Click *Reload plugins*,
then add `{"step": "<name>", "model": ..., "params": {...}}` to a flow version.
`plugins/dialogue_polish.py` is an example that tightens every line with the language model.
Delete the file and reload to remove the step; a flow that still names it skips it with a
message.

**External chat API.** Settings can point the planning steps at any OpenAI-compatible
`/chat/completions` endpoint. The key is written to `secrets.json` (mode 600) in the data
directory and is never logged or sent back to the browser.

## Tiers

| Tier | GPU memory | Video | Speech | Status |
| --- | --- | --- | --- | --- |
| A | 140 GB or more | MiniMax H3 reference-to-video, native audio | from the video model | runs end to end |
| B | 24 to 48 GB | Wan2.2-TI2V-5B from the approved still | Magpie TTS | catalog entry only; needs a speech endpoint in `trtmc-server` |

## Data layout

```text
short-drama-data/
  bundles/                 built bundles, named <model>-sm<arch>.bundle
  flows/<name>/v<N>.json   flow versions
  projects/<id>/
    novel.txt  plan.json  stills.json  stills/
    episodes/<NN>/raw/ clips/ sheets/ shots.json episode.srt episode.mp4 last_frame.png
  logs/trtmc-server.log
```

## Tests

```bash
CUDA_VISIBLE_DEVICES= python -m pytest apps/short_drama/tests
```

The tests use fake models; the FFmpeg test is skipped when FFmpeg is not installed.
